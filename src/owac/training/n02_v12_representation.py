"""Bounded RAW-only representation repair, before any π0.5 backend adaptation.

Local VICReg on exactly matched crop coordinates plus normalized RAW patch
prediction. The frozen P0 episode split supplies frames; no action labels,
RGB, RGB teacher or hidden simulator state enters this training objective.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.model_v12 import build_n02_v12_model
from owac.frameworks.n02_full.representation_v12 import RawPatchHead, local_vicreg
from owac.frameworks.n02_full.state import ObservationBundle
from owac.runtime.n02_v12_status import append_history, update_status
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12 import build_source

FORMAT = "n02-raw-representation-r8-v1"


def frame_pool(config, partition, *, per_episode=12):
    source = build_source(config, partition, history=8, batch_size=1)
    groups = defaultdict(list)
    for index in range(source.anchor_count):
        anchor = source.anchor(index)
        groups[anchor.relative_path].append(anchor.current_frame_index)
    frames, identities = [], []
    for path, indices in sorted(groups.items()):
        indices = sorted(set(indices))
        selected = [
            indices[i]
            for i in np.linspace(0, len(indices) - 1, min(per_episode, len(indices)), dtype=int)
        ]
        with np.load(Path(config["data"]["data_root"]) / path, allow_pickle=False) as data:
            cameras = [
                data[name + "_camera_raw_float32"][selected] for name in ("head", "left", "right")
            ]
            values = np.stack(cameras, axis=1)
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("RAW frame pool must preserve finite float32 loader units")
        frames.extend(values)
        identities.extend({"partition": partition, "path": path, "frame": int(i)} for i in selected)
    tensor = torch.from_numpy(np.stack(frames)).permute(0, 1, 4, 2, 3).contiguous()
    if tensor.shape[1:] != (3, 3, 240, 320):
        raise ValueError(f"unexpected RAW pool layout: {tuple(tensor.shape)}")
    return tensor, identities, source.identity()


def bundle(images, cameras):
    # B,V,C,H,W. Validity refers only to actual pixels; encoder supplies padding.
    return ObservationBundle(
        frames={
            camera: MeasurementFrame(
                values=images[:, i],
                camera_id=camera,
                valid_mask=torch.ones(
                    (len(images), *images.shape[-2:]), dtype=torch.bool, device=images.device
                ),
            )
            for i, camera in enumerate(cameras)
        },
        context=EmbodimentContext(),
    )


def views(images, generator):
    batch = len(images)
    crops, origins = [], []
    for view in range(2):
        current, positions = [], []
        for image in images:
            y = int(torch.randint(4, (), generator=generator)) * 16
            x = int(torch.randint(5, (), generator=generator)) * 16
            flipped = view == 1 and bool(torch.randint(2, (), generator=generator))
            if view == 1 and (y, x, flipped) == origins[0][len(current)]:
                x = (x + 16) % 80
            crop = image[..., y : y + 192, x : x + 256].clone()
            if flipped:
                crop = crop.flip(-1)
            gain = 2 ** float(torch.empty(()).uniform_(-0.25, 0.25, generator=generator))
            current.append(crop * gain)
            positions.append((y, x, flipped))
        crops.append(torch.stack(current))
        origins.append(positions)
    return torch.cat(crops), origins, batch


def matching(origins, index, *, height=12, width=16):
    grids = []
    for view in range(2):
        y, x, flipped = origins[view][index]
        positions = {}
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                source_col = width - 1 - col if flipped else col
                positions[(y + row * 16 + 8, x + source_col * 16 + 8)] = row * width + col
        grids.append(positions)
    common = sorted(grids[0].keys() & grids[1].keys())
    return [grids[0][p] for p in common], [grids[1][p] for p in common]


def deep_maps(spatial, cameras, height, width):
    return torch.stack(
        [spatial.levels[c + "::stride16"][..., : height // 16, : width // 16] for c in cameras],
        dim=1,
    )


def paired_embeddings(maps, origins, batch):
    left, right = [], []
    flat = maps.flatten(-2).transpose(-1, -2)
    for row in range(batch):
        i, j = matching(origins, row)
        for camera in range(3):
            left.append(flat[row, camera, i])
            right.append(flat[row + batch, camera, j])
    x, y = torch.cat(left).float(), torch.cat(right).float()
    return F.layer_norm(x, (x.shape[-1],)), F.layer_norm(y, (y.shape[-1],))


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.inference_mode()
def evaluate(model, head, pool, cameras, device, *, indices=None):
    model.eval()
    head.eval()
    if indices is None:
        indices = np.linspace(0, len(pool) - 1, min(6, len(pool)), dtype=int)
    rows = []
    for index in indices:
        raw = pool[index : index + 1].to(device)
        spatial = model.inference.encoder(bundle(raw, cameras))
        flipped = model.inference.encoder(bundle(raw.flip(-1), cameras))
        absent = model.inference.encoder(bundle(torch.zeros_like(raw), cameras))
        maps = deep_maps(spatial, cameras, 240, 320).flatten(0, 1).float()
        zero = deep_maps(absent, cameras, 240, 320).flatten(0, 1).float()
        normalized = F.layer_norm(maps.permute(0, 2, 3, 1), (maps.shape[1],)).permute(0, 3, 1, 2)
        normalized_zero = F.layer_norm(zero.permute(0, 2, 3, 1), (zero.shape[1],)).permute(
            0, 3, 1, 2
        )
        actual_loss = float(head(normalized, raw.flatten(0, 1)))
        blank_loss = float(head(normalized_zero, raw.flatten(0, 1)))
        reference = spatial.levels[cameras[0] + "::stride16"][..., :15, :20].float()
        changed = flipped.levels[cameras[0] + "::stride16"][..., :15, :20].float()
        tokens = model.inference.readout.spatial_output_tokens(spatial).float()
        flip_tokens = model.inference.readout.spatial_output_tokens(flipped).float()
        crops, origins, batch = views(raw.cpu(), torch.Generator().manual_seed(991 + int(index)))
        paired = model.inference.encoder(bundle(crops.to(device), cameras))
        pm = deep_maps(paired, cameras, 192, 256).float()
        x, y = paired_embeddings(pm, origins, batch)
        sim = F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T
        # Match only within each camera, avoiding false ambiguity between cameras.
        n = len(matching(origins, 0)[0])
        correct = []
        for camera in range(3):
            block = sim[camera * n : (camera + 1) * n, camera * n : (camera + 1) * n]
            correct.append(
                float((block.argmax(-1) == torch.arange(n, device=device)).float().mean())
            )
        rows.append(
            {
                "pool_index": int(index),
                "patch_loss": actual_loss,
                "blank_patch_loss": blank_loss,
                "deep_flip_relative_l2": float(
                    (reference - changed).norm() / reference.norm().clamp_min(1e-12)
                ),
                "fused_flip_relative_l2": float(
                    (tokens - flip_tokens).norm() / tokens.norm().clamp_min(1e-12)
                ),
                "crop_match_accuracy": float(np.mean(correct)),
                "random_match_accuracy": 1 / n,
            }
        )
    report = {
        "rows": rows,
        **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "pool_index"},
    }
    report["representation_gate_passed"] = (
        report["deep_flip_relative_l2"] >= 0.05
        and report["fused_flip_relative_l2"] >= 0.1
        and report["crop_match_accuracy"] >= 0.2
        and report["patch_loss"] <= 0.9 * report["blank_patch_loss"]
    )
    model.train()
    head.train()
    return report


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    config["model"]["scale_balance"] = True
    torch.set_num_threads(4)
    torch.manual_seed(1103)
    np.random.seed(1103)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    update_status(
        root,
        run_id=root.name,
        stage="P2",
        state="initialising",
        step=0,
        target_steps=args.updates,
        activity="构建 RAW 表征修复数据",
        metric_label="RAW局部重建误差",
        metric_kind="raw_representation",
        gpu=[int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])],
    )
    train, identities, train_identity = frame_pool(config, "train")
    dev, dev_ids, dev_identity = frame_pool(config, "dev", per_episode=4)
    train_paths = {r["path"] for r in identities}
    dev_paths = {r["path"] for r in dev_ids}
    assert not train_paths & dev_paths
    write_json(
        root / "data_identity.json",
        {
            "train": train_identity,
            "dev": dev_identity,
            "train_frames": identities,
            "dev_frames": dev_ids,
            "train_pool_shape": list(train.shape),
            "dev_pool_shape": list(dev.shape),
        },
    )
    model = build_n02_v12_model(config["model"]).to(device)
    encoder = model.inference.encoder
    initial = copy.deepcopy(encoder.state_dict())
    payload = torch.load(args.initial, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if args.mode == "fresh":
        encoder.load_state_dict(initial)
    else:
        current = encoder.state_dict()
        reset = ("backbone.stages.2.", "backbone.downsample_layers.2.", "proj.stride16.")
        current.update({k: v for k, v in initial.items() if k.startswith(reset)})
        encoder.load_state_dict(current)
    del initial, payload
    gc.collect()
    encoder.batch_cameras = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad_(
            not name.startswith(("backbone.stages.3.", "backbone.downsample_layers.3."))
        )
    head = RawPatchHead().to(device)
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters, "name": "encoder"},
            {"params": head.parameters(), "name": "raw_head"},
        ],
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        fused=True,
    )
    cameras = tuple(config["model"]["camera_ids"])
    model.train()
    head.train()
    baseline = evaluate(model, head, dev, cameras, device)
    write_json(root / "baseline.json", baseline)
    append_history(
        root, step=0, wall_time=0, train_loss=None, dev_representation_error=baseline["patch_loss"]
    )
    generator = torch.Generator().manual_seed(1103)
    timings = []
    last = baseline
    for step in range(1, args.updates + 1):
        start = time.perf_counter()
        ids = torch.randint(len(train), (args.batch,), generator=generator)
        raw = train[ids]
        crops, origins, batch = views(raw, generator)
        crops = crops.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            spatial = encoder(bundle(crops, cameras))
            maps = deep_maps(spatial, cameras, 192, 256)
        x, y = paired_embeddings(maps, origins, batch)
        vicreg, terms = local_vicreg(x, y)
        flattened = maps.flatten(0, 1).float()
        norm = F.layer_norm(flattened.permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2)
        reconstruction = head(norm, crops.flatten(0, 1))
        loss = vicreg + reconstruction
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite representation loss")
        loss.backward()
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], 1, error_if_nonfinite=True)
            group["lr"] = (
                1e-4
                * min(1, step / 20)
                * (0.25 + 0.75 * (1 + math.cos(math.pi * step / args.updates)) / 2)
            )
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        timings.append(seconds)
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "patch_loss": float(reconstruction.detach()),
            "vicreg": {k: float(v.detach()) for k, v in terms.items()},
            "step_seconds": seconds,
            "pool_indices": ids.tolist(),
        }
        with (root / "optimization.jsonl").open("a") as h:
            h.write(json.dumps(row) + "\n")
        if step % 50 == 0 or step == args.updates:
            last = evaluate(model, head, dev, cameras, device)
            write_json(root / f"dev_{step:07d}.json", last)
            torch.save(
                {
                    "format": FORMAT,
                    "step": step,
                    "encoder": cpu_state(encoder),
                    "raw_head": cpu_state(head),
                    "config": config,
                    "mode": args.mode,
                    "optimizer": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "data_identity": {"train": train_identity, "dev": dev_identity},
                    "dev": last,
                },
                root / f"checkpoint_{step:07d}.pt",
            )
            append_history(
                root,
                step=step,
                wall_time=float(sum(timings)),
                dev_representation_error=last["patch_loss"],
                train_loss=float(loss),
            )
            print(
                json.dumps(
                    {"step": step, "dev": last, "seconds_per_update": float(np.mean(timings))}
                ),
                flush=True,
            )
        update_status(
            root,
            run_id=root.name,
            stage="P2",
            state="running",
            step=step,
            target_steps=args.updates,
            activity="恢复 RAW 深层表征，π0.5 后端未训练",
            loss=float(loss),
            step_seconds=float(np.mean(timings[-20:])),
            dev_representation_error=last["patch_loss"],
            representation_metrics=last,
            metric_label="RAW局部重建误差",
            metric_kind="raw_representation",
            effective_batch_size=args.batch,
        )
    write_json(
        root / "result.json",
        {
            "mode": args.mode,
            "optimizer_updates": args.updates,
            "backend_updates": 0,
            "baseline": baseline,
            "final": last,
            "mean_step_seconds": float(np.mean(timings)),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "initial_sha256": hashlib.sha256(Path(args.initial).read_bytes()).hexdigest(),
        },
    )
    update_status(
        root, state="completed", activity="表征修复完成；等待验收后适配后端", step=args.updates
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--initial", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("reset-deep", "fresh"), required=True)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--batch", type=int, default=4)
    main(p.parse_args())
