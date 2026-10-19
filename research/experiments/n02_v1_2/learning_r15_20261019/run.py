"""Bounded budget/data/modality controls on the existing frozen spatial encoder."""

import argparse
import fcntl
import gc
import json
import runpy
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from owac.evaluation.n02_full.action_learning import (
    cpu_state,
    fit_action_head,
    full_targets,
    sample_order,
    selected_splits,
)
from owac.frameworks.n02_full.task_chain import (
    ResidualTokenAdapter,
    SpatialTaskEncoder,
    action_metrics,
    state_digest,
    visual_tokens,
)
from owac.training.n02_v12 import anchor_action_targets, build_source
from owac.training.n02_v12_task import source_records

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/n02-v12/learning-r15"
R14 = EVIDENCE.parent / "task_chain_r14_20261019"
R14OUT = ROOT / "outputs/n02-v12/task-chain-r14"
HELPERS = runpy.run_path(str(R14 / "run.py"))
CONFIG = json.loads(HELPERS["CONFIG"].read_text())
BUDGETS = (256, 512, 1024)
SEEDS = (15019, 15020)
FLOW_SEEDS = (188390828, 188390878)
write = HELPERS["write"]
torch_save = HELPERS["torch_save"]


def status(name, **fields):
    path = OUT / name / "status.json"
    previous = json.loads(path.read_text()) if path.exists() else dict(run_id=name)
    previous.update(fields, updated_at=time.time())
    write(path, previous)
    if fields.get("step", 0) % 64 == 0:
        print(json.dumps(previous), flush=True)


def publish():
    with (OUT / "publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reports = []
        for path in sorted(OUT.glob("*/report.json")):
            r = json.loads(path.read_text())
            if r.get("completed"):
                reports.append(
                    {
                        k: r[k]
                        for k in ("run_id", "modality", "demos", "seed", "actual_updates", "rows")
                    }
                )
        write(
            OUT / "progress.json",
            dict(reports=reports, planned_runs=14, completed_runs=len(reports), audited=False),
        )


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    status("prepare", state="running", activity="dataset", step=0)
    data = torch.load(R14OUT / "dataset.pt", weights_only=False)
    source = build_source(CONFIG, "train", history=1, batch_size=8)
    rows = source_records(source)
    normal = sorted({r["path"] for r in rows if r["light"] == "Normal"})
    old = json.loads((R14 / "selection.json").read_text())
    val = old["paths"]["val"]
    small = old["paths"]["train"]
    more = [p for p in normal if p not in val]
    if len(small) != 3 or len(more) != 5 or not set(small) < set(more):
        raise ValueError("predeclared comparison requires3 versus5 Normal train episodes")
    raw, states, actions, masks = (
        [x for x in data[k]] for k in ("raw", "state", "actions", "mask")
    )
    for path in sorted(set(more) - set(small)):
        episode = [r for r in rows if r["path"] == path and r["valid"] >= 10]
        stop = episode[0]["frames"] - 11
        times = sorted(set([0, 10, *np.linspace(0, stop, 10).astype(int).tolist()]))
        indices = sorted(
            {min(episode, key=lambda r: (abs(r["t"] - t), r["index"]))["index"] for t in times}
        )
        for index in indices:
            row = next(r for r in episode if r["index"] == index)
            batch = source.window(index, index + 1)
            y, mask = anchor_action_targets(source.targets(index, index + 1), 1)
            raw.append(
                np.stack(
                    [np.asarray(batch.observation.frames[c].values)[0] for c in data["cameras"]]
                )
            )
            states.append(np.asarray(batch.observation.context.proprio)[0])
            actions.append(y[0])
            masks.append(mask[0])
            data["records"].append(dict(**row, group=path, split="train", source_partition="train"))
    for key, value in zip(
        ("raw", "state", "actions", "mask"), (raw, states, actions, masks), strict=True
    ):
        data[key] = np.asarray(value)
    groups = {"3": small, "5": more}
    split3, split5 = (selected_splits(data["records"], groups[k]) for k in ("3", "5"))
    if split3["val"] != split5["val"] or split3["test"] != split5["test"]:
        raise ValueError("data quantity changed holdout")
    # Audit paired reference trajectory arrays and frame IDs before reading RGB features.
    rgb = np.empty((len(data["records"]), 3, 240, 320, 3), dtype=np.uint8)
    pairs = []
    for path in sorted({r["path"] for r in data["records"]}):
        original = Path(CONFIG["data"]["data_root"]) / path
        paired = Path(str(original).replace("/raw/", "/rgb/"))
        with np.load(original, allow_pickle=False) as a, np.load(paired, allow_pickle=False) as b:
            for key in ("actions", "robot_states", "left_endpose", "right_endpose"):
                np.testing.assert_array_equal(a[key], b[key])
            meta = json.loads(str(b["metadata_json"].item()))
            if meta.get("clean_protocol") != "original_lighting_joint_action_replay":
                raise ValueError("undeclared RGB reference generation")
            selected = [(i, r["t"]) for i, r in enumerate(data["records"]) if r["path"] == path]
            for c, camera in enumerate(("head", "left", "right")):
                images = b[camera + "_camera_rgb_uint8"]
                if len(images) != len(a["actions"]):
                    raise ValueError("paired frame count differs")
                for i, t in selected:
                    rgb[i, c] = images[t]
            pairs.append(
                dict(
                    raw=str(original),
                    rgb=str(paired),
                    action_state_ee_exact=True,
                    frames=len(a["actions"]),
                    rgb_protocol=meta["clean_protocol"],
                )
            )
    torch_save(OUT / "dataset.pt", data)
    torch_save(OUT / "rgb.pt", torch.from_numpy(rgb))
    write(
        EVIDENCE / "selection.json",
        dict(
            train_groups=groups,
            val_groups=val,
            test_groups=old["paths"]["test"],
            records=data["records"],
            counts={
                k: {s: len(i) for s, i in selected_splits(data["records"], g).items()}
                for k, g in groups.items()
            },
            test_lights=sorted({data["records"][i]["light"] for i in split3["test"]}),
            paired_rgb_audit=pairs,
            runtime_privileged_position_inputs=False,
            warning="Historical development set; RGB is clean original-lighting replay; "
            "reference gap does not isolate RAW format, architecture or pretraining.",
        ),
    )
    status("prepare", state="running", activity="RAW feature extraction", step=0)
    policy = HELPERS["load_policy"](OUT / "raw-loader")
    trainer = policy.trainer
    spatial = SpatialTaskEncoder(trainer.model.inference.encoder, trainer.model.inference.readout)
    spatial.set_joint_training(False)
    before = state_digest(spatial)
    expected = json.loads((R14 / "extraction.json").read_text())["spatial_sha_before"]
    if before != expected:
        raise ValueError("frozen RAW producer changed")
    base = torch.load(R14OUT / "tokens.pt", weights_only=True)
    values = [base]
    with torch.no_grad():
        for i in range(len(base), len(data["records"]), 8):
            x = spatial(
                HELPERS["observation"](
                    data["raw"][i : i + 8], data["state"][i : i + 8], data["cameras"]
                )
            )
            values.append(x.cpu())
    torch_save(OUT / "raw_tokens.pt", torch.cat(values))
    after = state_digest(spatial)
    if before != after:
        raise ValueError("RAW weights mutated")
    del values, base, spatial, trainer, policy
    gc.collect()
    torch.cuda.empty_cache()
    status("prepare", state="running", activity="native visual extraction", step=0)
    native_receipt = extract_native(rgb)
    write(
        EVIDENCE / "extraction.json",
        dict(
            completed=True,
            actual_updates=0,
            raw_sha_before=before,
            raw_sha_after=after,
            r14_tokens_reused_exact=True,
            raw_shape=[len(data["records"]), 768, 2048],
            native=native_receipt,
        ),
    )
    status("prepare", state="completed", activity="ready", step=0)


def extract_native(rgb):
    from safetensors import safe_open
    from starVLA.model.framework.VLM4A.PI0 import PI0
    from starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip import (
        SiglipVisionModel,
    )
    from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIMultiModalProjector
    from transformers.models.auto import CONFIG_MAPPING

    config = CONFIG_MAPPING["paligemma"]()
    config.vision_config.intermediate_size = 4304
    config.vision_config.projection_dim = 2048
    config.vision_config.projector_hidden_act = "gelu_fast"
    vision = SiglipVisionModel(config.vision_config).eval().requires_grad_(False)
    projector = OpenPIMultiModalProjector(config).eval().requires_grad_(False)
    path = Path("/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        for module, prefix in (
            (vision, "model.paligemma_with_expert.paligemma.model.vision_tower."),
            (projector, "model.paligemma_with_expert.paligemma.model.multi_modal_projector."),
        ):
            module.load_state_dict(
                {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)},
                strict=True,
            )
    vision, projector = vision.bfloat16().cuda(), projector.bfloat16().cuda()
    vision.vision_model.config._attn_implementation = "eager"
    before = (state_digest(vision), state_digest(projector))
    processor = SimpleNamespace(model_name="PI05", image_resolution=(224, 224))
    features = []
    with torch.inference_mode():
        for i in range(0, len(rgb), 8):
            images = torch.stack(
                [
                    PI0._image_to_tensor(processor, image)
                    for image in rgb[i : i + 8].reshape(-1, 240, 320, 3)
                ]
            )
            x = projector(vision(images.cuda().bfloat16()).last_hidden_state)
            features.append(x.reshape(-1, 768, 2048).float().cpu())
            status(
                "prepare",
                activity="native visual extraction",
                step=min(i + 8, len(rgb)),
                max_updates=len(rgb),
            )
    torch_save(OUT / "native_tokens.pt", torch.cat(features))
    after = (state_digest(vision), state_digest(projector))
    if before != after:
        raise ValueError("native encoder mutated")
    return dict(
        sha_before=before,
        sha_after=after,
        weights=str(path),
        strict_load=True,
        preprocessing="official PI0._image_to_tensor letterbox224; no8x8 pooling",
        shape=[len(rgb), 768, 2048],
        raw_teacher=False,
        disclosure="pretrained native SigLIP plus projector; clean original-lighting replay RGB",
    )


def heads():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = torch.load(OUT / "dataset.pt", weights_only=False)
    selection = json.loads((EVIDENCE / "selection.json").read_text())
    std = json.loads(Path(CONFIG["data"]["stats_path"]).read_text())["norm_stats"]["actions"]["std"]
    caches = {
        k: torch.load(OUT / (k + "_tokens.pt"), weights_only=True).cuda() for k in ("raw", "native")
    }
    for demos in (3, 5):
        for seed in SEEDS:
            for modality in ("state", "raw", "native"):
                name = f"{modality}-{demos}-s{seed}"
                directory = OUT / name
                directory.mkdir(exist_ok=False)
                status(
                    name,
                    state="running",
                    activity="small full-action head",
                    step=0,
                    modality=modality,
                    demos=demos,
                    seed=seed,
                    max_updates=1024,
                )

                def callback(row, name=name, directory=directory):
                    if row["step"] % 16 == 0:
                        status(
                            name, state="running", **{k: v for k, v in row.items() if k != "val"}
                        )
                    with (directory / "history.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")

                def checkpoint(budget, payload, directory=directory):
                    torch_save(directory / f"checkpoint_{budget:04d}.pt", payload)

                report, predictions = fit_action_head(
                    caches["native" if modality == "native" else "raw"],
                    data["state"],
                    data["actions"],
                    data["mask"],
                    data["records"],
                    selection["train_groups"][str(demos)],
                    std,
                    seed=seed,
                    state_only=modality == "state",
                    callback=callback,
                    checkpoint=checkpoint,
                )
                report.update(
                    run_id=name,
                    modality=modality,
                    demos=demos,
                    completed=True,
                    feature_producer="frozen",
                    closed_loops=0,
                )
                write(directory / "report.json", report)
                small_report = {k: v for k, v in report.items() if k not in ("order",)}
                write(EVIDENCE / (name + "_report.json"), small_report)
                write(directory / "sample_order.json", report["order"])
                np.savez(directory / "predictions.npz", **predictions)
                status(
                    name,
                    state="completed",
                    step=1024,
                    actual_updates=1024,
                    seconds_per_update=report["elapsed_s"] / 1024,
                    selected_val_mse=report["rows"][-1]["selected_validation_mse"],
                )
                publish()
    del caches
    gc.collect()
    torch.cuda.empty_cache()


def pi(demos):
    from owac.backends.low_rank import adapter_state, restore_adapters

    name = f"pi-{demos}-s{SEEDS[0]}"
    directory = OUT / name
    directory.mkdir(exist_ok=False)
    status(
        name,
        state="initialising",
        activity="pi frozen features",
        step=0,
        modality="pi",
        demos=demos,
        seed=SEEDS[0],
        max_updates=1024,
    )
    data = torch.load(OUT / "dataset.pt", weights_only=False)
    selection = json.loads((EVIDENCE / "selection.json").read_text())
    splits = selected_splits(data["records"], selection["train_groups"][str(demos)])
    policy = HELPERS["load_policy"](directory / "loader")
    trainer, device = policy.trainer, policy.trainer.device
    spatial = SpatialTaskEncoder(trainer.model.inference.encoder, trainer.model.inference.readout)
    spatial.set_joint_training(False)
    before = state_digest(spatial)
    if before != json.loads((EVIDENCE / "extraction.json").read_text())["raw_sha_before"]:
        raise ValueError("pi RAW representation changed")
    x = torch.load(OUT / "raw_tokens.pt", weights_only=True).to(device)
    states = torch.as_tensor(data["state"], device=device)
    target = full_targets(torch.as_tensor(data["actions"], device=device))
    mask = torch.as_tensor(data["mask"], device=device)
    lang, lang_mask, state = trainer.language(
        ["Grab the light brown cylindrical roller on the table."] * len(states), data["state"]
    )
    torch.manual_seed(SEEDS[0])
    adapter = ResidualTokenAdapter().to(device)
    parameters = list(adapter.parameters()) + [
        p for a in trainer.backend_adapters.values() for p in a.parameters()
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-3, fused=True)
    order = sample_order(splits["train"], 1024, SEEDS[0])
    write(directory / "sample_order.json", order)
    std = policy.stats["actions"]["std"]

    def saved():
        return dict(adapter=cpu_state(adapter), backend=adapter_state(trainer.backend_adapters))

    def restore(payload):
        adapter.load_state_dict(payload["adapter"], strict=True)
        restore_adapters(trainer.backend_adapters, payload["backend"])

    @torch.no_grad()
    def evaluate(split, seeds):
        indices = splits[split]
        predictions = []
        for seed in seeds:
            chunks = []
            for i in range(0, len(indices), 8):
                batch = indices[i : i + 8]
                noise = torch.randn(
                    (len(batch), 50, 32),
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(seed),
                )
                p = trainer.bridge.sample_actions_from_visual_tokens(
                    visual_tokens(adapter(x[batch]), data["cameras"]),
                    lang[batch],
                    lang_mask[batch],
                    state[batch],
                    noise=noise,
                    num_steps=10,
                )[..., :14]
                chunks.append(p.float().cpu().numpy())
            predictions.append(np.concatenate(chunks))
        per_seed = [
            action_metrics(p, data["actions"][indices], data["mask"][indices], std)
            for p in predictions
        ]
        avg = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
        return avg, np.stack(predictions)

    best_state = saved()
    val, _ = evaluate("val", FLOW_SEEDS[:1])
    best = val["normalized_control_mse"]
    best_step = 0
    history = [dict(step=0, val=val)]
    start = time.perf_counter()
    for step, indices in enumerate(order, 1):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device=device).manual_seed(SEEDS[0] + step)
        noise = torch.randn(target[indices].shape, device=device, generator=generator)
        flow_time = (
            torch.rand((len(indices),), device=device, generator=generator).pow(1 / 1.5) * 0.999
            + 0.001
        )
        loss = trainer.bridge.forward_from_visual_tokens(
            visual_tokens(adapter(x[indices]), data["cameras"]),
            lang[indices],
            lang_mask[indices],
            state[indices],
            target[indices],
            noise=noise,
            time=flow_time,
            action_mask=mask[indices],
        )["action_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite pi loss")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not np.isfinite(grad):
            raise FloatingPointError("nonfinite pi gradients")
        optimizer.step()
        torch.cuda.synchronize()
        row = dict(
            step=step,
            loss=float(loss.detach()),
            gradient_norm=grad,
            seconds_per_update=(time.perf_counter() - start) / step,
        )
        if step % 64 == 0:
            val, _ = evaluate("val", FLOW_SEEDS[:1])
            row["val"] = val
            history.append(row.copy())
            if val["normalized_control_mse"] < best:
                best, best_step, best_state = val["normalized_control_mse"], step, saved()
        if step in BUDGETS:
            torch_save(
                directory / f"checkpoint_{step:04d}.pt",
                dict(
                    selected=best_state,
                    last=saved(),
                    selected_step=best_step,
                    selected_validation_mse=best,
                ),
            )
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if step % 16 == 0:
            status(
                name,
                state="running",
                **{k: v for k, v in row.items() if k != "val"},
                selected_step=best_step,
                selected_val_mse=best,
                peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
    report = dict(
        run_id=name,
        modality="pi",
        demos=demos,
        seed=SEEDS[0],
        actual_updates=1024,
        parameters=sum(p.numel() for p in parameters),
        train_indices=splits["train"],
        history=history,
        elapsed_s=time.perf_counter() - start,
        rows=[],
        raw_sha_before=before,
        raw_sha_after=state_digest(spatial),
        closed_loops=0,
    )
    predictions = {}
    for budget in BUDGETS:
        payload = torch.load(
            directory / f"checkpoint_{budget:04d}.pt", map_location=device, weights_only=False
        )
        row = dict(
            budget=budget,
            selected_step=payload["selected_step"],
            selected_validation_mse=payload["selected_validation_mse"],
        )
        for version in ("selected", "last"):
            restore(payload[version])
            metrics = {}
            for split in ("train", "val", "test"):
                metrics[split], p = evaluate(split, FLOW_SEEDS)
                predictions[f"{budget}_{version}_{split}"] = p
            row[version] = metrics
        report["rows"].append(row)
    if report["raw_sha_before"] != report["raw_sha_after"]:
        raise ValueError("frozen visual model mutated")
    np.savez(directory / "predictions.npz", **predictions)
    report["completed"] = True
    write(directory / "report.json", report)
    write(EVIDENCE / (name + "_report.json"), report)
    status(
        name,
        state="completed",
        step=1024,
        actual_updates=1024,
        seconds_per_update=report["elapsed_s"] / 1024,
        selected_step=best_step,
        selected_val_mse=best,
    )
    publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "heads-pi5", "pi3"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "heads-pi5":
            heads()
            pi(5)
        else:
            pi(3)
    except Exception as error:
        write(
            OUT / (args.command + "-error.json"),
            dict(
                state="failed",
                command=args.command,
                failure_type=type(error).__name__,
                updated_at=time.time(),
            ),
        )
        raise
