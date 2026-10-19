"""Bounded paired spatial-only action fit and joint adaptation experiment."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n02_full.state import EmbodimentContext, ObservationBundle
from owac.frameworks.n02_full.task_chain import (
    ResidualTokenAdapter,
    SmallActionHead,
    SpatialTaskEncoder,
    action_metrics,
    masked_action_loss,
    split_groups,
    state_digest,
    visual_tokens,
)
from owac.runtime.n02_v12_robotwin import Policy
from owac.training.n02_v12 import anchor_action_targets, build_source
from owac.training.n02_v12_task import source_records

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/n02-v12/task-chain-r14"
CONFIG = EVIDENCE.parent / "training_r11_20261017/task_config.json"
CHECKPOINT = ROOT / "outputs/n02-v12/P2/raw-task-r11-task-s1103-a1/checkpoint_0000850.pt"
SEED = 14019
FLOW_SEEDS = (188390828, 188390878)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def torch_save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def observation(raw, state, cameras):
    """Current measured image/state only; no time, phase, group or privileged labels."""
    batch = len(raw)
    frames = {}
    for c, camera in enumerate(cameras):
        value = raw[:, c]
        valid = np.isfinite(value).all(1)
        frames[camera] = MeasurementFrame(
            values=value,
            valid_mask=valid,
            bad_mask=~valid,
            saturated_mask=np.zeros(valid.shape, dtype=bool),
            camera_id=camera,
            episode_ids=tuple("diagnostic" for _ in range(batch)),
            frame_ids=tuple("current" for _ in range(batch)),
        )
    return ObservationBundle(
        frames=frames,
        context=EmbodimentContext(proprio=state, proprio_available=True),
        episode_ids=tuple("diagnostic" for _ in range(batch)),
        anchor_ids=tuple("current" for _ in range(batch)),
    )


def load_policy(destination):
    policy = Policy(CONFIG, CHECKPOINT, destination)
    trainer = policy.trainer
    # The old optimizer is never used/resumed; free its moments before diagnostics.
    trainer.optimizer = None
    trainer.masked.requires_grad_(False)
    trainer.model.requires_grad_(False).eval()
    gc.collect()
    torch.cuda.empty_cache()
    return policy


def prepare():
    if (OUT / "dataset.pt").exists():
        raise FileExistsError("preserve existing r14 dataset")
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    sources = {s: build_source(config, s, history=1, batch_size=8) for s in ("train", "dev")}
    records = {s: source_records(v) for s, v in sources.items()}
    normal = sorted({r["path"] for r in records["train"] if r["light"] == "Normal"})
    if len(normal) < 4 or any("episode_000000." in p for p in normal):
        raise ValueError("need four distinct Normal training episodes excluding seed0")
    # Selection is lexicographic before any fit or outcome is measured.
    paths = {"train": normal[:3], "val": normal[3:4]}
    paths["test"] = sorted({r["path"] for r in records["dev"]})
    selected = []
    for split, group_paths in paths.items():
        partition = "dev" if split == "test" else "train"
        for group in group_paths:
            rows = [r for r in records[partition] if r["path"] == group and r["valid"] >= 10]
            stop = rows[0]["frames"] - 11
            times = sorted(set([0, 10, *np.linspace(0, stop, 10).astype(int).tolist()]))
            chosen = {
                min(rows, key=lambda r: (abs(r["t"] - t), r["index"]))["index"] for t in times
            }
            for index in sorted(chosen):
                row = next(r for r in rows if r["index"] == index)
                selected.append(dict(**row, group=group, split=split, source_partition=partition))
    split_groups(selected)
    raw, states, actions, masks = [], [], [], []
    cameras = tuple(config["model"]["camera_ids"])
    for row in selected:
        source, index = sources[row["source_partition"]], row["index"]
        batch = source.window(index, index + 1)
        y, valid = anchor_action_targets(source.targets(index, index + 1), 1)
        raw.append(np.stack([np.asarray(batch.observation.frames[c].values)[0] for c in cameras]))
        states.append(np.asarray(batch.observation.context.proprio)[0])
        actions.append(y[0])
        masks.append(valid[0])
    dataset = dict(
        raw=np.asarray(raw, dtype=np.float32),
        state=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool),
        records=selected,
        cameras=cameras,
    )
    # Verify single-current-frame extraction against the production causal history.
    history_source = build_source(config, "train", history=8, batch_size=8)
    row = selected[0]
    original = history_source.window(row["index"], row["index"] + 1)
    for c, camera in enumerate(cameras):
        np.testing.assert_array_equal(
            dataset["raw"][0, c], np.asarray(original.observation.frames[camera].values)[-1]
        )
    np.testing.assert_array_equal(dataset["state"][0], original.observation.context.proprio[-1])
    y, valid = anchor_action_targets(history_source.targets(row["index"], row["index"] + 1), 1)
    np.testing.assert_array_equal(dataset["actions"][0], y[0])
    np.testing.assert_array_equal(dataset["mask"][0], valid[0])
    torch_save(OUT / "dataset.pt", dataset)
    write(
        EVIDENCE / "selection.json",
        dict(
            paths=paths,
            records=selected,
            counts={s: len(v) for s, v in split_groups(selected).items()},
            production_history_current_frame_action_state_exact=True,
            disclosure="Parent RAW850 saw train groups; current splits are group-isolated. "
            "Old dev episodes are historical development data, not a new untouched benchmark.",
        ),
    )
    policy = load_policy(OUT / "extract-loader")
    trainer = policy.trainer
    encoder = SpatialTaskEncoder(trainer.model.inference.encoder, trainer.model.inference.readout)
    encoder.set_joint_training(False)
    before = state_digest(encoder)
    tokens = []
    parity = None
    with torch.no_grad():
        # Source-produced and standalone runtime observations must give identical tokens.
        ref = encoder(trainer.anchor_bundle(original).observation)
        stand = encoder(observation(dataset["raw"][:1], dataset["state"][:1], cameras))
        parity = float((ref - stand).abs().max())
        if parity != 0:
            raise ValueError("standalone observation changes spatial tokens")
        for i in range(0, len(selected), 8):
            x = encoder(
                observation(dataset["raw"][i : i + 8], dataset["state"][i : i + 8], cameras)
            )
            tokens.append(x.detach().cpu())
            print(
                json.dumps(dict(activity="extract", windows=min(i + 8, len(selected)))), flush=True
            )
    tokens = torch.cat(tokens)
    torch_save(OUT / "tokens.pt", tokens)
    if state_digest(encoder) != before:
        raise ValueError("frozen spatial producer mutated")
    write(
        EVIDENCE / "extraction.json",
        dict(
            completed=True,
            shape=list(tokens.shape),
            dtype=str(tokens.dtype),
            spatial_sha_before=before,
            spatial_sha_after=state_digest(encoder),
            runtime_source_token_max_difference=parity,
            updates=0,
            token_tensor_sha=state_digest(nn_buffer(tokens)),
        ),
    )


def nn_buffer(value):
    module = torch.nn.Module()
    module.register_buffer("values", value)
    return module


def train(branch, updates):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = torch.load(OUT / "dataset.pt", weights_only=False)
    splits = split_groups(data["records"])
    destination = OUT / branch
    destination.mkdir(exist_ok=False)
    write(
        destination / "status.json",
        dict(branch=branch, state="initialising", updated_at=time.time()),
    )
    policy = load_policy(destination / "loader")
    trainer, device = policy.trainer, policy.trainer.device
    if trainer.bridge.order_within_camera != "as_given":
        raise ValueError("r14 requires explicit camera-major row-major input order")
    spatial = SpatialTaskEncoder(trainer.model.inference.encoder, trainer.model.inference.readout)
    spatial.set_joint_training(False)
    initial_sha = state_digest(spatial)
    extraction = json.loads((EVIDENCE / "extraction.json").read_text())
    if initial_sha != extraction["spatial_sha_before"]:
        raise ValueError("paired frozen representations differ")
    tokens = torch.load(OUT / "tokens.pt", weights_only=True).to(device)
    if state_digest(nn_buffer(tokens)) != extraction["token_tensor_sha"]:
        raise ValueError("shared token cache changed")
    actions = torch.as_tensor(data["actions"], device=device)
    masks = torch.as_tensor(data["mask"], device=device)
    states = torch.as_tensor(data["state"], device=device)
    padded = trainer.bridge.prepare_control_targets(actions, device=device)
    instruction = "Grab the light brown cylindrical roller on the table."
    # Language+state encoding is precomputed from current measured state only.
    lang, lang_mask, native_state = trainer.language([instruction] * len(states), data["state"])
    stats = policy.stats["actions"]["std"]
    head = SmallActionHead().to(device)
    adapter = ResidualTokenAdapter().to(device)
    if branch == "small":
        head_parameters = list(head.parameters())
        head_lr = 1e-3
        trainer.bridge.base.requires_grad_(False)
    else:
        head_parameters = list(adapter.parameters())
        head_parameters += [p for a in trainer.backend_adapters.values() for p in a.parameters()]
        head_lr = 1e-4
    baseline_trainable = [n for n, p in trainer.bridge.base.named_parameters() if p.requires_grad]
    report = dict(
        branch=branch,
        seed=SEED,
        max_updates_per_phase=updates,
        common_spatial_sha=initial_sha,
        common_tokens_sha=extraction["token_tensor_sha"],
        stages=[],
        policy_closed_loops=0,
        horizon=50,
        offset=1,
        native_action_dims=32,
        actual_control_dims=14,
        state_input="current normalized measured joints14; no phase/index",
        head_parameters=sum(p.numel() for p in head_parameters),
        backend_trainable=baseline_trainable,
        objectives="small:direct masked32 MSE; pi:native masked32 flow; compare sampled14 metrics",
        initialization="same RAW850 visual/LoRA; small random; pi residual adapter identity",
        test_scope="historical frozen dev episodes; not new blind benchmark or closed-loop proof",
    )
    sample_rng = np.random.default_rng(SEED)
    # Same exact sequence for both branches and stages, sampled without test/val access.
    order = [
        sample_rng.choice(splits["train"], min(8, len(splits["train"])), replace=False).tolist()
        for _ in range(updates)
    ]
    write(destination / "sample_order.json", order)
    write(EVIDENCE / (branch + "_protocol.json"), report)

    def values(indices, joint):
        if not joint:
            return tokens[indices]
        return spatial(observation(data["raw"][indices], data["state"][indices], data["cameras"]))

    def predict(indices, joint, seed=FLOW_SEEDS[0], zero=False):
        x = values(indices, joint)
        if zero:
            x = torch.zeros_like(x)
        if branch == "small":
            return head(x, states[indices])[..., :14]
        x = adapter(x)
        noise = torch.randn(
            (len(indices), 50, 32),
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        # Return tensor without converting to numpy twice.
        return trainer.bridge.sample_actions_from_visual_tokens(
            visual_tokens(x, data["cameras"]),
            lang[indices],
            lang_mask[indices],
            native_state[indices],
            noise=noise,
            num_steps=10,
        )[..., :14]

    @torch.no_grad()
    def evaluate(split, joint, *, seeds=FLOW_SEEDS[:1], zero=False):
        indices = splits[split]
        all_preds = []
        for seed in seeds if branch == "pi" else seeds[:1]:
            chunks = []
            for i in range(0, len(indices), 8):
                batch = indices[i : i + 8]
                chunks.append(predict(batch, joint, seed=seed, zero=zero).float().cpu().numpy())
            all_preds.append(np.concatenate(chunks))
        metrics = [
            action_metrics(p, data["actions"][indices], data["mask"][indices], stats)
            for p in all_preds
        ]
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        return avg, np.stack(all_preds), metrics

    def save_state(path):
        from owac.backends.low_rank import adapter_state

        trainable = {
            n: p.detach().cpu().clone() for n, p in spatial.named_parameters() if p.requires_grad
        }
        torch_save(
            path,
            dict(
                head=head.state_dict(),
                adapter=adapter.state_dict(),
                backend=adapter_state(trainer.backend_adapters),
                spatial=trainable,
            ),
        )

    def load_state(path):
        from owac.backends.low_rank import restore_adapters

        saved = torch.load(path, map_location=device, weights_only=False)
        head.load_state_dict(saved["head"], strict=True)
        adapter.load_state_dict(saved["adapter"], strict=True)
        restore_adapters(trainer.backend_adapters, saved["backend"])
        with torch.no_grad():
            params = dict(spatial.named_parameters())
            for name, value in saved["spatial"].items():
                params[name].copy_(value)

    for phase in ("frozen", "joint"):
        joint = phase == "joint"
        spatial.set_joint_training(joint)
        groups = [dict(params=head_parameters, lr=head_lr, name="readout")]
        if joint:
            groups.append(
                dict(
                    params=[p for p in spatial.parameters() if p.requires_grad],
                    lr=1e-5,
                    name="spatial",
                )
            )
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-3, fused=True)
        phase_before = state_digest(spatial)
        selected_path = destination / (phase + "-selected.pt")
        phase_report = dict(
            phase=phase,
            trainable_spatial=[n for n, p in spatial.named_parameters() if p.requires_grad],
            history=[],
        )
        initial_val, _, _ = evaluate("val", joint)
        initial_train, _, _ = evaluate("train", joint)
        best, selected_step = initial_val["normalized_control_mse"], 0
        save_state(selected_path)
        phase_report.update(initial_train=initial_train, initial_val=initial_val)
        start = time.time()
        for step, indices in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)
            x = values(indices, joint)
            if branch == "small":
                loss = masked_action_loss(head(x, states[indices]), padded[indices], masks[indices])
            else:
                generator = torch.Generator(device=device).manual_seed(SEED + step)
                noise = torch.randn(padded[indices].shape, generator=generator, device=device)
                flow_time = torch.rand((len(indices),), generator=generator, device=device)
                flow_time = flow_time.pow(1 / 1.5) * 0.999 + 0.001
                result = trainer.bridge.forward_from_visual_tokens(
                    visual_tokens(adapter(x), data["cameras"]),
                    lang[indices],
                    lang_mask[indices],
                    native_state[indices],
                    padded[indices],
                    noise=noise,
                    time=flow_time,
                    action_mask=masks[indices],
                )
                loss = result["action_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss; stop without rollback")
            loss.backward()
            gradient_norms = {
                g["name"]: float(torch.nn.utils.clip_grad_norm_(g["params"], 1.0)) for g in groups
            }
            if not all(np.isfinite(v) for v in gradient_norms.values()):
                raise FloatingPointError("nonfinite gradient; stop before update")
            optimizer.step()
            torch.cuda.synchronize()
            row = dict(
                step=step,
                loss=float(loss.detach()),
                gradients=gradient_norms,
                elapsed_s=time.time() - start,
            )
            if step % 64 == 0 or step == updates:
                train_metrics, _, _ = evaluate("train", joint)
                val_metrics, _, _ = evaluate("val", joint)
                row.update(train=train_metrics, val=val_metrics)
                if val_metrics["normalized_control_mse"] < best:
                    best, selected_step = val_metrics["normalized_control_mse"], step
                    save_state(selected_path)
            with (destination / (phase + "-history.jsonl")).open("a") as log:
                log.write(json.dumps(row) + "\n")
            if step % 8 == 0 or step == 1:
                status = dict(
                    branch=branch,
                    phase=phase,
                    state="running",
                    step=step,
                    max_updates=updates,
                    seconds_per_update=(time.time() - start) / step,
                    latest_loss=row["loss"],
                    selected_step=selected_step,
                    selected_val_mse=best,
                    updated_at=time.time(),
                    peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    latest_validation=row.get("val"),
                    closed_loop_success="not evaluated",
                )
                write(EVIDENCE / (branch + "_status.json"), status)
                write(destination / "status.json", status)
                print(json.dumps(status), flush=True)
            if "val" in row:
                phase_report["history"].append(row)
        save_state(destination / (phase + "-final.pt"))
        phase_report.update(
            actual_updates=updates,
            selected_step=selected_step,
            elapsed_s=time.time() - start,
            spatial_sha_before=phase_before,
            spatial_sha_after_final=state_digest(spatial),
        )
        if not joint and phase_report["spatial_sha_after_final"] != initial_sha:
            raise ValueError("frozen visual parameters mutated")
        load_state(selected_path)
        preds = {}
        for split in ("train", "val", "test"):
            metrics, p, by_seed = evaluate(split, joint, seeds=FLOW_SEEDS)
            phase_report[split] = metrics
            phase_report[split + "_by_seed"] = by_seed
            preds[split] = p
        # Intervention cannot change targets or state. It is not a valid performance baseline.
        zero_metrics, zero_p, _ = evaluate("test", joint, seeds=FLOW_SEEDS[:1], zero=True)
        phase_report["test_zero_visual"] = zero_metrics
        preds["test_zero_visual"] = zero_p
        # Keep full per-window arrays for independent masked metric recomputation.
        np.savez(destination / (phase + "-predictions.npz"), **preds)
        phase_report["spatial_sha_selected"] = state_digest(spatial)
        report["stages"].append(phase_report)
        write(EVIDENCE / (branch + "_report.json"), report)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    status.update(state="completed", actual_updates=updates * 2, updated_at=time.time())
    write(EVIDENCE / (branch + "_status.json"), status)
    write(destination / "status.json", status)
    report.update(
        completed=True,
        actual_updates=updates * 2,
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    write(EVIDENCE / (branch + "_report.json"), report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "small", "pi"))
    parser.add_argument("--updates", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.updates <= 256:
        raise ValueError("r14 cap:256 updates per branch per phase")
    try:
        prepare() if args.command == "prepare" else train(args.command, args.updates)
    except Exception as error:
        if args.command != "prepare":
            status_path = OUT / args.command / "status.json"
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            previous.update(
                branch=args.command,
                state="failed",
                updated_at=time.time(),
                failure_type=type(error).__name__,
            )
            write(status_path, previous)
            write(EVIDENCE / (args.command + "_status.json"), previous)
        raise
