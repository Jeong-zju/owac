"""Freeze RAW weights and extract four spatial stages; labels are stored separately."""

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.runtime.n02_v12_robotwin import Policy, online_window
from owac.runtime.rawvla_robotwin import normalize_state

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
OUT = ROOT / "outputs/n02-v12/information-r13"
EVIDENCE = ROOT / "research/experiments/n02_v1_2/information_r13_20261019"
CAMS = ("head_camera", "left_camera", "right_camera")


def digest_model(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def wait_snapshot(path):
    deadline = time.monotonic() + 1200
    while True:
        try:
            return np.load(path, allow_pickle=False)
        except (FileNotFoundError, zipfile.BadZipFile, EOFError):
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def pool(value):
    # Position remains explicit; global-average pooling could hide small objects.
    return F.adaptive_avg_pool2d(value.float(), (8, 8)).cpu().numpy()


@torch.inference_mode()
def extract(args):
    destination = OUT / args.arm
    destination.mkdir(exist_ok=False)
    r12 = args.arm == "r12-one256"
    config = ROOT / (
        "research/experiments/n02_v1_2/sequence_r12_20261019/one_config.json"
        if r12
        else "research/experiments/n02_v1_2/training_r11_20261017/task_config.json"
    )
    checkpoint = ROOT / (
        "outputs/n02-v12/P2/raw-fewshot-r12-one-s1103-a1/checkpoint_0000256.pt"
        if r12
        else "outputs/n02-v12/P2/raw-task-r11-task-s1103-a1/checkpoint_0000850.pt"
    )
    policy = Policy(config, checkpoint, destination / "loader")
    trainer = policy.trainer
    trainer.model.eval().requires_grad_(False)
    trainer.bridge.base.eval().requires_grad_(False)
    if trainer.bridge.projection is not None:
        trainer.bridge.projection.eval().requires_grad_(False)
    trainer.optimizer.step = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("policy frozen"))
    before = digest_model(trainer.model)
    inf = trainer.model.inference
    captured = {}

    def encoder_hook(_module, _inputs, spatial):
        for name, level in [("shallow_stride4", "stride4"), ("deep_stride16", "stride16")]:
            values = [spatial.levels[c + "::" + level] for c in policy.cameras]
            captured[name] = np.concatenate([pool(v) for v in values], axis=0)

    # Use the actual production spatial fusion method, intercept only its detached return.
    original = inf.readout.spatial_output_tokens

    def spatial_output(features):
        value = original(features)
        captured["spatial_fusion"] = token_grid(value)
        return value

    def token_grid(value):
        shaped = value.reshape(1, 3, 16, 16, -1).permute(0, 1, 4, 2, 3).flatten(0, 1)
        return pool(shaped)

    native_handles = [
        inf.encoder.backbone.stages[i].register_forward_hook(
            lambda _m, _a, result, name=name: captured.update({name: pool(result)})
        )
        for i, name in [(0, "shallow_native"), (2, "deep_native")]
    ]
    handle = inf.encoder.register_forward_hook(encoder_hook)
    inf.readout.spatial_output_tokens = spatial_output
    all_records = []
    for panel in ("controlled", "cached"):
        chunks, ys, states, phases, records = [], [], [], [], []
        selection = json.loads((OUT / "selection.json").read_text())
        for selected in selection:
            scene = selected["scene"]
            if panel == "controlled":
                datafile = OUT / "snapshots" / f"scene_{scene:03d}.npz"
                with wait_snapshot(datafile) as a:
                    raw, y, state = a["raw"], a["y"], a["state"]
                indices = range(len(y))
            else:
                with np.load(selected["path"], allow_pickle=False) as a:
                    raw = np.stack([a[c + "_raw_float32"] for c in CAMS], axis=1)
                    state = a["actions"]
                    y = np.concatenate(
                        [a["left_endpose"][:, :3], a["right_endpose"][:, :3], state[:, [6, 13]]],
                        axis=1,
                    )
                indices = np.unique(np.rint(np.linspace(0, len(y) - 1, 4)).astype(int))
            for t in indices:
                example = dict(
                    step=int(t),
                    case_id=f"r13-{panel}-{scene}",
                    lang="Grab the light brown cylindrical roller on the table.",
                    normalized_state=normalize_state(state[t], policy.stats["state"]),
                    history=[
                        dict(step=int(i), image=list(raw[i]))
                        for i in ([t] if panel == "controlled" else range(max(0, t - 7), t + 1))
                    ],
                )
                window = online_window(
                    example, policy.cameras, history=8, representation=policy.representation
                )
                trainer.bridge.reset_episodes()
                visual = trainer.bridge.visual_tokens(window)
                captured["pre_pi_actual"] = token_grid(
                    trainer.bridge._project_visual_tokens(visual.values)
                )
                actual = {key: value.copy() for key, value in captured.items()}
                example["normalized_state"] = np.zeros(14, dtype=np.float32)
                window = online_window(
                    example, policy.cameras, history=8, representation=policy.representation
                )
                trainer.bridge.reset_episodes()
                fixed = trainer.bridge.visual_tokens(window)
                actual["pre_pi_fixed_state"] = token_grid(
                    trainer.bridge._project_visual_tokens(fixed.values)
                )
                chunks.append(actual)
                ys.append(y[t])
                states.append(state[t])
                phases.append([t / max(1, len(y) - 1)])
                records.append(
                    dict(
                        panel=panel,
                        scene=scene,
                        group=selected["path"],
                        split=selected["split"],
                        t=int(t),
                    )
                )
            print(json.dumps(dict(panel=panel, scene=scene, samples=len(records))), flush=True)
            (destination / "progress.json").write_text(json.dumps(dict(panel=panel, scene=scene)))
        arrays = {key: np.stack([c[key] for c in chunks]) for key in chunks[0]}
        arrays.update(
            y=np.asarray(ys),
            proprio=np.asarray(states),
            phase=np.asarray(phases),
            records=json.dumps(records),
        )
        # Frozen caches are regenerable. Avoid CPU deflate stalling the GPUs.
        temporary = destination / (panel + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(destination / (panel + ".npz"))
        all_records.extend(records)
    handle.remove()
    for native_handle in native_handles:
        native_handle.remove()
    inf.readout.spatial_output_tokens = original
    after = digest_model(trainer.model)
    if before != after:
        raise ValueError("frozen model mutated")
    receipt = dict(
        arm=args.arm,
        checkpoint=str(checkpoint),
        config=str(config),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        model_sha_before=before,
        model_sha_after=after,
        policy_updates=0,
        readout_grid=[8, 8],
        stages=list(chunks[0]),
        spatial_keys=list(policy.cameras),
        records=all_records,
        context=("pre_pi_actual contains current proprio; fixed_state uses normalized zero"),
        cached_labels="world EE convention from cached left/right_endpose; not TCP",
        controlled_labels="world TCP and object origin; independent controlled snapshots",
        holdout=(
            "independent for readout fitting; cached episodes may have been seen by frozen parent"
        ),
        completed_at=time.time(),
    )
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(completed=args.arm, samples=len(all_records))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("r11-task850", "r12-one256"), required=True)
    torch.set_num_threads(4)
    extract(p.parse_args())
