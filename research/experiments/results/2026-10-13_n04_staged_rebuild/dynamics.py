"""Frozen-weight memory interventions. Oracle fields are diagnostic-only inputs."""

import argparse
import statistics
from dataclasses import replace
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from oracles import OracleEgo
from run import ARCHIVE, ROOT, digest, save

from owac.evaluation.n04 import evaluate_condition
from owac.frameworks.n04.geometry import GridSpec, backward_grid_from_translation
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import build_encoder, config_from_json, load_checkpoint


def sequence_for(kind, seed):
    scene = SceneConfig(
        camera_translation=0,
        camera_rotation=0,
        object_velocity=(0.4, 1.0),
        stop_after=8 if kind == "stop" else None,
        seed=seed,
    )
    original = SyntheticSequenceGenerator(scene).sequence(
        batch=2, length=24, condition="object_only", seed=seed + 1
    )
    indices = list(range(24)) if kind != "reverse" else list(range(12)) + list(range(12, 0, -1))
    sequence = []
    noise_rng = torch.Generator().manual_seed(seed + 900)
    for t, idx in enumerate(indices):
        frame = original[idx]
        metadata = {**frame.metadata, "t": t}
        if t:
            metadata["object_moving"] = (
                (frame.true_object_translation - sequence[-1].true_object_translation).abs().sum(-1)
                > 1e-6
            ).tolist()
        measured = (
            frame.oracle_clean
            + torch.randn(frame.measured.shape, generator=noise_rng) * scene.noise_sigma
        ).clamp(0, 1)
        frame = replace(
            frame,
            metadata=metadata,
            measured=measured,
            saturated_mask=(measured >= 0.999).any(1, keepdim=True).float(),
        )
        if kind == "occluded" and 8 <= t < 12:
            frame = replace(
                frame,
                valid_mask=torch.zeros_like(frame.valid_mask),
                measured=torch.zeros_like(frame.measured),
            )
        sequence.append(frame)
    return scene, sequence


def intervention(encoder, sequence, mode):
    original = encoder.step
    index = 0
    hooks = []
    reference = {}
    core = getattr(encoder, "encoder", encoder)
    original_propagator = getattr(core, "propagator", None)
    oracle_ego = OracleEgo(original_propagator) if "ego" in mode else None
    if oracle_ego is not None:
        core.propagator = oracle_ego
    if "event" in mode:

        def event_hook(module, args, result):
            frame = reference["frame"]
            truth = F.adaptive_avg_pool2d(frame.object_mask.to(result), result.shape[-2:]) > 0.5
            moving = torch.tensor(frame.metadata["object_moving"], device=result.device)[
                :, None, None, None
            ]
            out = result.clone()
            out[:, 1:2] = torch.where(truth & moving, 15.0, -15.0)
            return out

        hooks.append(core.decomposer.head.register_forward_hook(event_hook))
    if mode == "no_change_use":

        def no_change(module, args):
            return args[0], torch.zeros_like(args[1]), args[2]

        hooks.extend(block.register_forward_pre_hook(no_change) for block in core.blocks)

    def step(self, values, context, state, **kwargs):
        nonlocal index
        frame = sequence[index]
        assert torch.equal(frame.measured.to(values), values)
        reference["frame"] = frame
        if oracle_ego is not None:
            oracle_ego.frame = frame
        if mode == "no_history":
            state = replace(
                state,
                change=[torch.zeros_like(x) for x in state.change]
                if isinstance(state.change, list)
                else torch.zeros_like(state.change),
            )
        elif "transport" in mode and index:
            previous = sequence[index - 1]
            delta = (frame.true_object_translation - previous.true_object_translation).to(values)
            assert not frame.true_camera_translation.any() and not frame.true_camera_rotation.any()
            changes = state.change if isinstance(state.change, list) else [state.change]
            moved = []
            for change in changes:
                h, w = change.shape[-2:]
                spec = GridSpec(h, w)
                mask = F.adaptive_max_pool2d(previous.object_mask.to(values), (h, w))
                # Include the signed edge trail as well as the previous object interior.
                mask = F.max_pool2d(mask, 5, 1, 2).clamp(0, 1)
                grid = backward_grid_from_translation(
                    spec,
                    delta[:, 0] * w / values.shape[-1],
                    delta[:, 1] * h / values.shape[-2],
                    batch=values.shape[0],
                    device=values.device,
                )
                moved.append(
                    change * (1 - mask)
                    + F.grid_sample(change * mask, grid, padding_mode="zeros", align_corners=False)
                )
            state = replace(state, change=moved if isinstance(state.change, list) else moved[0])
        index += 1
        return original(values, context, state, **kwargs)

    encoder.step = MethodType(step, encoder)

    def restore():
        encoder.step = original
        if oracle_ego is not None:
            core.propagator = original_propagator
        for hook in hooks:
            hook.remove()

    return restore


@torch.no_grad()
def run_one(encoder, kind, seed, mode):
    scene, sequence = sequence_for(kind, seed)
    restore = intervention(encoder, sequence, mode)
    try:
        report = evaluate_condition(
            encoder, condition="object_only", scene=scene, sequence=sequence
        )
    finally:
        restore()
    rows = report.trajectory
    event = 8 if kind == "stop" else 12
    recovery = next(
        (
            t - event
            for t in range(event, len(rows) - 1)
            if all(r["location_error_px"] <= 2 and r["target_iou"] >= 0.5 for r in rows[t : t + 2])
        ),
        None,
    )
    result = report.as_dict()
    result.pop("trajectory")
    result.update(kind=kind, seed=seed, mode=mode, recovery_frames=recovery)
    result["extra"].update(
        recovery_event=event,
        recovery_censored=recovery is None,
        hidden_error_px=statistics.fmean(r["location_error_px"] for r in rows[8:12])
        if kind == "occluded"
        else None,
        after_event_error_px=statistics.fmean(r["location_error_px"] for r in rows[event:]),
    )
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="s3b-combined")
    parser.add_argument("--split", choices=("development", "final"), default="development")
    args = parser.parse_args()
    torch.set_num_threads(4)
    seeds = (44000, 44001, 44002) if args.split == "development" else (45000, 45001, 45002)
    results = []
    destination = ARCHIVE / f"{args.stage}-dynamics-{args.split}.json"
    assert not destination.exists(), "Do not overwrite frozen dynamics"
    source = [
        *Path("src/owac/frameworks/n04").glob("*.py"),
        Path("src/owac/evaluation/n04.py"),
        Path(__file__),
    ]
    hashes = {str(p): digest(p) for p in source}
    for name in ("candidate", "predict_update_filter", "n04_no_change_branch"):
        for run in (0, 1, 2):
            checkpoint = ROOT / args.stage / name / f"run-{run}" / "checkpoint.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg, _ = config_from_json(Path("configs/encoders/n04/tiny.json"))
            cfg = cfg.with_model(**payload["model_config"])
            encoder = build_encoder(cfg, baseline=None if name == "candidate" else name)
            load_checkpoint(checkpoint, encoder=encoder)
            encoder.cuda().eval()
            modes = (
                (
                    "base",
                    "no_history",
                    "no_change_use",
                    "event",
                    "transport",
                    "transport_event",
                    "ego",
                    "ego_transport",
                    "ego_transport_event",
                )
                if name == "candidate"
                else ("base",)
            )
            for mode in modes:
                for kind in ("stop", "reverse", "occluded"):
                    for seed in seeds:
                        result, rows = run_one(encoder, kind, seed, mode)
                        results.append(
                            dict(name=name, run=run, checkpoint_sha256=digest(checkpoint), **result)
                        )
                        save(
                            ROOT
                            / args.stage
                            / name
                            / f"run-{run}"
                            / f"dynamics-{args.split}-{mode}-{kind}-{seed}.json",
                            rows,
                        )
            print("dynamic done", name, run, flush=True)
    assert all(digest(Path(p)) == sha for p, sha in hashes.items())
    save(destination, results)
    save(
        ARCHIVE / f"{args.stage}-dynamics-{args.split}-provenance.json",
        dict(
            source_sha256=hashes,
            seeds=seeds,
            complete=True,
            records=len(results),
            diagnostic_only_modes=[
                "event",
                "transport",
                "transport_event",
                "ego",
                "ego_transport",
                "ego_transport_event",
            ],
            note=(
                "Oracles are interventions, not mathematical optimality bounds. "
                "All-hidden frames bypass correction blocks."
            ),
        ),
    )


if __name__ == "__main__":
    main()
