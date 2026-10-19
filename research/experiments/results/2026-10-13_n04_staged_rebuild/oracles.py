"""Fixed-weight oracle interventions, never a deployable or trained capability score."""

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch

import owac.frameworks.n04.innovation as innovation_module
from owac.evaluation.n04 import evaluate_condition, json_safe
from owac.frameworks.n04.ego import PropagationResult
from owac.frameworks.n04.geometry import (
    AffineFlow,
    GridSpec,
    backward_grid_from_affine,
    warp_feature_field,
)
from owac.frameworks.n04.model import _resize_sampling_grid, linear_measurement_planes
from owac.frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from owac.training.n04 import build_encoder, config_from_json, load_checkpoint

ARCHIVE = Path("research/experiments/results/2026-10-13_n04_staged_rebuild")
ROOT = Path("outputs/n04-staged-rebuild-20261013")


class OracleEgo(torch.nn.Module):
    def __init__(self, original=None):
        super().__init__()
        self.original = original
        self.frame = None

    def forward(self, previous, spec, context):
        normal = self.original(previous, spec, context) if self.original is not None else None
        fields = previous if isinstance(previous, list) else [previous]
        flow = self.frame.oracle_camera_flow.to(fields[0]) / spec.pixel_stride
        matrix = torch.eye(2, device=flow.device)[None].expand(flow.shape[0], -1, -1).clone()
        matrix[:, :, 0] += flow[:, 0, 1] - flow[:, 0, 0]
        matrix[:, :, 1] += flow[:, 1, 0] - flow[:, 0, 0]
        grid = backward_grid_from_affine(spec, AffineFlow(matrix, flow[:, 0, 0]))
        warped = [
            warp_feature_field(
                f, _resize_sampling_grid(grid, f.shape[-2:]), GridSpec(*f.shape[-2:])
            ).values
            for f in fields
        ]
        valid = warp_feature_field(fields[0], grid, spec).validity[:, None]
        return PropagationResult(
            predicted=warped if isinstance(previous, list) else warped[0],
            validity=valid,
            uncertainty=normal.uncertainty if normal is not None else torch.full_like(valid, 0.05),
            forward_flow=flow,
            backward_grid=grid,
            geometry_mode="oracle_ego",
        )


def run_condition(encoder, condition, mode, gain_path="hint", matched_support=False):
    scene = SceneConfig(
        camera_translation=(0.6, 2.0),
        camera_rotation=(0, 0.7),
        object_velocity=(0.4, 1.8),
        illumination=(0.7, 1.3),
        stop_after=8 if condition == "object_only" else None,
        seed=43000,
    )
    sequence = SyntheticSequenceGenerator(scene).sequence(
        batch=2, length=16, condition=condition, seed=43001
    )
    original_prediction = encoder.predict_observation
    original_step = encoder.step
    original_gain = innovation_module.estimate_low_frequency_gain
    reference = {"frame": None, "previous": None}
    original_prop = encoder.propagator
    oracle = OracleEgo(original_prop) if "ego" in mode else None
    if oracle is not None:
        encoder.propagator = oracle
    hooks = []
    if "event" in mode:

        def event_hook(module, args, value):
            frame = reference["frame"]
            moving = torch.as_tensor(frame.metadata["object_moving"], device=value.device)[
                :, None, None, None
            ]
            truth = (
                torch.nn.functional.adaptive_avg_pool2d(
                    frame.object_mask.to(value), value.shape[-2:]
                )
                > 0.5
            )
            copy = value.clone()
            copy[:, 1:2] = torch.where(truth & moving, 15.0, -15.0)
            return copy

        hooks.append(encoder.decomposer.head.register_forward_hook(event_hook))
    if "gain" in mode:

        def oracle_gain(current, predicted, weight, **kwargs):
            frame, previous = reference["frame"], reference["previous"]
            ratio = (
                torch.ones(current.shape[0], device=current.device)
                if previous is None
                else (frame.true_illumination.to(current) / previous.true_illumination.to(current))
            )
            gain = ratio[:, None, None, None].expand_as(current)
            return gain, gain - 1

        innovation_module.estimate_low_frequency_gain = oracle_gain
    if ("gain" in mode and gain_path == "prediction") or matched_support:

        def compensated_prediction(self, history):
            if "gain" in mode and gain_path == "prediction":
                gain, _ = oracle_gain(history, history, None)
                history = history * gain
            if matched_support:
                frame = reference["frame"]
                support, _ = linear_measurement_planes(
                    frame.valid_mask.to(history) * (1 - frame.saturated_mask.to(history)),
                    torch.ones_like(frame.valid_mask).to(history),
                    encoder.config.input_mode,
                )
                history = history * support
            return original_prediction(history)

        encoder.predict_observation = MethodType(compensated_prediction, encoder)
    iterator = iter(sequence)

    def step(self, values, context, state, **kwargs):
        frame = next(iterator)
        assert torch.equal(frame.measured.to(values), values)
        reference["frame"] = frame
        if oracle is not None:
            oracle.frame = frame
        out = original_step(values, context, state, **kwargs)
        reference["previous"] = frame
        return out

    encoder.step = MethodType(step, encoder)
    try:
        report = evaluate_condition(encoder, condition=condition, scene=scene, sequence=sequence)
    finally:
        encoder.predict_observation = original_prediction
        encoder.step = original_step
        encoder.propagator = original_prop
        innovation_module.estimate_low_frequency_gain = original_gain
        for hook in hooks:
            hook.remove()
    result = report.as_dict()
    result.pop("trajectory")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--reference", choices=("task_state", "sensor_history"))
    parser.add_argument("--gain-path", choices=("hint", "prediction"), default="hint")
    parser.add_argument("--matched-support", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+", default=["camera_only", "object_only", "both", "illumination"]
    )
    parser.add_argument(
        "--modes", nargs="+", default=["base", "ego", "gain", "event", "ego_gain_event"]
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    sources = [*Path("src/owac/frameworks/n04").glob("*.py"), Path(__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    weights = {}
    rows = []
    for run in args.runs:
        checkpoint = ROOT / args.stage / "candidate" / f"run-{run}" / "checkpoint.pt"
        weights[str(checkpoint)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg, _ = config_from_json(Path("configs/encoders/n04/tiny.json"))
        cfg = cfg.with_model(**payload["model_config"])
        if args.reference:
            cfg = cfg.with_model(innovation_reference=args.reference)
        for mode in args.modes:
            encoder = build_encoder(cfg)
            load_checkpoint(checkpoint, encoder=encoder, allow_design_override=bool(args.reference))
            encoder.cuda().eval()
            for condition in args.conditions:
                result = run_condition(
                    encoder, condition, mode, args.gain_path, args.matched_support
                )
                rows.append(dict(run=run, mode=mode, **result))
        print("oracle run complete", args.stage, run, flush=True)
    suffix = "-motion-only"
    if args.runs != [0, 1, 2]:
        suffix += "-runs-" + "-".join(map(str, args.runs))
    if args.matched_support:
        suffix += "-matched-support"
    if args.gain_path != "hint":
        suffix += "-gain-" + args.gain_path
    if args.reference:
        suffix += "-reference-" + args.reference
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    destination = ARCHIVE / f"{args.stage}-oracles{suffix}.json"
    assert not destination.exists(), "Do not overwrite frozen oracle diagnostics"
    (ARCHIVE / f"{args.stage}-oracles{suffix}-provenance.json").write_text(
        json.dumps(
            dict(
                source_sha256=hashes,
                weight_sha256=weights,
                arguments=vars(args),
                records=len(rows),
                diagnostic_only=True,
            ),
            indent=2,
        )
        + "\n"
    )
    destination.write_text(json.dumps(json_safe(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
