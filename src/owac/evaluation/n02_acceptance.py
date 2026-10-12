"""Independent learned-model probes on matched measurements and held-out seeds.

Semantic slot parity is a declared synthetic training convention, not physical
geometry or a robot score. The gates are reported even when they fail. Analytic
Bayes results and untrained controls cannot satisfy these learned-model gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from owac.baselines.n02 import build_baseline
from owac.core.encoder_contract import ComputeBudget, MeasurementFrame, state_tensor_bytes
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import (
    SyntheticGenerator,
    SyntheticOptions,
    values_for_task,
)
from owac.training.n02 import episode_prior_context


def evaluate(checkpoint: Path, *, seeds=(100, 101, 102, 103), arm="candidate"):
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = N02Config.from_dict(payload["config"])
    trained_arm = payload.get("arm", "candidate")
    if arm != "candidate" and trained_arm != "candidate" and arm != trained_arm:
        raise ValueError("cross-architecture weights are not a trained baseline")
    selected = trained_arm if arm == "candidate" else arm
    model = N02Encoder(config) if selected == "candidate" else build_baseline(selected, config)
    model.load_state_dict(payload["model"])
    model.eval()
    config = model.config
    budget = ComputeBudget(config.cycles, config.history)
    options = SyntheticOptions(height=64, width=64, channels=3)
    records, latencies = [], []
    with torch.no_grad():
        for seed in seeds:
            generator = SyntheticGenerator(options, seed=seed, split="validation")
            observed = generator.generate(seed=seed, num_frames=8, prior_condition="correct")
            observed = replace(observed, prior=replace(observed.prior, log_odds=math.log(19)))
            # Intervene on the declared prior while keeping every raw byte fixed.
            paired = {
                "correct": observed,
                "misleading": replace(
                    observed,
                    prior=replace(
                        observed.prior,
                        condition="misleading",
                        log_odds=-abs(observed.prior.log_odds),
                    ),
                ),
                "missing": replace(observed, prior=replace(observed.prior, condition="missing")),
                "unidentifiable": generator.generate(
                    seed=seed, num_frames=8, prior_condition="uninformative", identifiable=False
                ),
                "stale": generator.generate(seed=seed, num_frames=8, prior_condition="stale"),
            }
            paired["stale"] = replace(
                paired["stale"], prior=replace(paired["stale"].prior, log_odds=math.log(19))
            )
            for condition, episode in paired.items():
                state, steps = None, []
                context = episode_prior_context([episode], config.K)
                for index, values in enumerate(episode.frames):
                    frame = MeasurementFrame(
                        values=torch.from_numpy(values).unsqueeze(0),
                        frame_ids=(index,),
                        episode_ids=(episode.episode_id,),
                        camera_id="synthetic_measurement",
                        metadata={"variance": options.noise_sigma**2},
                        metadata_available={"variance": True},
                    )
                    # Standalone baselines require an explicit initial state.
                    if state is None and selected in {"plain_recurrent", "particle_filter"}:
                        from owac.core.encoder_contract import CameraSpec

                        state = model.init_state(
                            1, CameraSpec("synthetic_measurement", 64, 64, 3, "rawvla_rgb3"), "cpu"
                        )
                    start = time.perf_counter()
                    output = model.step(frame, context, state, budget)
                    latencies.append((time.perf_counter() - start) * 1000)
                    state = model.detach_state(output.state)
                    representation = output.representation
                    weights = representation.weights[0, 0]
                    semantic = [float(weights[::2].sum()), float(weights[1::2].sum())]
                    truth = values_for_task(episode.task).index(episode.value_over_time[index])
                    geometry = representation.geometry[0, 0, :, 0]
                    steps.append(
                        {
                            "frame": index,
                            "semantic_mass": semantic,
                            "truth_semantic_mass": semantic[truth],
                            "cue_visible": bool(episode.cue_strength[index] > 0),
                            "geometry_proxy": geometry.tolist(),
                            "truth_proxy_nearest_error": float(
                                (geometry - (2 * truth - 1)).abs().min()
                            ),
                            "measurement_nll": float(representation.measurement_energy.min()),
                            "state_bytes": int(
                                output.cost.state_bytes or state_tensor_bytes(state)
                            ),
                        }
                    )
                repeated = model.step(frame, context, state, budget)
                duplicate_delta = float(
                    (repeated.representation.weights - output.representation.weights).abs().max()
                )
                prefix = [step for step in steps if not step["cue_visible"]]
                before_change = steps[max(0, episode.geometry_change_step - 1)]
                gate = {
                    "correct": steps[-1]["truth_semantic_mass"] >= 0.7,
                    "misleading": bool(prefix)
                    and prefix[-1]["truth_semantic_mass"] <= 0.25
                    and steps[-1]["truth_semantic_mass"] >= 0.7,
                    "missing": steps[-1]["truth_semantic_mass"] > 0.5,
                    "unidentifiable": all(
                        abs(step["semantic_mass"][0] - 0.5) <= 0.15 for step in steps
                    ),
                    "stale": before_change["truth_semantic_mass"] >= 0.7
                    and steps[-1]["truth_semantic_mass"] >= 0.7,
                }[condition]
                if weights.numel() == 1 or selected == "particle_filter":
                    gate = None
                    for step in steps:
                        step["semantic_mass"] = None
                        step["truth_semantic_mass"] = None
                records.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "measurement_sha256": hashlib.sha256(episode.frames.tobytes()).hexdigest(),
                        "declared_prior": context.hypothesis_prior.tolist(),
                        "steps": steps,
                        "gate_passed": gate,
                        "duplicate_weight_delta": duplicate_delta,
                        "last_ambiguous_prefix_frame": prefix[-1]["frame"] if prefix else None,
                    }
                )
    gates = {
        name: {
            "passed": sum(
                row["gate_passed"] is True for row in records if row["condition"] == name
            ),
            "total": sum(
                row["gate_passed"] is not None for row in records if row["condition"] == name
            ),
        }
        for name in paired
    }
    return {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": payload["step"],
        "trained_arm": trained_arm,
        "evaluated_arm": selected,
        "comparison_scope": "trained checkpoint"
        if selected == trained_arm
        else "same trained weights with rule intervention; not independently trained baseline",
        "scope": "binary semantic proxy; no physical or closed-loop score",
        "gate_definition": {
            "correct": "final true semantic mass >= .7",
            "misleading": "95% wrong prior; pre-cue true mass <= .25, post-cue >= .7",
            "missing": "final true semantic mass > .5",
            "stale": "true mass >= .7 both before and after actual geometry change",
            "unidentifiable": "all prefix and final semantic masses within .15 of .5",
        },
        "all_gates_passed": all(row["gate_passed"] for row in records)
        if any(row["gate_passed"] is not None for row in records)
        else None,
        "gate_applicability": "single-field/PF: proxy error and NLL only; semantic slots unverified"
        if selected in {"single_hypothesis", "plain_recurrent", "particle_filter"}
        else "declared slot-parity convention",
        "gates": gates,
        "records": records,
        "encoder_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        default="candidate",
        choices=(
            "candidate",
            "single_hypothesis",
            "plain_recurrent",
            "particle_filter",
            "dense_global_recurrent",
            "no_dependency_propagation",
            "no_contradiction_reset",
        ),
    )
    args = parser.parse_args()
    report = evaluate(args.checkpoint, arm=args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_gates_passed": report["all_gates_passed"], "gates": report["gates"]}))


if __name__ == "__main__":
    main()
