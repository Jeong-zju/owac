from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.hypotheses import neighbour_validity
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import SyntheticGenerator, SyntheticOptions, values_for_task
from owac.training.n02 import episode_prior_context, step_terms

Path("outputs/n02-source-review").mkdir(parents=True, exist_ok=True)

torch.set_num_threads(1)
CHECKPOINT = Path("outputs/n02-final-candidate/checkpoint.pt")
payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)


def build():
    model = N02Encoder(N02Config.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def replay(variant, condition, seed):
    model = build()
    config = model.config
    options = SyntheticOptions(height=64, width=64, channels=3)
    gen = SyntheticGenerator(options, seed=seed, split="validation")
    ep = gen.generate(seed=seed, num_frames=8, prior_condition="correct")
    ep = replace(ep, prior=replace(ep.prior, log_odds=math.log(19)))
    if condition == "misleading":
        ep = replace(ep, prior=replace(ep.prior, condition="misleading", log_odds=-math.log(19)))
    elif condition == "unidentifiable":
        ep = gen.generate(
            seed=seed, num_frames=8, prior_condition="uninformative", identifiable=False
        )
    if variant == "cue_removed":
        cleared = replace(ep, cue_strength=np.zeros_like(ep.cue_strength))
        atten = np.broadcast_to(ep.transmission[:, None, None], (len(ep.frames), 64, 64))
        if ep.occluder_mask is not None:
            atten = atten * ep.occluder_mask[None]
        frames = (
            ep.frames
            - (gen.clean_measurements(ep) - gen.clean_measurements(cleared)) * atten[:, None]
        )
        ep = replace(cleared, frames=frames.astype(ep.frames.dtype))
    context = episode_prior_context([ep], config.K)
    state = None
    rows = []
    calls = []
    frame_index = 0
    original = model.reviser.step

    def revision_hook(*args, **kwargs):
        normalized = args[1]
        out = original(*args, **kwargs)
        direct = out.contradiction
        extra = out.propagated & ~direct
        calls.append(
            {
                "cycle": kwargs["iteration"],
                "direct_by_slot": direct.sum((-1, -2)).flatten().tolist(),
                "extra_by_slot": extra.sum((-1, -2)).flatten().tolist(),
                "reset_by_slot": out.reset.sum((-1, -2)).flatten().tolist(),
                "residual_min_by_slot": normalized.amin((-1, -2)).flatten().tolist(),
                "residual_mean_by_slot": normalized.mean((-1, -2)).flatten().tolist(),
                "residual_max_by_slot": normalized.amax((-1, -2)).flatten().tolist(),
                "update_fraction": out.updated_fraction,
            }
        )
        return out

    model.reviser.step = revision_hook
    orig_initializer = model.initializer.forward

    def first_only(basis, *, context=None):
        if frame_index > 0 and context is not None:
            context = context.clone()
            context[:, : config.K] = 1 / config.K
        return orig_initializer(basis, context=context)

    if variant == "declared_context_first_only":
        model.initializer.forward = first_only
    if variant == "no_propagation":
        model.reviser.revision.propagation_enabled = False
    if variant == "no_reset":
        model.reviser.revision.reset_contradicted = False
    if variant.startswith("downweight_"):
        model.reviser.revision.downweight_contradicted = float(variant.rsplit("_", 1)[1])
    budget = ComputeBudget(config.cycles, config.history)
    with torch.no_grad():
        for frame_index, values in enumerate(ep.frames):
            calls = []
            if variant == "current_score_only" and state is not None:
                state = replace(state, weights=torch.full_like(state.weights, 1 / config.K))
            frame = MeasurementFrame(
                values=torch.from_numpy(values).unsqueeze(0),
                frame_ids=(frame_index,),
                episode_ids=(ep.episode_id,),
                camera_id="synthetic_measurement",
                metadata={"variance": options.noise_sigma**2},
                metadata_available={"variance": True},
            )
            out = model.step(frame, context, state, budget)
            state = model.detach_state(out.state)
            rep = out.representation
            E = rep.measurement_energy[0, 0]
            weights = rep.weights[0, 0]
            evidence = torch.softmax(-E / config.temperature, -1)
            truth = values_for_task(ep.task).index(ep.value_over_time[frame_index])
            rows.append(
                {
                    "frame": frame_index,
                    "cue": bool(ep.cue_strength[frame_index] > 0),
                    "weights": weights.tolist(),
                    "semantic0": float(weights[::2].sum()),
                    "truth_mass": float(weights[truth::2].sum()),
                    "evidence_semantic0": float(evidence[::2].sum()),
                    "evidence_true_mass": float(evidence[truth::2].sum()),
                    "measurement_energy": E.tolist(),
                    "uniform_evidence_ce_excess": float(
                        -torch.log_softmax(-E, -1).mean() - math.log(config.K)
                    ),
                    "revision": calls,
                }
            )
    return {
        "variant": variant,
        "condition": condition,
        "seed": seed,
        "frames_sha256": hashlib.sha256(ep.frames.tobytes()).hexdigest(),
        "declared_prior": context.hypothesis_prior.tolist(),
        "steps": rows,
    }


report = {
    "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    "checkpoint_step": payload["step"],
    "scope": (
        "source review; fixed learned weights; interventions are diagnosis, not accepted mechanisms"
    ),
    "replays": [],
}
for variant in [
    "standard",
    "declared_context_first_only",
    "current_score_only",
    "no_propagation",
    "no_reset",
    "cue_removed",
]:
    for condition in ["unidentifiable", "misleading", "correct"]:
        for seed in [100, 101, 102, 103]:
            report["replays"].append(replay(variant, condition, seed))

for variant in ("downweight_0", "downweight_1"):
    report["replays"].append(replay(variant, "misleading", 103))
downweight_rows = [r for r in report["replays"] if r["variant"].startswith("downweight_")]
downweight_weights = [torch.tensor([t["weights"] for t in r["steps"]]) for r in downweight_rows]
report["downweight_config_probe"] = {
    "values": [0.0, 1.0],
    "seed": 103,
    "posterior_bitwise_equal": torch.equal(*downweight_weights),
    "max_posterior_difference": float((downweight_weights[0] - downweight_weights[1]).abs().max()),
}

# Sparse source on the actual trained dependency table, versus identity fixture.
model = build()
cfg = model.config
rc = model.reviser
valid = neighbour_validity(5, 5, torch.device("cpu")).permute(1, 2, 0)[None, None]
learned = model.dependency_edge(1, 1, 5, 5, "cpu")
identity = torch.eye(cfg.K)[None, None, None, None, None].expand(1, 1, 5, 5, 4, cfg.K, cfg.K)
report["sparse_propagation"] = []
for edge_name, edge in [("identity_fixture", identity), ("trained_graph", learned)]:
    for slot in range(cfg.K):
        direct = torch.zeros(1, 1, cfg.K, 5, 5, dtype=torch.bool)
        direct[0, 0, slot, 2, 2] = True
        select, priority = rc.propagate(direct, edge, valid)
        report["sparse_propagation"].append(
            {
                "graph": edge_name,
                "source_slot": slot,
                "new_selected_by_slot": (select & ~direct).sum((-1, -2)).flatten().tolist(),
                "priority_neighbours_by_target_slot": [
                    priority[0, 0, k, 1, 2].item() for k in range(cfg.K)
                ],
            }
        )
# Two hops along an identity path with only one incoming predecessor per node.
rc.revision.propagation_hops = 2
path_edge = torch.zeros(1, 1, 1, 5, 4, cfg.K, cfg.K)
# neighbour_stack order (up, down, left, right); each node depends on its left neighbour.
path_edge[0, 0, 0, :, 2] = torch.eye(cfg.K)
path_valid = neighbour_validity(1, 5, torch.device("cpu")).permute(1, 2, 0)[None, None]
direct = torch.zeros(1, 1, cfg.K, 1, 5, dtype=torch.bool)
direct[0, 0, 0, 0, 0] = True
select, priority = rc.propagate(direct, path_edge, path_valid)
report["two_hop_path"] = {
    "selected": select[0, 0, 0, 0].tolist(),
    "priority": priority[0, 0, 0, 0].tolist(),
}

# Is the mixture objective sensitive to the actual posterior?
energy = torch.zeros(2, 1, 4, requires_grad=True)
weights = torch.tensor([[[0.01, 0.48, 0.01, 0.50]], [[0.25, 0.25, 0.25, 0.25]]], requires_grad=True)
rep = SimpleNamespace(
    measurement_energy=energy,
    weights=weights,
    geometry=torch.tensor([-1.0, 1.0, -1.0, 1.0])[None, None, :, None].expand(2, 1, 4, 1),
    h=torch.randn(2, 1, 4, 2, 2, 2),
)
terms = step_terms(
    rep,
    anchor=torch.tensor([0, 1]),
    wrong=torch.tensor([1, 0]),
    post_contradiction=torch.tensor([False, False]),
    observations=1,
    diversity_active=False,
    identifiable_now=torch.tensor([False, False]),
)
grad = torch.autograd.grad(
    terms["mixture"], (energy, weights), retain_graph=True, allow_unused=True
)
loss_grad = torch.autograd.grad(terms["mixture"] + terms["measurement"], energy)[0]
report["uniform_objective_probe"] = {
    "mixture": float(terms["mixture"].detach()),
    "mixture_energy_gradient": grad[0].tolist(),
    "mixture_posterior_gradient": None if grad[1] is None else grad[1].tolist(),
    "measurement_plus_mixture_energy_gradient": loss_grad.tolist(),
    "posterior_class0": [float(weights[i, 0, ::2].detach().sum()) for i in range(2)],
}
# All source negatives are dead under clamp; independent edge-variable gradient.
raw = torch.tensor([-0.1, 1.0, 0.1], requires_grad=True)
clamped = raw.clamp_min(0)
weight = clamped / clamped.sum()
weight[2].backward()
report["clamp_edge_probe"] = {"raw": raw.tolist(), "grad": raw.grad.tolist()}
logs = [
    json.loads(line)
    for line in Path("outputs/n02-final-candidate/train_log.jsonl").read_text().splitlines()
]
report["training_sampling"] = {
    "steps": len(logs),
    "steps_revision_active": sum(row["terms"]["revision"] is not None for row in logs),
    "frame_samples_total": sum(row["effective_samples"]["measurement"] for row in logs),
    "frame_samples_revision": sum(row["effective_samples"]["revision"] for row in logs),
    "training_prior_confidence_bounds": [1 / (1 + math.exp(-0.6)), 1 / (1 + math.exp(-1.6))],
    "acceptance_prior_confidence": 0.95,
}
Path("outputs/n02-source-review/report.json").write_text(json.dumps(report, indent=2) + "\n")
summary = {
    "sampling": report["training_sampling"],
    "sparse": report["sparse_propagation"],
    "two_hop": report["two_hop_path"],
    "objective": report["uniform_objective_probe"],
    "dead_config": report["downweight_config_probe"],
    "checkpoint_sha256": report["checkpoint_sha256"],
    "scope": report["scope"],
    "replays": [],
}
for r in report["replays"]:
    summary["replays"].append(
        {
            "variant": r["variant"],
            "condition": r["condition"],
            "seed": r["seed"],
            "truth_mass": [round(t["truth_mass"], 4) for t in r["steps"]],
            "semantic0": [round(t["semantic0"], 4) for t in r["steps"]],
            "current_evidence0": [round(t["evidence_semantic0"], 4) for t in r["steps"]],
            "direct_per_frame": [
                sum(sum(c["direct_by_slot"]) for c in t["revision"]) for t in r["steps"]
            ],
            "extra_per_frame": [
                sum(sum(c["extra_by_slot"]) for c in t["revision"]) for t in r["steps"]
            ],
        }
    )
Path("outputs/n02-source-review/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "replays"}, indent=2))
