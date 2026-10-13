"""Independent, frozen-weight spatial withdrawal/recompute intervention.

Run only after the preregistered DBSR baselines have reached their operational
plateau. All modes share the same full-size weights and a new sequence seed
namespace. This isolates the operator; it is not matched candidate retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from owac.baselines.burst_features import BurstFeatureConfig
from owac.frameworks.n01_dbsr import MODES, sequence_features, support_config

from .n01_burst import BurstProbe, parameter_digest
from .n01_burst_training import atomic_json, configure_runtime, training_sources
from .n01_content import ORACLE, make_protocol_batch, summarize_sequence
from .n01_content_data import CONDITIONS


def evaluate_sequence(model, batch, mode, protocol, support_overrides):
    error, count, mass, effective, graph_rows = [], [], [], [], []
    for representation, weights, record, graph in sequence_features(
        model,
        batch,
        mode,
        support_config(support_overrides),
        freeze_frame=protocol["freeze_frame"],
        encode_chunk=protocol["encode_chunk"],
    ):
        logits = model.head(representation)
        now = len(error)
        target = batch.target_xy[:, now].to(representation.device)
        error.append((model.head.soft_argmax(logits) - target).norm(dim=1).cpu())
        count.append(float((weights > 0).sum(1).float().mean()))
        mass.append(float(weights[:, 1:].sum(1).mean()))
        usable = weights.sum(1) > 0
        effective.append(
            float((weights.square().sum(1).clamp_min(1e-12).reciprocal() * usable).mean())
        )
        graph_rows.append(
            {**record, "applied_shared_open_fraction": float(graph.all(1).float().mean())}
        )
    metrics = summarize_sequence(torch.stack(error, 1), batch.visible, count, batch, protocol)
    metrics["contributing_feature_source_count_per_frame"] = metrics.pop(
        "selected_unique_frames_per_frame"
    )
    metrics["historical_feature_weight_mass_per_frame"] = mass
    metrics["effective_feature_source_count_per_frame"] = effective
    metrics["graph_per_frame"] = graph_rows
    return metrics


def source_snapshot():
    sources = training_sources()
    root = Path(__file__).resolve().parents[3]
    for path in [Path(__file__), root / "src/owac/frameworks/n01_dbsr.py"]:
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def probe(protocol, baseline_root, output):
    baseline_root, output = Path(baseline_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("probe output must be empty")
    # Audit all required baselines before consuming any mechanism holdout data.
    baseline_rows = {}
    for seed in protocol["seeds"]:
        for method in protocol["required_baselines"]:
            row = json.loads((baseline_root / f"{seed}-{method}/result.json").read_text())
            if not row["operationally_settled"]:
                raise ValueError(f"baseline {seed}-{method} is budget-limited, not settled")
            baseline_rows[f"{seed}-{method}"] = {
                "steps": row["steps"],
                "best_step": row["best_step"],
                "parameter_digest": row["best_parameter_digest"],
                "protocol": row["protocol"],
            }
    configure_runtime(protocol)
    sources = source_snapshot()
    payload = {
        "protocol": protocol,
        "sources": sources,
        "baseline_acceptance": baseline_rows,
        "scope": (
            "Frozen fully trained weights; param-free N01 spatial support intervention. "
            "No temporal source rejection and no matched graph-model retraining."
        ),
        "oracle_alignment_order": "RAW aligned before encoding in every intervention arm",
        "runs": [],
    }
    atomic_json(output / "run.json", payload)
    for seed in protocol["seeds"]:
        path = baseline_root / f"{seed}-dbsr_reference/best.pt"
        checkpoint = torch.load(path, map_location=protocol["device"], weights_only=True)
        model = BurstProbe(BurstFeatureConfig(**checkpoint["encoder_config"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(protocol["device"]).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        initial_digest = parameter_digest(model)
        if initial_digest != baseline_rows[f"{seed}-dbsr_reference"]["parameter_digest"]:
            raise ValueError("checkpoint differs from accepted baseline")
        interventions = [("none", "stock", "none")]
        interventions += [
            (f"{variant}_{mode}", variant, mode)
            for variant in protocol["support_variants"]
            for mode in MODES
            if mode != "none"
        ]
        for name, variant, mode in interventions:
            started = time.perf_counter()
            cases = {}
            with torch.no_grad():
                for index, condition in enumerate(CONDITIONS):
                    batch = make_protocol_batch(
                        protocol, seed + protocol["probe_seed_offset"], "test", index, condition
                    )
                    cases[condition] = evaluate_sequence(
                        model, batch, mode, protocol, protocol["support_variants"][variant]
                    )
            ordinary = [r for name, r in cases.items() if name != ORACLE]
            row = {
                "seed": seed,
                "mode": name,
                "support_variant": variant,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parameter_digest": initial_digest,
                "parameters": sum(p.numel() for p in model.parameters()),
                "cases": cases,
                "main_early_error_px": sum(r["early_error_px"] for r in ordinary) / len(ordinary),
                "main_mean_error_px": sum(r["mean_error_px"] for r in ordinary) / len(ordinary),
                "seconds": time.perf_counter() - started,
            }
            if parameter_digest(model) != initial_digest:
                raise RuntimeError("frozen probe modified model parameters")
            payload["runs"].append(row)
            atomic_json(output / "partial.json", payload)
            print(json.dumps({k: v for k, v in row.items() if k != "cases"}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if source_snapshot() != sources:
        raise RuntimeError("probe source changed during execution")
    # Every mode must have exactly the same measurements and supervision fixtures.
    for seed in protocol["seeds"]:
        rows = [r for r in payload["runs"] if r["seed"] == seed]
        for condition in CONDITIONS:
            for key in ["input_sha256", "fixture_sha256"]:
                if len({r["cases"][condition][key] for r in rows}) != 1:
                    raise RuntimeError("intervention inputs are not paired")
    atomic_json(output / "result.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    probe(json.loads(Path(args.config).read_text()), args.baselines, args.output)


if __name__ == "__main__":
    main()
