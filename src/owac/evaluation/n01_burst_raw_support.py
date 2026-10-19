"""Fresh holdout: original RAW evidence integration with unchanged DBSR layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from owac.baselines.burst_features import BurstFeatureConfig
from owac.frameworks.n01_dbsr_raw import MODES, sequence_features

from .n01_burst import BurstProbe, parameter_digest
from .n01_burst_training import atomic_json, configure_runtime, training_sources
from .n01_content import ORACLE, make_protocol_batch, summarize_sequence
from .n01_content_data import CONDITIONS


def evaluate_sequence(model, batch, mode, protocol):
    errors, counts, rows = [], [], []
    for feature, weights, stats in sequence_features(model, batch, mode, protocol):
        now = len(errors)
        target = batch.target_xy[:, now].to(feature.device)
        errors.append((model.head.soft_argmax(model.head(feature)) - target).norm(dim=1).cpu())
        counts.append(float((weights > 0).sum(1).float().mean()))
        rows.append(
            {
                "mean_variance_proxy": float(stats["variance"].mean()),
                "mean_age": float(stats["age"].mean()),
                "merges": stats["merges"],
                "splits": stats["splits"],
                "plane_open_fraction": stats["plane_open_fraction"],
                "unique_observation_count": sum(x >= 0 for x in stats["unique_source_ids"][0]),
                "derived_feature_is_new_measurement": False,
            }
        )
    metrics = summarize_sequence(torch.stack(errors, 1), batch.visible, counts, batch, protocol)
    metrics["contributing_feature_node_count_per_frame"] = metrics.pop(
        "selected_unique_frames_per_frame"
    )
    metrics["raw_domain_per_frame"] = rows
    return metrics


def sources():
    result = training_sources()
    root = Path(__file__).resolve().parents[3]
    for path in [Path(__file__), root / "src/owac/frameworks/n01_dbsr_raw.py"]:
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def probe(protocol, baselines, output):
    baselines, output = Path(baselines), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("output must be empty")
    for seed in protocol["seeds"]:
        for method in protocol["required_baselines"]:
            row = json.loads((baselines / f"{seed}-{method}/result.json").read_text())
            if not row["operationally_settled"]:
                raise ValueError("baseline is not settled")
    configure_runtime(protocol)
    snapshot = sources()
    payload = {
        "protocol": protocol,
        "sources": snapshot,
        "runs": [],
        "scope": (
            "Frozen mature weights; original per-plane RAW support-domain integration, "
            "fixed radius2/history8; no CNN topology change or new network"
        ),
        "feature_node_semantics": (
            "Extra current-frame or derived RAW-integral feature; raw unique observations "
            "remain at most8, feature nodes are not independent observations"
        ),
    }
    atomic_json(output / "run.json", payload)
    for seed in protocol["seeds"]:
        path = baselines / f"{seed}-dbsr_reference/best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = BurstProbe(BurstFeatureConfig(**checkpoint["encoder_config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(protocol["device"]).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        digest = parameter_digest(model)
        for mode in MODES:
            started = time.perf_counter()
            cases = {}
            with torch.no_grad():
                for index, condition in enumerate(CONDITIONS):
                    batch = make_protocol_batch(
                        protocol, seed + protocol["probe_seed_offset"], "test", index, condition
                    )
                    cases[condition] = evaluate_sequence(model, batch, mode, protocol)
            ordinary = [r for name, r in cases.items() if name != ORACLE]
            row = {
                "seed": seed,
                "mode": mode,
                "cases": cases,
                "parameters": sum(p.numel() for p in model.parameters()),
                "parameter_digest": digest,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "main_early_error_px": sum(r["early_error_px"] for r in ordinary) / len(ordinary),
                "main_mean_error_px": sum(r["mean_error_px"] for r in ordinary) / len(ordinary),
                "seconds": time.perf_counter() - started,
            }
            if parameter_digest(model) != digest:
                raise RuntimeError("parameters modified")
            payload["runs"].append(row)
            atomic_json(output / "partial.json", payload)
            print(json.dumps({k: v for k, v in row.items() if k != "cases"}), flush=True)
        del model
        torch.cuda.empty_cache()
    for seed in protocol["seeds"]:
        rows = [r for r in payload["runs"] if r["seed"] == seed]
        for condition in CONDITIONS:
            for key in ["input_sha256", "fixture_sha256"]:
                assert len({r["cases"][condition][key] for r in rows}) == 1
    if sources() != snapshot:
        raise RuntimeError("execution source changed")
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
