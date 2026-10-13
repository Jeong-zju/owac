"""Frozen C-weight intervention for optional N01 source reliability.

Fresh test seeds, no retraining/tuning, no selector calibration change. Reports
full transient/recovery metrics and selected history mass, not just frame count.
This is an exploratory rule intervention, not a trained capacity comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import MethodType

import torch

from ..frameworks.n01.config import N01Config
from .n01_content import ORACLE, ContentProbe, make_protocol_batch, parameter_hashes, run_sequence
from .n01_content_data import CONDITIONS


def digest(*tensors):
    checksum = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        checksum.update(str((value.shape, value.dtype)).encode())
        checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


def manifest():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src/owac/frameworks/n01").glob("*.py"))
    paths += [
        Path(__file__),
        Path(__file__).with_name("n01_content.py"),
        Path(__file__).with_name("n01_content_data.py"),
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def copy_intervention(checkpoint, mode, floor):
    """Explicit shared-weight copy; preserve the new mode's semantic buffer."""
    config = N01Config.from_dict(checkpoint["encoder_config"]).replace(
        source_reliability=mode,
        source_reliability_floor=floor,
    )
    model = ContentProbe(config)
    state = model.state_dict()
    with torch.no_grad():
        for name, value in checkpoint["model"].items():
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f"unmatched pretrained tensor: {name}")
            state[name].copy_(value)
    model.load_state_dict(state, strict=True)
    return model.eval()


def evaluate_case(model, batch, protocol):
    """Read-only diagnostic instrumentation; additional cost is disclosed."""
    encoder, records = model.encoder, []
    integrator = encoder.integrator
    original_statistics, original_step = integrator.domain_statistics, encoder.step

    def collect_statistics(self, state, edges, motion=None, estimator=None, depths=None):
        stats, connection = original_statistics(state, edges, motion, estimator, depths)
        if self.config.source_reliability == "none":
            current, _ = original_statistics(state, edges, motion, estimator, depths=(1,))
            # One depth in this fixed-horizon intervention. This extra diagnostic
            # integration is excluded from the actual encoder/selector result.
            if stats.mu.shape != current.mu.shape:
                raise ValueError("source probe expects exactly one fixed time depth")
            self.probe_history_mass = (
                1 - current.support_count / stats.support_count.clamp_min(1e-6)
            ).clamp(0, 1)
            self.probe_history_mass = torch.where(
                stats.support_count > 0, self.probe_history_mass, 0
            )
        else:
            self.probe_history_mass = self.last_history_mass
        return stats, connection

    def collect_step(self, frame, *, state):
        result = original_step(frame, state=state)
        selection = self.last_selection_weights
        indexes = self.last_selection_slice
        statistics = self.last_statistics
        mass = (selection * integrator.probe_history_mass[indexes]).sum(0).mean(1)
        age = (selection * statistics.age[indexes]).sum(0).mean(1)
        index = len(records)
        target = batch.target_xy[:, index]  # Reporting only, after encoding/readout inputs.
        rows = torch.arange(mass.shape[-2])[None, :, None]
        cols = torch.arange(mass.shape[-1])[None, None, :]
        roi = (rows - target[:, 0, None, None]).square() + (
            cols - target[:, 1, None, None]
        ).square() <= 4
        new = result.state
        observed = new.observed[:, :, None, None, None]
        valid = new.valid & observed
        reliability = (
            valid.float()
            if self.config.source_reliability == "none"
            else integrator.last_source_reliability
        )
        old_valid = valid[1:]
        denominator = old_valid.sum().clamp_min(1)
        support = new.support["camera"]
        records.append(
            {
                "history_mass_mean": float(mass.mean()),
                "roi_history_mass_mean": float(
                    ((mass * roi).sum((1, 2)) / roi.sum((1, 2)).clamp_min(1)).mean()
                ),
                "weighted_age_mean": float(age.mean()),
                "roi_weighted_age_mean": float(
                    ((age * roi).sum((1, 2)) / roi.sum((1, 2)).clamp_min(1)).mean()
                ),
                "old_source_reliability_mean": float(
                    (reliability[1:] * old_valid).sum() / denominator
                ),
                "old_source_rejected_fraction": float(
                    ((reliability[1:] == 0) & old_valid).sum() / denominator
                ),
                "raw_buffer_sha256": digest(
                    new.values, new.variance, new.valid, new.observed, new.frame_ids, new.timestamps
                ),
                "support_graph_sha256": digest(
                    support.edges, support.open_edges, support.merge_count, support.split_count
                ),
            }
        )
        return result

    integrator.domain_statistics = MethodType(collect_statistics, integrator)
    encoder.step = MethodType(collect_step, encoder)
    try:
        with torch.no_grad():
            _, metrics = run_sequence(model, batch, protocol)
    finally:
        integrator.domain_statistics, encoder.step = original_statistics, original_step
    metrics["source_diagnostics_per_frame"] = records
    return metrics


def run(probe, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(Path(probe["protocol_path"]).read_text())
    torch.set_num_threads(int(protocol["threads"]))
    source_start = manifest()
    receipts = []
    for training_seed, data_seed in zip(
        probe["training_seeds"], probe["evaluation_seeds"], strict=True
    ):
        path = Path(probe["checkpoints"].format(seed=training_seed))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["seed"] != training_seed or checkpoint["method"] != "C":
            raise ValueError("checkpoint provenance mismatch")
        models = {
            mode: copy_intervention(checkpoint, mode, float(probe["source_reliability_floor"]))
            for mode in probe["methods"]
        }
        hashes = {mode: parameter_hashes(model) for mode, model in models.items()}
        if any(value != next(iter(hashes.values())) for value in hashes.values()):
            raise ValueError("intervention parameters differ")
        runs = {mode: {} for mode in models}
        for index, condition in enumerate(CONDITIONS):
            batch = make_protocol_batch(protocol, data_seed, "test", index, condition)
            for mode, model in models.items():
                started = time.perf_counter()
                runs[mode][condition] = evaluate_case(model, batch, protocol)
                print(
                    json.dumps(
                        {
                            "training_seed": training_seed,
                            "data_seed": data_seed,
                            "condition": condition,
                            "mode": mode,
                            "early_error_px": runs[mode][condition]["early_error_px"],
                            "wall_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        comparisons = {}
        for condition in CONDITIONS:
            first, second = [runs[mode][condition] for mode in probe["methods"]]
            if first["fixture_sha256"] != second["fixture_sha256"]:
                raise ValueError("input/target fixture mismatch")
            for key in ("raw_buffer_sha256", "support_graph_sha256"):
                if [row[key] for row in first["source_diagnostics_per_frame"]] != [
                    row[key] for row in second["source_diagnostics_per_frame"]
                ]:
                    raise ValueError(f"protected {key} changed")
            comparisons[condition] = {
                key: second[key] - first[key]
                for key in ("early_error_px", "stable_error_px", "mean_error_px")
            }
        receipt = {
            "training_seed": training_seed,
            "evaluation_seed": data_seed,
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_step": checkpoint["step"],
            "parameters": hashes,
            "runs": runs,
            "differences_risk_minus_original": comparisons,
            "main_early_difference_px": sum(
                row["early_error_px"] for name, row in comparisons.items() if name != ORACLE
            )
            / 3,
        }
        receipts.append(receipt)
        (output / f"seed{training_seed}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    source_end = manifest()
    if source_start != source_end:
        raise ValueError("source changed during experiment")
    result = {
        "protocol": probe,
        "data_protocol": protocol,
        "source_start": source_start,
        "source_end": source_end,
        "paired_raw_and_graph_audit": "passed",
        "measurement_note": (
            "Frozen-weight intervention; extra baseline current-depth diagnostic work and "
            "hashing included in wall times, not an efficiency comparison."
        ),
        "records": receipts,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
