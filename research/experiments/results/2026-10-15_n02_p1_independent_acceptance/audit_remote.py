"""Read-only P1 acceptance audit; run from the delivered remote source root.

Does not train, change dictionaries/thresholds, or overwrite original receipts.
The optional geometry probe compares the delivered oracle's coordinate lookup
with direct indexing of the exact samples read by build_raw_query.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import inspect
import json
import platform
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(directory):
    path = directory / "scenario_records.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    result = {"sha256": sha(path), "rows": len(rows), "families": {}}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        item = {
            "rows": len(selected),
            "unique_scenes": len({row["scenario_id"] for row in selected}),
            "seeds": sorted({row["seed"] for row in selected}),
        }
        for key in ("correct", "corrected", "detected", "flagged"):
            if key in selected[0]:
                item[key] = sum(bool(row[key]) for row in selected)
        if family == "oracle":
            spreads = [max(row["scores"]) - min(row["scores"]) for row in selected]
            item.update(
                exact_all_equal=sum(value == 0 for value in spreads),
                spread_max=max(spreads),
                truth_unique_minimum=sum(
                    row["candidate_index"] == row["map_index"]
                    and row["scores"].count(min(row["scores"])) == 1
                    for row in selected
                ),
                first_records=selected[:2],
            )
        if family == "cue_swap":
            pairs = collections.defaultdict(dict)
            for row in selected:
                pairs[row["scenario_id"]][row["direction"]] = row["correct"]
            item["paired_both_correct"] = sum(
                pair.get("a", False) and pair.get("b", False) for pair in pairs.values()
            )
        for direction in ("a", "b"):
            group = [row for row in selected if row.get("direction") == direction]
            if group:
                item[direction] = {
                    "total": len(group),
                    "successes": sum(
                        bool(row.get("correct", row.get("corrected", False))) for row in group
                    ),
                }
        if family == "no_information":
            item["drift_max"] = max(row["drift"] for row in selected)
            item["within_0.05"] = sum(row["drift"] <= 0.05 for row in selected)
        if family in ("normal", "insufficiency"):
            item["verdict_tuples"] = dict(
                collections.Counter(str(row.get("verdicts")) for row in selected)
            )
            per_group = {}
            for row in selected:
                per_group.setdefault(row["scenario_id"], []).append(row)
            item["same_outcome_across_groups"] = sum(
                len(
                    {
                        (str(row.get("verdicts")), row.get("detected"), row.get("flagged"))
                        for row in group
                    }
                )
                == 1
                for group in per_group.values()
            )
        result["families"][family] = item
    return result


def main():
    import numpy as np
    import torch

    from owac.evaluation.n02_full import p1 as evaluation
    from owac.frameworks.n02_full import config, factors, synthetic
    from owac.frameworks.n02_full import model as model_module
    from owac.training import n02_full as training

    torch.set_num_threads(4)
    root = Path.cwd()
    cfg = config.load_config("configs/encoders/n02_full/p1_measurement.json")
    measurement = dict(cfg.training["measurement"])
    files = [
        *Path("src/owac/frameworks/n02_full").glob("*.py"),
        Path("src/owac/training/n02_full.py"),
        Path("src/owac/evaluation/n02_full/p1.py"),
        *Path("configs/encoders/n02_full").glob("p1*.json"),
    ]
    result = {
        "root": str(root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_marker_exists": (root / ".git").exists(),
        "source_hashes": {str(path): sha(path) for path in sorted(files)},
        "acceptance": {},
        "training_runs": {},
    }
    for name in ("p1-acceptance-interim", "p1-acceptance-5k", "p1-acceptance-5k-true"):
        directory = Path("outputs/n02-full") / name
        result["acceptance"][name] = read_records(directory)
    for directory in sorted(Path("outputs/n02-full/p1-formal").glob("p1-*")):
        entry = {"files": sorted(path.name for path in directory.iterdir())}
        log = directory / "train_log.jsonl"
        if log.exists():
            rows = [json.loads(line) for line in log.open() if line.strip()]
            entry["log_rows"] = len(rows)
            entry["first_log"] = rows[0] if rows else None
            entry["last_log"] = rows[-1] if rows else None
        manifest = directory / "run_manifest.json"
        if manifest.exists():
            entry["run_manifest"] = json.loads(manifest.read_text())
        cp = directory / "checkpoints/final.pt"
        if cp.exists() and "meas-5k" in directory.name:
            payload = training.load_checkpoint(cp)
            entry["checkpoint"] = {
                "path": str(cp),
                "bytes": cp.stat().st_size,
                "sha256": sha(cp),
                "keys": sorted(payload),
                "metadata": {
                    key: payload[key]
                    for key in (
                        "format",
                        "stage",
                        "updates",
                        "seed",
                        "config_hash",
                        "source_updates",
                        "sampler",
                        "metadata",
                    )
                    if key in payload
                },
            }
            del payload
        result["training_runs"][directory.name] = entry
    sensors = tuple(synthetic.SensorSpec(**entry) for entry in measurement["synthetic_sensors"])
    generator = synthetic.MeasurementModel(**measurement["measurement_model"])
    data_args = dict(
        sensors=sensors,
        measurement_model=generator,
        partitions=("train",),
        scenes=3000,
        frames=8,
        views_per_scene=1,
        seed=1103,
        canvas=tuple(measurement["canvas"]),
        action_horizon=measurement.get("action_horizon", 4),
    )
    dataset = synthetic.generate_dataset(**data_args)
    stub = SimpleNamespace(
        config=SimpleNamespace(**training.p1_component_config(cfg)), feature_dim=512
    )
    probes = []
    for sequence in dataset:
        task = factors.build_p1_task(sequence, frame_index=1)
        row_query = model_module.build_row_query(stub, task.inference.bundle)
        raw_query = model_module.build_raw_query(row_query)
        hypotheses = [
            evaluation.oracle_hypothesis(candidate, sequence.truth[1].object_ids)
            for candidate in task.dictionary.specs
        ]
        maps = [
            [
                generator.mean_variance(sensor, sequence.render_states[1], hypothesis=hypothesis)
                for sensor in sensors
            ]
            for hypothesis in hypotheses
        ]
        integer_means = []
        rounded_means = []
        full_means = []
        for hypothesis, camera_maps in zip(hypotheses, maps, strict=True):
            full_means.append(np.concatenate([m[0].reshape(-1) for m in camera_maps]))
            values = []
            for sensor, (mean, _) in zip(sensors, camera_maps, strict=True):
                _, y, x = row_query.sampling[sensor.camera_id]
                values.extend(np.asarray(mean)[y.numpy(), x.numpy()].reshape(-1).tolist())
            integer_means.append(np.asarray(values))
            mean_fn = evaluation._oracle_mean_variance(
                generator, sequence.render_states[1], {s.camera_id: s for s in sensors}, hypothesis
            )
            rounded_means.append(mean_fn(None, row_query.spec)[0].numpy())
        item = {
            "scene_id": sequence.scene_id,
            "world_size": task.dictionary.world_size(),
            "candidate_index": task.candidate_index,
            "query_points": len(row_query.spec.camera_ids),
            "observed_values_minmax": [
                float(raw_query.values.min()),
                float(raw_query.values.max()),
            ],
            "integer_vs_delivered_mean_abs_max": max(
                float(np.max(np.abs(a - b)))
                for a, b in zip(integer_means, rounded_means, strict=True)
            ),
            "candidate_mean_spread_query_integer": max(
                float(np.max(np.abs(a - integer_means[0]))) for a in integer_means
            ),
            "candidate_mean_spread_query_delivered": max(
                float(np.max(np.abs(a - rounded_means[0]))) for a in rounded_means
            ),
            "candidate_mean_spread_full_sensor": max(
                float(np.max(np.abs(a - full_means[0]))) for a in full_means
            ),
        }
        probes.append(item)
        if len(probes) == 8:
            break
    result["oracle_geometry_probes"] = probes
    result["static_wiring"] = {
        "step_bundle_has_declared_prior_argument": "declared_prior"
        in inspect.signature(model_module.step_bundle).parameters,
        "evaluator_model_step_mentions_declared_prior": "declared_prior"
        in inspect.getsource(evaluation._model_step),
        "model_module_mentions_declared_prior": "declared_prior" in inspect.getsource(model_module),
        "model_module_imports_checks": "from .checks" in inspect.getsource(model_module),
        "measurement_score_fields": [
            field.name
            for field in dataclasses.fields(
                __import__(
                    "owac.frameworks.n02_full.sensor", fromlist=["MeasurementScore"]
                ).MeasurementScore
            )
        ],
    }
    print(json.dumps(result, ensure_ascii=False, default=lambda value: str(value)))


if __name__ == "__main__":
    main()
