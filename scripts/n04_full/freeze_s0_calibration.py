"""Freeze model-independent spatial gates from audited training cameras only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from owac.raw.n04_live_collection import save_json, sha256


def freeze(audit_path: Path, parent_protocol: Path) -> dict:
    audit = json.loads(audit_path.read_text())
    if not audit["passed"]:
        raise ValueError("cannot freeze calibration from a failed data audit")
    groups = defaultdict(list)
    source_ids = defaultdict(set)
    for record in audit["records"]:
        if record["split"] != "train":
            if record["train_calibration"]:
                raise ValueError("non-training calibration leaked")
            continue
        for row in record["train_calibration"]:
            key = record["platform"] + "/" + row["camera"]
            groups[key].append(row)
            source_ids[key].add(record["source_trajectory_id"])
    references = {}
    for key, rows in sorted(groups.items()):
        f = float(np.median([r["focal_px"] for r in rows]))
        z = float(np.median([r["median_depth_m"] for r in rows]))
        references[key] = dict(
            train_sources=len(source_ids[key]),
            frame_samples=len(rows),
            focal_px=f,
            reference_depth_m=z,
            one_original_pixel_m=z / f,
            absolute_improvement_reference_m=z / f,
            visible_segment_tolerance_reference_m=z / f,
            inherited_point05px_reference_m=0.05 * z / f,
        )
    return dict(
        schema="n04-calibrated-simulation-protocol-v1",
        frozen=True,
        frozen_before_any_s1_training=True,
        parent_protocol_sha256=sha256(parent_protocol),
        data_audit_sha256=sha256(audit_path),
        calibration_source="training split only",
        camera_references=references,
        absolute_rule=(
            "relative improvement >=10% AND absolute improvement strictly above one "
            "original pixel at the scored Z/fx; 0.05px inherited floor is smaller"
        ),
        pointwise_rule_m="max(0.05, 1.0) * Z_m / fx_original_px",
        visible_segment_tolerance_m="Z_m / fx_original_px",
        selection_rule=(
            "Z/fx evaluated in the declared metric space; no fit to dev/final or model errors"
        ),
        pixel_coordinates="integer pixel centres; K converted before supervision",
        unchanged_hypothesis_gates=dict(
            h1_false_positive_budget=0.01,
            h1_false_positive_reduction=0.20,
            h1_recall_drop_max_pp=2.0,
            h2_stop_reverse_reduction=0.15,
            h2_missing_degradation_max=0.05,
        ),
        scope=(
            "new calibrated simulator extensions; no real sensor calibration claim; "
            "original nominal protocol retained as historical reference"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, freeze(args.audit, args.parent))


if __name__ == "__main__":
    main()
