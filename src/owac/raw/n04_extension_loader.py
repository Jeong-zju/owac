"""Explicit input/target boundary for S1 extension data; no implicit GT inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LiveExtensionSequence:
    """Saved paired data. State/action outcomes remain supervision-only."""

    def __init__(self, record: str | Path):
        self.path = Path(record)
        self.record = json.loads(self.path.read_text())
        if (
            self.record["schema"] != "n04-live-extension-v2"
            or self.record["metadata"].get("pixel_coordinates") != "integer_pixel_centres_top_left"
        ):
            raise ValueError("unexpected extension schema")

    def __len__(self):
        return len(self.record["frames"])

    def training_frame(self, index: int) -> dict:
        frame = self.record["frames"][index]
        with np.load(self.path.parent / frame["file"], allow_pickle=False) as arrays:
            inputs = {k: arrays[k].copy() for k in arrays.files if k.startswith("online_")}
            targets = {k: arrays[k].copy() for k in arrays.files if k.startswith("gt_")}
        return dict(
            inputs=inputs,
            targets=targets,
            timing=dict(frame),
            source_trajectory_id=self.record["source_trajectory_id"],
            split=self.record["split"],
        )

    def arrived_frames(self, decision_host_s: float) -> tuple[int, ...]:
        return tuple(
            f["index"]
            for f in self.record["frames"]
            if f["host_sample_s"] <= decision_host_s and f["host_arrival_s"] <= decision_host_s
        )


def rendered_training_frame(path: str | Path, index: int) -> dict:
    """Actually applies delayed-frame and wrong-calibration interventions.

    Input frame mapping is causal (never a future index). Actual world/geometry
    at target time stays separate. Association indices are diagnostic proposals,
    not true object IDs supplied to a deployed model.
    """
    with np.load(path, allow_pickle=False) as data:
        observed = int(data["input_frame_index"][index])
        if not 0 <= observed <= index:
            raise ValueError("noncausal frame mapping")
        return dict(
            inputs=dict(
                raw=data["online_rawvla_rgb3"][observed].copy(),
                K=data["input_K"][observed].copy(),
                sample_time_s=float(data["gt_time_s"][observed]),
                association_proposal=data["input_association"].copy(),
            ),
            targets={k: data[k][index].copy() for k in data.files if k.startswith("gt_")},
            measurement_frame=observed,
            target_frame=index,
        )
