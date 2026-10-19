"""Reproducible validation sampling and resumable real-source coverage receipts."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field


def validation_indices(length, count, seed, *, mode="fixed", step=0):
    if length < 1 or count < 1:
        raise ValueError("validation requires real nonempty samples")
    count = min(count, length)
    if mode == "fixed":
        # Spread across the whole dataset rather than taking adjacent windows from
        # the first trajectory. Identical indices and flow RNG at every checkpoint.
        return [i * length // count for i in range(count)]
    if mode != "random":
        raise ValueError("validation mode must be fixed or random")
    return random.Random(seed * 104729 + step).sample(range(length), count)


@dataclass
class Coverage:
    windows: set[int] = field(default_factory=set)
    trajectories: Counter = field(default_factory=Counter)
    conditions: Counter = field(default_factory=Counter)
    samples: int = 0

    def observe(self, index, meta):
        self.samples += 1
        self.windows.add(int(index))
        self.trajectories[str(meta.get("episode_key", "unknown"))] += 1
        condition = meta.get("condition", meta.get("lighting_domain", meta.get("variant_id")))
        self.conditions[str(condition) if condition is not None else "unavailable"] += 1

    def to_dict(self):
        return {
            "windows": sorted(self.windows),
            "trajectories": dict(self.trajectories),
            "conditions": dict(self.conditions),
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            set(data.get("windows", [])),
            Counter(data.get("trajectories", {})),
            Counter(data.get("conditions", {})),
            int(data.get("samples", 0)),
        )

    def report(self, dataset_length):
        return {
            "samples": self.samples,
            "distinct_windows": len(self.windows),
            "dataset_windows": dataset_length,
            "window_coverage": len(self.windows) / max(dataset_length, 1),
            "repeat_fraction": 1 - len(self.windows) / max(self.samples, 1) if self.samples else 0,
            "distinct_trajectories": len(self.trajectories),
            "trajectory_samples": dict(self.trajectories),
            "condition_samples": dict(self.conditions),
            "missing_condition_metadata": self.conditions.get("unavailable", 0),
        }
