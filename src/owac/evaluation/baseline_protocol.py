"""Small, portable evidence utilities for the frozen single-task baseline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wilson_interval(successes: int, count: int) -> list[float]:
    if not 0 <= successes <= count or count <= 0:
        raise ValueError("Expected a positive number of independent episodes")
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2))
    radius /= 1 + z * z / count
    return [max(0.0, center - radius), min(1.0, center + radius)]


def action_chunk(actions, start: int, horizon: int):
    """Repeat the final action within this episode, never cross episode boundaries."""
    import numpy as np

    if not 0 <= start < len(actions) or horizon <= 0:
        raise ValueError("Invalid action chunk bounds")
    return actions[np.minimum(np.arange(start, start + horizon), len(actions) - 1)]


def physical_success(*, predicate, lift, xy, height, fingers, open_command=True) -> bool:
    return bool(
        predicate
        and lift >= 0.10
        and xy <= 0.04
        and 0.025 < height < 0.09
        and min(fingers) >= 0.035
        and open_command
    )
