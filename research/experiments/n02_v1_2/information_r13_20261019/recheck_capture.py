"""Cold regenerate one held-out scene to verify saved image/label correspondence."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r13_capture", HERE / "capture.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.OUT
selected = module.selection()[40]
module.OUT = original / "cold_recheck"
module.selection = lambda: [selected]
os.chdir(module.SOURCE / "third_party/RoboTwin")
module.capture(SimpleNamespace(scenes=1))
rows = {}
with np.load(original / "snapshots/scene_040.npz") as before:
    with np.load(module.OUT / "snapshots/scene_040.npz") as after:
        for key in ("raw", "rgb", "y", "state"):
            rows[key] = dict(
                bitwise_equal=bool(np.array_equal(before[key], after[key])),
                max_absolute_difference=float(np.abs(before[key].astype(float) - after[key]).max()),
            )
(HERE / "capture_cold_recheck.json").write_text(
    json.dumps(
        dict(
            scene=40,
            rows=rows,
            policy_updates=0,
            policy_actions=0,
            scene_initialisation="standard scripted homing",
            note=("Atomic capture reproduces saved inputs and labels. No policy success claim."),
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(rows), flush=True)
