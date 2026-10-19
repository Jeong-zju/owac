"""Restore the archived CuRobo Python source and the author's simulator fixes."""

import importlib.util
import json
import shutil
from pathlib import Path

root = Path("/workspace/owac/rawvla")
archived = root / "records/baseline-archive/remote/source/third_party/RoboTwin/envs/curobo/src"
changed = []
for source in archived.rglob("*.py"):
    target = root / "curobo/src" / source.relative_to(archived)
    if not target.exists() or source.read_bytes() != target.read_bytes():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        changed.append(str(target))

sapien = Path(importlib.util.find_spec("sapien").origin).parent / "wrapper/urdf_loader.py"
text = sapien.read_text()
text = text.replace('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")')
text = text.replace('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"')
text = text.replace('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")')
sapien.write_text(text)
planner = Path(importlib.util.find_spec("mplib").origin).parent / "planner.py"
text = planner.read_text().replace(
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
)
planner.write_text(text)
record = {
    "basis": "archived working CuRobo Python files and author RoboTwin installation fixes",
    "curobo_python_files_changed": changed,
    "simulator_files": [str(sapien), str(planner)],
}
(root / "records/runtime-compatibility.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
