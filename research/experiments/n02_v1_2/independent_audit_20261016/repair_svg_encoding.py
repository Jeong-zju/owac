"""Lossless format repair of this audit's existing SVG pixel payloads; no model rerun."""

import base64
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from owac.evaluation.n02_full.v12_viz import png_bytes

root = Path("outputs/n02-v12/P2/independent-audit-20261016")
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
images, files = 0, []
checked = 0
for path in sorted(root.rglob("*.svg")):
    tree = ET.parse(path)
    changed = 0
    for element in tree.iter():
        for key, value in list(element.attrib.items()):
            if not value.startswith("data:image/png;base64,"):
                continue
            checked += 1
            raw = base64.b64decode(value.split(",", 1)[1])
            if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                continue
            width, height = int(element.attrib["width"]), int(element.attrib["height"])
            if len(raw) != width * height * 3 and "prediction_residual" in path.name:
                sensor_height, sensor_width = (240, 320) if "robotwin" in path.name else (256, 256)
                step = math.ceil(max(sensor_height, sensor_width) / width)
                height, width = sensor_height // step, sensor_width // step
            if len(raw) != width * height * 3:
                raise ValueError(f"{path}: unexpected pixel payload size")
            encoded = png_bytes(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
            element.set(key, "data:image/png;base64," + base64.b64encode(encoded).decode())
            changed += 1
    if changed:
        tree.write(path, encoding="utf-8")
        files.append(str(path))
        images += changed
receipt = {
    "image_payloads_checked": checked,
    "all_checked_payloads_are_png": True,
    "files_repaired": len(files),
    "image_payloads_repaired": images,
    "files": files,
    "operation": "PNG encoding of already saved pixels; no numerical image change",
}
(root / "svg_encoding_repair.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"files_repaired": len(files), "image_payloads_repaired": images}))
