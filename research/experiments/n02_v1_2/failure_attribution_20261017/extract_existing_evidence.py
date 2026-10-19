"""Read existing D1 weights, RAW cache metadata and sensor code; no inference or updates."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    root = Path(args.checkout)
    result = {"optimizer_updates": 0, "new_rollouts": 0, "model_inferences": 0, "arms": {}}
    for arm in ("narrow", "wide"):
        directory = root / "outputs/n02-v12/P2" / f"raw-backend-r8-fixedfront-{arm}-s1103-d1"
        pipeline = read(directory / "pipeline_result.json")
        checkpoint = directory / f"checkpoint_{pipeline['selected_step']:07d}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        reference = payload["dev_reference"]
        identity = payload["identity"]["train"]
        result["arms"][arm] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
            "selected_step": payload["step"],
            "cursor": payload["cursor"],
            "reference": {
                k: reference[k]
                for k in ("action_error", "sampled_control_mse", "sampled_per_window", "per_window")
            },
            "train_identity_scalars": {
                k: v for k, v in identity.items() if isinstance(v, (str, int, float, bool))
            },
            "train_identity_keys": list(identity),
        }
        del payload
    data_root = Path(read(directory / "report.json")["config"]["data"]["data_root"])
    metadata_keys = (
        "episode",
        "num_frames",
        "lighting_domain",
        "lighting_ev",
        "noise_seed",
        "raw_sensor_noise_enabled",
        "raw_sensor_shot_noise",
        "raw_sensor_read_noise",
        "raw_white_level",
        "raw_protocol",
        "representation",
    )
    result["cached_metadata"] = []
    for path in sorted((data_root / "data/robotwin2/raw/grab_roller").glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        result["cached_metadata"].append(
            {
                "relative_path": str(path.relative_to(data_root)),
                **{key: metadata.get(key) for key in metadata_keys},
            }
        )
    sensor = Path("/workspace/owac/rawvla/source/third_party/RoboTwin/envs/camera/raw_rgb10.py")
    source = sensor.read_text()
    tree = ast.parse(source)
    result["sensor_source"] = {
        "path": str(sensor),
        "sha256": file_hash(sensor),
        "definitions": {
            node.name: {"line": node.lineno, "source": ast.get_source_segment(source, node)}
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ("add_sensor_noise", "SensorNoiseParams", "hdr_linear_to_raw_float")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cache_files": len(result["cached_metadata"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
