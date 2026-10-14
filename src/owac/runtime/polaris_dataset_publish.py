"""Publish only a verified complete reconstructed corpus to the authorized HF namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_dataset_export import REPO_ID, release_ready
from owac.runtime.polaris_replay import utc

DIRECTORIES = {"data", "meta", "source_rgb", "scene_audit"}
FILES = {"source_provenance.json", "coverage.json", "training_loader_validation.json"}


def validate_source_coverage(dataset, inventory, protocol):
    coverage = json.loads((dataset / "coverage.json").read_text())
    release_ready(coverage, protocol)
    manifest = inventory / "episodes.jsonl"
    if sha256(manifest) != protocol["source_inventory_sha256"]:
        raise ValueError("Pinned original episode manifest changed")
    original = {r["id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    if set(coverage["source_ids"]) != set(original):
        raise ValueError("Full dataset identities do not match actual pinned source episodes")
    provenance = json.loads((dataset / "source_provenance.json").read_text())["episodes"]
    if len(provenance) != len(original):
        raise ValueError("Missing per-episode original provenance")
    for row in provenance:
        source = original[row["source_id"]]
        if row["frames"] != source["frames"]:
            raise ValueError("Source episode frame coverage differs")
        if row["source"]["record_sha256"] != source["record_sha256"]:
            raise ValueError("Source episode record hashes differ")
        if row["original_actions_modified"] or not row["physical_result"]["success"]:
            raise ValueError("Full expert release contains modified actions or failed task")
    if len({r["source_id"] for r in provenance}) != len(original):
        raise ValueError("Repeated episode provenance hides missing source identities")
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["total_episodes"] != len(original) or info["total_frames"] != coverage["source_frames"]:
        raise ValueError("Actual LeRobot metadata differs from claimed coverage")
    if coverage["raw_exact_value_checks"] != 6 * coverage["source_frames"]:
        raise ValueError("Not all three native RAW and noiseless CFA fields were validated")
    return coverage


def card(coverage):
    return f"""---
license: mit
language: [en]
tags: [robotics, isaac-sim, polaris, droid, pi05, raw, cfa, synthetic]
---
# Isaac / PolaRiS DROID: reconstructed paired RGB and native RAW

This is a **reconstructed derivative**, not an official exact recovery of the
original training scenes or RAW for the original JPEG pixels.

It covers {coverage["episodes"]} original public simulation expert episodes and
{coverage["source_frames"]:,} control frames from
[owhan/PolaRiS-datasets](https://huggingface.co/datasets/owhan/PolaRiS-datasets),
revision `dca86b8e186df63323eb375344a1977d92f508f7`.
The original corpus has 14 source scene folders and 17 task instructions.
It is the simulation cotraining portion; it does not include the complete pi05
pretraining corpus or the real DROID data in PolaRiS's mixed training recipe.

Original eight-dimensional float32 joint-position/gripper expert actions and
episode identities are retained. Objects and layouts were reconstructed from
the original RGB and robot trajectories, then physically replayed in Isaac Sim
6.1 with Arena's Franka / Robotiq asset. No post-initialization object teleport
was used. Physical task checks, reference-state tracking and scene/camera
correspondence assessments are retained in source_provenance.json.
Replay mode is recorded per episode: joint-target execution and an arm driven
by saved source joint states are distinct. Reference-driven arm replay preserves
original action labels but does not claim to execute the original arm commands.
{coverage.get("constraint_assisted_episodes", 0)} episodes use explicitly disclosed
source-action grasp assistance: a USD fixed joint is created at the actual grasp
pose when the original gripper closes within the allowed proximity, and disabled
when the original action opens the gripper.
Attachments can wait for approach readiness within the original closed cycle.
Gripper/object collisions are temporarily filtered while attached; environment
collisions and the released object's dynamics remain enabled.
These are constraint-assisted reconstruction trajectories, not free-contact expert replays
or policy benchmark
successes. Per-episode joint events, proximity and relative-pose errors are retained.
`scene_audit/` retains actual robot/object states, camera settings, sensor
metadata, capture timing and checksums separately from model observations.

## Paired observations and references

`observation.exterior_image_1_left`, `observation.exterior_image_2_left` and
`observation.wrist_image_left` contain new lossless RGB observations, 320 x 180.
`observation.raw.exterior_1_left`, `observation.raw.exterior_2_left` and
`observation.raw.wrist_left` contain corresponding native pre-ISP noisy GRBG
CFA DN. `observation.cfa_noiseless.*` preserves the native pre-noise CFA too.
All RAW is stored as unchanged uint32 row-major vectors of length 57600;
reshape to `(180, 320)` without treating it as an RGB image.

The two source exterior JPEG fields are byte-identical at **all 52,914 original
frames across all 316 episodes**. Their reconstructed fields are therefore
explicit aliases of one external camera, plus one independent wrist camera.
They are not three independently calibrated physical views.

Original JPEG bytes are preserved in `source_rgb/`, with decoded original RGB
in `reference.original_rgb.*` and original robot states in `reference.source_*`.
Native RAW is paired with the **new observation RGB**, not these original JPEGs.
Original camera calibration, exact background geometry and original exposure
were not published and are not claimed recovered. Per-episode correspondence
assessments describe approximations; task success is not proof of pixel alignment.

## Fine-tuning and loading

This is a LeRobot v2-style dataset validated with the pinned official OpenPI
DROID input transform. Training state uses actual replay arm joints and actual
finger joint divided by pi/4. Action chunks contain 15 unchanged original actions,
with end padding within the same source episode. Replay timing is nominal 15 Hz;
original capture timestamps were absent. Use scene/episode-based splits.

```python
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "{REPO_ID}",
    delta_timestamps={{"action": [i / 15 for i in range(15)]}},
)
sample = ds[0]
raw_dn = np.asarray(sample["observation.raw.exterior_1_left"], dtype=np.uint32)
raw_dn = raw_dn.reshape(180, 320)  # GRBG; original DN preserved
```

Stock pi05 uses RGB; a RAW-consuming model needs its own RAW input path. The
dataset loader verification is not a new trained policy or a benchmark score.
RAW is native simulator sensor synthesis, not physically calibrated real-camera
RAW. Sensor settings and reconstruction records are included as audit metadata.

## Attribution

Source dataset: owhan / PolaRiS authors, declared MIT. Original project:
[arhanjain/polaris](https://github.com/arhanjain/polaris).
Robot asset source: [owhan/DROID-sim-environments](https://huggingface.co/datasets/owhan/DROID-sim-environments).
Reconstruction and paired capture: Jeong-zju. This package does not redistribute
pi05 weights, private repository code, credentials, or original simulator binaries.
"""


def stage(dataset, inventory, protocol, output, source_license):
    coverage = validate_source_coverage(dataset, inventory, protocol)
    license_text = source_license.read_text()
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ValueError("Missing actual source license text/attribution")
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(DIRECTORIES | FILES):
        path = dataset / name
        if path.is_dir():
            shutil.copytree(path, output / name)
        elif path.is_file():
            shutil.copyfile(path, output / name)
        else:
            raise ValueError(f"Missing required dataset artifact: {name}")
    (output / "source_inventory").mkdir()
    shutil.copyfile(inventory / "episodes.jsonl", output / "source_inventory/episodes.jsonl")
    write_json(output / "release_protocol.json", protocol)
    (output / "README.md").write_text(card(coverage))
    (output / "LICENSE").write_text(license_text)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(output))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    write_json(
        output / "sha256-manifest.json",
        {
            "source_revision": protocol["source_revision"],
            "files": files,
            "staged_at_utc": utc(),
            "repo_id": REPO_ID,
        },
    )
    return files


def publish(output):
    protocol = json.loads((output / "release_protocol.json").read_text())
    validate_source_coverage(output, output / "source_inventory", protocol)
    manifest = json.loads((output / "sha256-manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        if sha256(output / name) != metadata["sha256"]:
            raise ValueError(f"Staged release changed after validation: {name}")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    if api.whoami()["name"] != "Jeong-zju":
        raise ValueError("Credential is not for the user-authorized publishing account")
    cli = ["uv", "tool", "run", "--from", "huggingface_hub==1.6.0", "hf"]
    subprocess.run(
        cli + ["repos", "create", REPO_ID, "--repo-type", "dataset", "--no-private", "--exist-ok"],
        check=True,
    )
    info = api.repo_info(REPO_ID, repo_type="dataset")
    if info.private:
        raise ValueError("Existing target is private; no content uploaded")
    subprocess.run(
        cli
        + [
            "upload-large-folder",
            REPO_ID,
            str(output),
            "--repo-type",
            "dataset",
            "--num-workers",
            "4",
        ],
        check=True,
    )
    info = api.repo_info(REPO_ID, repo_type="dataset")
    expected = manifest["files"] | {
        "sha256-manifest.json": {"sha256": sha256(output / "sha256-manifest.json")}
    }
    remote = {
        f.path: f
        for f in api.list_repo_tree(
            REPO_ID, repo_type="dataset", revision=info.sha, recursive=True, expand=True
        )
        if hasattr(f, "blob_id")
    }
    for name, metadata in expected.items():
        if name not in remote:
            raise ValueError(f"Uploaded file is missing: {name}")
        item = remote[name]
        if item.lfs is not None:
            digest = item.lfs.sha256
            if digest != metadata["sha256"]:
                raise ValueError(f"Remote LFS SHA256 differs: {name}")
        else:
            payload = (output / name).read_bytes()
            digest = hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload
            ).hexdigest()
            if digest != item.blob_id:
                raise ValueError(f"Remote Git blob differs: {name}")
    saved = hf_hub_download(REPO_ID, "sha256-manifest.json", repo_type="dataset", revision=info.sha)
    if sha256(Path(saved)) != expected["sha256-manifest.json"]["sha256"]:
        raise ValueError("Remote public manifest does not match local complete release")
    return {
        "repo_id": REPO_ID,
        "private": info.private,
        "commit": info.sha,
        "verified_files": len(expected),
        "verified_at_utc": utc(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    files = stage(args.dataset, args.inventory, protocol, args.output, args.source_license)
    if args.publish:
        result = publish(args.output)
        write_json(args.output.parent / "publication.json", result)
        print(json.dumps(result))
    else:
        print(json.dumps({"staged_files": len(files), "public_upload_complete": False}))


if __name__ == "__main__":
    main()
