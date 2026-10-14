"""Lossless native RAW fields alongside RGB in a LeRobot pi05 training dataset."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_reconstruction import FIELD_MAPPING, effective_status
from owac.runtime.polaris_replay import utc
from owac.runtime.polaris_replay_archive import verify_frames
from owac.runtime.polaris_replay_data import trajectories

REPO_ID = "Jeong-zju/isaac-arena-droid-rgb-raw-reconstructed"


def raw_feature(prefix, view):
    # Pinned LeRobot checks the literal word "image" in statistics keys. Native CFA
    # is an integer signal, so its key must not select that RGB-only statistics path.
    return f"observation.{prefix}.{view.replace('_image', '')}"


def load_source_record(source, inventory):
    shard = inventory / "source" / source["source_shard"]
    for index, digest, record in trajectories(shard):
        if index == source["source_record_index"]:
            if digest != source["record_sha256"]:
                raise ValueError("Original source record changed")
            return record
    raise ValueError("Missing original source record")


def release_ready(coverage, protocol):
    """A completed sample export cannot be presented as the completed public corpus."""
    expected = protocol["source_episodes"]
    if coverage["episodes"] != expected or len(set(coverage["source_ids"])) != expected:
        raise ValueError("Full release lacks all original source episode identities")
    if coverage["source_frames"] != protocol["source_control_frames"]:
        raise ValueError("Full release lacks all original source control frames")
    if coverage["views"] != protocol["source_rgb_fields"]:
        raise ValueError("Full release lacks a RGB/RAW schema field")
    if coverage["failed_or_unresolved_episodes"]:
        raise ValueError("Full release still has failed or unresolved scene replays")
    if not coverage["training_loader_validated"]:
        raise ValueError("Full release has not passed the actual training loader")
    if not coverage["per_episode_source_rgb_correspondence_review_complete"]:
        raise ValueError("Full release still lacks per-episode source RGB correspondence review")
    return True


def select_latest_replays(batches):
    """Retain all outcomes and let newer failures supersede older successes."""
    latest, outcomes = {}, []
    ordered = sorted(batches, key=lambda b: effective_status(b)["started_at_utc"])
    for batch in ordered:
        for row in effective_status(batch)["completed"]:
            latest[row["id"]] = {**row, "batch": batch}
            outcomes.append(
                {
                    "batch": str(batch),
                    "source_id": row["id"],
                    "infrastructure_error": row["infrastructure_error"],
                    "physical_success": (row.get("result") or {}).get("success"),
                }
            )
    accepted = [
        row
        for row in latest.values()
        if not row["infrastructure_error"] and row["result"]["success"]
    ]
    return accepted, outcomes


def export(batch, output, alias_audit, inventory):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    aliases = json.loads(alias_audit.read_text())
    if (
        aliases["episodes"] != 316
        or aliases["frames"] != 52914
        or aliases["identical_exterior_jpeg_frames"] != 52914
        or aliases["episodes_with_identical_exteriors"] != 316
    ):
        raise ValueError("Cannot alias RAW exterior fields without complete original-byte evidence")
    accepted, outcomes = select_latest_replays([batch] if isinstance(batch, Path) else batch)
    if not accepted:
        raise ValueError("No physically successful source replay is available to export")
    features = {
        "observation.joint_position": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.gripper_position": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (8,), "names": None},
        "replay_timestamp_seconds": {"dtype": "float64", "shape": (1,), "names": None},
        "source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "reference.source_joint_position": {"dtype": "float32", "shape": (7,), "names": None},
        "reference.source_gripper_position": {"dtype": "float32", "shape": (1,), "names": None},
    }
    for view in FIELD_MAPPING:
        features[f"observation.{view}"] = {
            "dtype": "image",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channel"],
        }
        features[f"reference.original_rgb.{view}"] = {
            "dtype": "image",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channel"],
        }
        for prefix in ("raw", "cfa_noiseless"):
            features[raw_feature(prefix, view)] = {
                "dtype": "uint32",
                # The pinned LeRobot stats merger assumes any rank-2 statistic is RGB.
                # Store native DN in row-major order and recover the sensor shape on load.
                "shape": (180 * 320,),
                "names": ["cfa_pixel_index_row_major"],
            }
    dataset = LeRobotDataset.create(
        REPO_ID,
        fps=15,
        root=output,
        robot_type="droid_jointpos",
        features=features,
        use_videos=False,
    )
    provenance = []
    samples = []
    for accepted_row in accepted:
        identity = accepted_row["id"]
        replay = accepted_row["batch"] / "runs" / identity / "replay"
        config = json.loads((replay / "config.json").read_text())
        result = json.loads((replay / "result.json").read_text())
        rows = [json.loads(x) for x in (replay / "frames.jsonl").read_text().splitlines()]
        checked = verify_frames(rows, result, config)
        audit = output / "scene_audit" / identity
        audit.mkdir(parents=True, exist_ok=False)
        for name in ("config.json", "dataset.json", "result.json"):
            shutil.copyfile(replay / name, audit / name)
        sensors = sorted(replay.glob("sensor-*.json"))
        if len(sensors) != len(set(FIELD_MAPPING.values())):
            raise ValueError("Missing native sensor metadata for a reconstructed view")
        for sensor in sensors:
            shutil.copyfile(sensor, audit / sensor.name)
        # Reference/evaluator states remain audit data, separate from model observations.
        with (audit / "states_and_capture.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        source_episode = Path(config["source_episode"])
        with np.load(source_episode / "trajectory.npz") as original:
            actions = original["actions"].copy()
            source_states = original["states"].copy()
        source = json.loads((source_episode / "source.json").read_text())
        record = load_source_record(source, inventory)
        if not np.array_equal(record["steps/action"].reshape(-1, 8), actions):
            raise ValueError("Copied episode actions differ from pinned source record")
        for index, row in enumerate(rows[: len(actions)]):
            if not np.array_equal(
                np.asarray(row["original_action"], dtype=np.float32), actions[index]
            ):
                raise ValueError("Replay changed source action labels")
            frame = {
                "observation.joint_position": np.asarray(row["joint_position"], dtype=np.float32),
                "observation.gripper_position": np.asarray(
                    [row["actual_gripper_joint"] / (np.pi / 4)], dtype=np.float32
                ),
                "action": actions[index],
                "task": source["instruction"],
                "replay_timestamp_seconds": np.asarray(
                    [row["simulation_time_seconds"]], dtype=np.float64
                ),
                "source_frame_index": np.asarray([index], dtype=np.int64),
                "reference.source_joint_position": source_states[index, :7],
                "reference.source_gripper_position": source_states[index, 7:8],
            }
            expected = {}
            for view, camera in FIELD_MAPPING.items():
                payload = record[f"steps/observation/{view}"][index]
                reference = output / "source_rgb" / identity / view / f"{index:05d}.jpg"
                reference.parent.mkdir(parents=True, exist_ok=True)
                reference.write_bytes(payload)
                with Image.open(io.BytesIO(payload)) as original_image:
                    frame[f"reference.original_rgb.{view}"] = np.asarray(original_image).copy()
                capture = row["capture"]["views"][camera]
                rgb_path = replay / capture["paired_rgb"]["path"]
                if sha256(rgb_path) != capture["paired_rgb"]["sha256"]:
                    raise ValueError("Paired RGB changed since native capture")
                with Image.open(rgb_path) as image:
                    rgb = np.asarray(image).copy()
                if rgb.dtype != np.uint8 or rgb.shape != (180, 320, 3):
                    raise ValueError("Invalid lossless paired RGB")
                frame[f"observation.{view}"] = rgb
                for prefix, buffer in [("raw", "3-noise"), ("cfa_noiseless", "2-cfa")]:
                    metadata = capture["files"][buffer]
                    path = replay / metadata["path"]
                    if sha256(path) != metadata["sha256"]:
                        raise ValueError("Native RAW changed since capture")
                    raw = np.fromfile(path, dtype="<u4").reshape(180, 320)
                    key = raw_feature(prefix, view)
                    frame[key] = raw.reshape(-1)
                    expected[key] = path
            dataset.add_frame(frame)
            chunk_indices = np.minimum(np.arange(index, index + 15), len(actions) - 1)
            samples.append((actions[chunk_indices].copy(), expected))
        dataset.save_episode()
        provenance.append(
            {
                "source_id": identity,
                "frames": len(actions),
                "source": source,
                "physical_result": result,
                "capture_validation": checked,
                "field_mapping": FIELD_MAPPING,
                "correspondence": config["source_rgb_correspondence"],
                "original_actions_modified": False,
                "replay_mode": config.get("replay_mode", "joint_targets"),
                "original_arm_actions_executed_as_targets": config.get("replay_mode")
                != "reference_arm_states",
                "grasp_assistance": result.get("grasp_assistance"),
                "free_contact_expert_replay": result.get("grasp_assistance") is None,
                "audit_metadata": str(audit.relative_to(output)),
            }
        )
    # Validate persisted values and real LeRobot chunks at episode boundaries.
    from openpi.models.model import ModelType
    from openpi.policies.droid_policy import DroidInputs

    dataset = LeRobotDataset(
        REPO_ID, root=output, delta_timestamps={"action": [i / 15 for i in range(15)]}
    )
    for index, (original_action, expected_raw) in enumerate(samples):
        restored = dataset[index]
        for key, path in expected_raw.items():
            original = np.fromfile(path, dtype="<u4").reshape(180, 320)
            saved = np.asarray(restored[key]).reshape(180, 320)
            if saved.dtype.kind not in "iu" or not np.array_equal(saved, original):
                raise ValueError(f"Persisted RAW was truncated, cast or changed: {index} {key}")
        action = np.asarray(restored["action"])
        if not np.array_equal(action, original_action):
            raise ValueError("Persisted LeRobot action differs from source")
        policy_input = {
            "observation/joint_position": np.asarray(restored["observation.joint_position"]),
            "observation/gripper_position": np.asarray(restored["observation.gripper_position"]),
            "observation/exterior_image_1_left": np.asarray(
                restored["observation.exterior_image_1_left"]
            ),
            "observation/wrist_image_left": np.asarray(restored["observation.wrist_image_left"]),
            "actions": action,
            "prompt": restored["task"],
        }
        transformed = DroidInputs(ModelType.PI05)(policy_input)
        if transformed["state"].shape != (8,) or transformed["actions"].shape != (15, 8):
            raise ValueError("Exported frame failed official pi05 DROID input transform")
    coverage = {
        "episodes": len(accepted),
        "source_ids": [r["id"] for r in accepted],
        "source_frames": len(samples),
        "views": list(FIELD_MAPPING),
        "failed_or_unresolved_episodes": 316 - len(accepted),
        "training_loader_validated": True,
        "constraint_assisted_episodes": sum(p["grasp_assistance"] is not None for p in provenance),
        "free_contact_episodes": sum(p["free_contact_expert_replay"] for p in provenance),
        "per_episode_source_rgb_correspondence_review_complete": all(
            p["correspondence"]["per_episode_scene_and_rgb_review_complete"] for p in provenance
        ),
        "full_dataset_complete": False,
        "publish_ready": False,
        "validated_at_utc": utc(),
        "raw_exact_value_checks": len(samples) * 6,
        "raw_storage_shape": [180 * 320],
        "raw_sensor_shape_height_width": [180, 320],
        "raw_storage_order": "row-major; uint32 DN unchanged; reshape, never normalize as RGB",
        "raw_feature_mapping": {view: raw_feature("raw", view) for view in FIELD_MAPPING},
        "source_rgb_preserved": "all original JPEG bytes in source_rgb; decoded reference fields",
        "raw_is_paired_with": "observation RGB in rebuilt scene; not reference.original_rgb",
        "pixel_alignment_with_original_jpeg_verified": False,
        "action_chunk_validation": "15 original actions; end padding stays within source episode",
    }
    write_json(
        output / "source_provenance.json",
        {"episodes": provenance, "all_replay_outcomes": outcomes},
    )
    write_json(output / "coverage.json", coverage)
    write_json(output / "training_loader_validation.json", {"coverage": coverage})
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-alias-audit", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.batch, args.output, args.view_alias_audit, args.inventory)))


if __name__ == "__main__":
    main()
