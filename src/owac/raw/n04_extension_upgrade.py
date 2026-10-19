"""Auditable v1 -> v2 input-boundary conversion; raw captures remain immutable."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from owac.raw.n04_live_collection import save_json, sha256


def upgrade(source: Path, output: Path):
    from owac.raw.n04_d1_capture import robot_proprio
    from owac.raw.n04_mechanism_data import _libero_env_from_npz

    output.mkdir(parents=True, exist_ok=False)
    maps = {}
    provenance = dict(source=str(source), records=[])
    for record_file in sorted(source.glob("episode-*/record.json")):
        record = json.loads(record_file.read_text())
        if record["schema"] not in ("n04-live-extension-v1", "n04-live-extension-v2"):
            raise ValueError("conversion only accepts v1 captures")
        dest = output / record_file.parent.name
        dest.mkdir()
        meta = record["metadata"]
        if meta["platform"] == "libero" and meta["task"] not in maps:
            env, _, _ = _libero_env_from_npz(meta["source_npz"], env_seed=meta["seed"])
            try:
                env.reset()
                sensors = robot_proprio(env)
                maps[meta["task"]] = dict(
                    qpos=sensors["audit_robot_qpos_indices"].tolist(),
                    qvel=sensors["audit_robot_qvel_indices"].tolist(),
                    joint_names=[
                        name
                        for robot in env.robots
                        for name in robot.robot_joints + robot.gripper_joints
                    ],
                    bddl_sha256=sha256(Path(meta["bddl"])),
                    method="indices read from same-version env.robots after reset",
                )
            finally:
                env.close()
        for frame in record["frames"]:
            old = record_file.parent / frame["file"]
            new = dest / frame["file"]
            if sha256(old) != frame["sha256"]:
                raise ValueError(f"changed source: {old}")
            original_hash = frame["sha256"]
            if meta["platform"] == "libero" and record["schema"] == "n04-live-extension-v1":
                with np.load(old, allow_pickle=False) as raw:
                    values = {key: raw[key] for key in raw.files}
                for key in ("qpos", "qvel"):
                    indices = np.asarray(maps[meta["task"]][key], dtype=np.int64)
                    full = values[f"online_{key}"]
                    values[f"gt_sim_{key}"] = full
                    values[f"online_{key}"] = full[indices]
                    values[f"audit_robot_{key}_indices"] = indices
                np.savez_compressed(new, **values)
            elif meta["platform"] == "robotwin2" and meta.get("pixel_coordinates") is None:
                with np.load(old, allow_pickle=False) as raw:
                    values = {key: raw[key] for key in raw.files}
                values["online_K"] = values["online_K"].copy()
                values["online_K"][:, :2, 2] -= 0.5
                np.savez_compressed(new, **values)
            else:
                os.link(old, new)  # same immutable pixels and hash, no needless duplicate GB
            frame["sha256"] = sha256(new)
            frame["original_capture_sha256"] = original_hash
        record["metadata"]["pixel_coordinates"] = "integer_pixel_centres_top_left"
        record["schema"] = "n04-live-extension-v2"
        record["conversion"] = dict(
            source_record=str(record_file),
            source_record_sha256=sha256(record_file),
            robot_sensor_mapping=maps.get(meta["task"]),
            change=(
                "LIBERO scene qpos/qvel moved to gt_sim_*; robot joints only online; "
                "RoboTwin K edge-centres -> integer pixel centres; other values unchanged"
            ),
        )
        save_json(dest / "record.json", record)
        provenance["records"].append(
            dict(
                source=str(record_file),
                source_sha256=sha256(record_file),
                target=str(dest / "record.json"),
                target_sha256=sha256(dest / "record.json"),
            )
        )
        save_json(output / "conversion.json", provenance)
    run = json.loads((source / "run.json").read_text())
    run["conversion_source"] = str(source)
    run["conversion_source_run_sha256"] = sha256(source / "run.json")
    save_json(output / "run.json", run)
    save_json(output / "robot-sensor-mappings.json", maps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    upgrade(args.source, args.output)


if __name__ == "__main__":
    main()
