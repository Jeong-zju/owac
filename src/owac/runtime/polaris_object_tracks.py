"""Infer versioned object/regrasp candidates from saved expert robot trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_replay import forward, utc


def infer_tracks(actions, states, tcp, association_radius=0.06):
    closing = np.flatnonzero((actions[:, 7] > 0.5) & (np.r_[0, actions[:-1, 7]] <= 0.5))
    opening = np.flatnonzero((actions[:, 7] <= 0.5) & (np.r_[0, actions[:-1, 7]] > 0.5))
    objects, cycles, unresolved = [], [], []
    for start in closing:
        end = next((int(i) for i in opening if i > start), None)
        if end is None:
            unresolved.append({"close": int(start), "reason": "no source release"})
            continue
        segment = states[start + 1 : end, 7]
        grip = float(np.median(segment)) if len(segment) else 1.0
        cycle = {
            "close": int(start),
            "open": end,
            "close_tcp": tcp[start].tolist(),
            "open_tcp": tcp[end].tolist(),
            "source_held_gripper_median": grip,
            "maximum_lift_m": float(tcp[start : end + 1, 2].max() - tcp[start, 2]),
            "xy_displacement_m": float(np.linalg.norm(tcp[end, :2] - tcp[start, :2])),
        }
        if end - start < 3 or (
            cycle["maximum_lift_m"] < 0.015 and cycle["xy_displacement_m"] < 0.03
        ):
            cycle.update(status="short_or_stationary_grasp_requires_review", object_index=None)
            unresolved.append(cycle)
            cycles.append(cycle)
            continue
        if grip > 0.95:
            # Thin objects can coexist with a nearly fully closed source gripper.
            # This is uncertainty to review, not proof that the grasp was empty.
            unresolved.append(
                {"close": int(start), "reason": "fully closed source gripper; verify object in RGB"}
            )
        candidates = [
            (float(np.linalg.norm(np.asarray(o["last_release_tcp"])[:2] - tcp[start, :2])), i)
            for i, o in enumerate(objects)
        ]
        nearest = min(candidates, default=(float("inf"), None))
        if nearest[0] <= association_radius:
            index = nearest[1]
        else:
            index = len(objects)
            objects.append(
                {
                    "initial_grasp_tcp": tcp[start].tolist(),
                    "initial_grasp_frame": int(start),
                    "grasp_cycles": [],
                }
            )
        objects[index]["last_release_tcp"] = tcp[end].tolist()
        objects[index]["last_release_frame"] = end
        objects[index]["grasp_cycles"].append([int(start), end])
        cycle.update(status="candidate_object_track", object_index=index)
        cycles.append(cycle)
    return {
        "candidate_objects": objects,
        "grasp_cycles": cycles,
        "unresolved_grasps": unresolved,
        "association_radius_m": association_radius,
        "object_states_recovered": False,
        "object_count_visually_verified": False,
        "basis": "source close/release and robot FK; spatial regrasp association is approximate",
    }


def run(inventory, kinematics, output, tcp_offset_m=0.131):
    output.mkdir(parents=True, exist_ok=False)
    kin = json.loads(kinematics.read_text())
    base = np.asarray(kin["base"])
    chain = [(np.asarray(a), np.asarray(b), axis) for a, b, axis in kin["chain"]]
    sources = list(map(json.loads, (inventory / "episodes.jsonl").read_text().splitlines()))
    manifest = []
    for source in sources:
        path = inventory / "episodes" / source["id"] / "source_numeric.npz"
        if sha256(path) != source["numeric_sha256"]:
            raise ValueError("Original expert arrays changed before object-track inference")
        with np.load(path) as data:
            actions, states = data["actions"], data["states"]
            tcp = np.asarray([forward(base, chain, q[:7], (tcp_offset_m, 0, 0))[1] for q in states])
            tracks = infer_tracks(actions, states, tcp)
        result = {
            "source_id": source["id"],
            "source_scene": source["source_scene_name"],
            "source_frames": source["frames"],
            "source_record_sha256": source["record_sha256"],
            "instruction": source["instructions"],
            "tcp_offset_m": tcp_offset_m,
            **tracks,
        }
        write_json(output / "episodes" / f"{source['id']}.json", result)
        manifest.append(
            {
                "source_id": source["id"],
                "candidate_object_count": len(tracks["candidate_objects"]),
                "unresolved_grasps": len(tracks["unresolved_grasps"]),
            }
        )
        write_json(
            output / "status.json",
            {
                "status": "complete" if len(manifest) == len(sources) else "running",
                "checked_episodes": len(manifest),
                "source_episodes": len(sources),
                "candidate_object_count": sum(r["candidate_object_count"] for r in manifest),
                "unresolved_grasps": sum(r["unresolved_grasps"] for r in manifest),
                "object_states_recovered": False,
                "kinematics_sha256": sha256(kinematics),
                "source_inventory_sha256": sha256(inventory / "episodes.jsonl"),
                "checked_at_utc": utc(),
            },
        )
        print(source["id"], len(tracks["candidate_objects"]), flush=True)
    write_json(output / "manifest.json", {"episodes": manifest})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tcp-offset-m", type=float, default=0.131)
    args = parser.parse_args()
    run(args.inventory, args.kinematics, args.output, args.tcp_offset_m)


if __name__ == "__main__":
    main()
