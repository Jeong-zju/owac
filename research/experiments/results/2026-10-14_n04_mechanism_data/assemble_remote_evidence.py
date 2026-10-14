#!/usr/bin/env python3
"""Merge the read-only remote probe outputs into the evidence file for build_feasibility.

Every value below is copied from a probe JSON produced during the S0 run; nothing is
inferred or synthesised.  Run with the repository venv from the worktree root:

    .venv/bin/python <this directory>/assemble_remote_evidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    payload = json.loads((HERE / name).read_text(encoding="utf-8"))
    return payload


def main() -> int:
    libero = load("remote-probe-libero-sim.json")
    contacts = load("remote-probe-libero-contacts.json")
    robotwin = load("remote-probe-robotwin-sim.json")
    inventory = load("remote-probe-inventory.json")
    npz = load("remote-probe-npz.json")["npz_fields"]

    robotwin_camera_methods = robotwin["camera_class_methods"]
    robotwin_scan = robotwin["api_source_scan"]
    libero_npz = npz["evidence"]["libero"]
    robotwin_npz = npz["evidence"]["robotwin2"]
    action_log = inventory["controller_logs"]["libero_actions_jsonl"]
    policy_record = inventory["controller_logs"]["robotwin_policy_record"]

    evidence = {
        "checked_at_utc": "2026-10-14T04:20:00Z",
        "host": "root@107.209.104.125:44327 (read-only probes)",
        "libero_sim": {
            "import_ok": bool(libero.get("import_ok")),
            "env_created": bool(libero.get("env_created")),
            "bddl_files_present": bool(libero.get("bddl_files_present")),
            "evidence": {
                "npz": libero.get("npz"),
                "bddl_file": libero.get("bddl_file"),
                "camera_intrinsics": libero.get("camera_intrinsics"),
                "depth_render": libero.get("depth_render"),
                "segmentation_render": libero.get("segmentation_render"),
                "state_replay": libero.get("state_replay"),
                "contacts": contacts,
            },
        },
        "robotwin_sim": {
            "import_ok": bool(robotwin.get("sapien_import_ok"))
            and bool(robotwin.get("camera_module_import_ok")),
            "assets_present": bool(robotwin.get("assets_present")),
            "task_config_present": bool(robotwin.get("task_config_present")),
            "evidence": {
                "sapien_version": robotwin.get("sapien_version"),
                "sapien_camera_api": robotwin.get("sapien_camera_api"),
                "camera_class_methods": robotwin_camera_methods,
                "api_source_scan": robotwin_scan,
                "task_config_files": robotwin.get("task_config_files"),
                "verification_level": "api and assets only: no live RoboTwin task environment was "
                "created in this round",
            },
        },
        "controller_logs": {
            "files_found": bool(action_log["exists"]) and policy_record is not None,
            "actions_present": bool(action_log["exists"]) and bool(action_log["lines"]),
            "evidence": {
                "libero_actions_jsonl": action_log,
                "libero_episodes_jsonl": inventory["controller_logs"]["libero_episodes_jsonl"],
                "robotwin_policy_record": policy_record,
                "robotwin_episodes_jsonl": inventory["controller_logs"]["robotwin_episodes_jsonl"],
                "note": "the LIBERO/robotwin logs belong to the frozen benchmark rollouts; the D3 "
                "training replay path has no separate control log",
            },
        },
        "replay": {
            "libero_pinned_head": bool(inventory["heads"]["libero_submodule"]),
            "robotwin_pinned_head": bool(inventory["heads"]["robotwin_submodule"]),
            "libero_state_replay_exact": bool(
                libero.get("state_replay", {}).get("exact_within_1e-9")
            ),
            "evidence": {
                "heads": inventory["heads"],
                "envs": inventory["envs"],
                "source_trajectories": inventory["source_trajectories"],
                "libero_state_replay": libero.get("state_replay"),
                "dataset": inventory["dataset"],
            },
        },
        "npz_fields": {
            "libero_actions_per_frame": bool(npz["libero_actions_per_frame"]),
            "robotwin_actions_per_frame": bool(npz["robotwin_actions_per_frame"]),
            "libero_three_channel_only": bool(npz["libero_three_channel_only"]),
            "robotwin_three_channel_only": bool(npz["robotwin_three_channel_only"]),
            "d0_synthetic_cfa": bool(npz["d0_synthetic_cfa"]),
            "evidence": {
                "libero": libero_npz,
                "robotwin2": robotwin_npz,
                "robotwin_actions_semantics": "the deployed producer stores "
                "`joint_action/vector` (14-D joint state) from the clean replay, not a commanded "
                "action; the commanded joint path lived in the source trajectory pkl",
            },
        },
        "geometry_fields": {
            "libero_K": bool(libero.get("camera_intrinsics", {}).get("K")),
            "robotwin_K": bool("get_intrinsic_matrix" in robotwin.get("sapien_camera_api", []))
            and bool(robotwin.get("camera_module_import_ok")),
            "libero_state_len_gt_0": bool(libero.get("npz", {}).get("states_shape", [0, 0])[1] > 0),
            "robotwin_robot_states": bool(robotwin_npz.get("robot_state_dim") == 16),
            "libero_depth_render": bool(libero.get("depth_render")),
            "robotwin_depth_api": bool("get_depth" in robotwin_camera_methods),
            "libero_segmentation_render": bool(libero.get("segmentation_render")),
            "robotwin_segmentation_api": bool("get_segmentation" in robotwin_camera_methods),
            "libero_contacts": bool(contacts.get("ok")),
            "robotwin_contacts_api": bool(
                robotwin_scan.get("envs/_base_task.py", {}).get("def get_scene_contact")
            ),
            # No live RoboTwin task environment was created this round; the API and asset checks
            # ran, the runtime extraction paths did not.
            "robotwin_K_executed": False,
            "robotwin_depth_executed": False,
            "robotwin_segmentation_executed": False,
            "robotwin_contacts_executed": False,
            "evidence": {
                "libero": {
                    "K": libero.get("camera_intrinsics", {}).get("K"),
                    "depth_render": libero.get("depth_render"),
                    "segmentation_render": libero.get("segmentation_render"),
                    "contacts": contacts,
                },
                "robotwin2": {
                    "sapien_camera_api": robotwin.get("sapien_camera_api"),
                    "camera_class_methods": robotwin_camera_methods,
                    "api_source_scan": robotwin_scan,
                    "npz_pose_keys": [key for key in robotwin_npz.get("keys", []) if "pose" in key],
                },
            },
        },
    }
    out = HERE / "remote-evidence.json"
    out.write_text(
        json.dumps(evidence, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"written": str(out), "bytes": out.stat().st_size}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
