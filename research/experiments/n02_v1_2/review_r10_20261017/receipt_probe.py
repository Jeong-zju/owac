"""Read existing remote metadata and command traces; no model or simulator runs."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    root = Path("outputs/n02-v12/review-r10-receipts")
    root.mkdir(parents=True, exist_ok=False)
    data_root = Path("/workspace/owac/rawvla/source/benchmark_data/RawVLA-Bench")
    files = sorted((data_root / "data/robotwin2/raw/grab_roller").glob("episode_*.npz"))
    metadata = []
    for p in files:
        with np.load(p, allow_pickle=False) as archive:
            m = json.loads(str(archive["metadata_json"].item()))
        metadata.append(
            dict(
                path=str(p.relative_to(data_root)),
                task=m.get("task_name"),
                task_description=m.get("task_description"),
                frames=m["num_frames"],
                clean_protocol=m.get("clean_protocol"),
            )
        )
    case = Path("outputs/n02-v12/representation-r8-rollouts/task-r9-control-a1-step0032")
    paths = [
        case / "control.jsonl",
        case / "actions.jsonl",
        case / "policy/inference.jsonl",
        Path("outputs/n02-v12/policy-audit-20261016/original-normal/actions.jsonl"),
    ]
    controls, actual, inference, native = [
        [json.loads(s) for s in p.read_text().splitlines()] for p in paths
    ]
    joints = [*range(6), *range(7, 13)]
    requested, executed, state = [
        np.array([r[k] for r in rows])
        for rows, k in [(controls, "requested"), (controls, "executed"), (actual, "state")]
    ]
    modified = requested - executed
    result = dict(
        optimizer_updates=0,
        new_closed_loops=0,
        metadata=metadata,
        task_descriptions_present=sum(bool(m["task_description"]) for m in metadata),
        live_languages=dict(Counter(r["language"] for r in inference)),
        raw_control=dict(
            commands=len(controls),
            modified_commands=int((np.abs(modified).max(1) > 1e-6).sum()),
            requested_executed_joint_rms_rad=float(np.mean(modified[:, joints] ** 2) ** 0.5),
            requested_executed_joint_max_rad=float(np.abs(modified[:, joints]).max()),
            requested_executed_gripper_max=float(np.abs(modified[:, [6, 13]]).max()),
            command_vs_measured_joint_max_rad=float(
                np.abs(executed[:, joints] - state[:, joints]).max()
            ),
            first_requested=requested[0].tolist(),
            first_executed=executed[0].tolist(),
        ),
        file_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    native_requested = np.array([r["action"] for r in native])
    limited = []
    previous = np.array(native[0]["state"])
    for command in native_requested:
        previous = previous + np.clip(command - previous, -0.05, 0.05)
        previous[[6, 13]] = previous[[6, 13]].clip(0, 1)
        limited.append(previous.copy())
    diff = np.array(limited) - native_requested
    result["native_offline_limiter_counterfactual"] = dict(
        scope="apply limiter to saved native commands; no new physical trajectory or success claim",
        commands=len(native),
        modified_commands=int((np.abs(diff).max(1) > 1e-6).sum()),
        joint_rms_difference_rad=float(np.mean(diff[:, joints] ** 2) ** 0.5),
        joint_max_difference_rad=float(np.abs(diff[:, joints]).max()),
        gripper_max_difference=float(np.abs(diff[:, [6, 13]]).max()),
    )
    (root / "receipt_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
