"""Recompute completion, action provenance, input pairing and video receipts."""

import json
import subprocess
from pathlib import Path

import numpy as np


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    root = Path("outputs/n02-v12/demos/r5-lighting-20261016")
    receipt = {"arms": {}, "pairs": [], "optimizer_updates": 0}
    for arm in ("raw", "rgb"):
        ledger = records(root / arm / "episodes.jsonl")
        assert len(ledger) == 5 and len({r["case_id"] for r in ledger}) == 5
        inference = records(root / f"{arm}-policy" / "inference.jsonl")
        load = json.loads((root / f"{arm}-policy" / "load_audit.json").read_text())
        cases = []
        for row in ledger:
            # Remote record paths are preserved verbatim; map their suffix to the local copy.
            directory = root / arm / row["case_id"] / f"attempt{row['attempt']}"
            actions = records(directory / "actions.jsonl")
            predictions = [r for r in inference if r["case_id"] == row["case_id"]]
            assert [r["step"] for r in actions] == list(range(row["num_steps"]))
            assert [r["step"] for r in predictions] == list(range(0, row["num_steps"], 50))
            predicted = np.concatenate([r["actions"] for r in predictions])[: len(actions)]
            executed = np.array([r["action"] for r in actions])
            difference = float(np.max(np.abs(predicted - executed)))
            assert difference == 0
            for prediction in predictions:
                assert prediction["finite"] and prediction["shape"] == [50, 14]
                stats = prediction["run_stats"]
                assert stats["prefix_builds"] == 1 and stats["encoder_calls"] == 1
                assert stats["denoise_steps"] == 10
                step = prediction["step"]
                expected = [0] * 8 if step == 0 else list(range(step - 7, step + 1))
                assert prediction["history_steps"] == expected
            video = directory / "rollout.mp4"
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration,size",
                    "-of",
                    "json",
                    str(video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cases.append(
                {
                    "case_id": row["case_id"],
                    "domain": row["entry"]["lighting_domain"],
                    "ev": row["entry"]["lighting_ev"],
                    "success": row["success"],
                    "steps": row["num_steps"],
                    "wall_seconds": row["wall_seconds"],
                    "video": str(video),
                    "video_probe": json.loads(probe.stdout),
                    "actual_minus_predicted_action_max": difference,
                    "decision_count": len(predictions),
                    "language": row["language"],
                    "returncode": row["returncode"],
                }
            )
        receipt["arms"][arm] = {
            "completed": len(ledger),
            "success": sum(int(r["success"]) for r in ledger),
            "checkpoint": load["checkpoint"],
            "checkpoint_sha256": load["checkpoint_sha256"],
            "inference_p50_seconds": float(np.median([r["inference_seconds"] for r in inference])),
            "inference_p95_seconds": float(
                np.percentile([r["inference_seconds"] for r in inference], 95)
            ),
            "cases": cases,
        }
    for case in receipt["arms"]["raw"]["cases"]:
        case_id = case["case_id"]
        left = np.load(root / "raw-policy" / f"{case_id}_first_input.npz")
        right = np.load(root / "rgb-policy" / f"{case_id}_first_input.npz")
        raw = {
            key: float(np.max(np.abs(left[key] - right[key])))
            for key in ("raw_0", "raw_1", "raw_2")
        }
        state_difference = float(np.max(np.abs(left["state"] - right["state"])))
        assert state_difference == 0 and max(raw.values()) == 0
        receipt["pairs"].append(
            {
                "case_id": case_id,
                "initial_state_max_abs_difference": state_difference,
                "initial_raw_max_abs_difference": raw,
                "raw_head_mean": float(left["raw_0"].mean()),
                "raw_dtype": str(left["raw_0"].dtype),
                "raw_shape": list(left["raw_0"].shape),
            }
        )
    destination = Path("research/experiments/n02_v1_2/lighting_demo_20261016/audit.json")
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                arm: {k: v for k, v in report.items() if k != "cases"}
                for arm, report in receipt["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
