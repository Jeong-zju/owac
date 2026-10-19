"""Independent receipt reduction: initial inputs, actual actions, video and objective probes."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("outputs/n02-v12/policy-audit-20261016")
    original = np.load(root / "original-normal/initial_observation.npz")
    report = {"optimizer_updates": 0, "cases": [], "training_probes": {}}
    for mode in ("original", "bridge", "bridge-unpadded", "n02-raw-native", "n02-rgb-native"):
        case = root / f"{mode}-normal"
        receipt = json.loads((case / "result.json").read_text())
        assert receipt["returncode"] == 0 and len(receipt["results"]) == 1
        result = receipt["results"][0]
        executed = rows(case / "actions.jsonl")
        predictions = rows(root / f"{mode}-policy/inference.jsonl")
        prediction = np.concatenate([row["actions"] for row in predictions])[: len(executed)]
        actual = np.array([row["action"] for row in executed])
        difference = float(np.max(np.abs(prediction - actual)))
        assert difference == 0
        assert [row["step"] for row in executed] == list(range(result["num_steps"]))
        if mode.startswith("n02-"):
            sample = np.load(root / f"{mode}-policy/audit-{mode}-normal_first_input.npz")
            load = json.loads((root / f"{mode}-policy/load_audit.json").read_text())
            assert load["legacy_weights_probe"] and load["optimizer_updates"] == 0
            assert load["policy_input_contract"] == "pi05-native-language-state-v1"
            for row in predictions:
                step = row["step"]
                assert row["history_steps"] == (
                    [0] * 8 if step == 0 else list(range(step - 7, step + 1))
                )
        else:
            sample = np.load(case / "initial_observation.npz")
            load = json.loads((root / f"{mode}-policy/load-audit.json").read_text())
        initial = {
            key: float(np.max(np.abs(original[key] - sample[key])))
            for key in ("state", "raw_0", "raw_1", "raw_2")
        }
        assert max(initial.values()) == 0
        video = case / "episode0.mp4"
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
        report["cases"].append(
            {
                "mode": mode,
                "success": result["success"],
                "steps": result["num_steps"],
                "language": result["language"],
                "entry": receipt["entry"],
                "wall_seconds": receipt["wall_seconds"],
                "initial_input_max_differences": initial,
                "executed_minus_predicted_action_max": difference,
                "decisions": len(predictions),
                "video": str(video),
                "video_sha256": digest(video),
                "video_probe": json.loads(probe.stdout),
                "weights": load,
            }
        )
    old_root = Path("outputs/n02-v12/demos/r5-lighting-20261016")
    report["old_r5_initial_input_max_differences"] = {}
    for arm in ("raw", "rgb"):
        sample = np.load(old_root / f"{arm}-policy/grab-roller-s0-Normal_first_input.npz")
        differences = {
            key: float(np.max(np.abs(original[key] - sample[key])))
            for key in ("state", "raw_0", "raw_1", "raw_2")
        }
        assert max(differences.values()) == 0
        report["old_r5_initial_input_max_differences"][arm] = differences
        probe = json.loads((root / f"trained-{arm}/probe.json").read_text())
        summaries = {}
        for condition in ("legacy", "native"):
            summaries[condition] = {
                key: float(np.mean([window[condition][key] for window in probe["dev_windows"]]))
                for key in (
                    "flow_mse_32",
                    "flow_mse_control14",
                    "sampled_action_mse_control14",
                    "first5_sample_mse",
                )
            }
        gradients = probe["objective_gradients"]
        report["training_probes"][arm] = {
            "step": probe["step"],
            "windows": len(probe["dev_windows"]),
            "means": summaries,
            "visual_token_rms_mean": float(
                np.mean([window["token_rms"] for window in probe["dev_windows"]])
            ),
            "gradient_window": probe["gradient_window"],
            "objective_gradients": gradients,
            "causal_to_action_gradient_l2_ratio": gradients["causal_raw"]["gradient_l2"]
            / gradients["action"]["gradient_l2"],
            "limitation": (
                "one training window; full-parameter norms, "
                "not a causal attribution or shared-layer conflict test"
            ),
        }
    native = np.array([row["action"] for row in rows(root / "original-normal/actions.jsonl")])
    bridge = np.array([row["action"] for row in rows(root / "bridge-normal/actions.jsonl")])
    report["native_bridge_full_executed_trajectory_max_difference"] = float(
        np.max(np.abs(native - bridge))
    )
    report["bridge_parity"] = json.loads((root / "bridge-parity/parity.json").read_text())
    report["r6_language_parity"] = json.loads((root / "r6-language-parity.json").read_text())
    (root / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": [
                    {key: row[key] for key in ("mode", "success", "steps")}
                    for row in report["cases"]
                ],
                "training_probes": report["training_probes"],
                "native_bridge_trajectory_max": report[
                    "native_bridge_full_executed_trajectory_max_difference"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
