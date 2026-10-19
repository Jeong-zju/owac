"""Audit actual bounded fit updates, landmark coverage and eligible checkpoints."""

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

METRICS = ("joint_rms_deg", "first_joint_max_deg", "gripper_mae", "gripper_max_error")
SEEDS = (188390828, 188390878, 188390928, 188390978)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root):
    evidence = root / "research/experiments/n02_v1_2/sequence_r12_20261019"
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - remote Python3.10
        "scope": "bounded development fit, not P3/P4 or closed-loop capability",
        "stage1": read(evidence / "stage1.json"),
        "stage2": {"arms": {}, "total_optimizer_updates": 0, "cap": 512},
    }
    for label in ("one", "three"):
        run = root / f"outputs/n02-v12/P2/raw-fewshot-r12-{label}-s1103-a1"
        status, config = read(run / "status.json"), read(run / "run_config.json")
        if status["state"] != "completed" or status["step"] != 256:
            raise ValueError(f"{label}: bounded run has not completed256 updates")
        updates = [
            json.loads(line) for line in (run / "optimization.jsonl").read_text().splitlines()
        ]
        if [r["step"] for r in updates] != list(range(1, 257)):
            raise ValueError(f"{label}: actual optimizer-update receipts differ from256")
        for row in updates:
            values = [v for v in row["terms"].values() if v is not None]
            values += list(row["gradient_norms_by_clip_group"].values())
            if not values or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{label}: nonfinite logged term or gradient norm")
        selected = set(config["task_adaptation"]["fewshot"]["episode_paths"])
        seen = set()
        for row in updates:
            for window in row["windows"]:
                paths, frames = window["anchors"], window["current_frame_indices"]
                if len(paths) != len(frames) or window["partition"] != "train":
                    raise ValueError("invalid actual training-window receipt")
                seen.update(zip(paths, frames, strict=True))
        if {path for path, _ in seen} != selected:
            raise ValueError("actual training paths differ from selected train episodes")
        sampler = read(run / "task_sampler.json")
        if len(seen) != sampler["reachable_valid_anchors"]:
            raise ValueError("actual updates have not covered every selected valid anchor")
        resume = read(run / "resume128_plan.json")
        if sha(run / "fit_0000000.json") != resume["initial_fit_sha256"]:
            raise ValueError("the original zero-update fit was overwritten")
        windows = read(run / "fit_windows.json")
        expected = set(
            itertools.product([(row["path"], row["t"], row["index"]) for row in windows], SEEDS)
        )
        fits, eligible = [], []
        for step in (0, 64, 128, 192, 256):
            fit = read(run / f"fit_{step:07d}.json")
            actual = {((r["path"], r["t"], r["index"]), r["seed"]) for r in fit["rows"]}
            if fit["step"] != step or actual != expected or len(fit["rows"]) != len(expected):
                raise ValueError("fit landmark/seed coverage differs from the declared protocol")
            maxima = {key: max(row[key] for row in fit["rows"]) for key in METRICS}
            if not all(math.isfinite(v) for row in fit["rows"] for v in (row[k] for k in METRICS)):
                raise ValueError("nonfinite fit error")
            if maxima != fit["maxima"]:
                raise ValueError("reported fit maxima disagree with the actual rows")
            gate = config["task_adaptation"]["fewshot"]["fit_gate"]
            if fit["limits"] != gate:
                raise ValueError("fit limits changed")
            passed = all(maxima[key] <= gate[key + "_max"] for key in METRICS)
            if passed != fit["passed"]:
                raise ValueError("reported fit verdict disagrees with the fixed limits")
            checkpoint = run / f"checkpoint_{step:07d}.pt"
            if step and not checkpoint.is_file():
                raise ValueError("saved fit point has no actual checkpoint")
            record = {"step": step, "passed": passed, "maxima": maxima}
            fits.append(record)
            if step and passed:
                eligible.append(
                    {**record, "checkpoint": str(checkpoint), "sha256": sha(checkpoint)}
                )
        report = read(run / "report.json")
        initial = read(run / "segments/initial128/report.json")
        checks = read(run / "checks.json")
        result["stage2"]["arms"][label] = {
            "run_id": config["run_id"],
            "actual_optimizer_updates": len(updates),
            "actual_anchor_exposures": sum(r["actual_anchors"] for r in updates),
            "unique_train_anchors_seen": len(seen),
            "only_selected_training_paths": True,
            "all_logged_terms_and_clip_norms_finite": True,
            "original_zero_update_fit_preserved": True,
            "fit_evaluations_per_checkpoint": len(expected),
            "fits": fits,
            "eligible_checkpoints": eligible,
            "final_checkpoint_sha256": sha(run / "checkpoint_0000256.pt"),
            "trained_parameters": report["trained_parameters"],
            "performance_initial128": initial["overview"],
            "performance_additional128": report["overview"],
            "save_resume_teacher_velocity_check": checks.get("save_resume"),
            "final_independent_dev": read(run / "dev_sampled_latest.json")["sampled_control_mse"],
        }
        result["stage2"]["total_optimizer_updates"] += len(updates)
    if result["stage2"]["total_optimizer_updates"] != 512:
        raise ValueError("bounded total must be512 actual updates")
    any_fit = any(a["eligible_checkpoints"] for a in result["stage2"]["arms"].values())
    result["stage3"] = {
        "status": "ready_for_normal_development_rollout"
        if any_fit
        else "not_started_fit_gate_failed",
        "completed_new_rollouts": 0,
        "max_new_rollouts": 3,
        "physical_success_criteria": "unchanged",
        "not_run_is_not_a_failure_or_success": True,
    }
    destination = evidence / "stage2_completion_audit.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"receipt": str(destination), "stage3": result["stage3"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/workspace/owac/worktrees/n02-measurement-completion")
    )
    collect(parser.parse_args().root)
