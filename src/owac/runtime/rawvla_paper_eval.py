"""Separate RoboTwin evaluation using the 632 original cache lighting conditions.

This is a candidate for the paper table denominator, not an authenticated copy
of the authors' unpublished rollout ledger. Full 3,250-case evaluation continues.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from owac.runtime.rawvla_full_eval import digest, paired_seed, rows, run_process, write_json


def get_model(arguments):
    from envs import _base_task

    from owac.runtime import rawvla_robotwin

    client = rawvla_robotwin.get_model(arguments)
    original = _base_task.apply_lighting_from_env

    def apply_original_conditions(task, *, seed=0):
        metadata = original(task, seed=seed)
        expected = json.loads(os.environ["RAWVLA_EXPECTED_ENTRY"])
        task.cameras.raw_sensor_noise_seed = int(expected["noise_seed"])
        task.cameras.raw_sensor_noise_frame_index = 0
        metadata["noise_seed"] = int(expected["noise_seed"])
        task.rawvla_lighting_metadata = dict(metadata)
        return metadata

    _base_task.apply_lighting_from_env = apply_original_conditions
    return client


def reset_model(model):
    from owac.runtime.rawvla_robotwin import reset_model as reset

    reset(model)


def eval(task, model, observation):
    from owac.runtime.rawvla_robotwin import eval as predict

    predict(task, model, observation)


def summarize(output, expected, finished=False):
    ledger = rows(output / "robotwin-paper/episodes.jsonl")
    identities = {row["case_id"] for row in ledger}
    if len(identities) != len(ledger):
        raise ValueError("Duplicate paper-condition episode")
    domains = defaultdict(lambda: {"total": 0, "success": 0})
    for row in ledger:
        domain = domains[row["entry"]["lighting_domain"]]
        domain["total"] += 1
        domain["success"] += int(row["success"])
    complete = len(ledger) == expected
    result = {
        "scope": "Additional 632 original-cache-condition rollouts; paper-table protocol inferred",
        "state": "complete" if complete else ("incomplete" if finished else "running"),
        "expected": expected,
        "completed": len(ledger),
        "success": sum(row["success"] for row in ledger),
        "domains": dict(domains),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if complete:
        result["success_rate_percent"] = 100 * result["success"] / expected
        for domain in domains.values():
            domain["success_rate_percent"] = 100 * domain["success"] / domain["total"]
    write_json(output / "robotwin-paper/summary.json", result)
    return result


def run(root, output, max_jobs=0):
    source = root / "source"
    selection = output / "robotwin-paper-conditions.json"
    selected = json.loads(selection.read_text())
    cases = selected["entries"]
    if len(cases) != 632 or selected["mismatches"]:
        raise ValueError("Unverified original conditions")
    directory = output / "robotwin-paper"
    directory.mkdir(exist_ok=True)
    protocol_file = directory / "protocol.json"
    if not protocol_file.exists():
        write_json(
            protocol_file,
            {
                "version": "owac-pi05-original-conditions-v1",
                "episodes": 632,
                "parent_protocol_sha256": digest(output / "protocol.json"),
                "selection_sha256": digest(selection),
                "code_sha256": digest(__file__),
                "weights": "Same frozen PI05 RPC service :18015",
                "image": "render_rgb (SAPIEN Color)",
                "lighting": "Original NPZ metadata EV, sample unit and noise seed",
                "published_default_rates_percent": {
                    "ExtremeLow": 17.32,
                    "Low": 51.18,
                    "Normal": 70.87,
                    "Over": 56.35,
                    "ExtremeOver": 38.40,
                    "Overall": 46.84,
                },
                "published_counts_inferred": {
                    "ExtremeLow": [22, 127],
                    "Low": [65, 127],
                    "Normal": [90, 127],
                    "Over": [71, 126],
                    "ExtremeOver": [48, 125],
                    "Overall": [296, 632],
                },
                "inference_limit": "Published percentages match cache domain denominators, "
                "but the authors did not expose the paper-specific policy rollout ledger",
            },
        )
    protocol = json.loads(protocol_file.read_text())
    if protocol["selection_sha256"] != digest(selection) or protocol["code_sha256"] != digest(
        __file__
    ):
        raise ValueError("Original-condition protocol changed")
    completed = {row["case_id"] for row in rows(directory / "episodes.jsonl")}
    jobs = 0
    for case in cases:
        case_id, entry = case["case_id"], case["entry"]
        if case_id in completed:
            continue
        seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
        for _ in range(2):
            case_root = directory / case_id
            prior = list(case_root.glob("attempt*"))
            attempt = 1 + max((int(path.name[7:]) for path in prior), default=0)
            attempt_root = case_root / f"attempt{attempt}"
            attempt_root.mkdir(parents=True)
            write_json(attempt_root / "entry.json", case)
            environment = os.environ.copy()
            environment.update(
                PYTHONPATH=f"{output}/frozen/src:{source}/starVLA:{source}/third_party/RoboTwin:"
                f"{source}/third_party/RoboTwin/script:"
                f"{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
                PATH=f"{root}/envs/robotwin/bin:" + environment["PATH"],
                CUDA_HOME="/usr/local/cuda-12.8",
                RAWVLA_ROBOTWIN_DENOISER="optix",
                ROBOTWIN_TEST_NUM="1",
                ROBOTWIN_EVAL_SEEDS=str(entry["dataset_seed"]),
                ROBOTWIN_FORCE_EVAL_SEED="1",
                ROBOTWIN_IMAGE_OBS_KEY="render_rgb",
                ROBOTWIN_LIGHTING_DOMAIN=entry["lighting_domain"],
                ROBOTWIN_LIGHTING_STRATEGY=entry["lighting_strategy"],
                ROBOTWIN_LIGHTING_SAMPLE_UNIT=str(entry["lighting_sample_unit"]),
                ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
                ROBOTWIN_POLICY_AUDIT_DIR=str(attempt_root),
                ROBOTWIN_POLICY_CHECKPOINT=str(
                    root / "checkpoints/pi05_robotwin/model.safetensors"
                ),
                START_EPISODE=str(entry["dataset_episode_index"]),
                RAWVLA_STEP_AUDIT=str(attempt_root / "actions.jsonl"),
                RAWVLA_FLOW_SEED=str(seed),
                RAWVLA_LANGUAGE_SEED=str(seed),
                RAWVLA_CASE_ID=f"{output.name}/{case_id}",
                RAWVLA_EXPECTED_ENTRY=json.dumps(entry),
                VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
            )
            command = [
                str(root / "envs/robotwin/bin/python"),
                str(output / "runtime/robotwin_eval.py"),
                "--config",
                str(output / "runtime/robotwin_deploy.yml"),
                "--policy_ckpt_path",
                str(root / "checkpoints/pi05_robotwin/model.safetensors"),
                "--overrides",
                "--task_name",
                entry["task_name"],
                "--task_config",
                "owac_rawvla_clean",
                "--ckpt_setting",
                "owac-pi05-original-conditions-v1",
                "--seed",
                "0",
                "--policy_name",
                "owac.runtime.rawvla_paper_eval",
            ]
            rc, seconds = run_process(
                command, source / "third_party/RoboTwin", environment, attempt_root / "run.log", 900
            )
            results = rows(attempt_root / "episode_results.jsonl")
            record = {
                "case_id": case_id,
                "entry": entry,
                "attempt": attempt,
                "returncode": rc,
                "wall_seconds": seconds,
                "directory": str(attempt_root),
                "source_metadata_sha256": case["metadata_sha256"],
            }
            if rc == 0 and len(results) == 1 and (attempt_root / "lighting.json").exists():
                if results[0]["eval_seed"] != entry["dataset_seed"]:
                    raise ValueError("Original-condition seed changed")
                record.update(results[0])
                record["actions_sha256"] = digest(attempt_root / "actions.jsonl")
                with (directory / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                completed.add(case_id)
            with (directory / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            summarize(output, 632)
            if case_id in completed:
                break
        jobs += 1
        if max_jobs and jobs >= max_jobs:
            return
    summarize(output, 632, finished=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/owac/rawvla", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-jobs", default=0, type=int)
    args = parser.parse_args()
    run(args.root, args.output, args.max_jobs)


if __name__ == "__main__":
    main()
