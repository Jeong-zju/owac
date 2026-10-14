"""Recover only the 33 missing full-manifest RoboTwin identities with audited expert guards."""

import argparse
import json
import os
import shutil
import signal
from pathlib import Path

from owac.runtime.rawvla_full_eval import (
    digest,
    paired_seed,
    rows,
    run_process,
    validate_manifest,
    write_json,
)
from owac.runtime.rawvla_repair import TASKS


def missing_cases(manifest, ledger, attempts):
    entries = validate_manifest(manifest, "robotwin")
    expected = {f"robotwin-{index:04d}": entry for index, entry in enumerate(entries)}
    completed = {row["case_id"]: row for row in ledger}
    if len(completed) != len(ledger) or len(completed) != 3217:
        raise ValueError("Expected exactly 3217 unique inherited completed cases")
    for case_id, row in completed.items():
        if row["entry"] != expected.get(case_id):
            raise ValueError("Inherited result differs from original manifest")
    failures = {row["case_id"]: row for row in attempts if row["returncode"] != 0}
    missing = sorted(set(expected) - set(completed))
    if len(missing) != 33:
        raise ValueError("Expected exactly the 33 missing original identities")
    cases = []
    for case_id in missing:
        entry = expected[case_id]
        if entry["task_name"] not in TASKS or case_id not in failures:
            raise ValueError("Missing case is outside diagnosed expert-check failures")
        cases.append({"case_id": case_id, "entry": entry, "old_failure": failures[case_id]})
    return cases


def prepare(parent, output, workers=3):
    manifest = json.loads((parent / "robotwin-manifest.json").read_text())
    ledger = rows(parent / "robotwin/episodes.jsonl")
    cases = missing_cases(manifest, ledger, rows(parent / "robotwin/attempts.jsonl"))
    if workers < 1 or workers > len(cases):
        raise ValueError("Invalid finite worker count")
    for case in cases:
        old = Path(case["old_failure"]["directory"])
        if rows(old / "actions.jsonl"):
            raise ValueError("Diagnosed failure already applied policy actions")
        text = (old / "run.log").read_text(errors="replace")
        if not any(
            message in text
            for message in (
                "target_pose cannot be None for move action.",
                "IndexError: list index out of range",
            )
        ):
            raise ValueError("Unexpected failure requires a separate repair diagnosis")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "frozen", output / "frozen")
    destination = output / "frozen/src/owac/runtime"
    for name in ("rawvla_repair.py", "rawvla_main_repair.py"):
        shutil.copy2(Path(__file__).with_name(name), destination / name)
    shutil.copytree(parent / "runtime", output / "runtime")
    for name in ("protocol.json", "libero-manifest.json", "robotwin-manifest.json"):
        shutil.copy2(parent / name, output / name)
    (output / "libero").symlink_to((parent / "libero").resolve())
    (output / "robotwin-paper").symlink_to((parent / "robotwin-paper").resolve())
    (output / "task_config").symlink_to((parent / "task_config").resolve())
    (output / "robotwin").mkdir()
    shutil.copy2(parent / "robotwin/episodes.jsonl", output / "robotwin/episodes.jsonl")
    script = output / "runtime/robotwin_eval.py"
    source = script.read_text()
    anchor = "                episode_info = TASK_ENV.play_once()\n"
    if source.count(anchor) != 1:
        raise ValueError("Frozen expert check anchor changed")
    script.write_text(
        source.replace(
            anchor,
            "                from owac.runtime.rawvla_repair import expert_check_with_repair\n"
            "                episode_info = expert_check_with_repair(TASK_ENV)\n",
        )
    )
    assignments = [cases[index::workers] for index in range(workers)]
    for index, selected in enumerate(assignments):
        worker = output / "workers" / str(index)
        worker.mkdir(parents=True)
        write_json(worker / "cases.json", selected)
    original = parent.parent / "pi05-paper-repair-20261013-v1/source-guards.json"
    guards = json.loads(original.read_text())
    if any(digest(name) != expected for name, expected in guards.items()):
        raise ValueError("Original simulator differs from validated five-case repair")
    shutil.copy2(original, output / "source-guards.json")
    parity = parent.parent / "pi05-paper-repair-20261013-v1/control-input-parity.json"
    shutil.copy2(parity, output / "control-input-parity.json")
    write_json(
        output / "repair-protocol.json",
        {
            "version": "owac-pi05-full-expert-repair-v1",
            "parent": str(parent),
            "expected": 3250,
            "inherited_completed": len(ledger),
            "parent_protocol_sha256": digest(parent / "protocol.json"),
            "manifest_sha256": digest(parent / "robotwin-manifest.json"),
            "inherited_ledger_sha256": digest(parent / "robotwin/episodes.jsonl"),
            "cases": cases,
            "workers": workers,
            "worker_cases_sha256": {
                str(index): digest(output / "workers" / str(index) / "cases.json")
                for index in range(workers)
            },
            "runner_sha256": digest(__file__),
            "expert_repair_sha256": digest(Path(__file__).with_name("rawvla_repair.py")),
            "patched_eval_sha256": digest(script),
            "original_eval_sha256": digest(parent / "runtime/robotwin_eval.py"),
            "scope": "Expert-only empty-plan guards; restored before original policy setup",
            "unchanged": "Original seed, paired lighting/noise and language, PI05 checkpoint/"
            "statistics, model/input/actions, physics, success test and step limit",
            "flow_seed_rule": "Original paired_seed(robotwin, task, dataset_seed)",
            "policy_server": "Same synchronously serialized PI05 RPC service localhost:18015",
        },
    )


def run(root, output, index):
    from owac.runtime.rawvla_job_guard import terminate_children

    protocol = json.loads((output / "repair-protocol.json").read_text())
    if (
        digest(__file__) != protocol["runner_sha256"]
        or digest(output / "runtime/robotwin_eval.py") != protocol["patched_eval_sha256"]
        or digest(Path(__file__).with_name("rawvla_repair.py")) != protocol["expert_repair_sha256"]
    ):
        raise ValueError("Prepared repair source changed")
    worker = output / "workers" / str(index)
    if digest(worker / "cases.json") != protocol["worker_cases_sha256"][str(index)]:
        raise ValueError("Worker assignment changed")
    cases = json.loads((worker / "cases.json").read_text())
    signal.signal(signal.SIGTERM, terminate_children)
    signal.signal(signal.SIGINT, terminate_children)
    completed = {row["case_id"] for row in rows(worker / "episodes.jsonl")}
    source = root / "source"
    manifest = json.loads((output / "robotwin-manifest.json").read_text())
    for selected in cases:
        case_id, entry = selected["case_id"], selected["entry"]
        if case_id in completed:
            continue
        seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
        for _ in range(2):
            case_root = output / "robotwin" / case_id
            prior = list(case_root.glob("attempt*"))
            number = 1 + max((int(path.name[7:]) for path in prior), default=0)
            directory = case_root / f"attempt{number}"
            directory.mkdir(parents=True)
            write_json(directory / "entry.json", entry)
            env = os.environ.copy()
            env.update(
                PYTHONPATH=f"{output}/frozen/src:{source}/starVLA:"
                f"{source}/third_party/RoboTwin:{source}/third_party/RoboTwin/script:"
                f"{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
                PATH=f"{root}/envs/robotwin/bin:" + env["PATH"],
                CUDA_HOME="/usr/local/cuda-12.8",
                RAWVLA_ROBOTWIN_DENOISER="optix",
                ROBOTWIN_TEST_NUM="1",
                ROBOTWIN_EVAL_SEEDS=str(entry["dataset_seed"]),
                ROBOTWIN_FORCE_EVAL_SEED="1",
                ROBOTWIN_IMAGE_OBS_KEY="render_rgb",
                ROBOTWIN_LIGHTING_DOMAIN=entry["lighting_domain"],
                ROBOTWIN_LIGHTING_STRATEGY=entry["lighting_strategy"],
                ROBOTWIN_LIGHTING_SAMPLE_UNIT="paired",
                ROBOTWIN_LIGHTING_BENCHMARK_SEED=str(manifest["benchmark_seed"]),
                ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
                ROBOTWIN_POLICY_AUDIT_DIR=str(directory),
                RAWVLA_STEP_AUDIT=str(directory / "actions.jsonl"),
                START_EPISODE=str(entry["dataset_episode_index"]),
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
                "owac-pi05-full-v1",
                "--seed",
                "0",
                "--policy_name",
                "owac.runtime.rawvla_robotwin",
            ]
            rc, seconds = run_process(
                command, source / "third_party/RoboTwin", env, directory / "run.log", 900
            )
            results = rows(directory / "episode_results.jsonl")
            record = {
                "case_id": case_id,
                "entry": entry,
                "attempt": number,
                "returncode": rc,
                "wall_seconds": seconds,
                "policy_flow_seed": seed,
                "directory": str(directory),
            }
            if rc == 0 and len(results) == 1 and (directory / "lighting.json").exists():
                if results[0]["eval_seed"] != entry["dataset_seed"]:
                    raise ValueError("Recovered physical seed changed")
                record.update(results[0])
                record["actions_sha256"] = digest(directory / "actions.jsonl")
                with (worker / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                completed.add(case_id)
            with (worker / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            write_json(
                worker / "status.json",
                {"expected": len(cases), "completed": len(completed), "last_case_id": case_id},
            )
            print(json.dumps({"worker": index, "case_id": case_id, "returncode": rc}), flush=True)
            if case_id in completed:
                break
    write_json(
        worker / "status.json",
        {
            "state": "complete" if len(completed) == len(cases) else "incomplete",
            "expected": len(cases),
            "completed": len(completed),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=Path("/workspace/owac/rawvla"))
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.parent, args.output, args.workers)
    else:
        run(args.root, args.output, args.worker)


if __name__ == "__main__":
    main()
