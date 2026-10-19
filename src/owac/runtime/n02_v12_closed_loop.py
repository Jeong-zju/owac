"""Bounded paired N02 lighting demonstration; never launches a full benchmark or training."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

from owac.runtime.rawvla_full_eval import digest, paired_seed, rows, run_process, write_json

DOMAINS = ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver")


def prepare(root, output, checkout):
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    source = root / "source"
    parent = source / (
        "experiments/simulation/robotwin/manifests/"
        "robotwin2_test_manifest_50_seeds_3250_rollouts.json"
    )
    manifest = json.loads(parent.read_text())
    # The first parent task and first initial seed, chosen without inspecting outcomes.
    first = manifest["entries"][0]
    cases = [
        next(
            dict(e, parent_index=i)
            for i, e in enumerate(manifest["entries"])
            if e["task_name"] == first["task_name"]
            and e["dataset_seed"] == first["dataset_seed"]
            and e["lighting_domain"] == domain
        )
        for domain in DOMAINS
    ]
    code = (source / "third_party/RoboTwin/script/eval_policy.py").read_text()
    code = "from owac.runtime.rawvla_full_eval import seed_instruction_generators\n" + code
    for exception in ("UnStableError", "Exception"):
        old = f"            except {exception} as e:\n"
        assert code.count(old) == 1
        code = code.replace(
            old, old + "                if eval_seeds:\n                    raise\n"
        )
    old = "        if use_dataset_instructions:\n"
    assert code.count(old) == 1
    code = code.replace(
        old, '        seed_instruction_generators(int(os.environ["RAWVLA_LANGUAGE_SEED"]))\n' + old
    )
    # Display RGB is for video; the model continues to consume float32 raw_linear.
    old = "            observation = TASK_ENV.get_obs()\n"
    assert code.count(old) == 1
    code = code.replace(
        old,
        old
        + '            head = observation["observation"]["head_camera"]\n'
        + '            head["rgb"] = head["render_rgb"]\n'
        # get_obs deep-copies now_obs; take_action writes that cached copy to video.
        + (
            '            TASK_ENV.now_obs["observation"]["head_camera"]["rgb"] = '
            'head["render_rgb"]\n'
        ),
    )
    old = "        video_save_dir = save_dir\n"
    assert code.count(old) == 1
    code = code.replace(old, '        video_save_dir = Path(os.environ["N02_CASE_VIDEO_DIR"])\n')
    compile(code, str(runtime / "robotwin_eval.py"), "exec")
    (runtime / "robotwin_eval.py").write_text(code)
    (output / "task_config").symlink_to(source / "third_party/RoboTwin/task_config")
    template = (
        source / "starVLA/examples/simBenchmarks/Robotwin/eval_files/deploy_policy.yml"
    ).read_text()
    for arm, port in (("raw", 18100), ("rgb", 18101)):
        (runtime / f"{arm}.yml").write_text(template.replace("port: 5694", f"port: {port}"))
    write_json(
        output / "protocol.json",
        {
            "scope": "10 demonstration episodes, 5 lighting conditions per model; not P4/100",
            "parent_manifest": str(parent),
            "parent_sha256": digest(parent),
            "benchmark_seed": manifest["benchmark_seed"],
            "entries": cases,
            "selection": "parent first task/initial seed, all five domains; no outcome selection",
            "checkpoints": {
                "raw": str(
                    checkout / "outputs/n02-v12/P2/raw-n02-robotwin-r5-s1103/checkpoint_0000900.pt"
                ),
                "rgb": str(
                    checkout / "outputs/n02-v12/P2/rgb-n02-robotwin-r5-s1103/checkpoint_0000500.pt"
                ),
            },
            "checkpoint_selection": "per-arm lowest fixed-dev among saved weights; unequal steps",
            "denoise_steps": 10,
            "chunk": 50,
            "replan": 50,
            "history": 8,
            "input": "three live float32 raw_linear cameras; RGB applies saved fixed ISP only",
            "video": "SAPIEN render_rgb head camera; display only",
            "renderer": "pinned RoboTwin ray tracing with OptiX instead of incompatible OIDN",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def wait_server(port, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"N02 policy did not become ready on {port}")


def run(root, output, checkout, arm):
    protocol = json.loads((output / "protocol.json").read_text())
    directory = output / arm
    directory.mkdir(exist_ok=True)
    completed = {row["case_id"] for row in rows(directory / "episodes.jsonl")}
    port = 18100 if arm == "raw" else 18101
    wait_server(port)
    source = root / "source"
    for entry in protocol["entries"]:
        case_id = f"grab-roller-s0-{entry['lighting_domain']}"
        if case_id in completed:
            continue
        case_root = directory / case_id
        attempt = len(list(case_root.glob("attempt*"))) + 1
        case = case_root / f"attempt{attempt}"
        case.mkdir(parents=True, exist_ok=False)
        write_json(case / "entry.json", entry)
        seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
        env = os.environ.copy()
        env.update(
            PYTHONPATH=f"{checkout}/src:{source}/starVLA:{source}/third_party/RoboTwin:"
            f"{source}/third_party/RoboTwin/script:"
            f"{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
            PATH=f"{root}/envs/robotwin/bin:" + env["PATH"],
            CUDA_HOME="/usr/local/cuda-12.8",
            RAWVLA_ROBOTWIN_DENOISER="optix",
            ROBOTWIN_TEST_NUM="1",
            ROBOTWIN_EVAL_SEEDS=str(entry["dataset_seed"]),
            ROBOTWIN_FORCE_EVAL_SEED="1",
            ROBOTWIN_IMAGE_OBS_KEY="raw_linear",
            ROBOTWIN_LIGHTING_DOMAIN=entry["lighting_domain"],
            ROBOTWIN_LIGHTING_STRATEGY=entry["lighting_strategy"],
            ROBOTWIN_LIGHTING_SAMPLE_UNIT="paired",
            ROBOTWIN_LIGHTING_BENCHMARK_SEED=str(protocol["benchmark_seed"]),
            ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
            ROBOTWIN_POLICY_AUDIT_DIR=str(case),
            ROBOTWIN_POLICY_CHECKPOINT=protocol["checkpoints"][arm],
            RAWVLA_STEP_AUDIT=str(case / "actions.jsonl"),
            START_EPISODE=str(entry["dataset_episode_index"]),
            RAWVLA_FLOW_SEED=str(seed),
            RAWVLA_LANGUAGE_SEED=str(seed),
            RAWVLA_CASE_ID=case_id,
            RAWVLA_EXPECTED_ENTRY=json.dumps(entry),
            N02_CASE_VIDEO_DIR=str(case),
            VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
        )
        write_json(
            directory / "status.json",
            {"state": "running", "case_id": case_id, "completed": len(completed), "target": 5},
        )
        command = [
            str(root / "envs/robotwin/bin/python"),
            str(output / "runtime/robotwin_eval.py"),
            "--config",
            str(output / f"runtime/{arm}.yml"),
            "--policy_ckpt_path",
            protocol["checkpoints"][arm],
            "--overrides",
            "--task_name",
            entry["task_name"],
            "--task_config",
            "owac_rawvla_clean",
            "--ckpt_setting",
            f"n02-r5-demo-{arm}",
            "--seed",
            "0",
            "--policy_name",
            "owac.runtime.n02_v12_robotwin",
        ]
        rc, seconds = run_process(
            command, source / "third_party/RoboTwin", env, case / "run.log", 900
        )
        results = rows(case / "episode_results.jsonl")
        record = {
            "case_id": case_id,
            "entry": entry,
            "returncode": rc,
            "wall_seconds": seconds,
            "arm": arm,
            "directory": str(case),
            "attempt": attempt,
        }
        if rc == 0 and len(results) == 1 and (case / "lighting.json").exists():
            record.update(results[0])
            if record["eval_seed"] != entry["dataset_seed"]:
                raise ValueError("simulator substituted an initial seed")
            log = (case / "run.log").read_text(errors="replace")
            if (case / "episode0.mp4").exists():
                (case / "episode0.mp4").rename(case / "rollout.mp4")
                record["video"] = str(case / "rollout.mp4")
            record["termination_reason"] = "success" if record["success"] else "step_limit"
            record["actions_sha256"] = digest(case / "actions.jsonl")
            record["log_has_success"] = "Success!" in log
            with (directory / "episodes.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            completed.add(case_id)
        with (directory / "attempts.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        ledger = rows(directory / "episodes.jsonl")
        write_json(
            directory / "status.json",
            {
                "state": "running" if rc == 0 else "error",
                "completed": len(ledger),
                "target": 5,
                "success": sum(int(r["success"]) for r in ledger),
                "last": record,
            },
        )
        print(json.dumps(record), flush=True)
        if rc != 0 or len(results) != 1:
            raise RuntimeError(f"infrastructure error, preserve identity for repair: {case}")
    ledger = rows(directory / "episodes.jsonl")
    write_json(
        directory / "status.json",
        {
            "state": "completed",
            "completed": len(ledger),
            "target": 5,
            "success": sum(int(r["success"]) for r in ledger),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=Path("/workspace/owac/rawvla"))
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("raw", "rgb"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.output, args.checkout)
    else:
        if args.arm is None:
            parser.error("run requires --arm")
        run(args.root, args.output, args.checkout, args.arm)


if __name__ == "__main__":
    main()
