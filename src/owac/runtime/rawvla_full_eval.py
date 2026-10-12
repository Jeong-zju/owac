"""Frozen full-manifest PI05 evaluation orchestration (no training)."""

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def paired_seed(platform, task, seed):
    text = f"owac-pi05-full-v1/{platform}/{task}/{seed}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def seed_instruction_generators(seed):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def validate_manifest(manifest, platform):
    entries = manifest["entries"]
    expected = 10000 if platform == "libero" else 3250
    if len(entries) != expected:
        raise ValueError(f"{platform}: {len(entries)} != {expected}")
    identities = []
    bases = defaultdict(set)
    for entry in entries:
        if platform == "libero":
            key = (entry["suite"], entry["task_id"], entry["init_state_index"])
        else:
            key = (entry["task_name"], entry["dataset_seed"])
        identities.append((*key, entry["lighting_domain"]))
        bases[key].add(entry["lighting_domain"])
    domains = {"ExtremeLow", "Low", "Normal", "Over", "ExtremeOver"}
    if len(set(identities)) != expected or any(values != domains for values in bases.values()):
        raise ValueError("Duplicate or incomplete paired manifest")
    return entries


def replace_once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f"Runtime instrumentation anchor changed: {old[:80]!r}")
    return code.replace(old, new)


def prepare(root, output, existing_libero=None):
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    source = root / "source"
    frozen = output / "frozen/src/owac"
    for package in ("", "runtime", "backends"):
        folder = frozen / package
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    for module in (
        "runtime/rawvla_full_eval.py",
        "runtime/rawvla_robotwin.py",
        "backends/rawvla.py",
    ):
        (frozen / module).write_bytes((root / "owac/src/owac" / module).read_bytes())
    manifests = {
        "libero": source
        / "experiments/simulation/libero/manifests"
        / "libero_manifest_50_init_states_10000_rollouts.json",
        "robotwin": source
        / "experiments/simulation/robotwin/manifests"
        / "robotwin2_test_manifest_50_seeds_3250_rollouts.json",
    }
    identities = {}
    for platform, path in manifests.items():
        manifest = json.loads(path.read_text())
        validate_manifest(manifest, platform)
        (output / f"{platform}-manifest.json").write_bytes(path.read_bytes())
        identities[platform] = {
            "source": str(path),
            "sha256": digest(path),
            "episodes": len(manifest["entries"]),
        }

    path = (
        source
        / "starVLA/examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py"
    )
    code = path.read_text()
    code = replace_once(
        code,
        "from __future__ import annotations\n",
        "from __future__ import annotations\nimport hashlib\n",
    )
    code = replace_once(
        code,
        "            client.reset()\n",
        """            client.reset()
            policy_seed_text = (f"owac-pi05-full-v1/libero/{args.task_suite}/"
                                f"{task_id}/{init_state_index}")
            policy_seed = int.from_bytes(
                hashlib.sha256(policy_seed_text.encode()).digest()[:4], "little")
            client.noise_rng = np.random.default_rng(policy_seed)
            episode_started = time.perf_counter()
            policy_steps = 0
""",
    )
    code = replace_once(
        code,
        '                    "success": bool(done),\n',
        '                    "success": bool(done),\n'
        '                    "num_steps": policy_steps,\n'
        '                    "policy_flow_seed": policy_seed,\n'
        '                    "wall_seconds": time.perf_counter() - episode_started,\n',
    )
    code = replace_once(
        code,
        "                obs, _, done, _ = env.step(action_plan.popleft().tolist())\n",
        """                applied_action = action_plan.popleft().tolist()
                if not np.isfinite(applied_action).all():
                    raise ValueError("Nonfinite policy action")
                with args.resume_path.with_suffix(".actions.jsonl").open("a") as action_stream:
                    action_stream.write(json.dumps({"task_suite": args.task_suite,
                                                    "task_id": task_id,
                                                    "episode_idx": ep_idx, "step": t,
                                                    "action": applied_action}) + "\\n")
                policy_steps += 1
                obs, _, done, _ = env.step(applied_action)
""",
    )
    compile(code, str(runtime / "libero_client.py"), "exec")
    (runtime / "libero_client.py").write_text(code)
    if existing_libero is not None:
        # Continue an already-frozen LIBERO run; select all entries, not outcomes.
        previous = json.loads((existing_libero / "protocol.json").read_text())
        if identities["libero"]["sha256"] != previous["manifests"]["libero"]["sha256"]:
            raise ValueError("Existing LIBERO manifest differs")
        (output / "libero").symlink_to(existing_libero / "libero")
        (runtime / "libero_client.py").write_bytes(
            (existing_libero / "runtime/libero_client.py").read_bytes()
        )

    path = source / "third_party/RoboTwin/script/eval_policy.py"
    code = path.read_text()
    code = "from owac.runtime.rawvla_full_eval import seed_instruction_generators\n" + code
    # Fixed manifest seeds must not enter the author's retry-the-same-seed loop.
    code = code.replace(
        "            except UnStableError as e:\n",
        "            except UnStableError as e:\n"
        "                if eval_seeds:\n                    raise\n",
    )
    code = code.replace(
        "            except Exception as e:\n",
        "            except Exception as e:\n"
        "                if eval_seeds:\n                    raise\n",
    )
    code = replace_once(
        code,
        "        if use_dataset_instructions:\n",
        '        seed_instruction_generators(int(os.environ["RAWVLA_LANGUAGE_SEED"]))\n'
        "        if use_dataset_instructions:\n",
    )
    compile(code, str(runtime / "robotwin_eval.py"), "exec")
    (runtime / "robotwin_eval.py").write_text(code)
    # The author's camera loader resolves ../task_config relative to __file__.
    (output / "task_config").symlink_to(source / "third_party/RoboTwin/task_config")
    template = source / "starVLA/examples/simBenchmarks/Robotwin/eval_files/deploy_policy.yml"
    (runtime / "robotwin_deploy.yml").write_text(
        template.read_text().replace("port: 5694", "port: 18015")
    )
    evidence_paths = [
        runtime / "libero_client.py",
        runtime / "robotwin_eval.py",
        runtime / "robotwin_deploy.yml",
        frozen / "runtime/rawvla_robotwin.py",
        frozen / "backends/rawvla.py",
        frozen / "runtime/rawvla_full_eval.py",
        source / "third_party/RoboTwin/task_config/owac_rawvla_clean.yml",
    ]
    protocol = {
        "version": "owac-pi05-full-v3",
        "total_episodes": 13250,
        "manifests": identities,
        "libero": {
            "image": "author rawvla_bench default_isp",
            "action_horizon": 10,
            "replan_steps": 5,
            "seed": 7,
            "port": 18010,
        },
        "robotwin": {
            "image": "render_rgb (SAPIEN Color, paper Table 2 Default)",
            "action_horizon": 50,
            "action_dim": 14,
            "normalization": "MEAN_STD",
            "state_padding": 0.0,
            "denoiser": "OptiX",
            "port": 18015,
            "expert_check": "language/scene eligibility only; no expert actions into policy",
            "fixed_seed": True,
            "episode_timeout_seconds": 900,
        },
        "flow_rng": "SHA256(platform, task, initial state) first 32 bits; "
        "independent per episode; paired across lighting domains",
        "language_rng": "paired episode seed; reset BOTH Python random and NumPy",
        "existing_libero_run": str(existing_libero) if existing_libero else None,
        "policy_failure": "task not solved by author max step count; included in denominator",
        "infrastructure_failure": "retain both bounded attempts for same entry; "
        "unresolved stays incomplete",
        "extra_training_steps": 0,
        "limitations": [
            "OptiX replaces incompatible SAPIEN bundled OIDN on Blackwell",
            "StarVLA image preprocessing; native LeRobot full numerical parity not established",
            "Explicit per-episode action noise improves repeatability; "
            "published RNG sequence unknown",
            "Full manifest coverage is not a claim to reproduce published numerical scores",
            "Main simulation Default PI05 only; learned ISP methods and real robots excluded",
        ],
        "code_sha256": {str(path): digest(path) for path in evidence_paths},
        "weights_sha256": {
            "libero": digest(
                source / "openpi_converted_protocol/pi05_libero_starvla/bfloat16/model.safetensors"
            ),
            "robotwin": digest(root / "checkpoints/pi05_robotwin/model.safetensors"),
            "robotwin_norm": digest(root / "checkpoints/pi05_robotwin/policy_norm_stats.json"),
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": 13250,
                "protocol_sha256": digest(output / "protocol.json"),
            }
        )
    )


def run_process(command, cwd, environment, log, timeout):
    started = time.perf_counter()
    with log.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "command": command,
                    "cwd": str(cwd),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
    return rc, time.perf_counter() - started


def summary(output, platform, expected, state="running"):
    if platform == "libero":
        completed = [
            row for path in (output / "libero").glob("*/*.episodes.jsonl") for row in rows(path)
        ]
    else:
        completed = rows(output / "robotwin" / "episodes.jsonl")
    domains = defaultdict(lambda: {"success": 0, "total": 0})
    for row in completed:
        entry = row.get("rawvla_bench_entry", row.get("entry", {}))
        domain = domains[entry["lighting_domain"]]
        domain["total"] += 1
        domain["success"] += int(row["success"])
    result = {
        "state": state,
        "expected": expected,
        "completed": len(completed),
        "success": sum(int(row["success"]) for row in completed),
        "domains": dict(domains),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output / f"{platform}-status.json", result)
    return result


def run_libero(root, output, max_jobs):
    source = root / "source"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{output}/frozen/src:{source}/starVLA:{source}/experiments/simulation/libero:{source}/starVLA/examples/simBenchmarks/LIBERO/eval_files/openpi",
        LIBERO_CONFIG_PATH=f"{source}/.local/libero",
        OPENPI_CONVERTED_ROOT=f"{source}/openpi_converted_protocol",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        TOKENIZERS_PARALLELISM="false",
    )
    jobs = 0
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        for task in range(10):
            directory = output / "libero" / suite
            directory.mkdir(parents=True, exist_ok=True)
            ledger = directory / f"task_{task}.episodes.jsonl"
            if len(rows(ledger)) == 250:
                continue
            command = [
                str(root / "envs/starvla/bin/python"),
                str(output / "runtime/libero_client.py"),
                "--model",
                "PI05",
                "--model-source",
                "openpi",
                "--host",
                "127.0.0.1",
                "--port",
                "18010",
                "--task-suite",
                suite,
                "--task-start",
                str(task),
                "--max-tasks",
                "1",
                "--image-mode",
                "rawvla_bench",
                "--rawvla-bench-representation",
                "default_isp",
                "--rawvla-bench-manifest",
                str(output / "libero-manifest.json"),
                "--resume-path",
                str(ledger),
                "--result-json",
                str(directory / f"task_{task}.json"),
                "--no-log-action-stats",
                "--debug-dump-dir",
                str(directory / f"task_{task}-input"),
            ]
            for attempt in range(1, 3):
                rc, seconds = run_process(
                    command,
                    source / "starVLA",
                    env,
                    directory / f"task_{task}-attempt{attempt}.log",
                    21600,
                )
                with (output / "libero-attempts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "suite": suite,
                                "task": task,
                                "attempt": attempt,
                                "returncode": rc,
                                "wall_seconds": seconds,
                            }
                        )
                        + "\n"
                    )
                summary(output, "libero", 10000)
                if rc == 0 and len(rows(ledger)) == 250:
                    break
            jobs += 1
            if max_jobs and jobs >= max_jobs:
                return
    result = summary(output, "libero", 10000)
    summary(output, "libero", 10000, "complete" if result["completed"] == 10000 else "incomplete")


def run_robotwin(root, output, max_jobs):
    source = root / "source"
    manifest = json.loads((output / "robotwin-manifest.json").read_text())
    entries = validate_manifest(manifest, "robotwin")
    directory = output / "robotwin"
    directory.mkdir(exist_ok=True)
    completed = {row["case_id"] for row in rows(directory / "episodes.jsonl")}
    # Begin with all 13 task/Normal/first-seed cases; preserve manifest identities.
    pilot = [
        i
        for i, entry in enumerate(entries)
        if entry["dataset_episode_index"] == 0 and entry["lighting_domain"] == "Normal"
    ]
    order = pilot + [i for i in range(len(entries)) if i not in pilot]
    jobs = 0
    for index in order:
        entry = entries[index]
        case_id = f"robotwin-{index:04d}"
        if case_id in completed:
            continue
        seed = paired_seed("robotwin", entry["task_name"], entry["dataset_seed"])
        for attempt in range(1, 3):
            case = directory / case_id / f"attempt{attempt}"
            if case.exists():
                # Never overwrite attempts after orchestration restart.
                attempt = max(int(p.name[7:]) for p in case.parent.glob("attempt*")) + 1
                case = case.parent / f"attempt{attempt}"
            case.mkdir(parents=True)
            write_json(case / "entry.json", entry)
            env = os.environ.copy()
            env.update(
                PYTHONPATH=f"{output}/frozen/src:{source}/starVLA:{source}/third_party/RoboTwin:{source}/third_party/RoboTwin/script:{source}/starVLA/examples/simBenchmarks/Robotwin/eval_files",
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
                ROBOTWIN_POLICY_AUDIT_DIR=str(case),
                RAWVLA_STEP_AUDIT=str(case / "actions.jsonl"),
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
                command, source / "third_party/RoboTwin", env, case / "run.log", 900
            )
            results = rows(case / "episode_results.jsonl")
            record = {
                "case_id": case_id,
                "attempt": attempt,
                "entry": entry,
                "returncode": rc,
                "wall_seconds": seconds,
                "policy_flow_seed": seed,
                "directory": str(case),
            }
            if rc == 0 and len(results) == 1 and (case / "lighting.json").exists():
                if results[0]["eval_seed"] != entry["dataset_seed"]:
                    raise ValueError("Seed changed during evaluation")
                record.update(results[0])
                record["actions_sha256"] = digest(case / "actions.jsonl")
                with (directory / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                completed.add(case_id)
            with (directory / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            summary(output, "robotwin", 3250)
            if case_id in completed:
                break
        jobs += 1
        if max_jobs and jobs >= max_jobs:
            return
    summary(output, "robotwin", 3250, "complete" if len(completed) == 3250 else "incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "libero", "robotwin", "status"))
    parser.add_argument("--root", default="/workspace/owac/rawvla", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-jobs", default=0, type=int)
    parser.add_argument("--existing-libero", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.output, args.existing_libero)
    elif args.command == "libero":
        verify_frozen(args.output)
        run_libero(args.root, args.output, args.max_jobs)
    elif args.command == "robotwin":
        verify_frozen(args.output)
        run_robotwin(args.root, args.output, args.max_jobs)
    else:
        for platform, expected in (("libero", 10000), ("robotwin", 3250)):
            print(json.dumps(summary(args.output, platform, expected)))


def verify_frozen(output):
    protocol = json.loads((output / "protocol.json").read_text())
    for path, expected in protocol["code_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen code changed: {path}")
    for platform, record in protocol["manifests"].items():
        if digest(output / f"{platform}-manifest.json") != record["sha256"]:
            raise ValueError(f"Frozen manifest changed: {platform}")


if __name__ == "__main__":
    main()
