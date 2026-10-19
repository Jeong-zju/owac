"""Launch only the selected r11 checkpoint's three bounded development cases."""

import argparse
import configparser
import hashlib
import json
import shlex
from pathlib import Path
from xmlrpc.client import ServerProxy

from supervisor.xmlrpc import SupervisorTransport


def main(domains):
    checkout = Path("/workspace/owac/worktrees/n02-measurement-completion")
    evidence = checkout / "research/experiments/n02_v1_2/closed_loop_r11_20261019"
    evidence.mkdir(parents=True, exist_ok=True)
    run = checkout / "outputs/n02-v12/P2/raw-task-r11-task-s1103-a1"
    selected = json.loads((run / "selection.json").read_text())["selected"]
    if not selected or selected["step"] != 850 or not selected["eligible"]:
        raise ValueError("expected the accepted offline candidate at step 850")
    checkpoint = Path(selected["checkpoint"])
    config = checkout / "research/experiments/n02_v1_2/training_r11_20261017/task_config.json"
    runner = checkout / "research/experiments/n02_v1_2/representation_r8_20261016/rollout.py"
    specs = [("Normal", 0, 18220), ("Low", 1, 18221), ("ExtremeLow", 0, 18220)]
    plan_path = evidence / "plan.json"
    plan = {
        "scope": "three fixed-seed r11 RAW development episodes; no training or P4",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "max_steps_per_case": 400,
        "timeout_seconds_per_case": 1800,
        "execution": json.loads(config.read_text())["execution"],
        "cases": [
            {
                "domain": d,
                "gpu": gpu,
                "port": port,
                "case": "r11-task850-" + d.lower() + "-a1",
                "supervisor": "owac-n02-r11-closed-" + d.lower() + "-a1",
            }
            for d, gpu, port in specs
        ],
    }
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing immutable plan differs")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.read("/etc/supervisor/supervisord.conf")
    server = ServerProxy(
        "http://localhost",
        transport=SupervisorTransport(None, None, cfg.get("supervisorctl", "serverurl")),
    )
    active = [
        p
        for p in server.supervisor.getAllProcessInfo()
        if p["statename"] in ("RUNNING", "STARTING")
    ]
    for case in plan["cases"]:
        if case["domain"] not in domains:
            continue
        output = checkout / "outputs/n02-v12/representation-r8-rollouts" / case["case"]
        if output.exists():
            raise FileExistsError(f"preserve existing attempt: {output}")
        if any(p["name"] == case["supervisor"] for p in active):
            raise ValueError("already running")
        if case["domain"] == "ExtremeLow" and any(
            p["name"] == "owac-n02-r11-closed-normal-a1" for p in active
        ):
            # Both policies use about 17 GiB on a 96-GiB card; independent processes
            # can overlap simulation waits without sharing policy or physics state.
            case = dict(case, port=18222)
            (evidence / "resource_amendment.json").write_text(
                json.dumps(
                    {
                        "case": case["case"],
                        "gpu": 0,
                        "planned_port": 18220,
                        "actual_port": 18222,
                        "reason": "overlap three independent trials within measured GPU headroom",
                        "changes_to_weights_data_seeds_physics_or_budget": False,
                    },
                    indent=2,
                )
                + "\n"
            )
        command = [
            str(checkout / "outputs/n02-full/runtime/py310-cu128/bin/python"),
            "-u",
            str(runner),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--port",
            str(case["port"]),
            "--domain",
            case["domain"],
            "--max-steps",
            "400",
            "--timeout",
            "1800",
        ]
        name = case["supervisor"]
        text = (
            f"[program:{name}]\ncommand={shlex.join(command)}\ndirectory={checkout}\n"
            f'environment=CUDA_VISIBLE_DEVICES="{case["gpu"]}",PYTHONUNBUFFERED="1",'
            f'OMP_NUM_THREADS="4",PYTHONPATH="{checkout}/src:/workspace/owac/rawvla/source/starVLA"\n'
            f"autostart=false\nautorestart=false\nstartsecs=1\nstopasgroup=true\nkillasgroup=true\n"
            f"stopsignal=TERM\nstopwaitsecs=30\nredirect_stderr=true\n"
            f"stdout_logfile={evidence}/{case['domain']}.log\nstdout_logfile_maxbytes=0\n"
        )
        path = Path("/etc/supervisor/conf.d") / (name + ".conf")
        if path.exists():
            raise FileExistsError(path)
        path.write_text(text)
    added, changed, removed = server.supervisor.reloadConfig()[0]
    wanted = {case["supervisor"] for case in plan["cases"] if case["domain"] in domains}
    if not wanted.issubset(set(added)):
        raise ValueError(f"expected only new programs: {added}, {changed}, {removed}")
    for name in sorted(wanted):
        server.supervisor.addProcessGroup(name)
        server.supervisor.startProcess(name, True)
        info = server.supervisor.getProcessInfo(name)
        print(json.dumps({k: info[k] for k in ("name", "statename", "pid")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domains", nargs="+", choices=("Normal", "Low", "ExtremeLow"))
    main(parser.parse_args().domains)
