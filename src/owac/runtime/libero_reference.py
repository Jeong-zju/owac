"""Supervise an official pretrained LIBERO reference without additional training."""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from owac.evaluation.baseline_protocol import file_sha256, wilson_interval, write_json


def serve(args):
    import numpy as np
    import torch
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    official = config.get_config("pi05_libero")
    # Disable compilation for bounded startup, keeping the pretrained computation.
    official = dataclasses.replace(
        official, model=dataclasses.replace(official.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(official, args.weights)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy, host="127.0.0.1", port=args.port, metadata=policy.metadata
    )

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    finally:
        write_json(
            args.metrics,
            {
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        )


def run(args):
    if not 1 <= args.trials <= 50:
        raise ValueError("Use 1 to 50 official initial states per task")
    # Avoid accidentally querying a previously running policy on this port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    shutil.copyfile(args.protocol, args.output / "protocol.json")
    status_path = args.output / "status.json"

    def status(stage, state, **extra):
        write_json(
            status_path,
            {
                "stage": stage,
                "status": state,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                **extra,
            },
        )

    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    repo = Path(__file__).resolve().parents[3]
    status("load_official_policy", "running")
    started = time.monotonic()
    server = None
    try:
        with (args.output / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "owac.runtime.libero_reference",
                    "serve",
                    "--weights",
                    str(args.weights),
                    "--port",
                    str(args.port),
                    "--seed",
                    str(protocol["seed"]),
                    "--metrics",
                    str(args.output / "policy_metrics.json"),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        "Official policy server exited during startup; see server.log"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("Official policy server startup exceeded 240 seconds")
            status("official_closed_loop_acceptance", "running")
            environment.update(
                MUJOCO_GL="egl",
                PYOPENGL_PLATFORM="egl",
                PYTHONPATH=f"{repo / 'src'}:{args.libero_root}",
                LIBERO_CONFIG_PATH=str(args.libero_config),
            )
            with (args.output / "evaluation.log").open("w") as evaluation_log:
                subprocess.run(
                    [
                        str(args.sim_python),
                        "-u",
                        "-m",
                        "owac.evaluation.libero_reference",
                        "--upstream-script",
                        str(repo / "third_party/openpi/examples/libero/main.py"),
                        "--output",
                        str(args.output / "evaluation"),
                        "--suite",
                        protocol["task_suite"],
                        "--trials",
                        str(args.trials),
                        "--seed",
                        str(protocol["seed"]),
                        "--port",
                        str(args.port),
                    ],
                    check=True,
                    env=environment,
                    stdout=evaluation_log,
                    stderr=subprocess.STDOUT,
                )
            summary_path = args.output / "evaluation/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["wilson_95"] = wilson_interval(summary["successes"], summary["episodes"])
            summary["episodes_sha256"] = file_sha256(args.output / "evaluation/episodes.jsonl")
            write_json(summary_path, summary)
            status("complete", "complete", summary=summary)
    except BaseException as error:
        status("failed", "failed", error=str(error))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        write_json(args.output / "timing.json", {"wall_seconds": time.monotonic() - started})


def archive(args):
    args.destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -p {args.ssh_port} -o BatchMode=yes",
            "--include=*/",
            "--include=*.json",
            "--include=*.jsonl",
            "--include=*.txt",
            "--exclude=*",
            f"{args.host}:{args.run_root}/",
            str(args.destination) + "/",
        ],
        check=True,
    )
    checked_at = datetime.now(UTC).isoformat()
    status = json.loads((args.destination / "status.json").read_text())
    summary_path = args.destination / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    episodes_path = args.destination / "evaluation/episodes.jsonl"
    rows = []
    if episodes_path.exists():
        rows = [
            json.loads(line)
            for line in episodes_path.read_text().splitlines(keepends=True)
            if line.endswith("\n")
        ]
    protocol = json.loads((args.destination / "protocol.json").read_text())
    if status["status"] == "complete":
        identities = {(row["task_id"], row["episode_index"]) for row in rows}
        expected = {
            (task_id, episode_id)
            for task_id in range(protocol["suite_tasks"])
            for episode_id in range(summary["expected_episodes"] // protocol["suite_tasks"])
        }
        if identities != expected or len(rows) != len(expected):
            raise ValueError("Missing or duplicated official task/initial-state identity")
        if sum(row["success"] for row in rows) != summary["successes"]:
            raise ValueError("Summary does not match per-episode successes")
        if summary["episodes_sha256"] != file_sha256(episodes_path):
            raise ValueError("Per-episode checksum mismatch")
        if summary["infrastructure_errors"] or any(row["status"] != "completed" for row in rows):
            raise ValueError("Infrastructure failures prevent accepting a policy score")
    write_json(args.destination / "check.json", {"checked_at_utc": checked_at})
    rate = summary.get("success_rate")
    score = (
        f"{summary['successes']} / {summary['episodes']} · {100 * rate:.1f}%"
        if rate is not None
        else "未测量"
    )
    table = "".join(
        "<tr>"
        f"<td>{row['task_id'] + 1}</td><td>{html.escape(row['task_name'])}</td>"
        f"<td>{row['successes']} / {row['episodes']}</td></tr>"
        for row in summary.get("per_task", [])
    )
    error = html.escape(status.get("error", ""))
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>官方 π0.5 / LIBERO RGB 参考</title><style>
body{{font:15px system-ui;background:#111827;color:#e5e7eb;margin:0}}
main{{max-width:980px;margin:auto;padding:24px}}h1{{font-size:24px;margin:0 0 12px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{padding:18px;background:#1f2937;border-radius:8px;flex:1}}
strong{{display:block;font-size:24px;margin:8px 0}}table{{width:100%;border-collapse:collapse}}
td,th{{padding:10px;text-align:left;border-bottom:1px solid #374151}}a{{color:#93c5fd}}
.muted{{color:#9ca3af}}.error{{color:#fca5a5;overflow-wrap:anywhere}}</style>
<main><h1>官方 π0.5 / 原版 LIBERO RGB 参考</h1>
<p class="error"><b>主 benchmark 准入未通过：此模拟器不提供原生 CFA/RAW。</b>
现成策略与原生采集必须同时满足。正式 benchmark 尚未选定；
Isaac Lab Arena DROID / 官方 jointpos π0.5 是待联合验收候选。
<a href="../../docs/decisions/0008-pretrained-and-native-cfa-required.md">当前选型要求</a></p>
<p>下方保留 LIBERO-Goal 全部 10 项任务与官方 pi05_libero 的 RGB 参考结果，额外训练 0 步。</p>
<div class="cards">
<div class="card">RGB 参考闭环<strong>{score}</strong>
{summary.get("episodes", 0)} / {summary.get("expected_episodes", 10)} 已完成</div>
<div class="card">运行状态<strong>{html.escape(status["status"])}</strong>
{html.escape(status["stage"])}</div>
<div class="card">基础设施异常<strong>{summary.get("infrastructure_errors", 0)}</strong>
异常不算策略失败分数</div></div>
<p class="muted">最后成功同步：{checked_at} UTC；定时检查关闭，请手动同步后刷新。</p>
<p class="error">{error}</p><table><thead><tr><th>任务</th><th>官方任务名</th>
<th>成功 / 完成</th></tr></thead><tbody>{table}</tbody></table>
<p>初始状态按官方文件和索引使用；相机、观测、动作、稳定等待与成功判据沿用官方入口。
本次每任务采样数以实际记录为准，不能当作官方 98.0% 的完整复现。</p>
<p>RGB 参考不具备原生 CFA/RAW。<a href="https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md">官方评测说明</a></p>
</main></html>"""
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(page)
    manifest = {
        "remote_run_root": args.run_root,
        "archived_at_utc": checked_at,
        "checkpoints_downloaded": False,
        "sha256": {
            str(path.relative_to(args.destination)): file_sha256(path)
            for path in sorted(args.destination.rglob("*"))
            if path.is_file() and path.name != "archive.json"
        },
        "html_sha256": file_sha256(args.html_output),
    }
    write_json(args.destination / "archive.json", manifest)
    for name, digest in manifest["sha256"].items():
        if file_sha256(args.destination / name) != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")
    print(json.dumps({"status": status, "summary": summary, "html": str(args.html_output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    policy = commands.add_parser("serve")
    policy.add_argument("--weights", type=Path, required=True)
    policy.add_argument("--port", type=int, default=18010)
    policy.add_argument("--seed", type=int, default=7)
    policy.add_argument("--metrics", type=Path, required=True)
    evaluation = commands.add_parser("run")
    evaluation.add_argument("--weights", type=Path, required=True)
    evaluation.add_argument("--protocol", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--sim-python", type=Path, required=True)
    evaluation.add_argument("--libero-config", type=Path, required=True)
    evaluation.add_argument("--libero-root", type=Path, required=True)
    evaluation.add_argument("--trials", type=int, default=1)
    evaluation.add_argument("--port", type=int, default=18010)
    backup = commands.add_parser("archive")
    backup.add_argument("--host", default="root@212.51.142.37")
    backup.add_argument("--ssh-port", type=int, default=16406)
    backup.add_argument("--run-root", required=True)
    backup.add_argument("--destination", type=Path, required=True)
    backup.add_argument(
        "--html-output", type=Path, default=Path("outputs/official-libero-baseline/progress.html")
    )
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
    elif args.command == "archive":
        archive(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
