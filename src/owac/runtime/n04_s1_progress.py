"""N04 S1 训练进度看板的数据同步器（在训练服务器上运行）。

只读运行状态，周期性写出 ``data.json`` 供 ``n04_s1_progress.html`` 渲染：

* 运行目录的 ``metrics.json``（dev 评测曲线 + 最新分组）、日志尾部、checkpoint 列表；
* supervisor program 状态（chain / 三个 seed）与 chain 日志；
* ``nvidia-smi`` 的逐卡占用。

用法（远端）::

    <encoder_python> -m owac.runtime.n04_s1_progress \
        --run-root /workspace/owac/research/n04-v1/runs/s1 \
        --out /workspace/owac/research/n04-v1/runs/s1/progress \
        --loop --interval 30

约束：不写入训练目录的任何文件（除 ``--out``），不启动/停止任何 job，只做只读采样与
速率换算。步进速率用两次采样的差分（状态存在 ``--out/state.json``），未变化时沿用上次值。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

__all__ = ["main", "sample_once"]

CST = timezone(timedelta(hours=8))
LOG_ERROR_PATTERN = re.compile(r"nan|traceback|error", re.IGNORECASE)
PROGRAMS = (
    "owac-n04-s1-chain",
    "owac-n04-s1-train-seed17",
    "owac-n04-s1-train-seed29",
    "owac-n04-s1-train-seed43",
)
REFERENCE = {
    "untrained_depth_mae_m": 0.7428,
    "threshold_mm_range": [1.7, 3.2],
}


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime | None) -> str | None:
    return None if moment is None else moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cst(moment: datetime | None) -> str | None:
    return None if moment is None else moment.astimezone(CST).strftime("%Y-%m-%d %H:%M CST")


def _run(command: list[str], timeout: float = 10.0) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _supervisor_statuses() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in PROGRAMS:
        line = _run(["supervisorctl", "status", name])
        if not line:
            entries.append({"name": name, "status": "UNKNOWN", "uptime": None})
            continue
        parts = line.split()
        status = parts[1] if len(parts) > 1 else "UNKNOWN"
        uptime = parts[4] if len(parts) > 4 else None
        entries.append(
            {
                "name": name,
                "status": status,
                "uptime": uptime,
                "raw": line,
            }
        )
    return entries


def _gpu_state() -> list[dict[str, Any]]:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    cards = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            cards.append(
                {
                    "index": int(fields[0]),
                    "used_mib": int(fields[1]),
                    "total_mib": int(fields[2]),
                    "util": int(fields[3]),
                }
            )
        except ValueError:
            continue
    return cards


def _active_run(
    run_root: Path, statuses: list[dict[str, Any]], preferred: str | None
) -> str | None:
    running = [
        entry["name"].rsplit("-", 1)[-1] for entry in statuses if entry["status"] == "RUNNING"
    ]
    running = [name for name in running if name.startswith("train-seed")]
    if preferred and (preferred in running or not running):
        return preferred
    if running:
        return running[0]
    candidates = sorted(
        (path for path in run_root.glob("train-seed*/metrics.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return None if not candidates else candidates[0].parent.name


def _read_evals(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    evals = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        evals.append(
            {
                "step": row.get("step"),
                "depth_mae_m": row.get("depth_mae_m"),
                "depth_norm_mae": row.get("depth_norm_mae"),
                "coverage_1sigma": row.get("coverage_1sigma"),
                "coverage_2sigma": row.get("coverage_2sigma"),
                "nll": row.get("depth_nll"),
                "sigma_m": row.get("variance_std_m"),
            }
        )
    return evals


def _groups_latest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    for row in reversed(rows if isinstance(rows, list) else []):
        groups = row.get("groups") if isinstance(row, dict) else None
        if isinstance(groups, dict) and groups:
            return groups
    return {}


def _log_lines(run_dir: Path, limit: int = 8) -> tuple[list[str], list[str], int]:
    """返回（当前 run 的日志尾部, 错误行, 错误计数）。

    日志文件可能含迁移前旧机的历史段：以最后一次 "data cache warmed" 作为当前段起点。
    """
    logs = sorted((run_dir / "logs").glob("*.log")) if (run_dir / "logs").exists() else []
    if not logs:
        return [], [], 0
    try:
        text = logs[-1].read_text(errors="replace")
    except OSError:
        return [], [], 0
    marker = text.rfind("data cache warmed")
    current = text[marker:] if marker >= 0 else text
    lines = [line for line in current.splitlines() if line.strip()]
    errors = [line for line in lines if LOG_ERROR_PATTERN.search(line)]
    return lines[-limit:], errors[-5:], len(errors)


def _checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for path in run_dir.glob("checkpoint*.pt"):
        match = re.search(r"step(\d+)", path.name)
        entries.append(
            {
                "name": path.name,
                "step": int(match.group(1)) if match else None,
                "bytes": path.stat().st_size,
                "mtime_utc": _iso(datetime.fromtimestamp(path.stat().st_mtime, UTC)),
            }
        )
    entries.sort(key=lambda entry: entry["step"] or 0, reverse=True)
    return entries


def _current_step(log_lines: list[str]) -> int | None:
    for line in reversed(log_lines):
        match = re.match(r"\s*step\s+(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def sample_once(
    run_root: Path,
    out_dir: Path,
    *,
    preferred_run: str | None = None,
) -> dict[str, Any]:
    """采样一次并写出 ``data.json``；返回写出的载荷。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = _supervisor_statuses()
    run_id = _active_run(run_root, statuses, preferred_run)
    now = _now()

    payload: dict[str, Any] = {
        "generated_utc": _iso(now),
        "generated_local": _cst(now),
        "stage": "S1",
        "run_id": run_id,
        "snapshot": "n04-s1-20261014T103419Z-d40c91817264",
        "programs": statuses,
        "chain": {},
        "train": {},
        "evals": [],
        "groups_latest": {},
        "checkpoints": [],
        "gpu": _gpu_state(),
        "errors": {"count": 0, "recent": []},
        "reference": dict(REFERENCE),
    }
    if run_id is None:
        payload["errors"]["recent"] = ["找不到训练 run 目录（metrics.json 缺失）"]
        (out_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    run_dir = run_root / run_id
    log_lines, error_lines, error_count = _log_lines(run_dir)
    payload["evals"] = _read_evals(run_dir)
    payload["groups_latest"] = _groups_latest(run_dir)
    payload["checkpoints"] = _checkpoints(run_dir)
    payload["errors"] = {"count": error_count, "recent": error_lines}

    chain_log = run_root / "chain" / "logs" / "chain.supervisor.log"
    chain_lines: list[str] = []
    if chain_log.exists():
        try:
            chain_lines = [
                line for line in chain_log.read_text(errors="replace").splitlines() if line
            ]
        except OSError:
            chain_lines = []
    chain_entry = next((entry for entry in statuses if entry["name"].endswith("-chain")), None)
    payload["chain"] = {
        "status": chain_entry["status"] if chain_entry else "UNKNOWN",
        "log_tail": chain_lines[-1] if chain_lines else None,
    }

    step = _current_step(log_lines)
    state_path = out_dir / "state.json"
    previous = _load_state(state_path)
    rate = previous.get("s_per_step")
    if step is not None and previous.get("step") is not None:
        delta_step = step - int(previous["step"])
        delta_time = (now - datetime.fromisoformat(previous["utc"])).total_seconds()
        if delta_step > 0 and delta_time > 0:
            rate = delta_time / delta_step
    payload["train"] = {
        "seed": int(run_id.split("seed")[-1]) if run_id.split("seed")[-1].isdigit() else None,
        "step": step,
        "declared_steps": 100000,
        "s_per_step": None if rate is None else round(float(rate), 3),
        "eta_utc": None,
        "eta_local": None,
        "latest_log": log_lines[-6:],
    }
    if step is not None and rate:
        eta = now + timedelta(seconds=(100000 - step) * float(rate))
        payload["train"]["eta_utc"] = _iso(eta)
        payload["train"]["eta_local"] = _cst(eta)
    best = [row["depth_mae_m"] for row in payload["evals"] if row.get("depth_mae_m")]
    if best:
        payload["reference"]["best_depth_mae_m"] = min(best)

    (out_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    state_path.write_text(
        json.dumps({"utc": now.isoformat(), "step": step, "s_per_step": rate}, ensure_ascii=False)
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--run", default=None, help="固定采样的 run id（默认自动取 RUNNING 的 seed）"
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0, help="循环采样间隔（秒）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    while True:
        sample_once(args.run_root, args.out, preferred_run=args.run)
        if not args.loop:
            return 0
        time.sleep(max(5.0, float(args.interval)))


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
