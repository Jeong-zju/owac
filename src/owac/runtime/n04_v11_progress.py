#!/usr/bin/env python3
"""N04 v1.1 训练实时监控：只读采样器 + 标准库静态服务（单文件，无第三方依赖）。

给 ``n04_v11_progress.html`` 提供数据。与旧的 ``n04_s1_progress.py``（三种子队列／
S1 深度验收快照）完全独立，不读、不写旧页面涉及的任何文件。

职责
  * 采样线程：每 ``--interval`` 秒（默认 12s）从下列只读来源聚合一份轻量快照
      <run>/progress.json      训练侧自己写的心跳／进度（可选，字段见 §7 约定）
      <run>/metrics.jsonl      训练与固定开发 loss 行（逐行 JSON）
      <run>/eval_results.jsonl 100 次闭环逐条结果
      <run>/eval100.json       固定 100 条列表（用于 planned，可选）
      <run>/artifacts/         可视化图片／视频（可选 index.json 提供说明）
      <run>/logs/*.log 等      日志尾部（有界读取）
      实际进程（只读扫描 /proc + nvidia-smi）
    快照先写临时文件再 ``os.replace`` 到 ``<out>/progress.json``，同时留在内存。
  * HTTP 线程：GET /                  页面（同目录 n04_v11_progress.html）
               GET /api/progress      当前快照 JSON（gzip + ETag/304 + keep-alive）
               GET /progress.json     快照的落盘副本
               GET /file/<stage>/<run>/<relpath>  只读图／视频（限 artifacts/ 内）
               GET /healthz           健康检查
  默认绑定 127.0.0.1:8090（远端 8080 被 jupyter 占用），不自行开公网入口。

只读与安全约束
  * 绝不写任何 run 目录；唯一写入是 ``--out`` 下的 progress.json / state.json 与 stdout。
  * 不加载 checkpoint、不递归扫描大数据目录、不占用训练 GPU；nvidia-smi 只查少量字段。
  * 不启动／停止／重启任何进程；扫描 /proc 只读 cmdline，不发送任何信号。
  * 不因 loss 不降或心跳延迟判训练失败——只报告事实（进程、数据年龄、错误行）。
  * 采样或服务出错只写进快照 ``warnings`` 并打印，训练进程不受影响。

用法（远端）::

    python3 n04_v11_progress.py \
        --run-root /workspace/owac/research/n04-v1/runs/v1.1 \
        --out /workspace/owac/research/n04-v1/monitor/state \
        --host 127.0.0.1 --port 8090 --interval 12

    # 调试：只采样一次打印摘要后退出，不起服务
    python3 n04_v11_progress.py --run-root ... --out ... --once
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

__all__ = ["sample_once", "main"]

SCHEMA = "n04-v11-monitor/1"
HERE = Path(__file__).resolve().parent
HTML_NAME = "n04_v11_progress.html"
DEFAULT_RUN_ROOT = "/workspace/owac/research/n04-v1/runs/v1.1"

CST = timezone(timedelta(hours=8))

# 采样规模上限：页面只显示轻量快照，不做大数据遍历。
MAX_RUNS = 16
MAX_POINTS = 400
MAX_METRICS_BYTES = 4 * 1024 * 1024
MAX_EVAL_BYTES = 4 * 1024 * 1024
MAX_LOG_BYTES = 128 * 1024
LOG_TAIL_LINES = 20
MAX_ARTIFACTS = 4
MAX_ARTIFACT_SCAN = 400  # artifacts/ 最多扫 400 个文件，不遍历大数据目录
MAX_PROGRESS_JSON_BYTES = 512 * 1024

MEDIA_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}
ARTIFACT_SUFFIXES = set(MEDIA_SUFFIXES) | {".json", ".txt", ".csv"}

ERROR_PATTERN = re.compile(
    r"traceback|error|exception|\bnan\b|\binf\b|killed|\boom\b|failed|错误|异常|崩溃|中断", re.I
)

# loss 序列候选键：先匹配到的键为准；键名决定页面标签（动作 loss / 辅助损失）。
# 键名可用点号表示嵌套字段（WS-D 的 metrics.jsonl 把各项损失写在 ``terms`` 里）。
# ``train_loss`` 是加权合计，``terms.action`` 才是动作 loss——标签必须区分，不能混为一谈。
SERIES_CANDIDATES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("train", "train_total", ("train_loss", "loss")),
    ("action", "action", ("terms.action", "action_loss", "train_action_loss", "loss_action")),
    ("dev", "dev_action", ("dev_loss", "dev_action_loss", "val_loss", "eval_loss")),
    (
        "corr",
        "correspondence",
        ("terms.correspondence", "correspondence_loss", "corr_loss", "correspondence"),
    ),
    ("forecast", "forecast", ("terms.forecast", "forecast_loss", "pred_loss", "forecast")),
    (
        "repr",
        "representation",
        ("terms.representation", "representation_loss", "repr_loss", "vicreg_loss"),
    ),
)


def _lookup(row: dict[str, Any], key: str) -> Any:
    """按点号路径取值（``terms.action``）；任一层缺失返回 None。"""
    node: Any = row
    for part in key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


PLATFORMS = ("libero", "robotwin")


# ---------------------------------------------------------------- 时间与数值


def _now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _local(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, CST).strftime("%Y-%m-%d %H:%M:%S CST")


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if out == out and abs(out) != float("inf") else None
    if isinstance(value, str):
        try:
            out = float(value.strip())
        except ValueError:
            return None
        return out if out == out and abs(out) != float("inf") else None
    return None


def _int(value: Any) -> int | None:
    out = _num(value)
    return None if out is None else int(out)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        out = value.strip()
        return out or None
    return str(value)


def _truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "成功", "success", "succeeded", "pass"}:
            return True
        if low in {"0", "false", "no", "n", "失败", "fail", "failed", "timeout", "error"}:
            return False
    return None


# ---------------------------------------------------------------- 有界读取


def _read_text_bounded(path: Path, max_bytes: int) -> str:
    """读取文件尾部至多 max_bytes；读不到返回空串。绝不整读无界大文件。"""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read(max_bytes)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _read_json(path: Path, max_bytes: int = MAX_PROGRESS_JSON_BYTES) -> Any:
    text = _read_text_bounded(path, max_bytes)
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _iter_jsonl(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    """逐行解析 JSONL；跳过残行，不因单行损坏而放弃整个文件。"""
    text = _read_text_bounded(path, max_bytes)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _has_any(run_dir: Path) -> bool:
    for name in ("progress.json", "metrics.jsonl", "eval_results.jsonl", "artifacts", "logs"):
        if (run_dir / name).exists():
            return True
    return False


# ---------------------------------------------------------------- 序列与评估


def _downsample(points: list[list[float]], limit: int = MAX_POINTS) -> list[list[float]]:
    if len(points) <= limit:
        return points
    stride = (len(points) + limit - 1) // limit
    out = points[::stride]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def _series_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    """从 metrics.jsonl 行提取曲线：{series_id: [[step, value], ...]} 与 {series_id: 来源键}。"""
    buckets: dict[str, list[list[float]]] = {sid: [] for sid, _, _ in SERIES_CANDIDATES}
    keys_used: dict[str, str] = {}
    for row in rows:
        step = _int(row.get("step") or row.get("global_step") or row.get("update"))
        if step is None:
            continue
        for sid, _, candidates in SERIES_CANDIDATES:
            for key in candidates:
                value = _num(_lookup(row, key))
                if value is None:
                    continue
                keys_used.setdefault(sid, key)
                buckets[sid].append([step, value])
                break
    series: dict[str, list[list[float]]] = {}
    for sid, points in buckets.items():
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        series[sid] = _downsample(points)
    return series, keys_used


def _norm_platform(row: dict[str, Any]) -> str | None:
    raw = " ".join(
        str(row.get(key) or "")
        for key in ("platform", "benchmark", "suite", "env", "task", "task_name")
    ).lower()
    if "libero" in raw:
        return "libero"
    if "robotwin" in raw or "robot_win" in raw or "robot-win" in raw:
        return "robotwin"
    value = _text(row.get("platform"))
    return value.lower() if value else None


def _eval_identity(row: dict[str, Any], platform: str | None, index: int) -> tuple:
    """闭环行身份：优先用写入方给的 ``identity``，否则按平台+任务+初态拼一个。"""
    identity = _text(row.get("identity"))
    if identity:
        return ("identity", identity)
    task = _text(row.get("task") or row.get("task_name") or row.get("suite"))
    ep = row.get("episode") if row.get("episode") is not None else row.get("episode_id")
    if ep is None:
        ep = row.get("initial_state") if row.get("initial_state") is not None else None
    if ep is None:
        ep = row.get("env_seed") if row.get("env_seed") is not None else row.get("seed")
    attempt = row.get("attempt") or row.get("try") or row.get("rep")
    if task is None and ep is None:
        return ("line", index)
    return (platform or "-", task, str(ep), _int(attempt))


def _eval_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """统计闭环结果：完成／成功／基础设施错误分开计数。

    与 ``training.n04_v11.summarize_eval_results`` 同口径：同一身份只按最后一行结算；
    ``status`` 为 ``success``/``failure`` 才进完成数，``infrastructure_error`` 单独计数、
    不进分母；任务超时算失败。
    """
    unique: dict[tuple, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        platform = _norm_platform(row)
        unique[_eval_identity(row, platform, index)] = row

    stats: dict[str, dict[str, int]] = {
        platform: {"completed": 0, "successes": 0, "errors": 0} for platform in PLATFORMS
    }
    total = {"completed": 0, "successes": 0, "errors": 0}
    failures: list[dict[str, Any]] = []

    for row in unique.values():
        platform = _norm_platform(row)
        bucket = stats.setdefault(
            platform or "other", {"completed": 0, "successes": 0, "errors": 0}
        )
        status = (_text(row.get("status")) or "").lower()
        success = _truthy(row.get("success"))
        if success is None:
            success = _truthy(row.get("result"))
        infra = _truthy(row.get("infrastructure_error"))
        if infra is None:
            infra = _truthy(row.get("infra_error"))
        if infra is None and isinstance(row.get("error"), bool):
            infra = row["error"] is True
        if status in {"infrastructure_error", "infra_error", "error", "exception", "异常"}:
            infra = True
        failure_status = status in {"failure", "fail", "timeout", "失败"}
        if infra:
            bucket["errors"] += 1
            total["errors"] += 1
            continue
        if status == "success" or (not failure_status and success is True):
            bucket["completed"] += 1
            bucket["successes"] += 1
            total["completed"] += 1
            total["successes"] += 1
            continue
        bucket["completed"] += 1
        total["completed"] += 1
        reason = (
            _text(row.get("failure_reason"))
            or _text(row.get("note"))
            or _text(row.get("reason"))
            or _text(row.get("detail"))
            or ("超时" if _truthy(row.get("timeout")) else None)
            or (status or None)
            or "未达到成功判据"
        )
        if _truthy(row.get("timeout")) and "超时" not in reason:
            reason = f"超时：{reason}"
        failures.append(
            {
                "platform": platform,
                "task": _text(row.get("task") or row.get("task_name") or row.get("suite")),
                "episode": _int(row.get("episode")) if row.get("episode") is not None else None,
                "reason": reason[:200],
                "at": _text(
                    row.get("updated_at")
                    or row.get("finished_at")
                    or row.get("time")
                    or row.get("ts")
                ),
            }
        )

    return {"stats": stats, "total": total, "failures": failures[-3:], "rows": len(unique)}


def _planned_from_list_file(run_dir: Path) -> dict[str, Any] | None:
    """固定 100 条列表（可选）：libero 50 + robotwin 50。"""
    for name in ("eval100.json", "eval_list.json", "eval_planned.json", "eval_plan.json"):
        data = _read_json(run_dir / name)
        items: list[Any] | None = None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("episodes", "items", "list"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        if not items:
            continue
        per = {platform: 0 for platform in PLATFORMS}
        other = 0
        for item in items:
            platform = _norm_platform(item) if isinstance(item, dict) else None
            if platform in per:
                per[platform] += 1
            else:
                other += 1
        return {
            "planned_total": len(items),
            "planned_per": per,
            "planned_other": other,
            "file": name,
        }
    return None


def _log_tail(run_dir: Path) -> tuple[list[str], str | None, str | None]:
    """返回（最后 20 行日志, 最近一条错误行, 日志文件名）。"""
    candidates: list[Path] = []
    for pattern in ("logs/*.log", "*.log", "logs/*.txt"):
        candidates.extend(path for path in run_dir.glob(pattern) if path.is_file())
    if not candidates:
        return [], None, None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    text = _read_text_bounded(newest, MAX_LOG_BYTES)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    tail = lines[-LOG_TAIL_LINES:]
    errors = [line for line in lines if ERROR_PATTERN.search(line)]
    return tail, (errors[-1] if errors else None), newest.name


def _artifact_list(
    run_dir: Path, stage: str, run: str, limit: int = MAX_ARTIFACTS
) -> list[dict[str, Any]]:
    root = run_dir / "artifacts"
    if not root.is_dir():
        return []
    captions: dict[str, Any] = {}
    index = _read_json(root / "index.json")
    items = (
        index
        if isinstance(index, list)
        else (index.get("files") if isinstance(index, dict) else None)
    )
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                name = _text(item.get("file") or item.get("name") or item.get("path"))
                if name:
                    captions[name] = item

    files: list[Path] = []
    for pattern in ("*", "*/*", "*/*/*"):
        if len(files) >= MAX_ARTIFACT_SCAN:
            break
        try:
            files.extend(path for path in root.glob(pattern) if path.is_file())
        except OSError:
            continue
    files = [path for path in files if path.suffix.lower() in ARTIFACT_SUFFIXES]
    files = [path for path in files if path.name != "index.json"]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        rel = path.relative_to(run_dir).as_posix()
        suffix = path.suffix.lower()
        kind = "video" if MEDIA_SUFFIXES.get(suffix, "").startswith("video/") else "image"
        meta = captions.get(path.name) or captions.get(rel) or {}
        out.append(
            {
                "name": path.name,
                "rel": rel,
                "url": f"/file/{stage}/{run}/{rel}",
                "kind": kind,
                "caption": _text(meta.get("caption") or meta.get("note") or meta.get("label")),
                "step": _int(meta.get("step")),
                "mtime": _iso(path.stat().st_mtime),
                "mtime_epoch": path.stat().st_mtime,
                "bytes": path.stat().st_size,
            }
        )
    return out


# ---------------------------------------------------------------- 进程与 GPU


def _scan_processes() -> list[dict[str, Any]]:
    """只读扫描 /proc 收集进程 cmdline（不发送信号、不修改任何进程）。"""
    entries: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return entries
    try:
        pids = [child.name for child in os.scandir(proc) if child.name.isdigit()]
    except OSError:
        return entries
    for pid in pids:
        try:
            raw = (proc / pid / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmdline = " ".join(
            part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
        )
        if len(cmdline) > 4000:
            cmdline = cmdline[:4000]
        entries.append(
            {
                "pid": int(pid),
                "cmdline": cmdline,
                "argv": [
                    part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
                ],
            }
        )
    return entries


def _run_pids(
    run_dir: Path, processes: Iterable[dict[str, Any]]
) -> tuple[list[int], str | None, list[int]]:
    """返回（强匹配 pid, 匹配来源, 弱匹配 pid）。强匹配 = cmdline 含完整 run 目录路径。"""
    strong: list[int] = []
    weak: list[int] = []
    source: str | None = None
    run_path = str(run_dir)
    name = run_dir.name
    processes = list(processes)
    live = {entry["pid"]: entry for entry in processes}

    def arguments(entry):
        return entry.get("argv") or entry["cmdline"].split()

    def references(entry):
        return any(
            arg == run_path or arg in (f"--output={run_path}", f"--run-dir={run_path}")
            for arg in arguments(entry)
        )

    for entry in processes:
        if run_path and references(entry):
            strong.append(entry["pid"])
            source = source or "cmdline"
            continue
        if name and any(Path(arg).name == name for arg in arguments(entry)):
            weak.append(entry["pid"])
    for pid_file in ("pid", "train.pid", "run.pid"):
        candidate = run_dir / pid_file
        if not candidate.is_file():
            continue
        text = _read_text_bounded(candidate, 64).strip()
        if text.isdigit():
            pid = int(text)
            # A PID file survives a crash and the PID may later be reused. It is
            # evidence only while that live process still names this run.
            if pid not in live or not references(live[pid]):
                continue
            if pid not in strong:
                strong.append(pid)
            source = source or f"{pid_file}"
    return strong, source, weak


def _gpu_state() -> tuple[list[dict[str, Any]], dict[int, int], dict[str, int], str | None]:
    """返回（逐卡状态, {pid: 显存MiB}, {uuid: 卡号}, 错误说明）。"""
    cards: list[dict[str, Any]] = []
    uuid_to_index: dict[str, int] = {}
    apps: dict[int, int] = {}
    error: str | None = None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:  # nvidia-smi 不在或超时
        return [], {}, {}, f"nvidia-smi 不可用：{exc}"
    if not out.strip():
        return [], {}, {}, "nvidia-smi 无输出"
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        used, total = _num(parts[2]), _num(parts[3])
        if used is None or total is None:
            continue
        index = _int(parts[0])
        if index is None:
            continue
        uuid_to_index[parts[1]] = index
        cards.append(
            {
                "index": index,
                "uuid": parts[1],
                "used_mib": used,
                "total_mib": total,
                "used_gb": round(used / 1024, 1),
                "total_gb": round(total / 1024, 1),
                "util_percent": _num(parts[4]) if len(parts) > 4 else None,
                "power_w": _num(parts[5]) if len(parts) > 5 else None,
                "power_limit_w": _num(parts[6]) if len(parts) > 6 else None,
                "temperature_c": _num(parts[7]) if len(parts) > 7 else None,
                "pids": [],
            }
        )
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            pid, used = _int(parts[0]), _num(parts[1])
            if pid is None or used is None:
                continue
            apps[pid] = apps.get(pid, 0) + int(used)
            for card in cards:
                if card["uuid"] == parts[2]:
                    card["pids"].append(pid)
    except (OSError, subprocess.SubprocessError):
        error = "nvidia-smi 进程查询失败"
    return cards, apps, uuid_to_index, error


# ---------------------------------------------------------------- 单 run 采样


def _discover_runs(
    run_root: Path, limit: int = MAX_RUNS
) -> tuple[list[tuple[str, Path]], list[str]]:
    """浅层发现 run 目录：<root>/<stage>/<run>/，也兼容 <root>/<run>/。"""
    warnings: list[str] = []
    if not run_root.is_dir():
        warnings.append(f"run 根目录不存在：{run_root}")
        return [], warnings
    found: list[tuple[str, Path]] = []

    def is_run_dir(path: Path) -> bool:
        return path.is_dir() and not path.name.startswith((".", "_")) and _has_any(path)

    try:
        children = sorted(
            path
            for path in run_root.iterdir()
            if path.is_dir() and not path.name.startswith((".", "_"))
        )
    except OSError as exc:
        return [], [f"无法列出 run 根目录：{exc}"]
    for child in children:
        if is_run_dir(child):
            found.append((child.name, child))
            continue
        try:
            grandchildren = sorted(
                path
                for path in child.iterdir()
                if path.is_dir() and not path.name.startswith((".", "_"))
            )
        except OSError:
            continue
        for grandchild in grandchildren:
            if is_run_dir(grandchild):
                found.append((f"{child.name}/{grandchild.name}", grandchild))
    if len(found) > limit:
        warnings.append(f"run 数量 {len(found)} 超过上限 {limit}，只采样最近更新的 {limit} 个")
        found = sorted(found, key=lambda item: _run_mtime(item[1]), reverse=True)[:limit]
    return found, warnings


def _run_mtime(run_dir: Path) -> float:
    newest = 0.0
    for name in ("progress.json", "metrics.jsonl", "eval_results.jsonl"):
        path = run_dir / name
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def _parse_run_key(run_key: str) -> tuple[str, str | None, str | None, int | None]:
    """从 '<stage>/<name>' 解析 stage / platform / arm / seed。"""
    stage, _, name = run_key.rpartition("/")
    stage = stage or name
    platform = None
    arm = None
    seed = None
    low = name.lower()
    for candidate in PLATFORMS:
        if low.startswith(candidate):
            platform = candidate
            break
    match = re.search(r"-([A-D])(?=-|$)", name)
    if match:
        arm = match.group(1)
    match = re.search(r"seed(\d+)", name)
    if match:
        seed = int(match.group(1))
    return stage, platform, arm, seed


def _sample_run(
    run_key: str,
    run_dir: Path,
    processes: list[dict[str, Any]],
    gpu_apps: dict[int, int],
    sampled_at: float,
    last_train_step: int | None = None,
    last_train_budget: int | None = None,
) -> dict[str, Any]:
    stage, platform, arm, seed = _parse_run_key(run_key)
    progress = _read_json(run_dir / "progress.json")
    if not isinstance(progress, dict):
        progress = {}

    metrics_rows = _iter_jsonl(run_dir / "metrics.jsonl", MAX_METRICS_BYTES)
    series, series_keys = _series_from_rows(metrics_rows)
    used_config = _read_json(run_dir / "config.used.json") or {}
    if used_config.get("owner") == "Codex":
        schedule = used_config.get("monitor", {})
        measured = []
        for row in metrics_rows:
            point_step = _int(row.get("step"))
            value = _num(row.get("dev_loss"))
            if point_step is None or value is None:
                continue
            early = point_step <= int(schedule.get("early_until", 0))
            interval = int(
                schedule.get("eval_every_early", 25) if early else schedule.get("eval_every", 25)
            )
            if row.get("phase") == "validate" or (
                point_step > 0 and point_step % max(interval, 1) == 0
            ):
                measured.append([point_step, value])
        # Cached validation values in ordinary training rows are not new measurements.
        series["dev"] = measured

    step = _int(progress.get("step"))
    if step is None:
        for sid in ("train", "dev"):
            if series.get(sid):
                step = int(series[sid][-1][0])
                break
    train_loss = _num(progress.get("train_loss"))
    dev_loss = _num(progress.get("dev_loss"))
    if series.get("train"):
        train_loss = train_loss if train_loss is not None else series["train"][-1][1]
    if series.get("dev"):
        dev_loss = dev_loss if dev_loss is not None else series["dev"][-1][1]
    if not series.get("train") and step is not None and train_loss is not None:
        series["train"] = [[step, train_loss]]
        series_keys.setdefault("train", "progress.json")
    elif (
        series.get("train")
        and step is not None
        and train_loss is not None
        and step > series["train"][-1][0]
    ):
        # 训练侧心跳比 JSONL 更新时补一个点，保证卡片数字与曲线一致。
        series["train"].append([step, train_loss])

    eval_rows = _iter_jsonl(run_dir / "eval_results.jsonl", MAX_EVAL_BYTES)
    eval_data = (
        _eval_from_rows(eval_rows)
        if eval_rows
        else {
            "stats": {p: {"completed": 0, "successes": 0, "errors": 0} for p in PLATFORMS},
            "total": {"completed": 0, "successes": 0, "errors": 0},
            "failures": [],
            "rows": 0,
        }
    )
    planned = _planned_from_list_file(run_dir)
    progress_eval = progress.get("eval") if isinstance(progress.get("eval"), dict) else {}

    planned_total = _int(progress_eval.get("planned"))
    planned_source: str | None = "progress.json" if planned_total is not None else None
    if planned_total is None and planned:
        planned_total = planned["planned_total"]
        planned_source = planned["file"]
    if planned_total is None and eval_data["rows"]:
        # 逐条结果已产生但还没有列表文件：按本轮协议的名义预算 100 显示（页面会注明来源）。
        planned_total = 100
        planned_source = "protocol-default"

    run_platform = (_text(progress.get("platform")) or platform or "").lower()

    def _platform_eval(name: str) -> dict[str, Any]:
        stats = eval_data["stats"].get(name, {"completed": 0, "successes": 0, "errors": 0})
        entry = progress_eval.get(name)
        entry = entry if isinstance(entry, dict) else {}

        def _pick(key: str) -> int:
            value = _int(entry.get(key))
            return stats[key] if value is None else value

        planned_p = _int(entry.get("planned"))
        if planned_p is None and planned:
            planned_p = planned["planned_per"].get(name)
        if (
            planned_p is None
            and run_platform == name
            and planned_total is not None
            and planned_source != "protocol-default"
        ):
            planned_p = planned_total  # 每平台一个 run 目录：顶层 planned 就是该平台的计划数
        if planned_p is None and planned_source == "protocol-default":
            planned_p = 50
        return {
            "completed": _pick("completed"),
            "planned": planned_p,
            "successes": _pick("successes"),
            "errors": _pick("errors"),
        }

    counters_libero = _platform_eval("libero")
    counters_robotwin = _platform_eval("robotwin")
    per_platform_from_progress = any(
        isinstance(progress_eval.get(name), dict) for name in PLATFORMS
    )
    completed_total = _int(progress_eval.get("completed"))
    successes_total = _int(progress_eval.get("successes"))
    errors_total = _int(progress_eval.get("errors"))
    if completed_total is None:
        completed_total = eval_data["total"]["completed"]
    if successes_total is None:
        successes_total = eval_data["total"]["successes"]
    if errors_total is None:
        errors_total = eval_data["total"]["errors"]
    # 训练侧只给了总数、平台分项来自逐条结果时，两处口径可能对不上；如实标注，不互相覆盖。
    counts_note = None
    if not per_platform_from_progress and completed_total is not None:
        other = eval_data["stats"].get("other", {"completed": 0, "successes": 0, "errors": 0})
        rows_total = (
            counters_libero["completed"] + counters_robotwin["completed"] + other["completed"]
        )
        if rows_total != completed_total:
            counts_note = (
                f"训练侧合计 {completed_total} 与逐条结果合计 {rows_total} 不一致"
                "（明细可能尚未写全），下表按逐条结果统计"
            )

    log_tail, last_error, log_name = _log_tail(run_dir)
    if not log_tail and isinstance(progress.get("log_tail"), list):
        log_tail = [_text(line) or "" for line in progress["log_tail"]][-LOG_TAIL_LINES:]

    pids, pid_source, weak_pids = _run_pids(run_dir, processes)
    if pids:
        job_status = "running"
        job_note = f"发现 {len(pids)} 个引用该 run 目录的进程"
    elif weak_pids:
        job_status = "running"
        job_note = f"发现 {len(weak_pids)} 个命令行包含 run 名称的进程（未含完整路径）"
    elif progress or metrics_rows or eval_rows:
        job_status = "no-process"
        job_note = "当前无引用该 run 的进程；数据可能来自更早的采样"
    else:
        job_status = "empty"
        job_note = "该 run 目录尚无任何数据文件"

    updated_ts = _num(progress.get("updated_at_epoch"))
    if updated_ts is None:
        raw = _text(progress.get("updated_at"))
        if raw:
            try:
                updated_ts = datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                updated_ts = None
    if updated_ts is None:
        newest = _run_mtime(run_dir)
        updated_ts = newest or None
    data_age = None if updated_ts is None else round(max(0.0, sampled_at - updated_ts), 1)

    reported_gpu = _num(progress.get("gpu_memory_gb"))
    if reported_gpu is None and metrics_rows:
        reported_gpu = _num(metrics_rows[-1].get("gpu_memory_gb"))
    gpu_memory_gb = reported_gpu
    gpu_source = "progress.json" if reported_gpu is not None else None
    if gpu_memory_gb is None and pids:
        used = sum(gpu_apps.get(pid, 0) for pid in pids)
        gpu_memory_gb = round(used / 1024, 1) if used else None
        gpu_source = "nvidia-smi" if gpu_memory_gb is not None else None

    # 评估／开发评测阶段：progress.json 的 step/budget_steps 是评估进度（planner 口径），
    # 不能当成训练 step 显示——训练 step 取 metrics.jsonl 的最后一行。
    phase_l = (_text(progress.get("phase")) or "").lower()
    eval_phase = phase_l in {"eval", "evaluation", "dev_eval", "dev-eval"}
    metrics_step = None
    if series.get("train"):
        metrics_step = int(series["train"][-1][0])
    train_step = None if eval_phase else step
    if metrics_step is not None and (eval_phase or train_step is None):
        train_step = metrics_step
    if eval_phase:
        # 评估期间训练 step 冻结：取曲线点与上次非评估采样所见训练步的较大者
        known_steps = [value for value in (train_step, last_train_step) if value is not None]
        train_step = max(known_steps) if known_steps else None
    budget_steps = _int(progress.get("budget_steps"))
    train_budget = None if eval_phase else budget_steps
    if train_budget is None and eval_phase:
        train_budget = last_train_budget  # 评估期间沿用上次训练预算（progress.json 里是评估计划）
    remaining = None
    if train_step is not None and train_budget is not None:
        remaining = max(0, train_budget - train_step)

    assessment = {}
    if used_config.get("owner") == "Codex":
        dev_points = series.get("dev", [])
        best_pair = min(dev_points, key=lambda point: point[1]) if dev_points else None
        best_value = _num((progress.get("activity") or {}).get("best_dev_loss"))
        if best_value is None and best_pair:
            best_value = best_pair[1]
        # A bounded metrics tail may omit the historical best; don't attach a
        # newer step to the older best score in that case.
        best_step = best_pair[0] if best_pair and best_pair[1] == best_value else None
        warmup = _int(used_config.get("optim", {}).get("warmup_steps"))
        assessment = {
            "warmup_steps": warmup,
            "warmup_complete": train_step is not None
            and warmup is not None
            and train_step >= warmup,
            "dev_windows": _int(used_config.get("data", {}).get("dev_windows")),
            "best_dev_loss": best_value,
            "best_dev_step": best_step,
            "convergence": "not_assessed",
            "representation_mode": used_config.get("loss", {}).get("representation_mode"),
            "representation_coefficients": used_config.get("loss", {}).get("representation", {}),
            "loss_weights": used_config.get("loss", {}).get("weights", {}),
            "compiled_modules": [
                key.removeprefix("compile_")
                for key, enabled in used_config.get("performance", {}).items()
                if key.startswith("compile_") and enabled
            ],
        }

    return {
        "run_id": run_key,
        "stage": _text(progress.get("stage")) or stage,
        "platform": _text(progress.get("platform")) or platform,
        "arm": _text(progress.get("arm")) or arm,
        "seed": _int(progress.get("seed")) if progress.get("seed") is not None else seed,
        "phase": _text(progress.get("phase")),
        "job": {
            "status": job_status,
            "note": job_note,
            "pids": pids or weak_pids,
            "pids_source": pid_source,
        },
        "updated_at": _iso(updated_ts),
        "updated_at_local": _local(updated_ts),
        "updated_at_epoch": updated_ts,
        "data_age_s": data_age,
        "step": step,
        "budget_steps": budget_steps,
        "phase_step": step if eval_phase else None,
        "train_step": train_step,
        "train_budget_steps": train_budget,
        "remaining_steps": remaining,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "owner": progress.get("owner"),
        "gpu_index": progress.get("gpu_index"),
        "assessment": assessment,
        "activity": {
            **progress.get("activity", {}),
            "validation_measurements": len(series.get("dev", [])),
            "last_validation_step": series["dev"][-1][0] if series.get("dev") else None,
        },
        "step_seconds_reported": _num(progress.get("step_seconds")),
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_memory_gb_source": gpu_source,
        "eval": {
            "completed": completed_total,
            "planned": planned_total,
            "successes": successes_total,
            "errors": errors_total,
            "libero": counters_libero,
            "robotwin": counters_robotwin,
            "rows": eval_data["rows"],
            "planned_source": planned_source,
            "counts_note": counts_note,
        },
        "recent_failures": eval_data["failures"],
        "series": series,
        "series_keys": series_keys,
        "artifacts": _artifact_list(run_dir, stage, run_dir.name),
        "log": {"tail": log_tail, "last_error": last_error, "source": log_name},
        "log_tail": log_tail,
        "sources": {
            "progress_json": bool(progress),
            "metrics_rows": len(metrics_rows),
            "eval_rows": eval_data["rows"],
        },
    }


# ---------------------------------------------------------------- 采样入口


def _load_state(out_dir: Path) -> dict[str, Any]:
    data = _read_json(out_dir / "state.json")
    return data if isinstance(data, dict) else {}


def _speed_info(
    run: dict[str, Any], previous: dict[str, Any], sampled_at: float
) -> tuple[float | None, str | None]:
    """Use completed update timings; Codex resumes must never infer speed from restored steps."""
    reported = run.get("step_seconds_reported")
    if reported and reported > 0:
        return float(reported), "reported"
    if run.get("owner") == "Codex":
        # Loading a checkpoint restores the step counter at once. That jump is
        # not newly executed training and cannot be divided by sampling time.
        return None, None
    history = previous.get("history") if isinstance(previous.get("history"), dict) else {}
    steps = history.get(run["run_id"])
    if isinstance(steps, list) and steps:
        window = [item for item in steps if isinstance(item, dict)]
        if window:
            first = window[0]
            current = run.get("train_step") if run.get("train_step") is not None else run["step"]
            if current is not None and isinstance(first.get("step"), (int, float)):
                delta_step = current - float(first["step"])
                delta_time = sampled_at - float(first.get("t") or 0)
                if delta_step > 0 and 5 <= delta_time <= 3600:
                    return delta_time / delta_step, "sampled"
    reported = run.get("step_seconds_reported")
    if reported and reported > 0:
        return float(reported), "reported"
    return None, None


def _update_history(
    payload: dict[str, Any], previous: dict[str, Any], sampled_at: float
) -> dict[str, Any]:
    old = previous.get("history") if isinstance(previous.get("history"), dict) else {}
    new_history: dict[str, Any] = {}
    for run in payload.get("runs", []):
        entry = (
            list(old.get(run["run_id"]) or []) if isinstance(old.get(run["run_id"]), list) else []
        )
        track_step = run.get("train_step") if run.get("train_step") is not None else run["step"]
        if track_step is not None:
            entry.append({"t": sampled_at, "step": track_step})
        entry = [item for item in entry if isinstance(item, dict)][-8:]
        new_history[run["run_id"]] = entry
    return new_history


def _attach_model_eval(payload: dict[str, Any]) -> None:
    """给每个 run 附上同一模型（stage+arm+seed）跨平台的 100 次闭环合计。

    每平台一个 run 目录（LIBERO 50 + RoboTwin 50），所以单个 run 只能看到自己那 50。
    缺失平台的计划数按本轮协议固定为 50，并标注来源，不冒充已读到的列表。
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for run in payload.get("runs", []):
        groups.setdefault((run.get("stage"), run.get("arm"), run.get("seed")), []).append(run)
    for run in payload.get("runs", []):
        group = groups[(run.get("stage"), run.get("arm"), run.get("seed"))]
        per_platform: dict[str, dict[str, Any]] = {}
        planned_from_runs = True
        for name in PLATFORMS:
            candidates = [
                member for member in group if (_text(member.get("platform")) or "").lower() == name
            ]
            if candidates:
                best = max(
                    candidates,
                    key=lambda member: (member.get("eval") or {}).get("completed") or 0,
                )
                entry = dict(best.get("eval") or {})
                sub = entry.get(name) if isinstance(entry.get(name), dict) else {}
                per_platform[name] = {
                    "completed": sub.get("completed", entry.get("completed")),
                    "planned": entry.get("planned"),
                    "successes": sub.get("successes", entry.get("successes")),
                    "errors": sub.get("errors", entry.get("errors")),
                    "source_run": best.get("run_id"),
                    "planned_source": entry.get("planned_source"),
                }
                if per_platform[name]["planned"] is None:
                    per_platform[name]["planned"] = 50
                    per_platform[name]["planned_source"] = "protocol-default"
                    planned_from_runs = False
            else:
                per_platform[name] = {
                    "completed": None,
                    "planned": 50,
                    "successes": None,
                    "errors": None,
                    "source_run": None,
                    "planned_source": "protocol-default",
                }
                planned_from_runs = False

        def _sum(key: str, entries: dict[str, dict[str, Any]] = per_platform) -> int | None:
            values = [entry.get(key) for entry in entries.values()]
            known = [value for value in values if isinstance(value, int)]
            return sum(known) if known else None

        run["model_eval"] = {
            "stage": run.get("stage"),
            "arm": run.get("arm"),
            "seed": run.get("seed"),
            "completed": _sum("completed"),
            "planned": _sum("planned"),
            "successes": _sum("successes"),
            "errors": _sum("errors"),
            "libero": per_platform.get("libero"),
            "robotwin": per_platform.get("robotwin"),
            "planned_source": "runs" if planned_from_runs else "protocol-default",
            "run_ids": sorted(str(member.get("run_id")) for member in group),
        }


def sample_once(
    run_root: Path, out_dir: Path, previous: dict[str, Any] | None = None
) -> dict[str, Any]:
    """采样一次：读只读来源 → 组装快照 → 原子写入 <out>/progress.json。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = previous or {}
    sampled_at = _now()
    warnings: list[str] = []

    runs_found, discovery_warnings = _discover_runs(run_root)
    warnings.extend(discovery_warnings)

    processes: list[dict[str, Any]] = []
    try:
        processes = _scan_processes()
    except Exception as exc:  # noqa: BLE001 - 采样绝不因单点失败中止
        warnings.append(f"进程扫描失败：{exc}")
    cards, gpu_apps, uuid_to_index, gpu_error = _gpu_state()
    if gpu_error:
        warnings.append(gpu_error)

    previous_train_steps = (
        previous.get("train_steps") if isinstance(previous.get("train_steps"), dict) else {}
    )
    previous_train_budgets = (
        previous.get("train_budgets") if isinstance(previous.get("train_budgets"), dict) else {}
    )
    runs: list[dict[str, Any]] = []
    for run_key, run_dir in runs_found:
        try:
            runs.append(
                _sample_run(
                    run_key,
                    run_dir,
                    processes,
                    gpu_apps,
                    sampled_at,
                    last_train_step=_int(previous_train_steps.get(run_key)),
                    last_train_budget=_int(previous_train_budgets.get(run_key)),
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{run_key} 采样失败：{type(exc).__name__}: {exc}")

    # GPU 逐卡：附上占用进程对应的 run（含弱匹配，便于"双卡在跑什么"一眼可见）。
    pid_to_run: dict[int, str] = {}
    for run in runs:
        for pid in run["job"]["pids"]:
            pid_to_run[pid] = run["run_id"]
    for card in cards:
        card_runs = sorted(
            {
                pid_to_run.get(pid, f"pid {pid}")
                for pid, used in gpu_apps.items()
                if pid in card.get("pids", []) and used
            }
        )
        card["runs"] = card_runs
        card.pop("uuid", None)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "sampled_at": _iso(sampled_at),
        "sampled_at_local": _local(sampled_at),
        "sampled_at_epoch": sampled_at,
        "run_root": str(run_root),
        "interval_s": previous.get("interval_s"),
        "runs": sorted(runs, key=lambda run: (-(run["updated_at_epoch"] or 0), run["run_id"])),
        "gpu": cards,
        "warnings": warnings,
    }
    repair_spec = _read_json(out_dir.parent / "repair-runs.json") or {}
    if repair_spec.get("active"):
        repair_rows = []
        allowed = (run_root.parent / "codex-diagnosis").resolve()
        for spec in repair_spec.get("runs", []):
            path = Path(str(spec.get("run_dir", ""))).resolve()
            if not path.is_relative_to(allowed):
                continue
            status = _read_json(path / "progress.json") or {}
            repair_rows.append(
                {
                    "platform": spec.get("platform"),
                    "step": _int(status.get("step")),
                    "budget": _int(status.get("budget_steps")),
                    "phase": status.get("phase"),
                    "updated_at": status.get("updated_at"),
                    "reload_verified": (path / "reload-verification.json").is_file(),
                }
            )
        payload["repair"] = {"runs": repair_rows, "formal_progress": False}

    # 步速／ETA：用上次采样做差分；不足两次有效采样则退回上报值。
    for run in payload["runs"]:
        speed, source = _speed_info(run, previous, sampled_at)
        run["step_seconds"] = None if speed is None else round(speed, 3)
        run["speed_source"] = source
        eta = None
        eta_at = None
        if speed and run["remaining_steps"]:
            eta = run["remaining_steps"] * speed
            eta_at = sampled_at + eta
        run["eta_seconds"] = None if eta is None else round(eta, 0)
        run["eta_at"] = _iso(eta_at)
        run["eta_at_local"] = _local(eta_at)

    _attach_model_eval(payload)
    payload["history"] = _update_history(payload, previous, sampled_at)
    # 记忆每个 run 最近一次非评估阶段看到的训练 step（评估阶段沿用，避免显示评估计数）。
    train_steps = dict(previous_train_steps)
    train_budgets = dict(previous_train_budgets)
    for run in payload["runs"]:
        if run.get("train_step") is not None:
            train_steps[run["run_id"]] = run["train_step"]
        if run.get("train_budget_steps") is not None:
            train_budgets[run["run_id"]] = run["train_budget_steps"]
    payload["train_steps"] = train_steps
    payload["train_budgets"] = train_budgets

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp = out_dir / ".progress.json.tmp"
    tmp.write_bytes(body)
    os.replace(tmp, out_dir / "progress.json")
    state_tmp = out_dir / ".state.json.tmp"
    state_tmp.write_text(
        json.dumps(
            {
                "sampled_at": sampled_at,
                "history": payload["history"],
                "train_steps": payload["train_steps"],
                "train_budgets": payload["train_budgets"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(state_tmp, out_dir / "state.json")
    return payload


# ---------------------------------------------------------------- HTTP 服务


class Snapshot:
    """线程安全保存最近一次成功快照。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: dict[str, Any] | None = None
        self._error: str | None = None

    def set(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._payload = payload
            self._error = None

    def set_error(self, error: str) -> None:
        with self._lock:
            self._error = error

    def get(self) -> tuple[dict[str, Any] | None, str | None]:
        with self._lock:
            return self._payload, self._error


def _is_within(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([str(child), str(parent)]) == str(parent)
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive：隧道下轮询更省往返
    server_version = "n04v11-progress"

    # -- 工具 --------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # 默认逐请求刷屏，改为静默失败请求
        if args and str(args[1] if len(args) > 1 else "") not in {"200", "206", "304"}:
            sys.stderr.write(f"HTTP {self.address_string()} {fmt % args}\n")

    def _headers(
        self, code: int, body: bytes, ctype: str, etag: str | None = None, cache: str = "no-cache"
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()

    def _send(
        self, code: int, body: bytes, ctype: str, etag: str | None = None, cache: str = "no-cache"
    ) -> None:
        accept = self.headers.get("Accept-Encoding") or ""
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache)
            self.end_headers()
            return
        if "gzip" in accept and len(body) > 1024:
            compressed = gzip.compress(body, 6)
            if len(compressed) < len(body):
                body = compressed
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache)
                self.send_header("X-Content-Type-Options", "nosniff")
                if etag:
                    self.send_header("ETag", etag)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
                return
        self._headers(code, body, ctype, etag, cache)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode()
        etag = f'"{hashlib.sha1(body).hexdigest()[:16]}"'
        self._send(code, body, "application/json; charset=utf-8", etag=etag)

    # -- 路由 --------------------------------------------------------------

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/api/progress":
            self._serve_progress()
        elif path == "/progress.json":
            self._serve_progress_file()
        elif path == "/healthz":
            payload, error = self.server.snapshot.get()  # type: ignore[attr-defined]
            self._json(
                {
                    "ok": payload is not None,
                    "sampled_at": (payload or {}).get("sampled_at"),
                    "runs": len((payload or {}).get("runs") or []),
                    "error": error,
                }
            )
        elif path.startswith("/file/"):
            self._serve_file(path[len("/file/") :])
        else:
            self._json({"error": "not found", "path": path}, code=404)

    def _serve_html(self) -> None:
        html = self.server.html_path  # type: ignore[attr-defined]
        try:
            body = html.read_bytes()
        except OSError as exc:
            self._send(500, f"页面文件不可读：{exc}".encode(), "text/plain; charset=utf-8")
            return
        self._send(200, body, "text/html; charset=utf-8")

    def _serve_progress(self) -> None:
        payload, error = self.server.snapshot.get()  # type: ignore[attr-defined]
        if payload is None:
            self._json(
                {"schema": SCHEMA, "runs": [], "warnings": [error or "采样器尚未产出快照"]},
                code=503,
            )
            return
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        etag = f'"{hashlib.sha1(body).hexdigest()[:16]}"'
        self._send(200, body, "application/json; charset=utf-8", etag=etag)

    def _serve_progress_file(self) -> None:
        path = self.server.out_dir / "progress.json"  # type: ignore[attr-defined]
        try:
            body = path.read_bytes()
        except OSError:
            self._json({"error": "progress.json 尚未生成"}, code=404)
            return
        etag = f'"{hashlib.sha1(body).hexdigest()[:16]}"'
        self._send(200, body, "application/json; charset=utf-8", etag=etag)

    def _serve_file(self, rel: str) -> None:
        parts = [part for part in rel.split("/") if part]
        if len(parts) < 3:
            self._json({"error": "路径需要 <stage>/<run>/<relpath>"}, code=404)
            return
        stage, run, rest = parts[0], parts[1], "/".join(parts[2:])
        if any(part in {".", ".."} for part in (stage, run)) or not rest:
            self._json({"error": "非法路径"}, code=404)
            return
        run_root = self.server.run_root  # type: ignore[attr-defined]
        run_dir = (run_root / stage / run).resolve()
        target = (run_dir / rest).resolve()
        artifacts_root = (run_dir / "artifacts").resolve()
        if not _is_within(target, artifacts_root) or not _is_within(target, run_dir):
            self._json({"error": "只读服务仅提供当前 run 的 artifacts/ 文件"}, code=403)
            return
        if not target.is_file():
            self._json({"error": "文件不存在", "path": rest}, code=404)
            return
        ctype = MEDIA_SUFFIXES.get(target.suffix.lower(), "application/octet-stream")
        try:
            size = target.stat().st_size
        except OSError:
            self._json({"error": "文件不可读"}, code=404)
            return
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match:
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else size - 1
                elif match.group(2):
                    start = max(0, size - int(match.group(2)))
                if 0 <= start <= end < size:
                    status = 206
                else:
                    start, end, status = 0, size - 1, 200
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=60")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with target.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except OSError:
            return


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[Handler],
        *,
        run_root: Path,
        out_dir: Path,
        html_path: Path,
        snapshot: Snapshot,
    ) -> None:
        super().__init__(address, handler)
        self.run_root = run_root
        self.out_dir = out_dir
        self.html_path = html_path
        self.snapshot = snapshot


# ---------------------------------------------------------------- 主流程


def _sampler_loop(
    stop: threading.Event,
    run_root: Path,
    out_dir: Path,
    interval: float,
    snapshot: Snapshot,
    interval_note: float,
) -> None:
    previous = _load_state(out_dir)
    while not stop.is_set():
        started = _now()
        try:
            payload = sample_once(run_root, out_dir, previous)
            payload["interval_s"] = interval_note
            previous = {"sampled_at": started, "history": payload.get("history")}
            snapshot.set(payload)
            runs = payload.get("runs") or []
            active = sum(1 for run in runs if (run.get("job") or {}).get("status") == "running")
            print(
                f"[{_iso(started)}] 采样完成 runs={len(runs)} 运行中={active}"
                f" 警告={len(payload.get('warnings') or [])}",
                flush=True,
            )
            for warning in payload.get("warnings") or []:
                print(f"[{_iso(started)}] 警告：{warning}", flush=True)
        except Exception as exc:  # noqa: BLE001 - 采样失败绝不影响下一次
            snapshot.set_error(f"{type(exc).__name__}: {exc}")
            print(f"[{_iso(started)}] 采样失败：{type(exc).__name__}: {exc}", flush=True)
        stop.wait(max(5.0, interval))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N04 v1.1 训练监控：只读采样器 + 静态服务")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(os.environ.get("N04_V11_RUN_ROOT") or DEFAULT_RUN_ROOT),
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="快照输出目录（默认 <run-root>/../../monitor/state）"
    )
    parser.add_argument("--html", type=Path, default=HERE / HTML_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--interval", type=float, default=12.0, help="采样间隔秒（建议 10-15）")
    parser.add_argument("--no-serve", action="store_true", help="只采样不启 HTTP 服务")
    parser.add_argument("--once", action="store_true", help="采样一次后退出（调试）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root: Path = args.run_root
    out_dir: Path = args.out or (run_root.parent.parent / "monitor" / "state")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.once:
        payload = sample_once(run_root, out_dir, _load_state(out_dir))
        summary = {
            "sampled_at": payload["sampled_at"],
            "run_root": payload["run_root"],
            "runs": [
                {
                    "run_id": run["run_id"],
                    "job": run["job"]["status"],
                    "step": run["step"],
                    "budget_steps": run["budget_steps"],
                    "train_loss": run["train_loss"],
                    "eval": run["eval"],
                    "warnings": payload["warnings"],
                }
                for run in payload["runs"]
            ],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    snapshot = Snapshot()
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sampler_loop,
        args=(stop, run_root, out_dir, args.interval, snapshot, args.interval),
        name="sampler",
        daemon=True,
    )
    sampler.start()

    if args.no_serve:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            stop.set()
        return 0

    server = Server(
        (args.host, args.port),
        Handler,
        run_root=run_root,
        out_dir=out_dir,
        html_path=args.html,
        snapshot=snapshot,
    )
    print(
        f"N04 v1.1 监控服务：http://{args.host}:{args.port}/  run-root={run_root}"
        f"  out={out_dir}  采样间隔={args.interval}s",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
