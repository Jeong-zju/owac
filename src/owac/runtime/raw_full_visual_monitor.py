# ruff: noqa: E501
"""Read-only localhost monitor for the two-GPU full-visual stage.

The old 8878 monitor and its historical page are never read or modified.  This
server binds to loopback only and exposes three fixed routes: ``/``, ``/api/status``
and ``/healthz``.  It never serves static files, checkpoint contents, directory
listings, or arbitrary run-root paths.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

EXPECTED_TRAINABLE = 428_579_952
TARGET_STEPS = 3116
TARGET_ANCHORS = 49_856
LOOPBACKS = {"127.0.0.1", "localhost", "::1"}
MAX_CURVE_POINTS = 240
MAX_JSONL_LINES = 4000
STALE_HEARTBEAT_SECONDS = 60.0
EXPECTED_RANKS = 2
SHARED_RAW_PROTOCOL = "owac.shared-rawvla.task6-normal.8-2-2.v1"


def _now() -> datetime:
    return datetime.now(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _tail_jsonl(path: Path, limit: int = MAX_JSONL_LINES) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return list(rows)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _age_seconds(value: Any, now: datetime | None = None) -> float | None:
    parsed = _parse_time(value)
    if parsed is None:
        return None
    return max(0.0, ((now or _now()) - parsed).total_seconds())


def _short_sha(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        return None
    return value[:12]


def _number(value: Any, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.{digits}g}"


def _bytes_label(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric) or numeric < 0:
        return "—"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = 0
    while numeric >= 1024.0 and index < len(units) - 1:
        numeric /= 1024.0
        index += 1
    return f"{numeric:.2f} {units[index]}"


def _seconds_label(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    hours = numeric / 3600.0
    return f"{numeric:.1f}s / {hours:.2f}h"


def _count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _sanitize_gpu(rows: Any) -> list[dict[str, Any]]:
    result = []
    if not isinstance(rows, Sequence):
        return result
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        result.append(
            {
                "rank": item.get("rank"),
                "local_rank": item.get("local_rank"),
                "allocated": _bytes_label(item.get("allocated_bytes")),
                "reserved": _bytes_label(item.get("reserved_bytes")),
                "max_allocated": _bytes_label(item.get("max_allocated_bytes")),
                "max_reserved": _bytes_label(item.get("max_reserved_bytes")),
            }
        )
    return result


def _proc_stat(pid: Any) -> dict[str, Any] | None:
    """Read actual /proc/<pid>/stat fields without trusting status claims."""
    try:
        numeric = int(pid)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    try:
        text = Path(f"/proc/{numeric}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        return {
            "pid": numeric,
            "state": fields[0],
            "pgid": int(fields[2]),
            "start_ticks": int(fields[19]),
        }
    except (ValueError, IndexError):
        return None


def _process_liveness(pid: Any, expected_start: Any, expected_pgid: Any) -> dict[str, Any]:
    stat = _proc_stat(pid)
    if stat is None:
        return {
            "classification": "missing",
            "alive": False,
            "actual_state": None,
            "actual_start_ticks": None,
            "actual_pgid": None,
        }
    expected_start_int = int(expected_start) if isinstance(expected_start, int) else None
    expected_pgid_int = int(expected_pgid) if isinstance(expected_pgid, int) else None
    if expected_start_int is None:
        classification = "unverified_startticks"
    elif stat["start_ticks"] != expected_start_int:
        classification = "reused"
    elif stat["state"] == "Z":
        classification = "zombie"
    elif expected_pgid_int is not None and stat["pgid"] != expected_pgid_int:
        classification = "pgid_changed"
    else:
        classification = "alive"
    return {
        "classification": classification,
        "alive": classification == "alive",
        "actual_state": stat["state"],
        "actual_start_ticks": stat["start_ticks"],
        "actual_pgid": stat["pgid"],
    }


def _process_rows(
    rows: Any,
    *,
    claimed_status: str,
    last_update_age: float | None,
) -> tuple[list[dict[str, Any]], str, Any]:
    items = (
        [item for item in rows if isinstance(item, Mapping)] if isinstance(rows, Sequence) else []
    )
    if not items:
        items = [{"rank": rank} for rank in range(EXPECTED_RANKS)]
    health: list[dict[str, Any]] = []
    for rank, item in enumerate(items):
        pid = item.get("pid")
        expected_start = item.get("pid_start_ticks")
        expected_pgid = item.get("pgid")
        live = _process_liveness(pid, expected_start, expected_pgid)
        health.append(
            {
                "rank": item.get("rank", rank),
                "pid": pid,
                "pid_start_ticks": expected_start,
                "pgid": expected_pgid,
                "actual_state": live["actual_state"],
                "actual_start_ticks": live["actual_start_ticks"],
                "actual_pgid": live["actual_pgid"],
                "classification": live["classification"],
                "alive": live["alive"],
            }
        )
    all_alive = bool(health) and all(item["alive"] for item in health)
    terminal = claimed_status in {"stopped", "completed", "error"}
    running_claim = claimed_status in {"running", "preparing"}
    stale = last_update_age is None or last_update_age > STALE_HEARTBEAT_SECONDS
    if running_claim:
        effective = (
            "running" if all_alive and not stale else ("stale" if all_alive else "stale_dead")
        )
    elif terminal:
        effective = (
            "stale_status_alive_process"
            if any(item["alive"] for item in health)
            else claimed_status
        )
    else:
        effective = claimed_status
    if any(item["alive"] for item in health):
        native_exit = None
    elif terminal:
        native_exit = None  # caller supplies status native_exit only when available
    else:
        native_exit = None
    return health, effective, native_exit


def _sanitize_validation(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        return None
    return {
        "step": record.get("step"),
        "valid_tokens": record.get("valid_tokens"),
        "feature_mse": record.get("feature_mse"),
        "feature_cosine": record.get("feature_cosine"),
        "feature_total": record.get("feature_total"),
        "relative_feature_mse": record.get("relative_feature_mse"),
        "sources": record.get("sources"),
        "windows": record.get("windows"),
        "episode_resets": record.get("episode_resets"),
        "is_best": record.get("is_best"),
        "checkpoint_sha256_12": _short_sha(record.get("checkpoint_sha256")),
        "timestamp": record.get("timestamp"),
    }


def _sanitize_selection(selection: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(selection, Mapping) or not selection:
        return None
    best = selection.get("best") if isinstance(selection.get("best"), Mapping) else selection
    return {
        "best_step": best.get("best_step", best.get("step")),
        "metric": selection.get("selection_metric")
        or selection.get("metric")
        or "validation_token_total",
        "value": best.get("best_value", best.get("value")),
        "checkpoint_sha256_12": _short_sha(
            best.get("checkpoint_sha256") or selection.get("checkpoint_sha256")
        ),
        "retained_model_only": bool(
            best.get("retained_model_only", selection.get("retained_model_only", False))
        ),
        "updated_at": selection.get("updated_at"),
    }


def _sanitize_last_checkpoint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "step": value.get("step"),
        "kind": value.get("kind"),
        "sha256_12": _short_sha(value.get("sha256")),
    }


def snapshot(run_root: str | Path) -> dict[str, Any]:
    """Build one bounded, path-free snapshot for the fixed read-only API."""
    root = Path(run_root)
    status = _read_json(root / "run-status.json")
    meta = _read_json(root / "run-meta.json")
    heartbeat = _read_json(root / "heartbeat.json")
    selection = _read_json(root / "selection.json")
    latest = _read_json(root / "latest.json")
    updates = _tail_jsonl(root / "updates.jsonl")
    validation_rows = _tail_jsonl(root / "validation.jsonl")
    latest_update = updates[-1] if updates else {}
    latest_validation = validation_rows[-1] if validation_rows else None
    now = _now()
    last_update_at = latest_update.get("timestamp") or heartbeat.get("updated_at")
    age = _age_seconds(last_update_at, now)
    state = str(status.get("status") or "waiting")
    gpu_rows = latest_update.get("gpu_by_rank") or status.get("gpu_by_rank") or []
    raw_processes = status.get("rank_processes") or latest_update.get("rank_processes") or []
    processes, effective_status, _process_native_exit = _process_rows(
        raw_processes,
        claimed_status=state,
        last_update_age=age,
    )
    train_curve = []
    for item in updates[-MAX_CURVE_POINTS:]:
        if not isinstance(item, Mapping):
            continue
        train_curve.append(
            {
                "step": item.get("new_global_updates"),
                "mse": item.get("loss_components", {}).get("mse")
                if isinstance(item.get("loss_components"), Mapping)
                else None,
                "cosine": item.get("loss_components", {}).get("cosine")
                if isinstance(item.get("loss_components"), Mapping)
                else None,
                "total": item.get("loss_components", {}).get("total")
                if isinstance(item.get("loss_components"), Mapping)
                else None,
                "lr": item.get("lr_used"),
                "wall": item.get("remote_gpu_wall_used_seconds"),
            }
        )
    split = meta.get("split") if isinstance(meta.get("split"), Mapping) else {}
    source_count = sum(len(value) for value in split.values()) if split else None
    split_counts = {
        "train": len(split.get("train", [])) if split else None,
        "validation": len(split.get("validation", [])) if split else None,
        "test": len(split.get("test", [])) if split else None,
    }
    data_contract = (
        meta.get("data_contract") if isinstance(meta.get("data_contract"), Mapping) else {}
    )
    protocol = data_contract.get("protocol")
    if protocol == SHARED_RAW_PROTOCOL:
        public_shared_used: bool | None = True
    elif protocol:
        public_shared_used = False
    else:
        public_shared_used = None
    result = {
        "schema": "owac.raw-full-visual.monitor.v1",
        "generated_at": now.isoformat(timespec="seconds"),
        "run_name": root.name,
        "status": effective_status,
        "status_claimed": state,
        "phase": status.get("phase") or heartbeat.get("phase"),
        "updated_at": status.get("updated_at") or heartbeat.get("updated_at"),
        "last_update_at": last_update_at,
        "last_update_age_seconds": age,
        "new_global_updates": int(status.get("new_global_updates") or 0),
        "target_new_global_updates": int(status.get("target_cumulative_updates") or TARGET_STEPS),
        "new_anchor_presentations": int(status.get("new_anchor_presentations") or 0),
        "target_new_anchor_presentations": int(
            status.get("target_anchor_presentations") or TARGET_ANCHORS
        ),
        "budget_update_equivalents": status.get("budget_update_equivalents"),
        "actual_scope": {
            "label": "学生整视觉头（stem、gate、位置编码、27 层 SigLIP、post-norm、原投影）",
            "trainable_parameters": int(status.get("full_visual_parameters") or EXPECTED_TRAINABLE),
            "expected_parameters": EXPECTED_TRAINABLE,
            "teacher_frozen": True,
            "language_action_backend_frozen": True,
            "teacher_backend_optimizer_members": False,
        },
        "source": {
            "source_count": source_count,
            "split": split_counts,
            "public_shared_dataset_used": public_shared_used,
            "test_used_for_selection": False,
            "note": "split 由 run-meta 记录；未提供时待核验，不默认50；共享RAW只读直接读取，旧50来源/缓存不用于本阶段",
        },
        "validation": _sanitize_validation(latest_validation),
        "selection": _sanitize_selection(selection),
        "last_checkpoint": _sanitize_last_checkpoint(status.get("last_checkpoint")),
        "latest_pointer": {
            "step": latest.get("new_global_updates"),
            "kind": latest.get("kind"),
            "sha256_12": _short_sha(latest.get("checkpoint_sha256")),
        },
        "gpu": _sanitize_gpu(gpu_rows),
        "processes": processes,
        "native_exit": (
            None
            if any(item["alive"] for item in processes)
            else (status.get("native_exit") if state in {"stopped", "completed", "error"} else None)
        ),
        "process_health": {
            "all_alive": bool(processes) and all(item["alive"] for item in processes),
            "any_alive": any(item["alive"] for item in processes),
            "stale_heartbeat": age is None or age > STALE_HEARTBEAT_SECONDS,
            "stale_after_seconds": STALE_HEARTBEAT_SECONDS,
        },
        "train_curve": train_curve,
        "loss_definition": {
            "feature": "FP32 masked MSE + 0.1 × cosine",
            "action_loss": False,
            "anchors_per_global_update": 16,
            "anchor_definition": "每个全局更新=2卡×每卡8个anchor；新阶段计数从0到3116，不是旧学生1872续训",
        },
    }
    try:
        result["budget"] = {
            "remote_gpu_wall_used_seconds": float(
                status.get("remote_gpu_wall_used_seconds") or 0.0
            ),
            "remote_gpu_wall_cap_seconds": float(
                status.get("remote_gpu_wall_cap_seconds") or 140841.2631939833
            ),
            "total_gpu_wall_cap_seconds": float(
                status.get("total_gpu_wall_cap_seconds") or 230400.0
            ),
            "output_bytes": int(status.get("output_bytes") or 0),
            "output_cap_bytes": int(status.get("output_cap_bytes") or 15396584412),
            "finalization_reserve_seconds": status.get("finalization_reserve_seconds"),
        }
    except (TypeError, ValueError):
        result["budget"] = {}
    return result


def _svg_curve(points: Sequence[Mapping[str, Any]]) -> str:
    values = []
    for item in points:
        try:
            value = float(item.get("total"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append((int(item.get("step") or 0), value))
    if len(values) < 2:
        return '<p class="muted">训练曲线将在首个真实更新后出现。</p>'
    width, height = 720, 180
    low = min(value for _, value in values)
    high = max(value for _, value in values)
    span = high - low or 1.0
    max_step = max(step for step, _ in values) or 1
    polyline = " ".join(
        f"{20 + (step / max_step) * (width - 40):.1f},"
        f"{height - 20 - ((value - low) / span) * (height - 40):.1f}"
        for step, value in values
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="feature total curve">'
        f'<polyline points="{polyline}" fill="none" stroke="#2b6cb0" stroke-width="2"/>'
        "</svg>"
    )


def _render_html(data: Mapping[str, Any]) -> str:
    scope = data.get("actual_scope", {})
    source = data.get("source", {})
    validation = data.get("validation") or {}
    selection = data.get("selection") or {}
    budget = data.get("budget", {})
    gpu_rows = data.get("gpu", [])
    process_rows = data.get("processes", [])
    step = int(data.get("new_global_updates") or 0)
    target = int(data.get("target_new_global_updates") or TARGET_STEPS)
    pct = min(100.0, 100.0 * step / target) if target else 0.0
    gpu_html = (
        "".join(
            f"<tr><td>{html.escape(str(item.get('rank')))}</td>"
            f"<td>{html.escape(str(item.get('allocated')))}</td>"
            f"<td>{html.escape(str(item.get('reserved')))}</td>"
            f"<td>{html.escape(str(item.get('max_allocated')))}</td>"
            f"<td>{html.escape(str(item.get('max_reserved')))}</td></tr>"
            for item in gpu_rows
        )
        or '<tr><td colspan="5" class="muted">等待两卡真实更新。</td></tr>'
    )
    process_html = (
        "".join(
            f"<tr><td>{html.escape(str(item.get('rank')))}</td>"
            f"<td>{html.escape(str(item.get('pid')))}</td>"
            f"<td>{html.escape(str(item.get('pid_start_ticks')))}</td>"
            f"<td>{html.escape(str(item.get('pgid')))}</td>"
            f"<td>{html.escape(str(item.get('classification')))}</td>"
            f"<td>{html.escape(str(item.get('actual_state')))}</td>"
            f"<td>{html.escape(str(item.get('alive')))}</td></tr>"
            for item in process_rows
        )
        or '<tr><td colspan="7" class="muted">尚未上报 rank 进程；/proc 活跃状态未知。</td></tr>'
    )
    age = data.get("last_update_age_seconds")
    age_label = "—" if age is None else f"{float(age):.1f}s"
    native = data.get("native_exit")
    process_health = data.get("process_health") or {}
    if native is not None:
        native_label = str(native)
    elif process_health.get("any_alive"):
        native_label = "unknown（进程仍活；未伪造原生退出）"
    else:
        native_label = "unknown（无可核验活跃进程；未伪造原生退出）"
    source_count = source.get("source_count")
    source_count_label = "待核验" if source_count is None else f"{source_count} 源"
    split = source.get("split") or {}
    split_label = " / ".join(
        "待核验" if split.get(name) is None else str(split.get(name))
        for name in ("train", "validation", "test")
    )
    heartbeat_label = "stale" if process_health.get("stale_heartbeat") else "fresh"
    val_total = validation.get("feature_total")
    val_mse = validation.get("feature_mse")
    val_cos = validation.get("feature_cosine")
    val_rel = validation.get("relative_feature_mse")
    best = selection.get("best_step")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>学生整视觉头 · 双5090 · 只读监控</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;background:#f7fafc;color:#1a202c}}
main{{max-width:1040px;margin:auto}} h1{{font-size:24px}} h2{{font-size:17px;margin-top:24px}}
.card{{background:white;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin:12px 0}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}}
.k{{color:#4a5568;font-size:13px}} .v{{font-weight:600}} .muted{{color:#718096}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:6px;border-bottom:1px solid #edf2f7}}
.bar{{height:12px;background:#edf2f7;border-radius:6px;overflow:hidden}} .bar>div{{height:100%;background:#3182ce}}
code{{background:#edf2f7;padding:1px 4px;border-radius:4px}}
</style></head><body><main>
<h1>学生整视觉头 · 双 RTX 5090 · 只读监控</h1>
<p class="muted">新阶段计数 0–{target}，不是旧学生1872续训。数据来自只读共享RAW的新来源清单/教师缓存；旧50来源不用于本阶段。旧本地8878页面保持停止历史，本页只读、固定路由、无检查点文件浏览。</p>
<div class="card"><div class="kv">
<div><div class="k">有效状态 / 声明状态 / phase</div><div class="v">{html.escape(str(data.get("status")))} / {html.escape(str(data.get("status_claimed")))} / {html.escape(str(data.get("phase")))}</div></div>
<div><div class="k">/proc + heartbeat</div><div class="v">{html.escape(heartbeat_label)}；all_alive={html.escape(str(process_health.get("all_alive")))}</div></div>
<div><div class="k">新阶段全局更新</div><div class="v">{step:,} / {target:,}</div><div class="bar"><div style="width:{pct:.2f}%"></div></div></div>
<div><div class="k">anchors</div><div class="v">{_count(data.get("new_anchor_presentations"))} / {_count(data.get("target_new_anchor_presentations"))}</div></div>
<div><div class="k">最后更新年龄</div><div class="v">{html.escape(age_label)}</div></div>
<div><div class="k">原生退出</div><div class="v">{html.escape(native_label)}</div></div>
</div></div>
<div class="card"><h2>实际训练范围</h2><p>{html.escape(str(scope.get("label")))}</p>
<div class="kv"><div><div class="k">可训练参数</div><div class="v">{_count(scope.get("trainable_parameters"))} / 预期 {_count(scope.get("expected_parameters"))}</div></div>
<div><div class="k">教师 / 语言动作后端</div><div class="v">冻结；不在 optimizer</div></div>
<div><div class="k">目标函数</div><div class="v">视觉 token FP32 masked MSE + 0.1×cosine；无动作损失</div></div></div></div>
<div class="card"><h2>数据与更新定义</h2><div class="kv">
<div><div class="k">来源 / 拆分</div><div class="v">{html.escape(source_count_label)}；train / val / test = {html.escape(split_label)}</div></div>
<div><div class="k">anchor</div><div class="v">每个全局更新 2卡×8=16；当前阶段新计数</div></div>
<div><div class="k">验证</div><div class="v">{html.escape(str(validation.get("windows") or "—"))} windows / {html.escape(str(validation.get("sources") or source.get("split", {}).get("validation") or "—"))} sources；test 不参与选型</div></div>
<div><div class="k">共享数据</div><div class="v">只读共享RAW直接读取；未复制；旧50来源不用于本阶段</div></div>
<div><div class="k">epoch / update 定义</div><div class="v">本页计数按新阶段全局更新，不按 epoch；sampler 内部 epoch 只决定排列顺序，不是训练轮次</div></div>
</div></div>
<div class="card"><h2>验证与选型</h2><div class="kv">
<div><div class="k">MSE / cosine / total</div><div class="v">{_number(val_mse)} / {_number(val_cos)} / {_number(val_total)}</div></div>
<div><div class="k">相对 feature MSE</div><div class="v">{_number(val_rel)}</div></div>
<div><div class="k">当前最佳可选 step</div><div class="v">{html.escape(str(best))}</div></div>
<div><div class="k">选型指标</div><div class="v">{html.escape(str(selection.get("metric")))}；最早并列优先</div></div>
<div><div class="k">最佳模型 SHA-12</div><div class="v"><code>{html.escape(str(selection.get("checkpoint_sha256_12") or "—"))}</code></div></div>
<div><div class="k">最后 checkpoint</div><div class="v">{html.escape(str(data.get("last_checkpoint")))}</div></div>
</div></div>
<div class="card"><h2>双卡显存</h2><table><thead><tr><th>rank</th><th>allocated</th><th>reserved</th><th>peak allocated</th><th>peak reserved</th></tr></thead><tbody>{gpu_html}</tbody></table></div>
<div class="card"><h2>预算</h2><div class="kv">
<div><div class="k">远程 aggregate GPU wall</div><div class="v">{_seconds_label(budget.get("remote_gpu_wall_used_seconds"))} / {_seconds_label(budget.get("remote_gpu_wall_cap_seconds"))}</div></div>
<div><div class="k">原始总 cap</div><div class="v">{_seconds_label(budget.get("total_gpu_wall_cap_seconds"))}</div></div>
<div><div class="k">新输出</div><div class="v">{_bytes_label(budget.get("output_bytes"))} / {_bytes_label(budget.get("output_cap_bytes"))}</div></div>
<div><div class="k">finalization reserve</div><div class="v">{_seconds_label(budget.get("finalization_reserve_seconds"))}</div></div>
</div></div>
<div class="card"><h2>训练曲线（feature total）</h2>{_svg_curve(data.get("train_curve", []))}</div>
<div class="card"><h2>rank 进程（实际 /proc 核验）</h2><table><thead><tr><th>rank</th><th>pid</th><th>start ticks</th><th>pgid</th><th>/proc 分类</th><th>state</th><th>alive</th></tr></thead><tbody>{process_html}</tbody></table>
<p class="muted">alive 需要 PID、startticks、pgid 与 /proc 实际值匹配且非 Z；missing/reused/zombie/pgid_changed 不会继续显示 running。原生退出仅在无活跃进程且运行时已上报时显示，存活期间为 null/unknown。</p></div>
<p class="muted">只读固定路由：/、/api/status、/healthz。无目录列表、无 checkpoint 内容/路径浏览。</p>
</main></body></html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "OWACFullVisualMonitor/1.0"
    run_root: Path = Path(".")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; img-src data:",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/healthz":
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", b'{"ok":true}\n')
            return
        if route == "/api/status":
            body = json.dumps(snapshot(self.run_root), ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", body + b"\n")
            return
        if route == "/":
            body = _render_html(snapshot(self.run_root)).encode("utf-8")
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", body)
            return
        self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")


def make_server(
    run_root: str | Path, host: str = "127.0.0.1", port: int = 8879
) -> ThreadingHTTPServer:
    if host not in LOOPBACKS:
        raise ValueError("full-visual monitor must bind to loopback only")
    root = Path(run_root).resolve()
    handler = type("BoundMonitorHandler", (MonitorHandler,), {"run_root": root})
    return ThreadingHTTPServer((host, int(port)), handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8879)
    args = parser.parse_args(argv)
    if args.host not in LOOPBACKS:
        parser.error("full-visual monitor must bind to 127.0.0.1/localhost")
    server = make_server(args.run_root, args.host, args.port)
    print(
        json.dumps(
            {
                "event": "monitor_listening",
                "url": f"http://{args.host}:{args.port}/",
                "run_root": str(Path(args.run_root).resolve()),
                "readonly": True,
                "routes": ["/", "/api/status", "/healthz"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
