#!/usr/bin/env python3
"""N02 v1.2 只读实时进度页（Python 标准库，无第三方依赖）。

对外接口：
  GET /                     单文件中文进度页（n02_v12_monitor.html）
  GET /api/status           所有已知 run 的 status 快照 + 部署状态 + GPU/CPU 采样
  GET /api/history?run_id=  该 run 的 history.jsonl（JSON 数组）
  GET /api/preview?run_id=&file=   该 run 目录内的预览图片文件

安全边界：
  * 只读 <run-root>/<stage>/<run_id>，stage ∈ {P2,P3,P4}，run_id 必须匹配 ^[A-Za-z0-9._-]+$；
    解析后的真实路径必须落在 run-root 内（挡住 ..、绝对路径与软链接逃逸）。
  * 响应只包含固定字段白名单，训练端写入 status.json 的其他字段（可能含凭据）不会回显。
  * 只监听 127.0.0.1。

部署：supervisor 程序 owac-n02-v12-monitor（默认 127.0.0.1:18080；
用户经 SSH -L 8081:localhost:18080 访问）。
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_RUN_ROOT = "/workspace/owac/worktrees/n02-measurement-completion/outputs/n02-v12"
DOWNLOAD_STATUS_PATH = "/workspace/owac/deployment/downloads.json"

STAGES = ("P2", "P3", "P4")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
STALE_SECONDS = 30.0
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
LOG_TAIL_LINES = 30
LOG_LINE_CHARS = 500
HISTORY_MAX_RECORDS = 20000
GPU_SAMPLE_SECONDS = 10.0
DEPLOYMENT_REFRESH_SECONDS = 10.0

# 只有这些字段会出现在 /api/status 里（其余一律不回显）。
STATUS_FIELDS = (
    "run_id",
    "stage",
    "arm",
    "platform",
    "seed",
    "state",
    "activity",
    "gpu",
    "step",
    "target_steps",
    "updated_at",
    "started_at",
    "step_seconds",
    "anchors_per_second",
    "loss",
    "dev_action_error",
    "metric_kind",
    "metric_label",
    "dev_representation_error",
    "representation_metrics",
    "checkpoint",
    "eval",
    "log_tail",
)

RUNNING_STATES = {"running", "run", "train", "training", "in_progress", "busy"}
DONE_STATES = {"done", "complete", "completed", "finished", "success", "succeeded"}
ERROR_STATES = {"error", "failed", "failure", "crashed", "exception"}

SLOT_ALIASES = (
    ("attention", ("attention", "attn", "heatmap", "heat_map", "overlay", "saliency")),
    ("residual", ("residual", "resid", "diff", "delta", "error_map", "errormap")),
    ("raw", ("raw", "bayer", "cfa", "hdr")),
    ("rgb", ("rgb", "srgb", "color", "colour")),
)
CAMERA_KEYS = ("camera", "camera_id", "cam", "camera_name")
FRAME_KEYS = ("frame", "frame_id", "frame_index")
LOOP_KEYS = ("inner_loop", "loop", "loop_index", "inner")

_LOCK = threading.Lock()
_STATE: dict = {
    "gpus": [],
    "gpu_sampled_at": None,
    "cpu": None,
    "deployment": {},
    "deployment_updated_at": None,
    "server_started_at": None,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_iso(value):
    """把 ISO8601（Z 或 +00:00）解析成 epoch 秒；失败返回 None。"""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        from datetime import datetime

        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def run_root() -> Path:
    return Path(os.environ.get("N02_V12_RUN_ROOT") or DEFAULT_RUN_ROOT)


def resolve_run_dir(root: Path, stage, run_id):
    """白名单校验：返回 (run_dir, None) 或 (None, 拒绝原因)。"""
    if stage not in STAGES:
        return None, "unknown stage"
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id) or run_id in (".", ".."):
        return None, "invalid run_id"
    root_real = Path(os.path.realpath(root))
    candidate = Path(os.path.realpath(root_real / stage / run_id))
    if candidate != root_real and root_real not in candidate.parents:
        return None, "path escapes run root"
    if not candidate.is_dir():
        return None, "run not found"
    return candidate, None


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _state_kind(state) -> str:
    text = str(state or "").strip().lower()
    if text in RUNNING_STATES:
        return "running"
    if text in DONE_STATES:
        return "done"
    if text in ERROR_STATES:
        return "error"
    return "idle" if not text else "other"


def _classify(key, name) -> str:
    for source in (str(key or "").lower(), str(name or "").lower()):
        for slot, aliases in SLOT_ALIASES:
            if any(alias in source for alias in aliases):
                return slot
    return "other"


def _preview_url(run_id: str, relative: str) -> str:
    return "/api/preview?run_id={}&file={}".format(
        urllib.parse.quote(run_id, safe=""), urllib.parse.quote(relative, safe="")
    )


def _tile_entry(run_id: str, relative) -> dict:
    text = str(relative).strip()
    if not text:
        return {}
    name = text.rsplit("/", 1)[-1]
    return {"name": name, "url": _preview_url(run_id, text), "path": text}


def normalize_previews(run_id, raw):
    """把训练端写的 preview_files（dict / 嵌套 dict / 记录列表）归一成前端结构。"""
    empty = {"options": {"cameras": [], "frames": [], "loops": []}, "tiles": [], "other": []}
    if not raw:
        return empty

    tiles: list[dict] = []
    other: list[dict] = []
    options = {"cameras": [], "frames": [], "loops": []}

    def new_tile(camera=None, frame=None, loop=None):
        return {
            "camera": camera,
            "frame": frame,
            "loop": loop,
            "slots": {s: [] for s, _ in SLOT_ALIASES},
        }

    def add(tile, key, value):
        entries = value if isinstance(value, (list, tuple)) else [value]
        for entry in entries:
            if isinstance(entry, dict):
                entry = entry.get("path") or entry.get("file") or entry.get("url")
            if not isinstance(entry, str) or not entry.strip():
                continue
            slot = _classify(key, entry)
            item = _tile_entry(run_id, entry)
            if slot == "other":
                other.append({"key": str(key or ""), **item})
            else:
                tile["slots"][slot].append(item)

    def fill(tile, mapping):
        for key, value in mapping.items():
            if key in CAMERA_KEYS + FRAME_KEYS + LOOP_KEYS:
                continue
            if isinstance(value, dict):
                nested = dict(value)
                nested.setdefault("camera", key)
                fill(new_tile_from(nested), value)
            else:
                add(tile, key, value)

    def new_tile_from(mapping):
        tile = new_tile(
            mapping.get("camera") or mapping.get("cam") or mapping.get("camera_id"),
            None,
            None,
        )
        for key in FRAME_KEYS:
            if key in mapping:
                tile["frame"] = mapping[key]
                break
        for key in LOOP_KEYS:
            if key in mapping:
                tile["loop"] = mapping[key]
                break
        tiles.append(tile)
        return tile

    if isinstance(raw, dict):
        flat = new_tile()
        for key, value in raw.items():
            if isinstance(value, dict):
                fill(new_tile_from({"camera": key}), value)
            else:
                add(flat, key, value)
        if any(flat["slots"].values()):
            tiles.append(flat)
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict) and any(
                key in item for key in ("raw", "rgb", "attention", "residual")
            ):
                fill(new_tile_from(item), item)
            elif isinstance(item, dict):
                for key, value in item.items():
                    add(new_tile(), key, value)
            else:
                add(new_tile(), "", item)

    for tile in tiles:
        for slot, values in (("camera", "cameras"), ("frame", "frames"), ("loop", "loops")):
            value = tile[slot]
            if value is not None and value not in options[values]:
                options[values].append(value)

    return {"options": options, "tiles": tiles, "other": other[:8]}


def _log_tail(value):
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            lines.extend(str(item).splitlines() or [""])
    else:
        return []
    out = [line[:LOG_LINE_CHARS] for line in lines if line.strip()][-LOG_TAIL_LINES:]
    return out


def is_sample_run(run_id: str, payload: dict) -> bool:
    if payload.get("ui_test") is True or payload.get("sample") is True:
        return True
    return str(run_id).startswith("ui-sample")


def _gpu_candidates(declared):
    """把训练端写的 gpu 字段归一成待匹配的 token 列表。

    训练端实际写法是 dict（如 {"cuda_visible_devices": "1", "device": "cuda", ...}），
    也兼容 int / "0,1" / 列表；无法解析（例如 GPU UUID）时返回空列表，交给 arm 规则兜底。
    """
    if isinstance(declared, dict):
        for key in ("cuda_visible_devices", "device_index", "index", "id", "gpu"):
            if key in declared:
                return _gpu_candidates(declared[key])
        return []
    if isinstance(declared, (list, tuple)):
        tokens: list[str] = []
        for item in declared:
            tokens.extend(_gpu_candidates(item))
        return tokens
    if declared is None or declared == "":
        return []
    return [part.strip() for part in str(declared).replace(";", ",").split(",") if part.strip()]


def _match_gpus(gpus, run_id, arm, declared):
    """按 status.gpu 匹配；没写就按 arm（raw→0 / rgb→1）推断，并标明来源。"""
    out: list[dict] = []
    wanted: list[int] = []
    for item in _gpu_candidates(declared):
        num = _as_float(item)
        if num is not None and float(num).is_integer():
            wanted.append(int(num))
    assigned_by = "status"
    if not wanted:
        arm_text = str(arm or "").lower()
        if "raw" in arm_text and len(gpus) >= 1:
            wanted, assigned_by = [0], "arm"
        elif "rgb" in arm_text and len(gpus) >= 2:
            wanted, assigned_by = [1], "arm"
        else:
            return []
    for gpu in gpus:
        if gpu.get("index") in wanted:
            out.append({**gpu, "assigned_by": assigned_by})
    return out


def load_run(root: Path, stage: str, run_id: str):
    run_dir, error = resolve_run_dir(root, stage, run_id)
    if run_dir is None:
        return None
    try:
        payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    with _LOCK:
        gpus = list(_STATE["gpus"])
    record = {field: payload.get(field) for field in STATUS_FIELDS}
    record["run_id"] = payload.get("run_id") or run_id
    record["stage"] = payload.get("stage") or stage
    record["is_sample"] = is_sample_run(run_id, payload)
    record["dir_name"] = run_id
    record["state_kind"] = _state_kind(payload.get("state"))
    record["log_tail"] = _log_tail(payload.get("log_tail"))
    record["previews"] = normalize_previews(run_id, payload.get("preview_files"))
    record["gpu_detail"] = _match_gpus(gpus, run_id, payload.get("arm"), payload.get("gpu"))
    record["history_available"] = (run_dir / "history.jsonl").is_file()
    record["has_status"] = (run_dir / "status.json").is_file()
    updated = parse_iso(payload.get("updated_at"))
    record["age_seconds"] = round(time.time() - updated, 1) if updated else None
    record["stale"] = bool(updated is None or (time.time() - updated) > STALE_SECONDS)
    started, steps = parse_iso(payload.get("started_at")), _as_float(payload.get("step"))
    target, speed = _as_float(payload.get("target_steps")), _as_float(payload.get("step_seconds"))
    elapsed = (updated - started) if (updated and started) else None
    record["elapsed_seconds"] = round(elapsed, 1) if elapsed and elapsed > 0 else None
    if (
        steps is not None
        and target is not None
        and speed is not None
        and target > steps
        and speed > 0
    ):
        record["eta_seconds"] = round((target - steps) * speed, 1)
    else:
        record["eta_seconds"] = None
    return record


def snapshot_runs(root: Path):
    records: list[dict] = []
    for stage in STAGES:
        stage_dir = root / stage
        if not stage_dir.is_dir():
            continue
        try:
            entries = sorted(os.listdir(stage_dir))
        except OSError:
            continue
        for run_id in entries:
            if not RUN_ID_RE.match(run_id) or run_id in (".", ".."):
                continue
            if not (stage_dir / run_id).is_dir():
                continue
            if not (stage_dir / run_id / "status.json").is_file():
                continue  # 只列出真正的 run：没有 status.json 的目录（临时/工作目录）不算作业
            record = load_run(root, stage, run_id)
            if record:
                records.append(record)
    records.sort(
        key=lambda item: (
            item["age_seconds"] is None,
            item["age_seconds"] or 0.0,
            item["dir_name"],
        )
    )
    return records


def _fresh_running_real(records):
    return [
        item
        for item in records
        if not item["is_sample"]
        and item["age_seconds"] is not None
        and item["state_kind"] == "running"
        and not item["stale"]
    ]


def _newest(records):
    return min(
        records,
        key=lambda item: item["age_seconds"] if item["age_seconds"] is not None else 1e9,
    )


def _primary_key(item):
    """主作业排序键（越小越优先）：真实优先 → 目标步数大 → 开始时间早 → 心跳新。

    同一个键跨所有 tier 使用，所以「正式训练 vs -profile 侧跑」「RAW 主臂 vs RGB 对照臂」
    按同一套语义选，不取决于哪个 run 恰好最后写盘。
    """
    target = _as_float(item.get("target_steps"))
    started = parse_iso(item.get("started_at"))
    age = item["age_seconds"] if item["age_seconds"] is not None else float("inf")
    return (
        0 if not item["is_sample"] else 1,
        -target if target is not None else float("inf"),
        started if started is not None else float("inf"),
        age,
    )


def _primary_of(records):
    """挑「主作业」：先看新鲜运行中的真实 run，再退到有数据的真实 run / 有数据的 run / 全部。"""
    if not records:
        return None
    groups = (
        _fresh_running_real(records),
        [item for item in records if not item["is_sample"] and item["age_seconds"] is not None],
        [item for item in records if item["age_seconds"] is not None],
        records,
    )
    for group in groups:
        if group:
            return min(group, key=_primary_key)["dir_name"]
    return None


def _pick(records, predicate):
    return _primary_of([item for item in records if predicate(item)])


def build_display(records) -> dict:
    """页面选择规则：先要有 status.json（age_seconds 非空），再优先真实作业，
    最后才轮到无数据的目录。

    运行根下可能出现别的 agent 建的临时目录（不是 run），不能让它们把页面顶成空白。
    """
    current = _primary_of(records)
    return {
        "current": current,
        "raw": _pick(records, lambda item: "raw" in str(item.get("arm") or "").lower()),
        "rgb": _pick(records, lambda item: "rgb" in str(item.get("arm") or "").lower()),
    }


def sample_gpus() -> list[dict]:
    try:
        text = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index = _as_float(parts[0])
        gpus.append(
            {
                "index": int(index) if index is not None else parts[0],
                "name": parts[1],
                "memory_used_mib": _as_float(parts[2]),
                "memory_total_mib": _as_float(parts[3]),
                "utilization": _as_float(parts[4]),
            }
        )
    return gpus


def sample_cpu():
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        return None
    return {
        "load1": round(load1, 2),
        "load5": round(load5, 2),
        "load15": round(load15, 2),
        "cores": os.cpu_count(),
    }


def sampler_loop() -> None:
    while True:
        gpus, cpu = sample_gpus(), sample_cpu()
        with _LOCK:
            if gpus:
                _STATE["gpus"] = gpus
                _STATE["gpu_sampled_at"] = utc_now()
            _STATE["cpu"] = cpu
        time.sleep(GPU_SAMPLE_SECONDS)


def deployment_loop() -> None:
    path = Path(DOWNLOAD_STATUS_PATH)
    while True:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("downloads.json is not an object")
            clean = {}
            for repo, row in payload.items():
                row = row if isinstance(row, dict) else {}
                clean[str(repo)] = {
                    key: row.get(key)
                    for key in ("status", "path", "updated_at", "files", "bytes")
                    if key in row
                }
            with _LOCK:
                _STATE["deployment"] = clean
                _STATE["deployment_updated_at"] = utc_now()
        except (OSError, ValueError):
            pass
        time.sleep(DEPLOYMENT_REFRESH_SECONDS)


def read_history(run_dir: Path):
    path = run_dir / "history.jsonl"
    real = Path(os.path.realpath(path))
    run_real = Path(os.path.realpath(run_dir))
    if real != run_real and run_real not in real.parents:
        return None
    if not real.is_file():
        return []
    try:
        text = real.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines()[-HISTORY_MAX_RECORDS:]:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def status_payload(root: Path) -> dict:
    records = snapshot_runs(root)
    with _LOCK:
        state = dict(_STATE)
    return {
        "updated_at": utc_now(),
        "stale_seconds": STALE_SECONDS,
        "run_root": str(root),
        "server": {
            "pid": os.getpid(),
            "started_at": state.get("server_started_at"),
            "port": _STATE.get("port"),
            "uptime_seconds": round(
                time.time() - (parse_iso(state.get("server_started_at")) or time.time()), 1
            ),
        },
        "gpus": state.get("gpus") or [],
        "gpu_sampled_at": state.get("gpu_sampled_at"),
        "gpu_sample_seconds": GPU_SAMPLE_SECONDS,
        "cpu": state.get("cpu"),
        "deployment": state.get("deployment") or {},
        "deployment_updated_at": state.get("deployment_updated_at"),
        "runs": records,
        "display": build_display(records),
    }


@functools.lru_cache(maxsize=1)
def _repaired_selector():
    # The deployed service runs this file directly with system Python, without
    # installing owac or setting PYTHONPATH. Load only the stdlib sibling.
    path = Path(__file__).with_name("n02_task_selection.py")
    spec = importlib.util.spec_from_file_location("n02_monitor_selection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load monitor selection helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_repaired_checkpoint


def repaired_selection(directory):
    return _repaired_selector()(directory)


def training_sample_fit(directory: Path | None, name: str = "fit_latest.json") -> dict | None:
    if directory is None:
        return None
    try:
        record = json.loads((directory / name).read_text())
        maxima, limits = record["maxima"], record["limits"]
        keys = ("joint_rms_deg", "first_joint_max_deg", "gripper_mae", "gripper_max_error")
        values = {key: float(maxima[key]) for key in keys}
        bounds = {key: float(limits[key + "_max"]) for key in keys}
        if not all(math.isfinite(v) and v >= 0 for v in (*values.values(), *bounds.values())):
            return None
        step = record["step"]
        if type(step) is not int or step < 0 or not record["rows"]:
            return None
        return {
            "step": step,
            "passed": all(values[key] <= bounds[key] for key in keys),
            "maxima": values,
            "limits": bounds,
            "evaluations": len(record["rows"]),
            "scope": "training-sample fit, not closed loop",
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def training_payload(root: Path) -> dict:
    """Only explicitly selected current training runs; old probes never become results."""
    try:
        active = json.loads((root / "active_training.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        active = {}
    cards = []
    for selected in active.get("runs", [])[:2]:
        stage, run_id = selected.get("stage", "P2"), selected.get("run_id")
        if stage not in STAGES or not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            continue
        directory, _ = resolve_run_dir(root, stage, run_id)
        record = load_run(root, stage, run_id) if directory is not None else None
        history = (read_history(directory) or []) if directory is not None else []
        metric_kind = (record or {}).get("metric_kind")
        error_key = (
            "dev_representation_error"
            if metric_kind == "raw_representation"
            else "sampled_control_mse"
            if metric_kind == "sampled_control"
            else "dev_action_error"
        )
        curve = [
            {"step": r.get("step"), "error": r.get(error_key)}
            for r in history
            if _as_float(r.get(error_key)) is not None
        ]
        # Saving and completing can record the same dev step twice.
        curve = list({r["step"]: r for r in curve}.values())
        baseline = curve[0]["error"] if curve and curve[0]["step"] == 0 else None
        latest = curve[-1] if curve else None
        try:
            quality = (
                json.loads((directory / "dev_sampled_latest.json").read_text()) if directory else {}
            )
        except (OSError, ValueError):
            quality = {}
        latest_training = latest
        selected_step = None
        if (
            active.get("phase") == "task_adaptation"
            and directory
            and (record or {}).get("state") in ("evaluating", "completed")
        ):
            try:
                chosen = json.loads((directory / "selection.json").read_text())["selected"]
                step = chosen["step"]
                if type(step) is not int or not 0 < step < 10_000_000 or not chosen["eligible"]:
                    raise ValueError("invalid selected checkpoint")
                selected_metrics = json.loads(
                    (directory / f"checkpoint_{step:07d}.metrics.json").read_text()
                )
                if selected_metrics.get("task_dev_manifest_sha256") != quality.get(
                    "task_dev_manifest_sha256"
                ):
                    raise ValueError("selected development protocol differs")
                quality = {**quality, **selected_metrics}
                latest = {"step": step, "error": selected_metrics["sampled_control_mse"]}
                selected_step = step
            except (OSError, ValueError, KeyError, TypeError):
                pass
        validation = None
        if active.get("phase") == "task_adaptation" and directory:
            try:
                before = json.loads((directory / "dev_baseline.json").read_text())
                opening_metric = (
                    "first10_mse"
                    if quality.get("sampled_metric") == "balanced_sampled_prefix14_mse"
                    else "mse"
                )
                initial_opening = before["task_control"]["strata"]["Normal/opening"][opening_metric]
                opening = quality["task_control"]["strata"]["Normal/opening"][opening_metric]
                improved = quality["sampled_control_mse"] < before["sampled_control_mse"]
                ratio = opening / initial_opening
                validation = {
                    "passed": improved and ratio <= 1.05,
                    "normal_opening_change_percent": (ratio - 1) * 100,
                    "label": "开场退化，当前权重不晋级闭环"
                    if ratio > 1.05
                    else "动作验证符合晋级门槛"
                    if improved
                    else "当前动作尚未优于训练前",
                }
                if quality.get("sampled_metric") in (
                    "balanced_sampled_prefix14_mse",
                    "balanced_sampled_control14_mse",
                ):
                    decision = repaired_selection(directory)
                    eligible = (
                        next(
                            (c for c in decision["candidates"] if c["step"] == latest["step"]), None
                        )
                        if latest
                        else None
                    )
                    validation = {
                        "passed": bool(eligible and eligible["eligible"]),
                        "normal_opening_change_percent": (ratio - 1) * 100,
                        "failed_guards": eligible["failed_guards"] if eligible else [],
                        "label": "离线门槛通过，仍须真实抓持验收"
                        if eligible and eligible["eligible"]
                        else "开场、关节或夹爪验证未通过，不晋级闭环",
                    }
                    if not latest or latest["step"] == 0:
                        validation = {
                            "passed": None,
                            "pending": True,
                            "label": "等待首次训练后留出验证",
                        }
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
                pass
        seconds = _as_float((record or {}).get("step_seconds"))
        actual_rate = _as_float((record or {}).get("anchors_per_second"))
        effective = int(selected.get("effective_batch_size", 2))
        fit_curve = []
        if directory:
            for path in sorted(directory.glob("fit_[0-9]*.json")):
                if not re.fullmatch(r"fit_[0-9]{7}\.json", path.name):
                    continue
                fit = training_sample_fit(directory, path.name)
                if fit is not None:
                    fit_curve.append({"step": fit["step"], "error": fit["maxima"]["joint_rms_deg"]})
        cards.append(
            {
                "label": selected.get("label", run_id),
                "run_id": run_id,
                "stage": stage,
                "state": (record or {}).get("state", "waiting"),
                "activity": (record or {}).get("activity", "等待启动"),
                "metric_kind": metric_kind,
                "metric_label": "前10步动作误差"
                if quality.get("sampled_metric") == "balanced_sampled_prefix14_mse"
                else (record or {}).get("metric_label"),
                "sampled_metric": quality.get("sampled_metric"),
                "representation_metrics": {
                    key: ((record or {}).get("representation_metrics") or {}).get(key)
                    for key in (
                        "crop_match_accuracy",
                        "deep_flip_relative_l2",
                        "fused_flip_relative_l2",
                        "representation_gate_passed",
                    )
                },
                "stale": (record or {}).get("stale", False),
                "updated_at": (record or {}).get("updated_at"),
                "step": (record or {}).get("step") or 0,
                "target": (record or {}).get("target_steps") or selected.get("target", 1000),
                "seconds_per_update": seconds,
                "samples_per_minute": 60 * actual_rate
                if actual_rate is not None
                else 60 * effective / seconds
                if seconds and seconds > 0
                else None,
                "eta_seconds": (record or {}).get("eta_seconds"),
                "loss": (record or {}).get("loss"),
                "latest_dev": latest,
                "latest_training_dev": latest_training,
                "selected_checkpoint_step": selected_step,
                "sampled_control_mse": quality.get("sampled_control_mse"),
                "task_control": quality.get("task_control"),
                "action_validation": validation,
                "training_fit": training_sample_fit(directory),
                "training_fit_curve": fit_curve,
                "spatial_diversity": quality.get("spatial_diversity_per_window"),
                "baseline_dev": baseline,
                "best_dev": min(r["error"] for r in curve) if curve else None,
                "improvement_percent": 100 * (baseline - latest["error"]) / baseline
                if baseline and baseline > 0 and latest
                else None,
                "curve": curve,
                "gpu": (record or {}).get("gpu_detail", []),
                "checkpoint": (record or {}).get("checkpoint"),
            }
        )
    cases = repair_cases(root) if any("raw-repair-" in c["run_id"] for c in cards) else []
    completed = [c for c in cases if c["success"] is not None]
    return {
        "updated_at": utc_now(),
        "title": active.get("title", "N02 训练"),
        "phase": active.get("phase")
        if active.get("phase") in ("raw_fusion", "backend", "task_adaptation", "fewshot_fit")
        else None,
        "phase_detail": active.get("phase_detail", "")
        if isinstance(active.get("phase_detail", ""), str)
        else "",
        "runs": cards,
        "closed_loop": {
            "completed": len(completed),
            "success": sum(c["accepted_success"] is True for c in completed),
            "benchmark_success": sum(c["success"] is True for c in completed),
            "pending_review": sum(c["accepted_success"] is None for c in completed),
            "successful_case": next((c for c in completed if c["accepted_success"] is True), None),
        },
    }


def demo_directory(root: Path) -> Path | None:
    """Only a selected demonstration below run-root/demos may be served."""
    try:
        selected = json.loads((root / "active_closed_loop.json").read_text())
        relative = Path(selected["directory"])
        if relative.is_absolute() or ".." in relative.parts:
            return None
        directory = (root / relative).resolve()
        allowed = (root / "demos").resolve()
        if allowed not in directory.parents or not directory.is_dir():
            return None
        return directory
    except (OSError, ValueError, KeyError, TypeError):
        return None


def demo_rows(path: Path) -> list[dict]:
    """The standalone stdlib monitor must not import the installed training package."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue  # An episode append may still be in progress.
        if isinstance(record, dict):
            records.append(record)
    return records


def demo_payload(root: Path) -> dict:
    directory = demo_directory(root)
    if directory is None:
        return {"cases": [], "arms": {}}
    protocol = json.loads((directory / "protocol.json").read_text())
    result = {"cases": [], "arms": {}, "scope": protocol["scope"]}
    for arm in ("raw", "rgb"):
        status_path = directory / arm / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        result["arms"][arm] = {
            key: status.get(key) for key in ("state", "completed", "target", "success", "case_id")
        }
    ledgers = {
        arm: {row["case_id"]: row for row in demo_rows(directory / arm / "episodes.jsonl")}
        for arm in ("raw", "rgb")
    }
    for arm, ledger in ledgers.items():
        result["arms"][arm]["completed"] = len(ledger)
        result["arms"][arm]["success"] = sum(int(row["success"]) for row in ledger.values())
    for entry in protocol["entries"]:
        case_id = f"grab-roller-s0-{entry['lighting_domain']}"
        item = {
            "case_id": case_id,
            "domain": entry["lighting_domain"],
            "ev": entry["lighting_ev"],
            "task": entry["task_name"],
            "arms": {},
        }
        for arm in ("raw", "rgb"):
            record = ledgers[arm].get(case_id)
            if record is not None:
                item["arms"][arm] = {
                    key: record.get(key)
                    for key in ("success", "num_steps", "wall_seconds", "termination_reason")
                }
                if record.get("video"):
                    item["arms"][arm]["video"] = f"/api/demo-video?arm={arm}&case_id={case_id}"
        result["cases"].append(item)
    return result


def policy_audit_video(root: Path, mode: str | None) -> Path | None:
    if mode not in ("original", "bridge", "bridge-unpadded", "n02-raw-native", "n02-rgb-native"):
        return None
    root = root.resolve()
    directory = (root / "policy-audit-20261016").resolve()
    if root not in directory.parents:
        return None
    video = (directory / f"{mode}-normal" / "episode0.mp4").resolve()
    if directory in video.parents and video.is_file() and video.suffix == ".mp4":
        return video
    return None


def repair_cases(root: Path) -> list[dict]:
    directory = root / "repair-r7-20261016"
    cases = []
    for path in sorted(directory.glob("*-step*/result.json")):
        if directory.resolve() not in path.resolve().parents:
            continue
        try:
            row = json.loads(path.read_text())
            episode = row.get("results", [])
            official = episode[0]["success"] if len(episode) == 1 else None
            # A simulator flag alone does not establish a physically completed grasp.
            review_path = path.parent / "physical_assessment.json"
            review = json.loads(review_path.read_text()) if review_path.is_file() else {}
            cases.append(
                {
                    "case_id": path.parent.name,
                    "kind": row["kind"],
                    "step": row["step"],
                    "success": official,
                    "accepted_success": review.get("accepted_success") if official else False,
                    "review_note": review.get("reason", "原判据触发，抓取结果待复核")
                    if official
                    else "原判据未通过",
                    "num_steps": episode[0].get("num_steps") if len(episode) == 1 else None,
                    "returncode": row["returncode"],
                    "video": "/api/repair-r7-video?case=" + path.parent.name
                    if repair_video(root, path.parent.name)
                    else None,
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return cases


def repair_video(root: Path, case: str | None) -> Path | None:
    if not isinstance(case, str) or not re.fullmatch(
        r"(warm|fresh)-step\d{4}(?:-attempt[1-9]\d*)?", case
    ):
        return None
    directory = (root / "repair-r7-20261016").resolve()
    path = (directory / case / "episode0.mp4").resolve()
    if directory in path.parents and path.is_file():
        return path
    return None


def physics_video(root: Path, case: str | None) -> Path | None:
    if case not in ("raw-time-continued", "native-continued", "raw-live-continued"):
        return None
    root = root.resolve()
    directory = (root / "physics-audit-20261016").resolve()
    path = (directory / case / "episode0.mp4").resolve()
    if root in directory.parents and directory in path.parents and path.is_file():
        return path
    return None


def representation_video(root: Path, case: str | None) -> Path | None:
    if not isinstance(case, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,95}", case):
        return None
    root = root.resolve()
    directory = (root / "representation-r8-rollouts").resolve()
    target = (directory / case / "episode0.mp4").resolve()
    if root in directory.parents and directory in target.parents and target.is_file():
        return target
    return None


def representation_cases(root: Path) -> list[dict]:
    root = root.resolve()
    directory = (root / "representation-r8-rollouts").resolve()
    if root not in directory.parents:
        return []
    result = []
    for protocol in sorted(directory.glob("*/protocol.json")):
        if directory not in protocol.resolve().parents:
            continue
        case = protocol.parent.name
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,95}", case):
            continue
        try:
            declared = json.loads(protocol.read_text())
            receipt = protocol.with_name("result.json")
            if directory not in receipt.resolve().parents:
                continue
            record = json.loads(receipt.read_text()) if receipt.is_file() else {}
            physical = record.get("physical") or {}
            motion_path = protocol.with_name("motion.json")
            motion = json.loads(motion_path.read_text()) if motion_path.is_file() else {}
            control_rows = demo_rows(protocol.with_name("control.jsonl"))
            assessed = (
                record.get("returncode") == 0
                and physical.get("action_source") == "live_raw_policy"
                and physical.get("diagnostic_only") is False
                and isinstance(physical.get("physical_accepted"), bool)
            )
            result.append(
                {
                    "case_id": case,
                    "lighting_domain": declared.get("lighting_domain", "Normal"),
                    "lighting_ev": declared.get("entry", {}).get("lighting_ev"),
                    "state": "completed" if assessed else "error" if record else "running",
                    "accepted_success": physical["physical_accepted"] if assessed else None,
                    "hold_seconds": physical.get("max_hold_seconds"),
                    "control_steps": physical.get("control_steps"),
                    "progress_steps": len(control_rows),
                    "motion": motion,
                    "replan_interval": declared.get("replan_interval"),
                    "checkpoint_step": declared.get("checkpoint_step"),
                    "joint_delta_limit": declared.get("joint_delta_limit"),
                    "upstream_triggered": physical.get("first_official_trigger") is not None,
                    "video": "/api/representation-video?case=" + case
                    if assessed and representation_video(root, case)
                    else None,
                }
            )
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return result


def active_development(root: Path) -> dict | None:
    try:
        record = json.loads((root / "active_development.json").read_text())
        cases = record["cases"]
        if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
            return None
        if not all(re.fullmatch(r"[a-z][a-z0-9-]{0,95}", c["case_id"]) for c in cases):
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


EXECUTION_WITNESSES = {
    "native-a1": "10步重规划＋0.05限幅",
    "native-nolimit-a1": "10步重规划，取消限幅",
    "native-original-a1": "恢复原生50步执行",
    "native-fixed-r12-a1": "r12：原生50步＋固定采样噪声",
}


def execution_video(root: Path, case: str | None) -> Path | None:
    if case not in EXECUTION_WITNESSES:
        return None
    directory = (root / "repair-r11-20261017").resolve()
    target = (directory / case / "episode0.mp4").resolve()
    if root.resolve() in directory.parents and directory in target.parents and target.is_file():
        return target
    return None


def execution_cases(root: Path) -> list[dict]:
    """Independent native RGB witnesses; these never enter RAW success counts."""
    directory = (root / "repair-r11-20261017").resolve()
    if root.resolve() not in directory.parents:
        return []
    result = []
    for case, label in EXECUTION_WITNESSES.items():
        path = directory / case / "result.json"
        if directory not in path.resolve().parents or not path.is_file():
            continue
        try:
            row = json.loads(path.read_text())
            protocol = json.loads(path.with_name("protocol.json").read_text())
            physical = row.get("physical") or {}
            assessed = (
                row.get("returncode") == 0
                and protocol.get("scope") == "native RGB execution witness"
                and physical.get("diagnostic_only") is False
                and isinstance(physical.get("physical_accepted"), bool)
                and row.get("accepted_success") == physical["physical_accepted"]
            )
            result.append(
                {
                    "case_id": case,
                    "label": label,
                    "policy": "native_rgb_reference",
                    "accepted_success": physical["physical_accepted"] if assessed else None,
                    "steps": physical.get("control_steps"),
                    "hold_seconds": physical.get("max_hold_seconds"),
                    "video": "/api/execution-video?case=" + case
                    if assessed and execution_video(root, case)
                    else None,
                }
            )
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return result


def task_chain_payload(root: Path) -> dict:
    """Read fixed, local r14 diagnostic status files without exposing arbitrary fields."""
    fields = (
        "branch",
        "phase",
        "state",
        "step",
        "max_updates",
        "seconds_per_update",
        "latest_loss",
        "selected_step",
        "selected_val_mse",
        "updated_at",
        "peak_memory_gib",
        "actual_updates",
        "closed_loop_success",
    )
    cards = []
    for branch in ("small", "pi"):
        path = (root / "task-chain-r14" / branch / "status.json").resolve()
        if root.resolve() not in path.parents or not path.is_file():
            cards.append(dict(branch=branch, state="waiting"))
            continue
        try:
            value = json.loads(path.read_text())
            card = {key: value[key] for key in fields if key in value}
            # A stopped writer must never keep appearing as live training.
            card["stale"] = time.time() - float(value.get("updated_at", 0)) > 180
            cards.append(card)
        except (OSError, ValueError, TypeError):
            cards.append(dict(branch=branch, state="unavailable"))
    comparison = None
    path = (root / "task-chain-r14/comparison.json").resolve()
    if root.resolve() in path.parents and path.is_file():
        try:
            raw = json.loads(path.read_text())
            keys = (
                "branch",
                "phase",
                "train_joint_deg",
                "test_joint_deg",
                "normal_joint_deg",
                "gripper_mae",
                "fit_passed",
                "selected_step",
            )
            comparison = dict(
                rows=[{k: r[k] for k in keys if k in r} for r in raw["rows"]],
                state_only_test_deg=raw.get("state_only_test_deg"),
            )
        except (OSError, ValueError, TypeError, KeyError):
            comparison = None
    return dict(runs=cards, comparison=comparison, scope="完整50步离线动作诊断；不代表闭环成功")


def learning_payload(root: Path) -> dict:
    """Fixed r15 namespace; preserve separation from historical r14 results."""
    directory = root / "learning-r15"

    def read(relative):
        path = (directory / relative).resolve()
        if root.resolve() not in path.parents:
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    fields = (
        "run_id",
        "state",
        "activity",
        "step",
        "max_updates",
        "modality",
        "demos",
        "seed",
        "actual_updates",
        "seconds_per_update",
        "selected_val_mse",
        "updated_at",
    )
    names = [
        f"{m}-{d}-s{s}" for d in (3, 5) for s in (15019, 15020) for m in ("state", "raw", "native")
    ] + [f"pi-{d}-s15019" for d in (3, 5)]
    cards = []
    for name in ["prepare", *names]:
        value = read(name + "/status.json")
        if value:
            card = {k: value[k] for k in fields if k in value}
            card["stale"] = time.time() - float(value.get("updated_at", 0)) > 180
            cards.append(card)
    public = read("progress.json")
    reports = []
    metrics = ("normalized_control_mse", "joint_rms_deg", "gripper_mae", "windows")
    for raw in public.get("reports", []):
        if raw.get("run_id") not in names:
            continue
        r = {k: raw[k] for k in ("run_id", "modality", "demos", "seed") if k in raw}
        r["rows"] = []
        for value in raw.get("rows", []):
            row = {k: value[k] for k in ("budget", "selected_step") if k in value}
            for version in ("selected", "last"):
                row[version] = {
                    s: {k: value[version][s][k] for k in metrics} for s in ("train", "val", "test")
                }
                if "normal_test" in value[version]:
                    row[version]["normal_test"] = {
                        k: value[version]["normal_test"][k] for k in metrics
                    }
            r["rows"].append(row)
        reports.append(r)
    errors = [read(name + "-error.json") for name in ("prepare", "heads-pi5", "pi3")]
    failed = [
        {k: r[k] for k in ("command", "failure_type", "updated_at") if k in r} for r in errors if r
    ]
    return dict(
        runs=cards,
        reports=reports,
        planned_runs=14,
        completed_runs=len(reports),
        audited=bool(public.get("audited")),
        failures=failed,
        scope="冻结编码器的50步动作诊断；历史留出Normal/Over；RGB为原光照重放参照",
    )


def residual_payload(root: Path) -> dict:
    """Read only the fixed r16 experiment, with public-field whitelists."""
    directory = root / "residual-r16"

    def read(relative):
        path = (directory / relative).resolve()
        if root.resolve() not in path.parents:
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    names = [f"{m}-s{s}" for s in (15019, 15020) for m in ("state", "raw", "native")]
    fields = (
        "run_id",
        "state",
        "modality",
        "seed",
        "step",
        "max_updates",
        "actual_updates",
        "seconds_per_update",
        "selected_gate",
        "selected_step",
        "selected_val_mse",
        "updated_at",
    )
    cards = []
    for name in names:
        value = read(name + "/status.json")
        card = {k: value[k] for k in fields if k in value}
        card.setdefault("run_id", name)
        card.setdefault("state", "waiting")
        card["stale"] = bool(value) and time.time() - float(value.get("updated_at", 0)) > 180
        cards.append(card)
    public = read("progress.json")
    metrics = ("normalized_control_mse", "joint_rms_deg", "gripper_mae", "windows")

    def metric_set(value):
        return {
            s: {k: v[k] for k in metrics if k in v}
            for s, v in value.items()
            if s in ("train", "val", "test", "normal_test", "over_test")
        }

    reports = []
    for raw in public.get("reports", []):
        if raw.get("run_id") not in names:
            continue
        r = {k: raw[k] for k in ("run_id", "modality", "seed") if k in raw}
        r["baseline"] = metric_set(raw.get("baseline", {}))
        r["rows"] = []
        for value in raw.get("rows", []):
            row = {k: value[k] for k in ("budget", "selected_step", "selected_gate") if k in value}
            for version in ("selected", "last"):
                row[version] = metric_set(value.get(version, {}))
            r["rows"].append(row)
        reports.append(r)
    failures = []
    for seed in (15019, 15020):
        value = read(f"seed{seed}-error.json")
        if value:
            failures.append({k: value[k] for k in ("seed", "failure_type") if k in value})
    return dict(
        runs=cards,
        reports=reports,
        planned_runs=6,
        completed_runs=len(reports),
        audited=bool(public.get("audited")),
        failures=failures,
        scope="固定状态基线＋有界残差；50步离线动作；历史开发留出Normal/Over",
    )


def joint_payload(root: Path) -> dict:
    """Read only the fixed r17 experiment, with public-field whitelists."""
    directory = root / "residual-joint-r17"

    def read(relative):
        path = (directory / relative).resolve()
        if root.resolve() not in path.parents:
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    names = [
        f"{m}-s{s}" for s in (17019, 17020) for m in ("baseline", "state", "raw", "native", "joint")
    ]
    fields = (
        "run_id",
        "state",
        "modality",
        "seed",
        "step",
        "max_updates",
        "actual_updates",
        "seconds_per_update",
        "selected_gate",
        "selected_step",
        "selected_val_mse",
        "updated_at",
    )
    cards = []
    for name in names:
        value = read(name + "/status.json")
        card = {k: value[k] for k in fields if k in value}
        card.setdefault("run_id", name)
        card.setdefault("state", "waiting")
        card["stale"] = bool(value) and time.time() - float(value.get("updated_at", 0)) > 180
        cards.append(card)
    public = read("progress.json")
    metrics = ("normalized_control_mse", "joint_rms_deg", "gripper_mae", "windows")

    def metric_set(value):
        return {
            s: {k: v[k] for k in metrics if k in v}
            for s, v in value.items()
            if s in ("train", "val", "test", "normal_test", "over_test", "home_test")
        }

    reports = []
    for raw in public.get("reports", []):
        if raw.get("run_id") not in names:
            continue
        r = {k: raw[k] for k in ("run_id", "modality", "seed") if k in raw}
        r["baseline"] = metric_set(raw.get("baseline", {}))
        r["rows"] = []
        for value in raw.get("rows", []):
            row = {k: value[k] for k in ("budget", "selected_step", "selected_gate") if k in value}
            for version in ("selected", "last"):
                row[version] = metric_set(value.get(version, {}))
            r["rows"].append(row)
        reports.append(r)
    failures = []
    for seed in (17019, 17020):
        value = read(f"train-{seed}-error.json") or read(f"prepare-{seed}-error.json")
        if value:
            failures.append({k: value[k] for k in ("seed", "failure_type") if k in value})
    collection = read("capture_progress.json")
    collection = {k: collection[k] for k in ("planned", "attempted", "accepted") if k in collection}
    preparation = read("prepare/status.json")
    preparation = {
        k: preparation[k] for k in ("state", "activity", "step", "max_updates") if k in preparation
    }
    return dict(
        collection=collection,
        preparation=preparation,
        runs=cards,
        reports=reports,
        planned_runs=10,
        completed_runs=len(reports),
        audited=bool(public.get("audited")),
        failures=failures,
        scope="正常光新场景；冻结RAW与联合RAW、状态/原生视觉残差；专家演示成功不等于策略成功",
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "n02-v12-monitor"
    run_root_override = None

    def _root(self) -> Path:
        return Path(self.run_root_override or run_root())

    def _send(self, code, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _video(self, path: Path) -> None:
        """Serve validated local videos with bounded reads and browser seeking."""
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", requested.strip())
            valid = match is not None and bool(match[1] or match[2]) and size > 0
            if valid:
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), size - 1) if match[2] else size - 1
                else:
                    length = int(match[2])
                    start = max(0, size - length)
                    valid = length > 0
                valid = valid and 0 <= start <= end < size
            if not valid:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = source.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _query(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        return {key: values[-1] for key, values in urllib.parse.parse_qs(parsed.query).items()}

    def _reject(self, reason: str, run_id=None) -> None:
        code = 404 if reason in ("run not found", "unknown stage") else 400
        self._json(code, {"error": reason, "run_id": run_id})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/joint":
                html_path = Path(__file__).with_name("n02_joint.html")
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/joint":
                self._json(200, joint_payload(self._root()))
                return
            if path == "/residual":
                html_path = Path(__file__).with_name("n02_residual.html")
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/residual":
                self._json(200, residual_payload(self._root()))
                return
            if path == "/learning":
                html_path = Path(__file__).with_name("n02_learning.html")
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/learning":
                self._json(200, learning_payload(self._root()))
                return
            if path == "/task-chain":
                html_path = Path(__file__).with_name("n02_task_chain.html")
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/task-chain":
                self._json(200, task_chain_payload(self._root()))
                return
            if path == "/closed-loop":
                html_path = Path(__file__).with_name("n02_v12_demo.html")
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/closed-loop-demo":
                self._json(200, demo_payload(self._root()))
                return
            if path == "/api/repair-r7":
                self._json(200, {"cases": repair_cases(self._root())})
                return
            if path == "/api/execution-cases":
                self._json(200, {"cases": execution_cases(self._root())})
                return
            if path == "/api/execution-video":
                video = execution_video(self._root(), self._query().get("case"))
                if video is None:
                    self._json(404, {"error": "execution witness video not available"})
                else:
                    self._video(video)
                return
            if path == "/api/representation-cases":
                self._json(
                    200,
                    {
                        "cases": representation_cases(self._root()),
                        "current_development": active_development(self._root()),
                    },
                )
                return
            if path == "/api/representation-video":
                video = representation_video(self._root(), self._query().get("case"))
                if video is None:
                    self._json(404, {"error": "representation rollout video not available"})
                else:
                    self._video(video)
                return
            if path == "/api/physics-video":
                video = physics_video(self._root(), self._query().get("case"))
                if video is None:
                    self._json(404, {"error": "physics review video not available"})
                else:
                    self._video(video)
                return
            if path == "/api/repair-r7-video":
                video = repair_video(self._root(), self._query().get("case"))
                if video is None:
                    self._json(404, {"error": "repair video not available"})
                else:
                    self._video(video)
                return
            if path == "/api/demo-video":
                self._serve_demo_video()
                return
            if path == "/api/policy-audit-video":
                video = policy_audit_video(self._root(), self._query().get("mode"))
                if video is None:
                    self._json(404, {"error": "audit video not available"})
                else:
                    self._video(video)
                return
            if path in ("/", "/index.html", "/details"):
                html_path = Path(__file__).with_name("n02_v12_monitor.html")
                if path != "/details":
                    html_path = Path(__file__).with_name("n02_v12_training.html")
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            if path == "/api/status":
                self._json(200, status_payload(self._root()))
                return
            if path == "/api/training":
                self._json(200, training_payload(self._root()))
                return
            if path == "/api/history":
                query = self._query()
                if query.get("stage") is None:
                    run_dir, error = self._locate(query.get("run_id"))
                else:
                    run_dir, error = resolve_run_dir(
                        self._root(), query.get("stage"), query.get("run_id")
                    )
                if run_dir is None:
                    self._reject(error, query.get("run_id"))
                    return
                records = read_history(run_dir)
                self._json(200, records if records is not None else [])
                return
            if path == "/api/preview":
                self._serve_preview()
                return
            self._json(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except OSError:
            self._json(500, {"error": "internal error"})

    def _locate(self, run_id):
        """history 接口允许只给 run_id（在 stage 白名单里逐个找）。"""
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id) or run_id in (".", ".."):
            return None, "invalid run_id"
        for stage in STAGES:
            run_dir, error = resolve_run_dir(self._root(), stage, run_id)
            if run_dir is not None:
                return run_dir, None
        return None, "run not found"

    def _serve_demo_video(self):
        query = self._query()
        directory = demo_directory(self._root())
        arm, case_id = query.get("arm"), query.get("case_id")
        if directory is not None and arm in ("raw", "rgb"):
            for record in demo_rows(directory / arm / "episodes.jsonl"):
                if record["case_id"] != case_id or not record.get("video"):
                    continue
                video = Path(record["video"]).resolve()
                if directory in video.parents and video.suffix == ".mp4" and video.is_file():
                    self._video(video)
                    return
        self._json(404, {"error": "video not available"})

    def _serve_preview(self) -> None:
        query = self._query()
        run_id, relative = query.get("run_id"), query.get("file")
        run_dir, error = self._locate(run_id)
        if run_dir is None:
            self._reject(error, run_id)
            return
        if not isinstance(relative, str) or not relative.strip() or "\x00" in relative:
            self._reject("invalid file", run_id)
            return
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            self._reject("invalid file", run_id)
            return
        target = Path(os.path.realpath(run_dir / candidate))
        if target != run_dir and run_dir not in target.parents:
            self._reject("invalid file", run_id)
            return
        if target.suffix.lower() not in IMAGE_SUFFIXES or not target.is_file():
            self._reject("invalid file", run_id)
            return
        suffix = target.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }[suffix]
        self._send(200, target.read_bytes(), content_type)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        self._json(405, {"error": "read-only monitor: POST not allowed"})

    def log_message(self, fmt, *args) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="N02 v1.2 read-only monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--run-root", default=None)
    args = parser.parse_args()

    if args.run_root:
        os.environ["N02_V12_RUN_ROOT"] = args.run_root
    root = run_root()
    root.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        _STATE["server_started_at"] = utc_now()
        _STATE["port"] = args.port
    Handler.run_root_override = str(root)

    threading.Thread(target=sampler_loop, daemon=True).start()
    threading.Thread(target=deployment_loop, daemon=True).start()

    while True:
        try:
            httpd = ThreadingHTTPServer((args.host, args.port), Handler)
            break
        except OSError as exc:
            print(f"[monitor] bind {args.host}:{args.port} failed ({exc}); retry in 5s", flush=True)
            time.sleep(5)
    print(
        f"[monitor] serving http://{args.host}:{args.port}/ run_root={root} pid={os.getpid()}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
