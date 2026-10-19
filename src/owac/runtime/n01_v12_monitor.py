"""Read-only loopback progress service and writer library for the N01 v1.2 P3 runs.

The service is a single-page monitor for long training/evaluation loops that run on a
remote server with two GPUs (RAW queue on GPU0, RGB queue on GPU1). It is deliberately
small and decoupled from training:

* writers are called by training/evaluation workers and never block training;
* the HTTP service only reads files under ``--run-root`` and never touches torch, models,
  datasets or full logs;
* the page can be closed, the SSH session can drop, or the service can be restarted at
  any time without affecting a single training process.

Storage layout under the run root (nothing outside it is ever read or written)::

    monitor/workers/<role>.json     one file per worker, replaced atomically
    monitor/status.json             aggregated status produced by :func:`reconcile`
    monitor/metrics.jsonl           appended metric points, one JSON object per line
    monitor/logs/<role>.jsonl       appended worker log lines
    monitor/cases/index.json        case index ``{"artifacts": [...]}``
    monitor/artifacts/...           images/videos served by ``GET /artifacts/<path>``
    monitor/gpus.json               last GPU sample (background thread, ~8 s)
    monitor/overrides.json          optional orchestrator state/activity override

Step semantics: :meth:`RunStateWriter.metric` records whatever ``step`` the caller
passes, and callers must pass the real optimizer update step. A gradient-accumulation
micro-step is not an update step and must never be reported as one; passing a micro-step
counter silently rescales every reported curve.

Typical use::

    from owac.runtime.n01_v12_monitor import RunStateWriter

    with RunStateWriter(run_root, "libero-raw_n01-seed101", method="raw_n01",
                        input_domain="RAW", data_kind="hdr_derived",
                        platform="libero", seed=101, planned_steps=50_000,
                        gpu_index=0) as run:
        run.update(step=step, phase="training")
        run.metric("train_action_loss", step=step, value=loss, unit="loss",
                   label="训练 action loss")
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import datetime

try:  # The remote supervisor environment may still be the Python 3.10 policy env.
    from datetime import UTC
except ImportError:  # pragma: no cover - Python 3.10 only.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 - `timezone.utc` is the only 3.10 alias.

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

__all__ = [
    "ARTIFACTS_DIR",
    "CASES_RELATIVE",
    "DEFAULT_GPU_INTERVAL_S",
    "DEFAULT_HEARTBEAT_TIMEOUT_S",
    "DEFAULT_RECONCILE_INTERVAL_S",
    "GPUS_NAME",
    "GPU_QUERY_COMMAND",
    "LOG_TAIL_LINES",
    "MAX_POINTS_PER_SERIES",
    "METRICS_NAME",
    "MONITOR_DIR",
    "OVERRIDES_NAME",
    "STATUS_FRESH_S",
    "STATUS_NAME",
    "GpuSampler",
    "MonitorState",
    "ReconcileLoop",
    "RunStateWriter",
    "artifacts_dir",
    "atomic_write_json",
    "build_server",
    "downsample",
    "ensure_loopback",
    "iso_now",
    "logs_dir",
    "main",
    "monitor_dir",
    "read_json",
    "read_metrics",
    "reconcile",
    "sample_gpus",
    "serve",
    "tail_lines",
    "workers_dir",
    "write_overrides",
]

STAGE_DEFAULT = "P3"
MONITOR_HTML_NAME = "n01_v12_monitor.html"
MONITOR_DIR = "monitor"
WORKERS_DIR = "workers"
LOGS_DIR = "logs"
ARTIFACTS_DIR = "artifacts"
CASES_RELATIVE = Path("cases") / "index.json"
STATUS_NAME = "status.json"
METRICS_NAME = "metrics.jsonl"
GPUS_NAME = "gpus.json"
OVERRIDES_NAME = "overrides.json"

LOG_TAIL_LINES = 100
MAX_POINTS_PER_SERIES = 2000
MAX_METRICS_BYTES = 64 * 1024 * 1024
# The worker files are written once per *logged* step, so the staleness bound has to
# exceed (log_every x slowest step time) or a healthy run is reported as abnormally
# exited.  Measured: LIBERO logs every 20 steps at ~5.5 s/step (~110 s between writes) and
# RoboTwin at ~27 s/step (~540 s), against which the previous 90 s default flagged every
# gap -- the page reported a running LIBERO arm as "heartbeat expired, possibly abnormal
# exit" while it was training at 86% GPU utilisation, and a reader would have believed it.
DEFAULT_HEARTBEAT_TIMEOUT_S = 1800.0
DEFAULT_GPU_INTERVAL_S = 8.0
DEFAULT_RECONCILE_INTERVAL_S = 5.0
STATUS_FRESH_S = 30.0
DEV_LOOP_BUDGET = 100
CHUNK_BYTES = 64 * 1024
SPEED_IDLE_S = 120.0

GPU_QUERY_COMMAND = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit",
    "--format=csv,noheader,nounits",
]

CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}

STATE_TEXT = {
    "implementing": "编程中",
    "checking": "检查中",
    "waiting_resource": "等待资源",
    "training": "训练中",
    "evaluating": "评测中",
    "stage_end": "阶段结束",
    "failed": "失败",
    "needs_user": "需要用户处理",
}

NORMAL_EXIT_REASONS = frozenset({"completed", "done", "success", "stage_end", "finished"})
RESERVED_KEYS = frozenset({"role", "started_at", "recent_log", "log_path", "monitor_dir"})
NUMERIC_FIELDS = frozenset(
    {"step", "planned_steps", "seed", "gpu_index", "pid", "expected_rollouts", "finished"}
)
_SNAPSHOT_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def iso_now() -> str:
    """Return the current UTC time as ``2026-10-15T09:51:05Z`` (the format the page parses)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_from(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp from the monitor files; ``None`` when unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def monitor_dir(run_root: str | os.PathLike[str]) -> Path:
    return Path(run_root) / MONITOR_DIR


def workers_dir(run_root: str | os.PathLike[str]) -> Path:
    return monitor_dir(run_root) / WORKERS_DIR


def logs_dir(run_root: str | os.PathLike[str]) -> Path:
    return monitor_dir(run_root) / LOGS_DIR


def artifacts_dir(run_root: str | os.PathLike[str]) -> Path:
    return monitor_dir(run_root) / ARTIFACTS_DIR


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Read one small JSON file; never raises (missing/corrupt files yield ``default``)."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


def _sanitize(value: Any) -> Any:
    """Make a value JSON-safe: non-finite floats become null, scalars become plain types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _iso_from(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    if isinstance(value, int):
        return value
    return str(value)


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Write ``payload`` to a sibling temporary file and replace the target atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_sanitize(payload), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)


def append_jsonl(path: str | os.PathLike[str], payload: Any) -> None:
    """Append one JSON line with a single ``O_APPEND`` write (safe across workers)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_sanitize(payload), ensure_ascii=False, allow_nan=False) + "\n"
    handle = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(handle, line.encode("utf-8"))
    finally:
        os.close(handle)


def tail_lines(path: str | os.PathLike[str], tail: int, *, block_bytes: int = 8192) -> list[str]:
    """Read only the last ``tail`` lines of a file, seeking from the end (never whole-file)."""
    if tail <= 0:
        return []
    source = Path(path)
    data = b""
    position = 0
    try:
        with source.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            while position > 0 and data.count(b"\n") <= tail:
                step = min(block_bytes, position)
                position -= step
                stream.seek(position)
                data = stream.read(step) + data
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if position > 0 and lines:
        lines = lines[1:]
    return lines[-tail:]


def _iter_jsonl(
    path: str | os.PathLike[str], *, max_bytes: int = MAX_METRICS_BYTES
) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, reading at most the trailing ``max_bytes``."""
    source = Path(path)
    try:
        with source.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if size > max_bytes:
                stream.seek(size - max_bytes)
                stream.readline()
            else:
                stream.seek(0)
            for raw in stream:
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except OSError:
        return


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value)) if math.isfinite(value) else None
    text = re.sub(r"[^0-9.+-]", "", str(value))
    if text in {"", "-", "+", ".", "-.", "+."}:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_role(role: str) -> str:
    text = str(role).strip()
    if not text or text in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", text):
        raise ValueError(f"role 必须是 [A-Za-z0-9._-]+ 的短标识，收到 {role!r}")
    if text.startswith("."):
        raise ValueError(f"role 不能以点开头：{role!r}")
    return text


def ensure_loopback(host: str) -> str:
    """Return a loopback bind host, or raise ``ValueError`` with a Chinese explanation."""
    text = str(host).strip()
    if not text:
        raise ValueError("拒绝启动：绑定地址为空；本服务只允许绑定 loopback（127.0.0.1 / ::1）")
    if text.lower() in {"localhost", "localhost.localdomain"}:
        return text.lower()
    try:
        address = ipaddress.ip_address(text.strip("[]"))
    except ValueError as error:
        raise ValueError(
            f"拒绝启动：无法解析绑定地址 {host!r}；本服务只允许 127.0.0.1 / ::1 / localhost"
        ) from error
    if not address.is_loopback:
        raise ValueError(
            f"拒绝启动：{host} 不是 loopback 地址。本页只读且按用户要求不对外公开，"
            "请使用 127.0.0.1 或 ::1（不要绑定 0.0.0.0）"
        )
    return text


# --------------------------------------------------------------------------------------
# writing side
# --------------------------------------------------------------------------------------


def _warn(message: str, strict: bool) -> None:
    """Monitoring must never take a training process down: warn, or raise in strict mode."""
    if strict:
        raise RuntimeError(message)
    print(f"[n01-v12-monitor] {message}", file=sys.stderr, flush=True)


class RunStateWriter:
    """Progress writer owned by one training/evaluation worker.

    Every worker writes its own ``monitor/workers/<role>.json`` through a temporary file
    plus :func:`os.replace`, so concurrent workers never contend on one JSON document.

    Each call is fail-soft by default (:meth:`update`, :meth:`log`, :meth:`metric`,
    :meth:`rollouts`): a monitoring error prints one line to stderr and returns instead of
    interrupting training. Pass ``strict=True`` to raise instead (useful in tests).

    Callers must heartbeat: call :meth:`update` (or :meth:`heartbeat`) at least every
    ``heartbeat_timeout_s / 3`` seconds, otherwise :func:`reconcile` marks the run as
    suspicious instead of reporting it as running. ``pid`` in the written file is always
    :func:`os.getpid` of the process that performed the write.
    """

    def __init__(
        self,
        run_root: str | os.PathLike[str],
        role: str,
        *,
        method: str | None = None,
        input_domain: str | None = None,
        data_kind: str | None = None,
        platform: str | None = None,
        seed: int | None = None,
        planned_steps: int | None = None,
        stage: str = STAGE_DEFAULT,
        state: str = "training",
        phase: str = "training",
        activity: str | None = None,
        expected_rollouts: int | None = None,
        gpu_index: int | None = None,
        strict: bool = False,
    ) -> None:
        self.run_root = Path(run_root)
        self.role = _safe_role(role)
        self.strict = strict
        self._lock = threading.Lock()
        self.worker_path = workers_dir(self.run_root) / f"{self.role}.json"
        self.log_path = logs_dir(self.run_root) / f"{self.role}.jsonl"
        existing = read_json(self.worker_path, {})
        existing = existing if isinstance(existing, dict) else {}
        self._snapshot: dict[str, Any] = {
            "role": self.role,
            "method": method or existing.get("method"),
            "input_domain": input_domain or existing.get("input_domain"),
            "data_kind": data_kind or existing.get("data_kind"),
            "platform": platform or existing.get("platform"),
            "seed": seed if seed is not None else existing.get("seed"),
            "planned_steps": (
                planned_steps if planned_steps is not None else existing.get("planned_steps")
            ),
            "expected_rollouts": (
                expected_rollouts
                if expected_rollouts is not None
                else existing.get("expected_rollouts")
            ),
            "gpu_index": gpu_index if gpu_index is not None else existing.get("gpu_index"),
            "stage": stage or existing.get("stage") or STAGE_DEFAULT,
            "state": state,
            "phase": phase,
            "activity": activity,
            "step": _as_int(existing.get("step")),
            "speed": _as_float(existing.get("speed")),
            "speed_unit": "step/s",
            "alive": True,
            "finished": False,
            "exit_reason": None,
            "started_at": existing.get("started_at") or iso_now(),
            "updated_at": iso_now(),
            "finished_at": None,
            "rollouts": existing.get("rollouts")
            if isinstance(existing.get("rollouts"), dict)
            else None,
            "recent_log": existing.get("recent_log")
            if isinstance(existing.get("recent_log"), list)
            else [],
        }
        self._speed_step: int | None = self._snapshot["step"]
        self._speed_at = time.monotonic()
        self._flush()

    # -- context manager ---------------------------------------------------------

    def __enter__(self) -> RunStateWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            self.finish(exit_reason=f"异常退出：{exc_type.__name__}", state="failed")
        elif not self._snapshot.get("finished"):
            self.finish()
        return None

    # -- writes ------------------------------------------------------------------

    def update(self, **fields: Any) -> dict[str, Any]:
        """Merge fields into the worker snapshot and write it atomically.

        Common fields: ``step`` (real optimizer step), ``phase``, ``state``, ``activity``,
        ``planned_steps``, plus any extra JSON-safe key the page may read.
        """
        try:
            with self._lock:
                step = _as_int(fields.get("step"))
                if "step" in fields and step is not None:
                    self._note_step(step)
                for key, value in fields.items():
                    if key in RESERVED_KEYS:
                        _warn(
                            f"忽略受保护字段 {key!r}（role/started_at 由 writer 维护）", self.strict
                        )
                        continue
                    if key in NUMERIC_FIELDS and value is not None:
                        self._snapshot[key] = _as_int(value)
                    else:
                        self._snapshot[key] = _sanitize(value)
                self._touch_locked()
                self._flush_locked()
            return self.snapshot()
        except Exception as error:  # noqa: BLE001 - monitoring must never break training
            _warn(f"{self.role} update 失败：{error}", self.strict)
            return self.snapshot()

    def heartbeat(self, **fields: Any) -> dict[str, Any]:
        """Alias of :meth:`update` for periodic liveness writes."""
        return self.update(**fields)

    def log(self, message: str, level: str = "info") -> None:
        """Append one plain-text log line and keep the last lines inside the snapshot."""
        try:
            entry = {
                "ts": iso_now(),
                "level": str(level or "info"),
                "role": self.role,
                "message": str(message),
            }
            with self._lock:
                append_jsonl(self.log_path, entry)
                kept = list(self._snapshot.get("recent_log") or [])
                kept.append(entry)
                self._snapshot["recent_log"] = kept[-LOG_TAIL_LINES:]
                self._touch_locked()
                self._flush_locked()
        except Exception as error:  # noqa: BLE001 - see class docstring
            _warn(f"{self.role} log 失败：{error}", self.strict)

    def metric(
        self,
        name: str,
        step: int | None,
        value: float,
        *,
        unit: str = "",
        label: str | None = None,
        t: float | None = None,
    ) -> None:
        """Append one metric point to ``monitor/metrics.jsonl``.

        ``step`` is the caller's real optimizer update step. A gradient-accumulation
        micro-step is *not* an update step: reporting the micro-step counter rescales the
        whole curve and must be avoided.
        """
        try:
            series = str(name or "").strip()
            number = _as_float(value)
            index = _as_int(step)
            if not series:
                _warn("metric 名称不能为空，已忽略该点", self.strict)
                return
            if number is None:
                _warn(f"metric {series} 的 value 不是有限数，已忽略该点", self.strict)
                return
            entry = {
                "name": series,
                # Without the role the arms are indistinguishable in the stream: the raw and
                # rgb runs write the same metric names to the same file from two processes,
                # so a curve of `val_action_loss` silently averaged two models (measured: 12
                # validation points, every row with no role, two values per step).
                "role": str(self.role) if self.role else "",
                "label": str(label or series),
                "unit": str(unit or ""),
                "step": index,
                "value": number,
                "t": int(t) if t is not None else int(time.time()),
            }
            append_jsonl(monitor_dir(self.run_root) / METRICS_NAME, entry)
        except Exception as error:  # noqa: BLE001 - see class docstring
            _warn(f"{self.role} metric 失败：{error}", self.strict)

    def rollouts(
        self,
        completed: int | None,
        expected: int | None = None,
        successes: int | None = None,
        failures: int | None = None,
        infra_pending: int | None = None,
        note: str | None = None,
    ) -> None:
        """Record development-loop rollout counts (``None`` means "not measured yet")."""
        payload = {
            "completed_rollouts": _as_int(completed),
            "expected_rollouts": _as_int(expected),
            "successes": _as_int(successes),
            "failures": _as_int(failures),
            "infra_pending": _as_int(infra_pending),
            "note": None if note is None else str(note),
            "updated_at": iso_now(),
        }
        self.update(rollouts=payload, expected_rollouts=_as_int(expected))

    def finish(self, exit_reason: str | None = None, state: str | None = None) -> dict[str, Any]:
        """Mark the run finished; the summary then reports ``alive: false`` with a reason."""
        try:
            with self._lock:
                self._snapshot["alive"] = False
                self._snapshot["finished"] = True
                self._snapshot["state"] = state or "stage_end"
                self._snapshot["exit_reason"] = exit_reason or "completed"
                self._snapshot["finished_at"] = iso_now()
                self._snapshot["speed"] = 0.0
                self._touch_locked()
                self._flush_locked()
        except Exception as error:  # noqa: BLE001 - see class docstring
            _warn(f"{self.role} finish 失败：{error}", self.strict)
        return self.snapshot()

    # -- reads -------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._snapshot, ensure_ascii=False))

    # -- internals ---------------------------------------------------------------

    def _note_step(self, step: int) -> None:
        now = time.monotonic()
        elapsed = now - self._speed_at
        if elapsed > SPEED_IDLE_S:
            self._snapshot["speed"] = 0.0
            self._speed_step, self._speed_at = step, now
        elif elapsed >= 1.0:
            if self._speed_step is not None and step >= self._speed_step:
                self._snapshot["speed"] = (step - self._speed_step) / elapsed
            self._speed_step, self._speed_at = step, now
        if self._snapshot.get("step") is None or step >= int(self._snapshot["step"]):
            self._snapshot["step"] = step

    def _touch_locked(self) -> None:
        self._snapshot["updated_at"] = iso_now()
        self._snapshot["pid"] = os.getpid()
        self._snapshot["log_path"] = str(self.log_path)

    def _flush_locked(self) -> None:
        atomic_write_json(self.worker_path, self._snapshot)

    def _flush(self) -> None:
        try:
            with self._lock:
                self._touch_locked()
                self._flush_locked()
        except Exception as error:  # noqa: BLE001 - see class docstring
            _warn(f"{self.role} 首次写入失败：{error}", self.strict)


def write_overrides(
    run_root: str | os.PathLike[str],
    *,
    stage: str | None = None,
    state: str | None = None,
    activity: str | None = None,
    tasks: list[dict[str, Any]] | None = None,
    p4_total: int | None = None,
    p4_extra: int | None = None,
    merge: bool = True,
) -> dict[str, Any]:
    """Record orchestrator-owned status fields that cannot be derived from worker files.

    The page reads ``stage``/``state``/``activity`` from the aggregated status; states such
    as ``implementing``, ``checking``, ``waiting_resource`` or ``needs_user`` describe the
    conductor, not a worker, so the conductor writes them here. Passing ``merge=False``
    replaces the whole override document.
    """
    path = monitor_dir(run_root) / OVERRIDES_NAME
    payload = read_json(path, {}) if merge else {}
    payload = payload if isinstance(payload, dict) else {}
    fields = {
        "stage": stage,
        "state": state,
        "activity": activity,
        "tasks": tasks,
        "p4_total": p4_total,
        "p4_extra": p4_extra,
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    atomic_write_json(path, payload)
    return payload


# --------------------------------------------------------------------------------------
# GPU sampling (background thread only; never called per optimizer step)
# --------------------------------------------------------------------------------------


def _parse_gpu_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        return None
    index = _as_int(parts[0])
    extended = len(parts) >= 7 and _as_int(parts[-5]) is not None
    fields = 5 if extended else 3
    used = _as_int(parts[-fields])
    total = _as_int(parts[-fields + 1])
    if index is None or used is None or total is None:
        return None
    return {
        "index": index,
        "name": ",".join(parts[1:-fields]).strip(),
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_pct": _as_int(parts[-fields + 2]),
        **(
            {"power_w": _as_float(parts[-2]), "power_limit_w": _as_float(parts[-1])}
            if extended
            else {}
        ),
    }


def sample_gpus(*, timeout_s: float = 10.0, command: list[str] | None = None) -> dict[str, Any]:
    """Sample physical GPUs once; a missing ``nvidia-smi`` degrades to an empty list."""
    sampled_at = iso_now()
    argv = list(command or GPU_QUERY_COMMAND)
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except FileNotFoundError:
        return {
            "gpus": [],
            "sampled_at": sampled_at,
            "error": "未找到 nvidia-smi：GPU 采样已降级为空列表（不影响服务与训练）",
            "unavailable": True,
        }
    except subprocess.TimeoutExpired:
        return {
            "gpus": [],
            "sampled_at": sampled_at,
            "error": f"nvidia-smi 超时（>{timeout_s:g}s）：本次采样跳过",
            "unavailable": True,
        }
    except OSError as error:
        return {
            "gpus": [],
            "sampled_at": sampled_at,
            "error": f"nvidia-smi 无法执行：{error}",
            "unavailable": True,
        }
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else "无错误输出"
        return {
            "gpus": [],
            "sampled_at": sampled_at,
            "error": f"nvidia-smi 返回码 {result.returncode}：{reason}",
            "unavailable": True,
        }
    gpus = []
    for line in result.stdout.splitlines():
        parsed = _parse_gpu_line(line)
        if parsed is not None:
            gpus.append(parsed)
    return {"gpus": gpus, "sampled_at": sampled_at, "error": None, "unavailable": False}


class GpuSampler(threading.Thread):
    """Background thread that samples GPUs every few seconds and caches them on disk."""

    def __init__(
        self,
        run_root: str | os.PathLike[str],
        *,
        interval_s: float = DEFAULT_GPU_INTERVAL_S,
        timeout_s: float = 10.0,
        command: list[str] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__(name="n01-monitor-gpu", daemon=True)
        self.run_root = Path(run_root)
        self.interval_s = max(1.0, float(interval_s))
        self.timeout_s = float(timeout_s)
        self.command = command
        self.strict = strict
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last: dict[str, Any] = {"gpus": [], "sampled_at": None, "error": None}

    def run(self) -> None:
        while True:
            try:
                self.sample_once()
            except Exception as error:  # noqa: BLE001 - a sampler must not kill the service
                _warn(f"GPU 采样失败：{error}", self.strict)
            if self._stop.wait(self.interval_s):
                return

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout_s)

    def sample_once(self) -> dict[str, Any]:
        payload = sample_gpus(timeout_s=self.timeout_s, command=self.command)
        with self._lock:
            self.last = payload
        atomic_write_json(monitor_dir(self.run_root) / GPUS_NAME, payload)
        return payload

    def cached(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.last)


def read_gpu_cache(run_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the cached GPU sample written by :class:`GpuSampler` (empty when absent)."""
    payload = read_json(monitor_dir(run_root) / GPUS_NAME, {})
    if not isinstance(payload, dict):
        return {"gpus": [], "sampled_at": None, "error": None}
    gpus = payload.get("gpus")
    return {
        "gpus": gpus if isinstance(gpus, list) else [],
        "sampled_at": payload.get("sampled_at"),
        "error": payload.get("error"),
    }


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def downsample(
    points: list[dict[str, Any]], limit: int = MAX_POINTS_PER_SERIES
) -> list[dict[str, Any]]:
    """Thin a point list uniformly to at most ``limit`` points, keeping first and last."""
    total = len(points)
    if limit <= 0 or total <= limit:
        return list(points)
    stride = total / float(limit)
    thinned = [points[min(total - 1, int(index * stride))] for index in range(limit)]
    thinned[0] = points[0]
    thinned[-1] = points[-1]
    return thinned


def read_metrics(
    run_root: str | os.PathLike[str], *, limit: int = MAX_POINTS_PER_SERIES
) -> dict[str, Any]:
    """Aggregate ``monitor/metrics.jsonl`` into the ``{"series": [...]}`` chart shape.

    Series are keyed by ``(name, role)``, not by name alone.  Two arms write the same metric
    names to this one file from two processes, so keying by name merged the raw and rgb
    curves into a single line whose points alternated between two different models.
    """

    path = monitor_dir(run_root) / METRICS_NAME
    series: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        name = str(row.get("name") or "").strip()
        value = _as_float(row.get("value"))
        if not name or value is None:
            continue
        # Rows written before the role was recorded are kept but labelled as unattributed
        # rather than folded into an arm: they cannot be assigned to one, and guessing
        # would be exactly the conflation this keying removes.
        role = str(row.get("role") or "").strip()
        key = (name, role)
        entry = series.get(key)
        if entry is None:
            label = str(row.get("label") or name)
            entry = {
                "name": name,
                "role": role or None,
                "label": f"{label} [{role}]" if role else f"{label} [未标注 arm]",
                "unit": str(row.get("unit") or ""),
                "points": [],
            }
            series[key] = entry
        # The row label is NOT copied over the series label here: doing so overwrote the
        # "[role]" suffix added above, leaving two series with identical names in the
        # legend -- which is the visual form of the very conflation this keying removes.
        if row.get("unit"):
            entry["unit"] = str(row["unit"])
        entry["points"].append(
            {"step": _as_int(row.get("step")), "value": value, "t": _as_int(row.get("t"))}
        )
    for entry in series.values():
        points = sorted(
            entry["points"],
            key=lambda point: (
                point["step"] is None,
                point["step"] if point["step"] is not None else 0,
                point["t"] if point["t"] is not None else 0,
            ),
        )
        entry["points"] = downsample(points, limit)
        entry["point_count"] = len(points)
        entry["downsampled"] = len(entry["points"]) < len(points)
    return {"series": list(series.values())}


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------


def _run_entry(payload: dict[str, Any], *, now: datetime, timeout_s: float) -> dict[str, Any]:
    updated = parse_iso(payload.get("updated_at"))
    age = (now - updated).total_seconds() if updated is not None else None
    finished = bool(payload.get("finished"))
    fresh = age is not None and age <= timeout_s
    alive = (not finished) and fresh
    suspicious = (not finished) and not fresh
    exit_reason = payload.get("exit_reason")
    if suspicious and not exit_reason:
        expired = "未知时间" if age is None else f"{age:.0f} 秒"
        exit_reason = f"心跳过期（{expired}未更新，worker 未调用 finish，可能异常退出）"
    started = parse_iso(payload.get("started_at"))
    finished_at = parse_iso(payload.get("finished_at"))
    if finished and started is not None and finished_at is not None:
        elapsed = (finished_at - started).total_seconds()
    elif started is not None:
        elapsed = (now - started).total_seconds()
    else:
        elapsed = None
    rollouts = payload.get("rollouts")
    rollouts = rollouts if isinstance(rollouts, dict) else None
    return {
        **{
            key: payload.get(key)
            for key in (
                "step_seconds",
                "samples_per_second",
                "data_wait_fraction",
                "execution",
                "action_loss",
                "motion_loss",
                "validation_loss",
                "validation_step",
                "baseline_validation_loss",
                "history_accepted_fraction",
                "peak_gpu_gib",
                "step_progress_at",
                "checkpoint_step",
                "checkpoint_at",
                "activity",
            )
        },
        "role": payload.get("role"),
        "method": payload.get("method"),
        "input_domain": payload.get("input_domain"),
        "data_kind": payload.get("data_kind"),
        "platform": payload.get("platform"),
        "seed": _as_int(payload.get("seed")),
        "pid": _as_int(payload.get("pid")),
        "gpu_index": _as_int(payload.get("gpu_index")),
        "alive": alive,
        "finished": finished,
        "suspicious": suspicious,
        "phase": payload.get("phase"),
        "state": payload.get("state"),
        "step": _as_int(payload.get("step")),
        "planned_steps": _as_int(payload.get("planned_steps")),
        "started_at": payload.get("started_at"),
        "updated_at": payload.get("updated_at"),
        "finished_at": payload.get("finished_at"),
        "exit_reason": exit_reason,
        "heartbeat_age_s": None if age is None else round(age, 3),
        "elapsed_s": None if elapsed is None else round(max(0.0, elapsed), 3),
        "speed": _as_float(payload.get("speed")) if alive else 0.0,
        "speed_unit": payload.get("speed_unit") or "step/s",
        "rollouts": rollouts,
        "recent_log": [
            entry for entry in (payload.get("recent_log") or []) if isinstance(entry, dict)
        ][-LOG_TAIL_LINES:],
    }


def _aggregate(values: list[int | None], mode: str) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) if mode == "sum" else max(present)


def _dev_loop(runs: list[dict[str, Any]], budget: int, mode: str) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for run in runs:
        method = str(run.get("method") or run.get("role") or "").strip()
        if not method:
            continue
        group = groups.setdefault(method, {"method": method, "runs": [], "platforms": []})
        group["runs"].append(run)
        platform = str(run.get("platform") or "").strip()
        if platform and platform not in group["platforms"]:
            group["platforms"].append(platform)
    by_method = []
    for group in groups.values():
        rollouts = [run.get("rollouts") or {} for run in group["runs"]]
        by_method.append(
            {
                "method": group["method"],
                "completed_rollouts": _aggregate(
                    [_as_int(item.get("completed_rollouts")) for item in rollouts], mode
                ),
                "expected_rollouts": _aggregate(
                    [_as_int(item.get("expected_rollouts")) for item in rollouts], mode
                ),
                "successes": _aggregate(
                    [_as_int(item.get("successes")) for item in rollouts], mode
                ),
                "failures": _aggregate([_as_int(item.get("failures")) for item in rollouts], mode),
                "infra_pending": _aggregate(
                    [_as_int(item.get("infra_pending")) for item in rollouts], mode
                ),
                "platforms": group["platforms"],
                "roles": [run.get("role") for run in group["runs"]],
            }
        )
    by_method.sort(key=lambda item: str(item["method"]))
    completed = _aggregate([item["completed_rollouts"] for item in by_method], mode)
    note = (
        "completed_rollouts 为各 method 的汇总，"
        + (
            "按 method 相加（适用于各 worker 各自占用开发闭环分区）"
            if mode == "sum"
            else "取最大（保守，不把并行 worker 相加成双倍）"
        )
        + f"；开发闭环总预算 budget_rollouts={budget}。"
    )
    return {
        "completed_rollouts": completed,
        "budget_rollouts": budget,
        "by_method": by_method,
        "aggregate": mode,
        "note": note,
    }


def _annotate_gpus(gpus: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add queue/job labels only where a worker declared its own ``gpu_index``."""
    annotated = [dict(gpu) for gpu in gpus if isinstance(gpu, dict)]
    for gpu in annotated:
        index = _as_int(gpu.get("index"))
        if index is None:
            continue
        for run in runs:
            if _as_int(run.get("gpu_index")) != index:
                continue
            if not gpu.get("queue") and run.get("input_domain"):
                gpu["queue"] = run.get("input_domain")
            if not gpu.get("job") and run.get("role"):
                gpu["job"] = run.get("role")
            break
    return annotated


def _artifact_url(run_root: Path, item: dict[str, Any]) -> str:
    raw = str(item.get("url") or "").strip()
    if not raw or raw.startswith(("http://", "https://", "//", "data:")):
        return raw
    base = artifacts_dir(run_root).resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
        except OSError:
            return raw
        if resolved.is_relative_to(base):
            return f"{ARTIFACTS_DIR}/{resolved.relative_to(base).as_posix()}"
        return raw
    return raw.lstrip("/")


def read_case_index(run_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read ``monitor/cases/index.json`` and normalize artifact URLs for the page."""
    root = Path(run_root)
    payload = read_json(monitor_dir(root) / CASES_RELATIVE, {})
    if isinstance(payload, dict):
        raw_items = payload.get("artifacts")
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = None
    items = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        artifact = {str(key): _sanitize(value) for key, value in item.items()}
        artifact["url"] = _artifact_url(root, artifact)
        items.append(artifact)
    return items


def _resolve_state(runs: list[dict[str, Any]], override: str | None) -> str | None:
    if override:
        return override
    if not runs:
        return None
    alive = [run for run in runs if run["alive"]]
    failed = [
        run
        for run in runs
        if run.get("state") == "failed"
        or (not run["alive"] and run.get("exit_reason") not in NORMAL_EXIT_REASONS)
    ]
    if failed and not alive:
        return "failed"
    if alive:
        states = {str(run.get("state")) for run in alive if run.get("state")}
        if states == {"evaluating"}:
            return "evaluating"
        if "training" in states:
            return "training"
        return sorted(states)[0] if states else None
    return "stage_end"


def _pick_current(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    alive = [run for run in runs if run["alive"]]
    pool = alive or runs
    ordered = sorted(pool, key=lambda run: (str(run.get("updated_at") or ""), str(run.get("role"))))
    current = dict(ordered[-1])
    if not alive:
        current["note"] = "当前没有心跳正常的 worker，此处显示最近一次 run（已停止）"
    return current


def _activity_for(
    runs: list[dict[str, Any]], current: dict[str, Any] | None, state: str | None
) -> str:
    if not runs:
        return "尚未收到任何 worker 心跳（monitor/workers 为空，等待训练或评测进程登记）"
    alive = [run for run in runs if run["alive"]]
    suspicious = [run for run in runs if run.get("suspicious")]
    if not alive and suspicious:
        names = "、".join(str(run.get("role")) for run in suspicious)
        return f"所有 worker 心跳均已过期：{names}；未调用 finish，可能异常退出，请核对进程"
    if current is None:
        return STATE_TEXT.get(state or "", "状态未登记")
    method = current.get("method") or current.get("role") or "未知 run"
    domain = current.get("input_domain") or "输入域未登记"
    step = current.get("step")
    step_text = f"，step {step}" if step is not None else ""
    if current.get("alive"):
        return f"{method}（{domain}）正在{STATE_TEXT.get(state or '', state or '运行')}{step_text}"
    return f"{method}（{domain}）已停止（{current.get('exit_reason') or '原因未登记'}）"


def reconcile(
    run_root: str | os.PathLike[str],
    *,
    stage: str = STAGE_DEFAULT,
    heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
    dev_loop_budget: int = DEV_LOOP_BUDGET,
    dev_loop_aggregate: str = "max",
    gpus: list[dict[str, Any]] | None = None,
    log_tail: int = LOG_TAIL_LINES,
    write: bool = True,
) -> dict[str, Any]:
    """Merge all worker files, the case index and the GPU cache into the status document.

    This is the only place that produces ``monitor/status.json``. It reads small JSON
    documents only (never metrics, never full logs) so it is safe to run from a cron job,
    from the service loop, or by hand. Returns the same dict it writes.
    """
    root = Path(run_root)
    now = datetime.now(UTC)
    overrides = read_json(monitor_dir(root) / OVERRIDES_NAME, {})
    overrides = overrides if isinstance(overrides, dict) else {}
    worker_payloads = []
    for path in sorted(workers_dir(root).glob("*.json")):
        payload = read_json(path, None)
        if isinstance(payload, dict):
            worker_payloads.append((path, payload))
    runs = [
        _run_entry(payload, now=now, timeout_s=heartbeat_timeout_s)
        for _, payload in worker_payloads
    ]
    mode = "sum" if str(dev_loop_aggregate).lower() == "sum" else "max"
    cache = read_gpu_cache(root)
    if gpus is None:
        gpus = cache["gpus"]
    recent_log = []
    for _, payload in worker_payloads:
        recent_log.extend(
            entry for entry in (payload.get("recent_log") or []) if isinstance(entry, dict)
        )
    recent_log.sort(key=lambda entry: str(entry.get("ts") or ""))
    current = _pick_current(runs)
    state_override = overrides.get("state")
    resolved_state = _resolve_state(runs, str(state_override) if state_override else None)
    activity = overrides.get("activity") or _activity_for(runs, current, resolved_state)
    status = {
        "stage": str(overrides.get("stage") or stage or STAGE_DEFAULT),
        "state": resolved_state,
        "activity": str(activity),
        "updated_at": iso_now(),
        "current_run": current,
        "runs": runs,
        "resources": {
            "gpus": _annotate_gpus(list(gpus), runs),
            "gpu_sampled_at": cache["sampled_at"] if gpus else None,
            "dev_loop": _dev_loop(runs, dev_loop_budget, mode),
        },
        "artifacts": read_case_index(root),
        "recent_log": recent_log[-log_tail:],
        "closed_loop_probe": {
            role: payload
            for role in ("raw", "rgb")
            if (payload := read_json(monitor_dir(root) / f"probe20k-{role}.json", {}))
        },
    }
    for key in ("tasks", "p4_total", "p4_extra", "training_plan"):
        if key in overrides:
            status[key] = overrides[key]
    if write:
        atomic_write_json(monitor_dir(root) / STATUS_NAME, status)
    return status


# --------------------------------------------------------------------------------------
# read model shared by the HTTP handler
# --------------------------------------------------------------------------------------


class MonitorState:
    """Disk-backed read model. Nothing is cached across restarts beyond file mtimes."""

    def __init__(
        self,
        run_root: str | os.PathLike[str],
        *,
        stage: str = STAGE_DEFAULT,
        html_path: str | os.PathLike[str] | None = None,
        heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
        dev_loop_budget: int = DEV_LOOP_BUDGET,
        dev_loop_aggregate: str = "max",
        log_tail: int = LOG_TAIL_LINES,
        status_fresh_s: float = STATUS_FRESH_S,
        metrics_limit: int = MAX_POINTS_PER_SERIES,
    ) -> None:
        self.run_root = Path(run_root)
        self.stage = stage
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.dev_loop_budget = dev_loop_budget
        self.dev_loop_aggregate = dev_loop_aggregate
        self.log_tail = log_tail
        self.status_fresh_s = status_fresh_s
        self.metrics_limit = metrics_limit
        self.status_path = monitor_dir(self.run_root) / STATUS_NAME
        self.html_file = (
            Path(html_path)
            if html_path is not None
            else Path(__file__).with_name(MONITOR_HTML_NAME)
        )
        self._lock = threading.Lock()

    def refresh(self) -> dict[str, Any]:
        """Force one aggregation and write ``status.json``."""
        with self._lock:
            return reconcile(
                self.run_root,
                stage=self.stage,
                heartbeat_timeout_s=self.heartbeat_timeout_s,
                dev_loop_budget=self.dev_loop_budget,
                dev_loop_aggregate=self.dev_loop_aggregate,
                log_tail=self.log_tail,
            )

    def status(self) -> dict[str, Any]:
        """Serve the aggregated status; aggregate on demand when the cache is missing/stale."""
        try:
            age = time.time() - self.status_path.stat().st_mtime
        except OSError:
            age = None
        if age is not None and age <= self.status_fresh_s:
            payload = read_json(self.status_path, None)
            if isinstance(payload, dict):
                return payload
        return self.refresh()

    def metrics(self) -> dict[str, Any]:
        return read_metrics(self.run_root, limit=self.metrics_limit)

    def artifacts(self) -> dict[str, Any]:
        return {"artifacts": read_case_index(self.run_root)}

    def logs(self, tail: int | None = None) -> dict[str, Any]:
        count = max(1, min(int(tail if tail is not None else self.log_tail), 2000))
        directory = logs_dir(self.run_root)
        entries: list[dict[str, Any]] = []
        sources: list[str] = []
        if directory.is_dir():
            for path in sorted(directory.glob("*.jsonl")):
                sources.append(path.name)
                for line in tail_lines(path, count):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except ValueError:
                        payload = None
                    if not isinstance(payload, dict):
                        entries.append(
                            {"ts": None, "level": "info", "message": text, "role": path.stem}
                        )
                        continue
                    entries.append(
                        {
                            "ts": payload.get("ts"),
                            "level": payload.get("level") or "info",
                            "message": str(payload.get("message") or ""),
                            "role": payload.get("role") or path.stem,
                        }
                    )
        entries.sort(key=lambda entry: str(entry.get("ts") or ""))
        return {"logs": entries[-count:], "tail": count, "sources": sources}


# --------------------------------------------------------------------------------------
# HTTP service
# --------------------------------------------------------------------------------------


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    text = header.strip().lower()
    if not text.startswith("bytes=") or "," in text:
        return None
    spec = text[len("bytes=") :].strip()
    if "-" not in spec:
        return None
    first, _, last = spec.partition("-")
    try:
        if not first:
            length = int(last)
            if length <= 0 or size <= 0:
                return None
            return max(0, size - length), size - 1
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size:
        return None
    end = min(end, size - 1)
    if end < start:
        return None
    return start, end


def _resolve_inside(base: Path, relative: str) -> Path | None:
    """Resolve ``relative`` under ``base``; ``None`` when it escapes (path traversal)."""
    if not relative or "\x00" in relative:
        return None
    candidate = Path(relative)
    if candidate.is_absolute():
        return None
    try:
        base_resolved = base.resolve()
        resolved = (base_resolved / candidate).resolve()
    except OSError:
        return None
    if resolved == base_resolved or not resolved.is_relative_to(base_resolved):
        return None
    return resolved


class MonitorRequestHandler(BaseHTTPRequestHandler):
    """Read-only request handler. It only ever reads files below the run root."""

    monitor_state: MonitorState
    server_version = "N01V12Monitor/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._responded = False
        try:
            self._route(urlsplit(self.path).path, parse_qs(urlsplit(self.path).query))
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception as error:  # noqa: BLE001 - never leak a traceback to the browser
            if not getattr(self, "_responded", False):
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"服务内部错误：{error}"}
                )

    do_HEAD = do_GET

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[n01-v12-monitor] {self.address_string()} - {fmt % args}\n")

    # -- routing -----------------------------------------------------------------

    def _route(self, path: str, query: dict[str, list[str]]) -> None:
        state = self.monitor_state
        if path in {"/", "/index.html"}:
            self._serve_html()
        elif path == "/api/status":
            self._send_json(HTTPStatus.OK, state.status())
        elif path == "/api/metrics":
            self._send_json(HTTPStatus.OK, state.metrics(), pretty=False)
        elif path == "/api/artifacts":
            self._send_json(HTTPStatus.OK, state.artifacts())
        elif path == "/api/logs":
            raw = (query.get("tail") or [str(state.log_tail)])[0]
            self._send_json(HTTPStatus.OK, state.logs(_as_int(raw) or state.log_tail))
        elif path.startswith(f"/{ARTIFACTS_DIR}/"):
            self._serve_artifact(unquote(path[len(ARTIFACTS_DIR) + 2 :]))
        else:
            self._send_error_json(
                HTTPStatus.NOT_FOUND,
                f"未知路径：{path}；本服务只提供 /、/api/status、/api/metrics、"
                f"/api/artifacts、/api/logs 与 /{ARTIFACTS_DIR}/<相对路径>",
            )

    # -- responses ---------------------------------------------------------------

    def _send_bytes(
        self,
        data: bytes,
        status: HTTPStatus,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self._responded = True
        if self.command != "HEAD" and data:
            self.wfile.write(data)

    def _send_json(self, status: HTTPStatus, payload: Any, *, pretty: bool = True) -> None:
        data = json.dumps(
            _sanitize(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        ).encode("utf-8")
        self._send_bytes(data, status, "application/json; charset=utf-8")

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message, "status": int(status)})

    # -- static / artifacts ------------------------------------------------------

    def _serve_html(self) -> None:
        page = self.monitor_state.html_file
        if not page.is_file():
            body = (
                "监控页面文件尚未就绪：服务端没有找到 n01_v12_monitor.html"
                f"（期望路径 {page}）。数据接口 /api/status、/api/metrics、/api/artifacts "
                "仍可正常访问。\n"
            ).encode()
            self._send_bytes(body, HTTPStatus.SERVICE_UNAVAILABLE, "text/plain; charset=utf-8")
            return
        try:
            data = page.read_bytes()
        except OSError as error:
            self._send_bytes(
                f"监控页面文件无法读取：{error}\n".encode(),
                HTTPStatus.SERVICE_UNAVAILABLE,
                "text/plain; charset=utf-8",
            )
            return
        self._send_bytes(data, HTTPStatus.OK, "text/html; charset=utf-8")

    def _serve_artifact(self, raw_relative: str) -> None:
        base = artifacts_dir(self.monitor_state.run_root)
        target = _resolve_inside(base, raw_relative)
        if target is None:
            self._send_error_json(
                HTTPStatus.FORBIDDEN,
                "拒绝访问：请求路径必须指向 run 根目录下 monitor/artifacts 之内的文件",
            )
            return
        try:
            if not target.is_file():
                self._send_error_json(HTTPStatus.NOT_FOUND, f"案例文件不存在：{raw_relative}")
                return
            size = target.stat().st_size
        except OSError as error:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"案例文件无法读取：{error}")
            return
        content_type = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            parsed = _parse_range(range_header, size)
            if parsed is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self._responded = True
                return
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self._responded = True
        if self.command == "HEAD":
            return
        remaining = length
        with target.open("rb") as stream:
            stream.seek(start)
            while remaining > 0:
                chunk = stream.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def build_server(
    state: MonitorState, host: str = "127.0.0.1", port: int = 18080
) -> ThreadingHTTPServer:
    """Create (but do not start) the loopback HTTP server; refuses non-loopback hosts."""
    bind_host = ensure_loopback(host)
    handler = type("N01V12MonitorHandler", (MonitorRequestHandler,), {"monitor_state": state})
    server = ThreadingHTTPServer((bind_host, int(port)), handler)
    server.daemon_threads = True
    return server


class ReconcileLoop(threading.Thread):
    """Background aggregation loop; keeps ``status.json`` fresh without blocking requests."""

    def __init__(
        self, state: MonitorState, interval_s: float = DEFAULT_RECONCILE_INTERVAL_S
    ) -> None:
        super().__init__(name="n01-monitor-reconcile", daemon=True)
        self.state = state
        self.interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()

    def run(self) -> None:
        while True:
            try:
                self.state.refresh()
            except Exception as error:  # noqa: BLE001 - keep serving without aggregation
                print(f"[n01-v12-monitor] 汇总失败：{error}", file=sys.stderr, flush=True)
            if self._stop.wait(self.interval_s):
                return

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout_s)


def serve(
    state: MonitorState,
    host: str = "127.0.0.1",
    port: int = 18080,
    *,
    enable_gpu: bool = True,
    gpu_interval_s: float = DEFAULT_GPU_INTERVAL_S,
    reconcile_interval_s: float = DEFAULT_RECONCILE_INTERVAL_S,
) -> int:
    """Serve forever on loopback; only this explicit call starts background threads."""
    server = build_server(state, host, port)
    sampler = GpuSampler(state.run_root, interval_s=gpu_interval_s) if enable_gpu else None
    loop = ReconcileLoop(state, reconcile_interval_s)
    if sampler is not None:
        sampler.start()
    loop.start()
    address = server.server_address
    print(
        f"N01 v1.2 只读监控服务：http://{address[0]}:{address[1]}/ "
        f"（run-root {state.run_root}；仅 loopback，页面关闭或 SSH 断开不影响训练）",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("收到中断，正在停止监控服务…", flush=True)
    finally:
        loop.stop()
        if sampler is not None:
            sampler.stop()
        server.server_close()
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m owac.runtime.n01_v12_monitor",
        description="N01 v1.2 P3 只读进度服务：读取 run 根目录下的 monitor/ 状态，不控制训练。",
    )
    parser.add_argument("--run-root", type=Path, required=True, help="训练 run 根目录（只读）")
    parser.add_argument(
        "--host", default="127.0.0.1", help="绑定地址，必须是 loopback（默认 127.0.0.1）"
    )
    parser.add_argument("--port", type=int, default=18080, help="监听端口（默认 18080）")
    parser.add_argument("--stage", default=STAGE_DEFAULT, help="阶段名（默认 P3）")
    parser.add_argument(
        "--once", action="store_true", help="只做一次 reconcile 并打印 status JSON 后退出"
    )
    parser.add_argument(
        "--html", type=Path, default=None, help="监控页面路径（默认与本模块同目录）"
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=DEFAULT_HEARTBEAT_TIMEOUT_S,
        help=(
            "心跳过期秒数，超过即标记为可疑（默认 1800）。必须大于"
            "（log_every × 最慢平台的单步耗时），否则健康 run 会被误报为异常退出",
        ),
    )
    parser.add_argument(
        "--reconcile-interval",
        type=float,
        default=DEFAULT_RECONCILE_INTERVAL_S,
        help="汇总间隔秒（默认 5）",
    )
    parser.add_argument(
        "--gpu-interval",
        type=float,
        default=DEFAULT_GPU_INTERVAL_S,
        help="GPU 采样间隔秒（默认 8）",
    )
    parser.add_argument("--no-gpu", action="store_true", help="不采样 GPU（gpus 返回空列表）")
    parser.add_argument(
        "--dev-loop-budget", type=int, default=DEV_LOOP_BUDGET, help="开发闭环总预算（默认 100）"
    )
    parser.add_argument(
        "--dev-loop-aggregate",
        choices=("max", "sum"),
        default="max",
        help="并行 worker 的闭环计数汇总方式（默认 max，保守不翻倍）",
    )
    parser.add_argument(
        "--log-tail", type=int, default=LOG_TAIL_LINES, help="日志尾部行数（默认 100）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: serve on loopback, or ``--once`` for a single aggregation."""
    args = _build_parser().parse_args(argv)
    try:
        ensure_loopback(args.host)
    except ValueError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    run_root = args.run_root.expanduser()
    state = MonitorState(
        run_root,
        stage=args.stage,
        html_path=args.html,
        heartbeat_timeout_s=args.heartbeat_timeout,
        dev_loop_budget=args.dev_loop_budget,
        dev_loop_aggregate=args.dev_loop_aggregate,
        log_tail=args.log_tail,
    )
    if args.once:
        payload = reconcile(
            run_root,
            stage=args.stage,
            heartbeat_timeout_s=args.heartbeat_timeout,
            dev_loop_budget=args.dev_loop_budget,
            dev_loop_aggregate=args.dev_loop_aggregate,
            log_tail=args.log_tail,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    try:
        return serve(
            state,
            args.host,
            args.port,
            enable_gpu=not args.no_gpu,
            gpu_interval_s=args.gpu_interval,
            reconcile_interval_s=args.reconcile_interval,
        )
    except OSError as error:
        print(f"无法启动监控服务：绑定 {args.host}:{args.port} 失败（{error}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
