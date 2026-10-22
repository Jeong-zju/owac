"""Read-only localhost monitor for the OWAC C-full RAW->RAW token distillation run.

The monitor is intentionally small and standard-library only.  It exposes exactly
three read-only routes and reads only fixed file names below the configured run
root; there is no path parameter, no upload, and no start/stop endpoint::

    GET/HEAD /              Chinese single-page monitor (repo-native HTML asset)
    GET/HEAD /index.html    same asset
    GET/HEAD /api/snapshot  honest JSON snapshot of the configured run root

CLI::

    python -m owac.runtime.raw_distillation_monitor --config <new.json> \
        [--run-root <dir>] [--host 127.0.0.1] [--port 8878] [--poll-seconds 5]

The page polls ``/api/snapshot`` every five seconds.  Unknown values are ``null``
in JSON and ``未知`` on the page; the monitor never invents a curve, a metric, a
checkpoint, a GPU number, or a running process.  A recorded PID is checked against
``/proc/<pid>/stat`` field 22 (``starttime``) so a dead or reused PID can never be
displayed as running.  Append-only JSONL files are read with torn-last-line
tolerance (a final line without a trailing newline is ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

SNAPSHOT_SCHEMA = "owac.raw-distillation-monitor.snapshot.v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8878
DEFAULT_POLL_SECONDS = 5
HEARTBEAT_STALE_SECONDS = 30.0
MAX_SERIES_RECORDS = 6000
MAX_PARENT_RECORDS = 10000
MAX_JSONL_BYTES = 32 * 1024 * 1024
MAX_CONFIG_BYTES = 16 * 1024 * 1024
MAX_WARNINGS = 20

ROUTE_ROOT = "/"
ROUTE_INDEX = "/index.html"
ROUTE_SNAPSHOT = "/api/snapshot"
SERVED_ROUTES = frozenset((ROUTE_ROOT, ROUTE_INDEX, ROUTE_SNAPSHOT))
HTML_NAME = "raw_distillation_monitor.html"

# docs/contract: outputs/raw-distillation-c-full-20261021-123313/harness/round-01/
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/raw-distillation-c-full-20261021-123313"

RUN_STATUS_VALUES = (
    "preparing",
    "prepared",
    "running",
    "stopping",
    "stopped",
    "stopped_at_update_limit",
    "completed",
    "error",
)
ACTIVE_STATUS_VALUES = frozenset(("preparing", "running", "stopping"))
HEARTBEAT_PHASE_MAP = {
    "preparing": "preparing",
    "running": "running",
    "stopping": "stopping",
    # A legacy heartbeat "finished" is not proof of successful completion; only
    # the runner's durable final status may say completed.
    "finished": "stopped",
    "prepared": "prepared",
    "stopped": "stopped",
    "completed": "completed",
    "error": "error",
}
PROCESS_STATE_TEXT = {
    "alive": "存活（starttime 匹配）",
    "dead": "已结束（PID 不存在）",
    "reused": "PID 已被复用（starttime 不匹配）",
    "unknown": "未知（无法核验）",
}
CLOSED_LOOP = {
    "status": "not_run",
    "episodes": 0,
    "control_restored": False,
    "note": "本阶段未授权闭环；不代表控制已恢复",
}


# ---------------------------------------------------------------------------
# Small, strict value normalizers.  Unknown/typed-wrong values become None so the
# page can render 未知 instead of a fabricated zero.
# ---------------------------------------------------------------------------


def as_int(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def first_int(mapping, *keys):
    """First present integer field, for actual-producer/legacy key compatibility."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            value = as_int(mapping.get(key))
            if value is not None:
                return value
    return None


def as_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def as_bool(value):
    return value if isinstance(value, bool) else None


def clean_text(value, limit=400):
    if not isinstance(value, str):
        return None
    text = value.replace("\x00", "").strip()
    if not text:
        return None
    return text[:limit]


def basename(value):
    text = clean_text(value, 1000)
    if text is None:
        return None
    text = text.replace("\\", "/").rstrip("/")
    name = text.rsplit("/", 1)[-1]
    return name or None


def sanitize(value, depth=0):
    """Bounded JSON-safe copy used for open-ended diagnostic/error payloads."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return clean_text(value, 500)
    if depth >= 4:
        return None
    if isinstance(value, (list, tuple)):
        return [sanitize(item, depth + 1) for item in value[:100]]
    if isinstance(value, dict):
        packed = {}
        for key, item in list(value.items())[:80]:
            name = clean_text(str(key), 80)
            if name:
                packed[name] = sanitize(item, depth + 1)
        return packed
    return None


def parse_iso(value):
    text = clean_text(value, 64)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def now_iso(now=None):
    return (now or datetime.now().astimezone()).isoformat(timespec="seconds")


def age_seconds(value, now):
    parsed = parse_iso(value)
    if parsed is None:
        return None
    return round(max(0.0, (now - parsed).total_seconds()), 1)


def display_path(path):
    """Repository-relative path when possible, otherwise a basename (never a raw absolute path)."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return candidate.name


def sha256_file(path):
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def read_json_file(path, warnings, name=None):
    path = Path(path)
    label = name or path.name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"{label}: 无法解析（{type(exc).__name__}）")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"{label}: 顶层不是 JSON 对象")
        return None
    return payload


def read_jsonl(path, warnings, limit=MAX_SERIES_RECORDS, name=None):
    """Parse an append-only JSONL file.

    A final line without a trailing newline is ignored as torn.  Earlier malformed
    lines are skipped but counted so the page can disclose the partial read.
    """
    path = Path(path)
    label = name or path.name
    result = {
        "present": False,
        "rows": [],
        "torn_tail": False,
        "invalid_lines": 0,
        "truncated": False,
    }
    if not path.is_file():
        return result
    result["present"] = True
    try:
        raw = path.read_bytes()
    except OSError:
        warnings.append(f"{label}: 无法读取")
        return result
    if len(raw) > MAX_JSONL_BYTES:
        result["truncated"] = True
        raw = raw[-MAX_JSONL_BYTES:]
        first_newline = raw.find(b"\n")
        raw = raw[first_newline + 1 :] if first_newline >= 0 else b""
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return result
    if not text.endswith("\n"):
        result["torn_tail"] = True
    lines = text.splitlines()
    if result["torn_tail"]:
        # The final unterminated segment may still be mid-write; never parse it.
        lines = lines[:-1]
    parsed = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            result["invalid_lines"] += 1
            continue
        if isinstance(record, dict):
            parsed.append(record)
        else:
            result["invalid_lines"] += 1
    if len(parsed) > limit:
        result["truncated"] = True
        parsed = parsed[-limit:]
    result["rows"] = parsed
    return result


def read_proc_stat(pid):
    """Return (state character, starttime ticks) from /proc/<pid>/stat."""
    pid = as_int(pid)
    if pid is None or pid <= 0:
        raise ValueError("invalid pid")
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    close = stat.rfind(")")
    if close < 0:
        raise ValueError("malformed /proc stat")
    fields = stat[close + 2 :].split()
    # Field 3 is the first token after comm; starttime is field 22 -> index 19.
    if len(fields) < 20:
        raise ValueError("short /proc stat")
    try:
        return fields[0], int(fields[19])
    except ValueError as exc:
        raise ValueError("non-numeric /proc starttime") from exc


def read_proc_start_ticks(pid):
    """Return /proc/<pid>/stat field 22 (starttime ticks) or raise OSError/ValueError."""
    return read_proc_stat(pid)[1]


def proc_state(pid, recorded_start_ticks):
    """Honest PID liveness: dead or reused may never be reported as alive/running."""
    pid = as_int(pid)
    recorded = as_int(recorded_start_ticks)
    state = {
        "pid": pid,
        "recorded_start_ticks": recorded,
        "observed_start_ticks": None,
        "observed_process_state": None,
        "alive": None,
        "state": "unknown",
        "zombie": False,
    }
    if pid is None or pid <= 0:
        return state
    try:
        observed_state, observed = read_proc_stat(pid)
    except FileNotFoundError:
        state.update(state="dead", alive=False)
        return state
    except (OSError, ValueError):
        return state
    state["observed_start_ticks"] = observed
    state["observed_process_state"] = observed_state
    if observed_state == "Z":
        # A zombie is not alive even when its starttime still matches: it has
        # exited and only its unreaped process-table entry remains.
        state.update(state="dead", alive=False, zombie=True)
        return state
    if recorded is None:
        return state
    if observed != recorded:
        state.update(state="reused", alive=False)
    else:
        state.update(state="alive", alive=True)
    return state


# ---------------------------------------------------------------------------
# Per-file record normalizers.  Only known fields leave the monitor.
# ---------------------------------------------------------------------------


def normalize_checkpoint(raw):
    if not isinstance(raw, dict):
        return None
    checkpoint = {
        "step": as_int(raw.get("step")),
        "path": basename(raw.get("path")),
        "sha256": clean_text(raw.get("sha256"), 80),
        "kind": clean_text(raw.get("kind"), 32),
    }
    if all(checkpoint[key] is None for key in ("step", "path", "sha256")):
        return None
    return checkpoint


def normalize_action_diagnostic(raw):
    """Diagnostic-only action error; always marked as unable to select or stop."""
    if not isinstance(raw, dict):
        return None
    diagnostic = {
        "metric": clean_text(raw.get("metric"), 120),
        "value": as_float(raw.get("value")),
        "checkpoint_sha256": clean_text(raw.get("checkpoint_sha256"), 80),
        "timestamp": clean_text(raw.get("timestamp"), 64),
        "source": clean_text(raw.get("source"), 32),
        "used_for_selection": False,
        "used_for_stopping": False,
    }
    if all(
        diagnostic[key] is None
        for key in ("metric", "value", "checkpoint_sha256", "timestamp", "source")
    ):
        return None
    return diagnostic


def normalize_update(raw):
    if not isinstance(raw, dict):
        return None
    cumulative = as_int(raw.get("cumulative_updates"))
    if cumulative is None:
        return None
    endpoints = raw.get("sampled_source_endpoints")
    return {
        "step": cumulative,
        "cumulative_updates": cumulative,
        "new_stage_updates": as_int(raw.get("new_stage_updates")),
        "parent_cumulative_updates": as_int(raw.get("parent_cumulative_updates")),
        "timestamp": clean_text(raw.get("timestamp"), 64),
        "anchor_presentations": as_int(raw.get("anchor_presentations")),
        "new_stage_anchor_presentations": as_int(raw.get("new_stage_anchor_presentations")),
        "sampled_source_count": len(endpoints) if isinstance(endpoints, list) else None,
        "feature_mse": as_float(raw.get("feature_mse")),
        "feature_cosine": as_float(raw.get("feature_cosine")),
        "feature_total": as_float(raw.get("feature_total")),
        "loss_total": as_float(raw.get("loss_total")),
        "lr_used": as_float(raw.get("lr_used")),
        "lr_next": as_float(raw.get("lr_next")),
        "trainable_parameter_count": as_int(raw.get("trainable_parameter_count")),
        "trainable_tensor_count": as_int(raw.get("trainable_tensor_count")),
        "finite_gradients": as_bool(raw.get("finite_gradients")),
        "all_gradients_nonzero": as_bool(raw.get("all_gradients_nonzero")),
        "elapsed_seconds": as_float(raw.get("elapsed_seconds")),
        "new_stage_wall_seconds": as_float(raw.get("new_stage_wall_seconds")),
        "gpu_allocated_bytes": as_int(raw.get("gpu_allocated_bytes")),
        "gpu_reserved_bytes": as_int(raw.get("gpu_reserved_bytes")),
        "host_rss_bytes": as_int(raw.get("host_rss_bytes")),
        "output_bytes": as_int(raw.get("output_bytes")),
        "checkpoint": normalize_checkpoint(raw.get("checkpoint")),
    }


def normalize_validation(raw):
    if not isinstance(raw, dict):
        return None
    step = as_int(raw.get("step"))
    if step is None:
        return None
    return {
        "step": step,
        "new_stage_step": as_int(raw.get("new_stage_step")),
        "timestamp": clean_text(raw.get("timestamp"), 64),
        "split": clean_text(raw.get("split"), 32),
        "sources": as_int(raw.get("sources")),
        "trajectories": as_int(raw.get("trajectories")),
        "windows": as_int(raw.get("windows")),
        "anchors": as_int(raw.get("anchors")),
        "episode_resets": as_int(raw.get("episode_resets")),
        "valid_tokens": as_int(raw.get("valid_tokens")),
        "scalar_denominator": as_int(raw.get("scalar_denominator")),
        "feature_mse": as_float(raw.get("feature_mse")),
        "feature_cosine": as_float(raw.get("feature_cosine")),
        "feature_total": as_float(raw.get("feature_total")),
        "cosine_weight": as_float(raw.get("cosine_weight")),
        "per_source": sanitize(raw.get("per_source")),
        "selection_metric": clean_text(raw.get("selection_metric"), 80),
        "selection_eligible": as_bool(raw.get("selection_eligible")),
        "is_best": as_bool(raw.get("is_best")),
        "checkpoint_path": basename(raw.get("checkpoint_path")),
        "checkpoint_sha256": clean_text(raw.get("checkpoint_sha256"), 80),
        "test_used": as_bool(raw.get("test_used")),
        "action_diagnostic": normalize_action_diagnostic(raw.get("action_diagnostic")),
    }


def normalize_parent_point(raw):
    if not isinstance(raw, dict):
        return None
    step = as_int(raw.get("cumulative_updates"))
    if step is None:
        return None
    return {
        "step": step,
        "cumulative_updates": step,
        "event": clean_text(raw.get("event"), 40),
        "timestamp": clean_text(raw.get("timestamp"), 64),
        "anchor_presentations": as_int(raw.get("anchor_presentations")),
        "feature_mse": as_float(raw.get("feature_mse")),
        "feature_cosine": as_float(raw.get("feature_cosine")),
        "feature_total": as_float(raw.get("feature_total")),
        "learning_rate": as_float(raw.get("learning_rate")),
        "gpu_allocated_bytes": as_int(raw.get("gpu_allocated_bytes")),
        "gpu_reserved_bytes": as_int(raw.get("gpu_reserved_bytes")),
    }


def normalize_parent_baseline(raw):
    if not isinstance(raw, dict):
        return None
    baseline = {
        "schema": clean_text(raw.get("schema"), 120),
        "step": as_int(raw.get("step")),
        "checkpoint_sha256": clean_text(raw.get("checkpoint_sha256"), 80),
        "feature_mse": as_float(raw.get("feature_mse")),
        "feature_cosine": as_float(raw.get("feature_cosine")),
        "feature_total": as_float(raw.get("feature_total")),
        "valid_tokens": as_int(raw.get("valid_tokens")),
        "scalar_denominator": as_int(raw.get("scalar_denominator")),
        "windows": as_int(raw.get("windows")),
        "sources": as_int(raw.get("sources")),
        "episode_resets": as_int(raw.get("episode_resets")),
    }
    if all(baseline[key] is None for key in baseline if key != "schema"):
        return None
    return baseline


def normalize_acceptance(raw):
    acceptance = {
        "present": False,
        "path": "acceptance/phase-0051.json",
        "schema": None,
        "status": None,
        "step": None,
        "checkpoint_sha256": None,
        "reviewer": None,
        "note": None,
        "accepted_at": None,
        "passed": None,
        "matches_last_checkpoint": None,
        "matches_selection_checkpoint": None,
    }
    if not isinstance(raw, dict):
        return acceptance
    acceptance.update(
        {
            "present": True,
            "schema": clean_text(raw.get("schema"), 120),
            "status": clean_text(raw.get("status"), 40),
            "step": as_int(raw.get("step")),
            "checkpoint_sha256": clean_text(raw.get("checkpoint_sha256"), 80),
            "reviewer": clean_text(raw.get("reviewer"), 120),
            "note": clean_text(raw.get("note"), 500),
            "accepted_at": clean_text(raw.get("accepted_at"), 64)
            or clean_text(raw.get("timestamp"), 64)
            or clean_text(raw.get("updated_at"), 64),
            "passed": as_bool(raw.get("passed")),
        }
    )
    return acceptance


def normalize_legacy_selector(raw):
    if not isinstance(raw, dict):
        return None
    selector = {
        "cleared": as_bool(raw.get("cleared")),
        "metric": clean_text(raw.get("metric"), 120),
        "best_step": as_int(raw.get("best_step")),
        "best_value": as_float(raw.get("best_value")),
        "checkpoint_sha256": clean_text(raw.get("checkpoint_sha256"), 80),
        "source": clean_text(raw.get("source"), 200),
        "declared_used_for_selection": as_bool(raw.get("used_for_selection")),
        # Frozen contract: the old pilot action selector never selects this stage.
        "used_for_selection": False,
    }
    if all(selector[key] is None for key in ("metric", "best_step", "best_value")):
        return None
    return selector


def build_selection(run_status, sidecar):
    selection = {
        "available": False,
        "source": None,
        "metric": None,
        "direction": None,
        "tie_break": None,
        "best_step": None,
        "best_value": None,
        "best_metric": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "valid_tokens": None,
        "scalar_denominator": None,
        "feature_mse": None,
        "feature_cosine": None,
        "parent_c50_baseline": None,
        "legacy_action_selector": None,
        "updated_at": None,
        "test_used": None,
        "test_selection": None,
    }
    if isinstance(sidecar, dict):
        selection["source"] = "selection.json"
        selection["metric"] = clean_text(sidecar.get("selection_metric"), 80)
        selection["direction"] = clean_text(sidecar.get("direction"), 20)
        selection["tie_break"] = clean_text(sidecar.get("tie_break"), 40)
        selection["best_step"] = as_int(sidecar.get("best_step"))
        selection["best_value"] = as_float(sidecar.get("best_value"))
        selection["checkpoint_path"] = basename(sidecar.get("checkpoint_path"))
        selection["checkpoint_sha256"] = clean_text(sidecar.get("checkpoint_sha256"), 80)
        selection["valid_tokens"] = as_int(sidecar.get("valid_tokens"))
        selection["scalar_denominator"] = as_int(sidecar.get("scalar_denominator"))
        selection["feature_mse"] = as_float(sidecar.get("feature_mse"))
        selection["feature_cosine"] = as_float(sidecar.get("feature_cosine"))
        parent = sidecar.get("parent_c50_baseline")
        if isinstance(parent, dict):
            selection["parent_c50_baseline"] = {
                "step": as_int(parent.get("step")),
                "checkpoint_sha256": clean_text(parent.get("checkpoint_sha256"), 80),
                "feature_total": as_float(parent.get("feature_total")),
            }
        selection["legacy_action_selector"] = normalize_legacy_selector(
            sidecar.get("legacy_action_selector")
        )
        selection["updated_at"] = clean_text(sidecar.get("updated_at"), 64)
        selection["test_used"] = as_bool(sidecar.get("test_used"))
        selection["test_selection"] = as_bool(sidecar.get("test_selection"))
    best = run_status.get("best") if isinstance(run_status, dict) else None
    if isinstance(best, dict):
        if selection["best_step"] is None:
            selection["best_step"] = as_int(best.get("step"))
        if selection["best_value"] is None:
            selection["best_value"] = as_float(best.get("value"))
        if selection["checkpoint_sha256"] is None:
            selection["checkpoint_sha256"] = clean_text(best.get("checkpoint_sha256"), 80)
        selection["best_metric"] = clean_text(best.get("metric"), 80)
        if selection["source"] is None and selection["best_step"] is not None:
            selection["source"] = "run-status.json"
    status_metric = clean_text(
        run_status.get("selection_metric") if isinstance(run_status, dict) else None, 80
    )
    if selection["metric"] is None:
        selection["metric"] = status_metric
    if selection["best_metric"] is None:
        selection["best_metric"] = selection["metric"]
    selection["available"] = bool(
        selection["source"] is not None or selection["best_step"] is not None
    )
    return selection


def build_progress(run_status, train_series, parent_history, supervisor=None):
    status = run_status if isinstance(run_status, dict) else {}
    last_update = train_series[-1] if train_series else {}
    supervisor = supervisor if isinstance(supervisor, dict) else {}

    def pick(*values):
        return next((value for value in values if value is not None), None)

    cumulative = pick(
        as_int(status.get("cumulative_updates")), last_update.get("cumulative_updates")
    )
    parent_cumulative = pick(
        as_int(status.get("parent_cumulative_updates")),
        last_update.get("parent_cumulative_updates"),
        (parent_history.get("baseline") or {}).get("step"),
        parent_history.get("last_step"),
    )
    new_stage_updates = pick(
        as_int(status.get("new_stage_updates")), last_update.get("new_stage_updates")
    )
    if new_stage_updates is None and cumulative is not None and parent_cumulative is not None:
        # Deterministic accounting from two real counters, never a default of 0.
        new_stage_updates = max(0, cumulative - parent_cumulative)
    target = as_int(status.get("target_cumulative_updates"))
    new_stage_target = (
        target - parent_cumulative
        if target is not None and parent_cumulative is not None and target >= parent_cumulative
        else None
    )
    wall, wall_source = None, None
    supervisor_wall = as_float(supervisor.get("new_stage_wall_seconds"))
    if supervisor_wall is not None:
        wall, wall_source = supervisor_wall, "supervisor-state.json"
    elif as_float(status.get("new_stage_wall_seconds")) is not None:
        wall, wall_source = as_float(status.get("new_stage_wall_seconds")), "run-status.json"
    elif last_update.get("new_stage_wall_seconds") is not None:
        wall, wall_source = last_update.get("new_stage_wall_seconds"), "updates.jsonl"
    wall_cap = pick(
        as_float(supervisor.get("new_stage_wall_cap_seconds")),
        as_float(status.get("new_stage_wall_cap_seconds")),
    )
    output, output_source = None, None
    supervisor_output = as_int(supervisor.get("new_stage_output_bytes"))
    if supervisor_output is not None:
        output, output_source = supervisor_output, "supervisor-state.json"
    elif as_int(status.get("output_bytes")) is not None:
        output, output_source = as_int(status.get("output_bytes")), "run-status.json"
    elif last_update.get("output_bytes") is not None:
        output, output_source = last_update.get("output_bytes"), "updates.jsonl"
    output_cap = pick(
        as_int(supervisor.get("new_stage_output_cap_bytes")),
        as_int(status.get("output_cap_bytes")),
    )
    gpu = None
    for row in reversed(train_series):
        if row.get("gpu_allocated_bytes") is not None or row.get("gpu_reserved_bytes") is not None:
            gpu = {
                "allocated_bytes": row.get("gpu_allocated_bytes"),
                "reserved_bytes": row.get("gpu_reserved_bytes"),
                "host_rss_bytes": row.get("host_rss_bytes"),
                "record_step": row.get("step"),
                "source": "updates.jsonl",
            }
            break
    progress_fraction = None
    if cumulative is not None and target is not None and target > 0:
        # Keep full-precision ratios in the API; the page rounds for display.
        progress_fraction = min(max(cumulative / target, 0.0), 1.0)
    # New-stage fraction uses only new-stage counters.  If the parent start is
    # unknown we return None (not 0) so the page cannot fake a 0% start.
    new_stage_fraction = None
    if (
        parent_cumulative is not None
        and new_stage_updates is not None
        and new_stage_target is not None
        and new_stage_target > 0
    ):
        new_stage_fraction = min(max(new_stage_updates / new_stage_target, 0.0), 1.0)
    wall_fraction = None
    if wall is not None and wall_cap is not None and wall_cap > 0:
        wall_fraction = min(max(wall / wall_cap, 0.0), 1.0)
    disk_fraction = None
    if output is not None and output_cap is not None and output_cap > 0:
        disk_fraction = min(max(output / output_cap, 0.0), 1.0)
    anchor_cap = pick(
        as_int(status.get("anchor_presentations_cap")),
        as_int(status.get("target_anchor_presentations")),
        as_int(status.get("anchor_cap")),
    )
    return {
        "cumulative_updates": cumulative,
        "new_stage_updates": new_stage_updates,
        "parent_cumulative_updates": parent_cumulative,
        "target_cumulative_updates": target,
        "new_stage_target_updates": new_stage_target,
        "stop_after_update": as_int(status.get("stop_after_update")),
        "anchor_presentations": pick(
            as_int(status.get("anchor_presentations")), last_update.get("anchor_presentations")
        ),
        "new_stage_anchor_presentations": pick(
            as_int(status.get("new_stage_anchor_presentations")),
            last_update.get("new_stage_anchor_presentations"),
        ),
        "anchor_presentations_cap": anchor_cap,
        "new_stage_wall_seconds": wall,
        "new_stage_wall_cap_seconds": wall_cap,
        "wall_source": wall_source,
        "wall_semantics": (
            "supervisor cumulative new-stage wall seconds (all phases, no truncation)"
        ),
        "output_bytes": output,
        "output_cap_bytes": output_cap,
        "output_source": output_source,
        "output_scope": "all_new_run_root_bytes",
        "output_scope_description": (
            "run root 下全部 NEW 真实文件（含 harness 日志/回执）；排除符号链接，"
            "按 inode 去重硬链接；不是仅检查点或部分目录"
        ),
        "progress_fraction": progress_fraction,
        "new_stage_fraction": new_stage_fraction,
        "wall_fraction": wall_fraction,
        "disk_fraction": disk_fraction,
        "gpu": gpu,
    }


def build_heartbeat(raw, now):
    if not isinstance(raw, dict):
        return None
    age = age_seconds(raw.get("updated_at"), now)
    return {
        "pid": as_int(raw.get("pid")),
        "pid_start_ticks": as_int(raw.get("pid_start_ticks")),
        "pgid": as_int(raw.get("pgid")),
        "seq": as_int(raw.get("seq")),
        "updated_at": clean_text(raw.get("updated_at"), 64),
        "phase": clean_text(raw.get("phase"), 40),
        "cumulative_updates": as_int(raw.get("cumulative_updates")),
        "age_seconds": age,
        "stale": None if age is None else age > HEARTBEAT_STALE_SECONDS,
    }


def build_process(run_status, heartbeat, heartbeat_block, now, supervisor=None):
    status = run_status if isinstance(run_status, dict) else {}
    runner_pid = as_int(status.get("pid"))
    runner_ticks = as_int(status.get("pid_start_ticks"))
    runner_pgid = as_int(status.get("pgid"))
    runner_source = "run-status.json" if runner_pid is not None else None
    if runner_pid is None and isinstance(heartbeat, dict):
        runner_pid = as_int(heartbeat.get("pid"))
        runner_ticks = as_int(heartbeat.get("pid_start_ticks"))
        runner_pgid = as_int(heartbeat.get("pgid"))
        if runner_pid is not None:
            runner_source = "heartbeat.json"
    runner_liveness = proc_state(runner_pid, runner_ticks)
    heartbeat_age = heartbeat_block.get("age_seconds") if heartbeat_block else None
    heartbeat_stale = heartbeat_block.get("stale") if heartbeat_block else None
    process = {
        "pid": runner_pid,
        "pid_start_ticks": runner_ticks,
        "pgid": runner_pgid,
        "alive": runner_liveness["alive"],
        "state": runner_liveness["state"],
        "zombie": runner_liveness.get("zombie", False),
        "observed_start_ticks": runner_liveness["observed_start_ticks"],
        "observed_process_state": runner_liveness["observed_process_state"],
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_stale": heartbeat_stale,
        "status_age_seconds": age_seconds(status.get("updated_at"), now),
        "source": runner_source,
        "liveness_checked_at": now_iso(now),
        "runner_pid": runner_pid,
        "runner_pid_start_ticks": runner_ticks,
        "runner_pgid": runner_pgid,
        "runner_alive": runner_liveness["alive"],
        "runner_state": runner_liveness["state"],
        "active_source": "runner" if runner_pid is not None else None,
    }
    # While the supervisor's verified current child is loading the model, the
    # old prepare/runner PID may already be dead and its run-status may still say
    # prepared.  Use the verified live child as the active process, but retain
    # the old runner PID/state for audit and never infer liveness from "running".
    supervisor = supervisor if isinstance(supervisor, dict) else {}
    if supervisor.get("live_verified") is True:
        child_pid = as_int(supervisor.get("child_pid"))
        child_ticks = as_int(
            supervisor.get("child_start_ticks") or supervisor.get("child_pid_start_ticks")
        )
        child_liveness = proc_state(child_pid, child_ticks)
        differs = runner_pid != child_pid or runner_liveness.get("alive") is not True
        if differs and child_liveness.get("alive") is True:
            supervisor_heartbeat = supervisor.get("heartbeat") or {}
            process.update(
                {
                    "pid": child_pid,
                    "pid_start_ticks": child_ticks,
                    "pgid": as_int(supervisor.get("child_pgid")),
                    "alive": True,
                    "state": "alive",
                    "zombie": False,
                    "observed_start_ticks": child_liveness["observed_start_ticks"],
                    "observed_process_state": child_liveness["observed_process_state"],
                    "heartbeat_age_seconds": supervisor_heartbeat.get("age_seconds"),
                    "heartbeat_stale": supervisor_heartbeat.get("stale"),
                    "source": "supervisor-state.json:child",
                    "active_source": "supervisor_child",
                }
            )
    return process


def effective_status(declared, process, heartbeat_block, now, supervisor=None):
    """Return (status, stale_reason).  Never reports running without verified PID+heartbeat."""
    supervisor = supervisor if isinstance(supervisor, dict) else {}
    if (
        supervisor.get("live_verified") is True
        and process.get("active_source") == "supervisor_child"
    ):
        return "preparing", None
    if declared in {"prepared", "stopped", "completed", "error"}:
        # prepare's successful receipt is a terminal prepare state: its old PID
        # may be dead while a fresh supervisor child is loading the model.
        if (
            declared == "prepared"
            and supervisor.get("live_verified") is True
            and process.get("active_source") == "supervisor_child"
        ):
            return "preparing", None
        return declared, None
    if declared not in ACTIVE_STATUS_VALUES:
        return declared, None
    if process.get("pid") is None:
        # A monitor may legitimately start before the runner writes any process state.
        return ("preparing", None) if declared == "preparing" else ("unknown", "process_missing")
    state = process.get("state")
    if state == "dead":
        return "stale", "process_dead"
    if state == "reused":
        return "stale", "process_reused"
    if state != "alive":
        return "unknown", "process_unverified"
    heartbeat_fresh = bool(
        heartbeat_block
        and heartbeat_block.get("age_seconds") is not None
        and heartbeat_block["age_seconds"] <= HEARTBEAT_STALE_SECONDS
    )
    status_age = process.get("status_age_seconds")
    grace = status_age is not None and status_age <= HEARTBEAT_STALE_SECONDS
    if heartbeat_fresh or grace:
        return declared, None
    if heartbeat_block is None:
        return "stale", "heartbeat_missing"
    return "stale", "heartbeat_stale"


def build_status(
    run_status, heartbeat, heartbeat_block, process, now, data_present=False, supervisor=None
):
    status = run_status if isinstance(run_status, dict) else {}
    raw = clean_text(status.get("status"), 64)
    declared_raw = raw
    if raw in RUN_STATUS_VALUES:
        declared = raw
    elif isinstance(heartbeat, dict):
        declared = HEARTBEAT_PHASE_MAP.get(clean_text(heartbeat.get("phase"), 40), "unknown")
        declared_raw = raw or clean_text(heartbeat.get("phase"), 40)
    elif data_present:
        declared = "unknown"
    else:
        declared = "preparing"
    effective, stale_reason = effective_status(
        declared, process, heartbeat_block, now, supervisor=supervisor
    )
    status_detail = None
    if effective == "preparing" and process.get("active_source") == "supervisor_child":
        status_detail = "loading_model"
    elif declared == "prepared":
        status_detail = (
            "loading_model" if effective == "preparing" else "preflight_complete_waiting_training"
        )
    elif effective == "stopped":
        status_detail = "stopped"
    phase = clean_text(status.get("phase"), 40) or (
        clean_text(heartbeat.get("phase"), 40) if isinstance(heartbeat, dict) else None
    )
    return {
        "status": effective,
        "status_declared": declared,
        "status_declared_raw": declared_raw,
        "status_detail": status_detail,
        "phase": phase,
        "stale": effective == "stale",
        "stale_reason": stale_reason,
        "updated_at": clean_text(status.get("updated_at"), 64),
        "started_at": clean_text(status.get("started_at"), 64),
        "finished_at": clean_text(status.get("finished_at"), 64),
        "native_exit": as_int(status.get("native_exit")),
    }


def build_error(run_status):
    raw = run_status.get("error") if isinstance(run_status, dict) else None
    if raw is None or raw is False or raw == "" or raw == [] or raw == {}:
        return None
    if isinstance(raw, str):
        return clean_text(raw, 2000)
    if isinstance(raw, (dict, list)):
        packed = sanitize(raw)
        return None if packed in (None, "", [], {}) else packed
    return None


def build_last_checkpoint(run_status, validation_series, selection):
    checkpoint = normalize_checkpoint(
        run_status.get("last_checkpoint") if isinstance(run_status, dict) else None
    )
    if checkpoint is not None:
        checkpoint["source"] = "run-status.json"
        return checkpoint
    for row in reversed(validation_series):
        if row.get("checkpoint_path") is None and row.get("checkpoint_sha256") is None:
            continue
        return {
            "step": row.get("step"),
            "path": row.get("checkpoint_path"),
            "sha256": row.get("checkpoint_sha256"),
            "kind": None,
            "source": "validation.jsonl",
        }
    if selection.get("checkpoint_path") is not None or selection.get("checkpoint_sha256"):
        return {
            "step": selection.get("best_step"),
            "path": selection.get("checkpoint_path"),
            "sha256": selection.get("checkpoint_sha256"),
            "kind": "selected",
            "source": "selection.json",
        }
    return None


def build_action_diagnostic(validation_series):
    for row in reversed(validation_series):
        diagnostic = row.get("action_diagnostic")
        if not isinstance(diagnostic, dict):
            continue
        packed = dict(diagnostic)
        packed["record_step"] = row.get("step")
        packed["record_checkpoint_sha256"] = row.get("checkpoint_sha256")
        return packed
    return None


def build_parent_history(parent_file, parent_baseline, train_series):
    seen = {}
    for raw in parent_file["rows"]:
        point = normalize_parent_point(raw)
        if point is not None:
            seen[point["step"]] = point
    points = [seen[step] for step in sorted(seen)]
    last_update = train_series[-1] if train_series else {}
    return {
        "present": parent_file["present"],
        "available": bool(points),
        "source": "parent-history.jsonl" if parent_file["present"] else None,
        "points": points,
        "point_count": len(points),
        "last_step": points[-1]["step"] if points else None,
        "baseline": parent_baseline,
        "baseline_present": parent_baseline is not None,
        "parent_cumulative_updates": (parent_baseline or {}).get("step")
        or (points[-1]["step"] if points else last_update.get("parent_cumulative_updates")),
        "torn_tail": parent_file["torn_tail"],
        "invalid_lines": parent_file["invalid_lines"],
        "truncated": parent_file["truncated"],
    }


def build_supervisor(state, heartbeat, acceptance, now):
    if state is None and heartbeat is None:
        return None
    state = state if isinstance(state, dict) else {}
    heartbeat_raw = heartbeat if isinstance(heartbeat, dict) else {}

    # Actual supervisor producer keys: supervisor_pid/supervisor_pid_start_ticks,
    # child_pid/child_pgid/child_start_ticks.  Keep legacy aliases only as
    # fallbacks for previously written files; do not invent producer keys.
    supervisor_pid = first_int(state, "supervisor_pid", "pid")
    if supervisor_pid is None:
        supervisor_pid = first_int(heartbeat_raw, "pid")
    supervisor_ticks = first_int(state, "supervisor_pid_start_ticks", "pid_start_ticks")
    if supervisor_ticks is None:
        supervisor_ticks = first_int(heartbeat_raw, "pid_start_ticks")
    child_pid = first_int(state, "child_pid")
    if child_pid is None:
        child_pid = first_int(heartbeat_raw, "child_pid")
    child_ticks = first_int(state, "child_start_ticks", "child_pid_start_ticks")
    if child_ticks is None:
        child_ticks = first_int(heartbeat_raw, "child_start_ticks", "child_pid_start_ticks")
    child_pgid = first_int(state, "child_pgid")
    if child_pgid is None:
        child_pgid = first_int(heartbeat_raw, "child_pgid")

    supervisor_proc = proc_state(supervisor_pid, supervisor_ticks)
    child_proc = proc_state(child_pid, child_ticks)
    command = state.get("runner_command")
    if isinstance(command, str):
        command = [command]
    command_basenames = None
    if isinstance(command, list):
        names = [basename(item) for item in command[:40]]
        command_basenames = [name for name in names if name]
    heartbeat_block = None
    if heartbeat_raw:
        age = age_seconds(heartbeat_raw.get("updated_at"), now)
        heartbeat_block = {
            "present": True,
            "seq": as_int(heartbeat_raw.get("seq")),
            "updated_at": clean_text(heartbeat_raw.get("updated_at"), 64),
            "status": clean_text(heartbeat_raw.get("status"), 40),
            "age_seconds": age,
            "stale": None if age is None else age > HEARTBEAT_STALE_SECONDS,
            "pid": first_int(heartbeat_raw, "pid"),
            "pid_start_ticks": first_int(heartbeat_raw, "pid_start_ticks"),
            "child_pid": first_int(heartbeat_raw, "child_pid"),
            "child_pgid": first_int(heartbeat_raw, "child_pgid"),
            "child_start_ticks": first_int(
                heartbeat_raw, "child_start_ticks", "child_pid_start_ticks"
            ),
            "child_pid_start_ticks": first_int(
                heartbeat_raw, "child_pid_start_ticks", "child_start_ticks"
            ),
        }
    heartbeat_age = heartbeat_block["age_seconds"] if heartbeat_block else None
    heartbeat_fresh = heartbeat_age is not None and heartbeat_age <= HEARTBEAT_STALE_SECONDS
    state_age = age_seconds(state.get("updated_at"), now)
    # A fresh supervisor requires the verified supervisor PID plus either a fresh
    # heartbeat or a fresh durable state update.  The state "running" string is
    # never used as process liveness proof.
    supervisor_fresh = supervisor_proc["alive"] is True and (
        heartbeat_fresh or (state_age is not None and state_age <= HEARTBEAT_STALE_SECONDS)
    )
    child_verified = child_proc["state"] == "alive"
    live_verified = bool(supervisor_fresh and child_verified)
    acceptance_block = None
    if isinstance(state.get("acceptance"), dict):
        acceptance_block = normalize_acceptance(state.get("acceptance"))
    return {
        "present": True,
        "source": "supervisor-state.json"
        if state
        else "supervisor-heartbeat.json"
        if heartbeat
        else None,
        "schema": clean_text(state.get("schema"), 120),
        "status": clean_text(state.get("status"), 60),
        "phase": clean_text(state.get("phase"), 60),
        # Primary fields expose the actual verified producer identity.
        "pid": supervisor_proc["pid"],
        "pid_start_ticks": supervisor_proc["recorded_start_ticks"],
        "supervisor_pid": supervisor_proc["pid"],
        "supervisor_pid_start_ticks": supervisor_proc["recorded_start_ticks"],
        "alive": supervisor_proc["alive"],
        "process_state": supervisor_proc["state"],
        "obs_start_ticks": supervisor_proc["observed_start_ticks"],
        "obs_process_state": supervisor_proc["observed_process_state"],
        "supervisor_fresh": supervisor_fresh,
        "state_age_seconds": state_age,
        "heartbeat_age_seconds": heartbeat_age,
        "child_pid": child_proc["pid"],
        "child_pid_start_ticks": child_proc["recorded_start_ticks"],
        "child_start_ticks": child_proc["recorded_start_ticks"],
        "child_pgid": child_pgid,
        "child_alive": child_proc["alive"],
        "child_process_state": child_proc["state"],
        "child_verified": child_verified,
        "live_verified": live_verified,
        "child_native_exit": (
            None if child_proc["alive"] is True else as_int(state.get("child_native_exit"))
        ),
        "new_stage_wall_seconds": as_float(state.get("new_stage_wall_seconds")),
        "new_stage_wall_cap_seconds": as_float(state.get("new_stage_wall_cap_seconds")),
        "new_stage_output_bytes": as_int(state.get("new_stage_output_bytes")),
        "new_stage_output_cap_bytes": as_int(state.get("new_stage_output_cap_bytes")),
        "last_error": build_error({"error": state.get("last_error")}),
        "acceptance": acceptance_block,
        "phase_history": sanitize(state.get("phase_history"))
        if isinstance(state.get("phase_history"), list)
        else None,
        "runner_command_basenames": command_basenames,
        "started_at": clean_text(state.get("started_at"), 64),
        "updated_at": clean_text(state.get("updated_at"), 64),
        "heartbeat": heartbeat_block,
        "acceptance_receipt": acceptance,
    }


def build_config(config_path, warnings):
    block = {
        "present": False,
        "filename": None,
        "sha256": None,
        "size_bytes": None,
    }
    if config_path is None:
        return block
    path = Path(config_path)
    block["filename"] = path.name
    try:
        stat = path.stat()
    except OSError:
        warnings.append(f"{path.name}: 配置文件不存在或不可读（配置哈希未知）")
        return block
    block["present"] = path.is_file()
    block["size_bytes"] = stat.st_size
    if stat.st_size <= MAX_CONFIG_BYTES:
        block["sha256"] = sha256_file(path)
    else:
        warnings.append(f"{path.name}: 配置文件过大，未计算哈希（未知）")
    return block


def resolve_run_root(config_path=None, run_root=None):
    if run_root is not None:
        return Path(run_root).expanduser()
    if config_path is not None:
        payload = read_json_file(config_path, [], name=Path(config_path).name)
        candidate = payload.get("run_root") if isinstance(payload, dict) else None
        if isinstance(candidate, str) and candidate.strip():
            path = Path(candidate.strip()).expanduser()
            return path if path.is_absolute() else REPO_ROOT / path
    return DEFAULT_RUN_ROOT


def build_disk_block(progress):
    return {
        "new_stage_output_bytes": progress.get("output_bytes"),
        "new_stage_output_cap_bytes": progress.get("output_cap_bytes"),
        "fraction": progress.get("disk_fraction"),
        "source": progress.get("output_source"),
        "scope": progress.get("output_scope"),
        "scope_description": progress.get("output_scope_description"),
        "cap_covers_all_new_run_root_bytes": True,
    }


# ---------------------------------------------------------------------------
# Snapshot assembly.
# ---------------------------------------------------------------------------


def build_snapshot(run_root, config_path=None, poll_seconds=DEFAULT_POLL_SECONDS, now=None):
    """Build the read-only snapshot for exactly one configured run root."""
    warnings = []
    root = Path(run_root) if run_root is not None else DEFAULT_RUN_ROOT
    moment = now or datetime.now().astimezone()
    config = build_config(config_path, warnings)
    run_status = read_json_file(root / "run-status.json", warnings) or {}
    heartbeat_raw = read_json_file(root / "heartbeat.json", warnings)
    heartbeat_block = build_heartbeat(heartbeat_raw, moment)
    selection_raw = read_json_file(root / "selection.json", warnings)
    parent_baseline_raw = read_json_file(root / "parent-baseline.json", warnings)
    supervisor_raw = read_json_file(root / "supervisor-state.json", warnings)
    supervisor_heartbeat_raw = read_json_file(root / "supervisor-heartbeat.json", warnings)
    acceptance_raw = read_json_file(root / "acceptance" / "phase-0051.json", warnings)
    updates_file = read_jsonl(root / "updates.jsonl", warnings, limit=MAX_SERIES_RECORDS)
    validation_file = read_jsonl(root / "validation.jsonl", warnings, limit=MAX_SERIES_RECORDS)
    parent_file = read_jsonl(root / "parent-history.jsonl", warnings, limit=MAX_PARENT_RECORDS)
    train_series = [
        row for row in (normalize_update(raw) for raw in updates_file["rows"]) if row is not None
    ]
    validation_series = [
        row
        for row in (normalize_validation(raw) for raw in validation_file["rows"])
        if row is not None
    ]
    parent_baseline = normalize_parent_baseline(parent_baseline_raw)
    parent_history = build_parent_history(parent_file, parent_baseline, train_series)
    selection = build_selection(run_status, selection_raw)
    last_checkpoint = build_last_checkpoint(run_status, validation_series, selection)
    acceptance = normalize_acceptance(acceptance_raw)
    last_sha = last_checkpoint.get("sha256") if last_checkpoint else None
    if last_sha and acceptance.get("checkpoint_sha256"):
        acceptance["matches_last_checkpoint"] = last_sha == acceptance["checkpoint_sha256"]
    selection_sha = selection.get("checkpoint_sha256")
    if selection_sha and acceptance.get("checkpoint_sha256"):
        acceptance["matches_selection_checkpoint"] = (
            selection_sha == acceptance["checkpoint_sha256"]
        )
    supervisor = build_supervisor(supervisor_raw, supervisor_heartbeat_raw, acceptance, moment)
    process = build_process(run_status, heartbeat_raw, heartbeat_block, moment, supervisor)
    data_present = bool(train_series or validation_series or selection_raw or heartbeat_raw)
    status = build_status(
        run_status,
        heartbeat_raw,
        heartbeat_block,
        process,
        moment,
        data_present=data_present,
        supervisor=supervisor,
    )
    progress_block = build_progress(run_status, train_series, parent_history, supervisor)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "generated_at": now_iso(moment),
        "poll_seconds": as_int(poll_seconds) or DEFAULT_POLL_SECONDS,
        "run_root": display_path(root),
        "run_root_basename": root.name,
        "config": config,
        "status": status["status"],
        "status_declared": status["status_declared"],
        "status_declared_raw": status["status_declared_raw"],
        "status_detail": status["status_detail"],
        "phase": status["phase"],
        "stale": status["stale"],
        "stale_reason": status["stale_reason"],
        "status_updated_at": status["updated_at"],
        "status_started_at": status["started_at"],
        "status_finished_at": status["finished_at"],
        "native_exit": status["native_exit"],
        "process": process,
        "heartbeat": heartbeat_block,
        "progress": progress_block,
        "train_series": train_series,
        "validation_series": validation_series,
        "selection": selection,
        "last_checkpoint": last_checkpoint,
        "error": build_error(run_status),
        "disk": build_disk_block(progress_block),
        "action_diagnostic": build_action_diagnostic(validation_series),
        "closed_loop": dict(CLOSED_LOOP),
        "parent_history": parent_history,
        "supervisor": supervisor,
        "acceptance": acceptance,
        "read_warnings": warnings[:MAX_WARNINGS],
        "series_read": {
            "updates": {
                "present": updates_file["present"],
                "torn_tail": updates_file["torn_tail"],
                "invalid_lines": updates_file["invalid_lines"],
                "truncated": updates_file["truncated"],
            },
            "validation": {
                "present": validation_file["present"],
                "torn_tail": validation_file["torn_tail"],
                "invalid_lines": validation_file["invalid_lines"],
                "truncated": validation_file["truncated"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Read-only HTTP surface.
# ---------------------------------------------------------------------------


class SnapshotHandler(BaseHTTPRequestHandler):
    server_version = "owac-raw-distillation-monitor"
    run_root = None
    config_path = None
    poll_seconds = DEFAULT_POLL_SECONDS

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _method_not_allowed(self):
        body = json.dumps(
            {"error": "read-only monitor: only GET and HEAD are allowed"},
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def __getattr__(self, name):
        # Any HTTP method other than GET/HEAD resolves to an honest 405.
        if name.startswith("do_"):
            return self._method_not_allowed
        raise AttributeError(name)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        try:
            if parsed.query or parsed.path not in SERVED_ROUTES:
                self._json(404, {"error": "not found"})
                return
            if parsed.path in (ROUTE_ROOT, ROUTE_INDEX):
                asset = Path(__file__).with_name(HTML_NAME)
                self._send(200, asset.read_bytes(), "text/html; charset=utf-8")
                return
            snapshot = build_snapshot(
                self.run_root or DEFAULT_RUN_ROOT,
                self.config_path,
                poll_seconds=self.poll_seconds,
            )
            self._json(200, snapshot)
        except (BrokenPipeError, ConnectionResetError):
            return
        except OSError as exc:
            self._json(500, {"error": f"internal read error: {type(exc).__name__}"})

    def do_HEAD(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self.do_GET()

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        return


def create_server(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    run_root=None,
    config_path=None,
    poll_seconds=DEFAULT_POLL_SECONDS,
):
    """Create a ThreadingHTTPServer for tests and the CLI; callers own shutdown."""
    root = Path(run_root) if run_root is not None else DEFAULT_RUN_ROOT

    class _BoundHandler(SnapshotHandler):
        pass

    _BoundHandler.run_root = root
    _BoundHandler.config_path = Path(config_path) if config_path is not None else None
    _BoundHandler.poll_seconds = poll_seconds
    server = ThreadingHTTPServer((host, int(port)), _BoundHandler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OWAC C-full RAW->RAW token distillation read-only monitor"
    )
    parser.add_argument("--config", type=Path, required=True, help="frozen new-stage config JSON")
    parser.add_argument("--run-root", type=Path, default=None, help="configured run root")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        print(
            f"[monitor] refusing non-loopback host {args.host!r} (local-only monitor)",
            file=sys.stderr,
            flush=True,
        )
        return 2
    poll_seconds = args.poll_seconds if 1 <= args.poll_seconds <= 60 else DEFAULT_POLL_SECONDS
    root = resolve_run_root(args.config, args.run_root)
    try:
        server = create_server(args.host, args.port, root, args.config, poll_seconds)
    except OSError as exc:
        print(
            f"[monitor] bind {args.host}:{args.port} failed ({exc})",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        f"[monitor] serving http://{args.host}:{args.port}/ "
        f"run_root={display_path(root)} config={args.config.name} pid={os.getpid()}",
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
