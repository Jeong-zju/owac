"""Unified P0 CLI for the N02 full (v1.1) line: inventory, parity, profile, report.

    python -m owac.evaluation.n02_full.p0 inventory --config ... --output ...
    python -m owac.evaluation.n02_full.p0 parity    --config ... --output ...
    python -m owac.evaluation.n02_full.p0 profile   --config ... --output ...
    python -m owac.evaluation.n02_full.p0 report    --config ... --output ...

Every subcommand writes one JSON document (``<output>/<subcommand>.json``; the
report writes ``report.json``, ``metrics.json``, ``run_manifest.json`` and
``stage_decision.json``). Missing upstream modules, missing data roots and
upstream call failures become ``status="not_run"`` with a recorded reason and
exit code 0 -- the CLI never fabricates a passing result and never claims
"import works" as "algorithm verified". A malformed config is the only error
that exits non-zero.

Profile target protocol (``profile_targets.build_profile_targets(config)``)
--------------------------------------------------------------------------
The suite returns a sequence of targets. Each target is either an object or a
mapping and must expose:

* a name: ``name`` (fallback ``id``, then ``target_<index>``; for a dict
  container the container key is used as the fallback name);
* one update callable: object attribute / mapping key ``update``, ``step``,
  ``run`` or ``fn``, or a callable object itself (a ``torch.nn.Module`` is
  callable, so its ``forward`` is the update).

Optional metadata, read from the entry and then from its ``profile_meta`` dict,
never invented:

* ``warmup`` -- called once before the warmup updates;
* ``parameter_count`` / ``num_parameters`` / ``num_params`` / ``params`` -- int
  (``profile_meta["params"]`` is read directly, never recomputed by guessing);
* ``batch_size``, ``sequence_length``, ``frames_per_update``,
  ``throughput_unit`` (``"frames"`` or ``"samples"``);
* ``composed_parts`` -- names of measured targets this target composes;
* ``notes`` -- sequence of strings.

Module targets are driven from their declared ``profile_meta`` (the frozen
component contract returns ``dict[str, nn.Module]``): the update input is
``zeros(input_shape)`` in the module's dtype/device, a required ``mask`` parameter
gets the patch mask derived from that shape, parameters that carry a default are
left to the module, and any other required parameter makes the target
``not_measured`` with that reason instead of a guessed input. A module target is
moved to the requested CUDA device for its measurement and returned to cpu
afterwards, so each target's peak memory is its own residency.

Timing is real wall time around real update calls; peak CUDA memory comes from
``torch.cuda.max_memory_allocated/reserved`` and stays ``null`` on CPU with an
entry in the ``not_measured`` column. Every ``torch.cuda`` call happens only when
the effective device type is ``cuda`` and CUDA is usable, so a ``--device cpu``
run on a CUDA host never touches a CUDA API (``torch.cuda.get_device_name("cpu")``
raises there) and reports ``device_name="cpu"`` with ``cuda_queried=false``. The
output is always split into the three columns ``measured`` /
``composed_estimate`` / ``not_measured``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from owac.frameworks.n02_full import config as n02_config

__all__ = ["SUBCOMMANDS", "build_parser", "main"]

#: Stable subcommand names; ``--help`` must list exactly these.
SUBCOMMANDS = ("inventory", "parity", "profile", "report")

DEFAULT_CONFIG = "configs/encoders/n02_full/p0.json"
DEFAULT_OUTPUT_ROOT = "outputs/n02-full/p0"

LOCK_PATH = "configs/encoders/n02_full/third_party.lock.json"

DATA_MODULE = "owac.frameworks.n02_full.data"
PARITY_MODULE = "owac.frameworks.n02_full.components.parity_suite"
PROFILE_MODULE = "owac.frameworks.n02_full.components.profile_targets"

#: Status vocabulary of this CLI's own stage payloads.
_STAGE_STATUSES = ("ok", "not_run", "incomplete", "fail")
#: Extra status words accepted verbatim from an upstream suite. ``pass`` is a real
#: ``parity_suite.run_all`` verdict (its checks passed); rewriting it into
#: "incomplete" would hide a genuine result, so it is recorded as reported.
_UPSTREAM_STATUSES = (*_STAGE_STATUSES, "pass")
#: Statuses that mean "this evidence is a successful, usable result". ``ok`` is
#: this CLI's own word, ``pass`` is the upstream suite's.
_SUCCESSFUL_STATUSES = ("ok", "pass")
_VERDICTS = ("pass", "fail", "not_run", "incomplete")

#: Keys that may carry the data/partition hash of an inventory run.
_DATA_HASH_KEYS = ("data_hash", "dataset_hash", "partition_hash", "manifest_hash")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path) -> str | None:
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    """Write canonical JSON (sorted keys, UTF-8, indent 2) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def _now() -> str:
    # datetime.UTC is 3.11+; this module keeps the 3.10-compatible spelling.
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _json_safe(payload: Any) -> Any:
    """Round-trip through json.dumps to prove the payload is serializable."""
    return json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


def git_sha() -> str | None:
    """HEAD commit of this checkout, or ``None`` when git cannot be queried."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def interpreter_info() -> dict:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
    }


def _load_attr(module_name: str, attr: str) -> tuple[Any | None, str | None]:
    """Import ``module_name`` and read ``attr``; return (value, reason-if-missing)."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return None, f"{module_name} is not importable: {exc}"
    except Exception as exc:  # noqa: BLE001 - import-time failure must not crash the CLI
        return None, f"importing {module_name} raised {type(exc).__name__}: {exc}"
    target = getattr(module, attr, None)
    if target is None:
        return None, f"{module_name}.{attr} is missing"
    return target, None


def _call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any | None, str | None, str | None]:
    """Call an upstream entry point; return (result, reason, traceback-tail)."""
    try:
        return fn(*args, **kwargs), None, None
    except Exception as exc:  # noqa: BLE001 - a failing upstream call is a not_run reason
        tail = traceback.format_exc(limit=12)
        return None, f"{type(exc).__name__}: {exc}", tail


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _find_key(payload: Any, names: tuple[str, ...], depth: int = 3) -> Any:
    """First value found under one of ``names`` within a bounded nesting depth."""
    if depth < 0:
        return None
    if isinstance(payload, dict):
        for name in names:
            value = payload.get(name)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            found = _find_key(value, names, depth - 1)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_key(value, names, depth - 1)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def _lock_summary(path: str | Path = LOCK_PATH) -> dict:
    """Summary of the tracked upstream source lock, never its full content."""
    target = Path(path)
    summary: dict[str, Any] = {
        "path": str(target),
        "available": False,
        "reason": "",
        "file_sha256": None,
        "canonical_sha256": None,
        "entry_count": None,
        "entries": [],
        "entries_truncated": False,
    }
    if not target.is_file():
        summary["reason"] = "third_party.lock.json not found; upstream pins are not recorded yet"
        return summary
    summary["file_sha256"] = sha256_file(target)
    try:
        parsed = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        summary["reason"] = f"third_party.lock.json is not readable JSON: {exc}"
        return summary
    summary["available"] = True
    summary["canonical_sha256"] = hashlib.sha256(canonical_json(parsed).encode("utf-8")).hexdigest()
    if isinstance(parsed, list):
        entries = parsed
    elif isinstance(parsed, dict):
        embedded = parsed.get("entries")
        entries = embedded if isinstance(embedded, list) else list(parsed.values())
    else:
        entries = []
    summary["entry_count"] = len(entries)
    listed: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            listed.append({"index": index, "value_type": type(entry).__name__})
            continue
        listed.append(
            {
                "index": index,
                "id": entry.get("id") or entry.get("name") or entry.get("repo") or entry.get("url"),
                "commit": entry.get("commit") or entry.get("revision") or entry.get("sha"),
                **{
                    key: entry[key]
                    for key in ("sha256", "file_sha256", "hash", "artifact_sha256", "license")
                    if key in entry
                },
            }
        )
    limit = 40
    summary["entries"] = listed[:limit]
    summary["entries_truncated"] = len(listed) > limit
    return summary


def _runtime_probe(config: n02_config.N02FullConfig) -> dict:
    """Existing runtime probe payload, referenced by its source path."""
    root = Path(str(config.paths.get("output_root") or DEFAULT_OUTPUT_ROOT))
    path = root / "p0" / "runtime" / "probe.json"
    payload: dict[str, Any] = {
        "source": str(path),
        "available": False,
        "reason": "",
        "payload": None,
    }
    if not path.is_file():
        payload["reason"] = "runtime probe not found; run the runtime probe under this path first"
        return payload
    try:
        payload["payload"] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        payload["reason"] = f"runtime probe is not readable JSON: {exc}"
        return payload
    payload["available"] = True
    payload["file_sha256"] = sha256_file(path)
    return payload


def _dataset_root_issue(cache_root: str | None, dataset_format: str | None) -> str | None:
    if not cache_root:
        return "config data.cache_root is not configured"
    root = Path(cache_root)
    if not root.exists():
        return f"config data.cache_root does not exist: {root}"
    if root.is_file():
        return f"config data.cache_root is a file, not a directory: {root}"
    try:
        empty = next(root.iterdir(), None) is None
    except OSError as exc:
        return f"config data.cache_root cannot be listed: {exc}"
    if empty:
        return f"config data.cache_root is empty: {root}"
    if not dataset_format:
        return "config data.dataset_format is not configured"
    return None


def _audit_limits(max_files: int | None, evidence_kind: str | None, result: Any) -> dict:
    """What limit actually applied to the dataset audit.

    ``max_files`` is the cap this run requested (``None`` leaves the upstream
    default, i.e. read every discovered file). ``truncated`` means "not every
    discovered file was read": the upstream report is authoritative when it
    states it (``truncated_source="upstream_report"``), otherwise the value is
    derived from the requested cap (``"derived"``). ``evidence_kind`` is the
    effective caller declaration, taken from the report when present.
    ``audit_called`` records whether the upstream audit ran at all -- on the
    early not_run paths (missing root, missing module) nothing was passed.
    """
    reported = result if isinstance(result, dict) else {}
    kind = reported.get("evidence_kind")
    if isinstance(kind, str) and kind:
        kind_source = "upstream_report"
    else:
        kind = evidence_kind
        kind_source = "requested" if evidence_kind is not None else "default"
    upstream_truncated = reported.get("truncated")
    if isinstance(upstream_truncated, bool):
        truncated, truncated_source = upstream_truncated, "upstream_report"
    else:
        truncated, truncated_source = max_files is not None, "derived"
    return {
        "max_files": max_files,
        "evidence_kind": kind,
        "evidence_kind_source": kind_source,
        "truncated": truncated,
        "truncated_source": truncated_source,
        "files_checked": reported.get("files_checked"),
        "files_total": reported.get("files_total"),
        "audit_called": result is not None,
    }


def stage_inventory(cfg: n02_config.N02FullConfig, args: argparse.Namespace) -> dict:
    output = Path(args.output)
    device = args.device or str(cfg.runtime.get("device", "cpu"))
    payload: dict[str, Any] = {
        "stage": "inventory",
        "status": "not_run",
        "reason": "",
        "generated_at": _now(),
        "config_path": str(args.config),
        "config_hash": n02_config.config_hash(cfg),
        "git_sha": git_sha(),
        "device": device,
        "interpreter": interpreter_info(),
        "data": {
            "cache_root": cfg.data.get("cache_root"),
            "dataset_format": cfg.data.get("dataset_format"),
            "protocol_verified": cfg.data.get("protocol_verified"),
        },
        "dataset": None,
        "audit_limits": _audit_limits(args.max_files, args.evidence_kind, None),
        "upstream_lock": _lock_summary(),
        "runtime_probe": _runtime_probe(cfg),
    }

    audit, missing = _load_attr(DATA_MODULE, "audit_dataset_root")
    if missing is not None:
        payload["reason"] = missing
        write_json(output / "inventory.json", payload)
        return payload

    issue = _dataset_root_issue(cfg.data.get("cache_root"), cfg.data.get("dataset_format"))
    if issue is not None:
        payload["reason"] = issue
        write_json(output / "inventory.json", payload)
        return payload

    # Only pass the limits the caller actually asked for: an explicit cap must
    # reach the audit (never be silently dropped), while an unset value keeps the
    # upstream default and stays compatible with an audit that takes no keywords.
    audit_kwargs: dict[str, Any] = {}
    if args.max_files is not None:
        audit_kwargs["max_files"] = args.max_files
    if args.evidence_kind is not None:
        audit_kwargs["evidence_kind"] = args.evidence_kind

    if cfg.data.get("stats_path"):
        audit_kwargs["stats_path"] = cfg.data["stats_path"]

    result, reason, tail = _call(
        audit, cfg.data.get("cache_root"), cfg.data.get("dataset_format"), **audit_kwargs
    )
    if reason is not None:
        payload["reason"] = f"audit_dataset_root failed: {reason}"
        payload["error_traceback"] = tail
        write_json(output / "inventory.json", payload)
        return payload

    payload["dataset"] = _json_safe(result)
    payload["audit_limits"] = _audit_limits(args.max_files, args.evidence_kind, result)
    declared = result.get("status") if isinstance(result, dict) else None
    if declared in _UPSTREAM_STATUSES:
        payload["status"] = declared
        if declared not in _SUCCESSFUL_STATUSES:
            payload["reason"] = str(result.get("reason", "")) or (
                "audit_dataset_root reported this status"
            )
    else:
        payload["status"] = "ok"
    write_json(output / "inventory.json", payload)
    return payload


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------


def stage_parity(cfg: n02_config.N02FullConfig, args: argparse.Namespace) -> dict:
    output = Path(args.output)
    device = args.device or str(cfg.runtime.get("device", "cpu"))
    provenance: dict[str, Any] = {
        "stage": "parity",
        "config_path": str(args.config),
        "config_hash": n02_config.config_hash(cfg),
        "git_sha": git_sha(),
        "device": device,
        "generated_at": _now(),
    }

    run_all, missing = _load_attr(PARITY_MODULE, "run_all")
    if missing is not None:
        payload = dict(provenance)
        payload.update({"status": "not_run", "reason": missing})
        write_json(output / "parity.json", payload)
        return payload

    result, reason, tail = _call(run_all, cfg, device=device)
    if reason is not None:
        payload = dict(provenance)
        payload.update(
            {
                "status": "not_run",
                "reason": f"parity_suite.run_all failed: {reason}",
                "error_traceback": tail,
            }
        )
        write_json(output / "parity.json", payload)
        return payload

    if isinstance(result, dict):
        body = _json_safe(result)
    elif dataclasses.is_dataclass(result) and not isinstance(result, type):
        body = _json_safe(dataclasses.asdict(result))
    elif callable(getattr(result, "as_dict", None)):
        body = _json_safe(result.as_dict())
    else:
        payload = dict(provenance)
        payload.update(
            {
                "status": "not_run",
                "reason": (
                    "parity_suite.run_all must return a dict (or a dataclass with as_dict); "
                    f"got {type(result).__name__}"
                ),
            }
        )
        write_json(output / "parity.json", payload)
        return payload

    for key, value in provenance.items():
        body.setdefault(key, value)
    body.setdefault("status", "ok")
    if body.get("status") not in _UPSTREAM_STATUSES:
        body["status"] = "incomplete"
        body.setdefault("reason", "parity_suite returned an unrecognized status")
    write_json(output / "parity.json", body)
    return body


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


class _ProfileTarget:
    """Duck-typed view over one ``build_profile_targets`` entry.

    Accepted shapes: an object or mapping with a ``name`` and one update callable
    (``update``/``step``/``run``/``fn``, or the object itself when callable), or a
    ``torch.nn.Module`` whose ``forward`` is the update. Metadata is read from the
    entry first and then from its ``profile_meta`` dict, which is how the frozen
    component contract exposes ``batch_size``/``input_shape``/``params``/``notes``.
    Nothing is guessed: a missing field stays ``None`` and is reported as not
    measured instead of being filled with a plausible number.
    """

    def __init__(self, raw: Any, index: int, fallback_name: str | None = None) -> None:
        self.raw = raw
        self.index = index
        self.fallback_name = fallback_name
        self.notes: tuple[str, ...] = ()
        note_value = self._get("notes", "note")
        if isinstance(note_value, str):
            self.notes = (note_value,)
        elif isinstance(note_value, (list, tuple)):
            self.notes = tuple(str(item) for item in note_value)

    def meta(self) -> dict:
        """The declared ``profile_meta`` mapping, or an empty dict."""
        meta = getattr(self.raw, "profile_meta", None)
        return dict(meta) if isinstance(meta, dict) else {}

    def _get(self, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(self.raw, dict):
                if self.raw.get(name) is not None:
                    return self.raw[name]
            else:
                value = getattr(self.raw, name, None)
                if value is not None:
                    return value
        meta = self.meta()
        for name in names:
            if meta.get(name) is not None:
                return meta[name]
        return default

    @property
    def name(self) -> str:
        value = self._get("name", "id", "target_id", default=None)
        if value is not None:
            return str(value)
        if self.fallback_name:
            return str(self.fallback_name)
        return f"target_{self.index}"

    @property
    def update(self) -> Any:
        value = self._get("update", "step", "run", "fn")
        if value is not None:
            return value
        if callable(self.raw):
            return self.raw
        return None

    def warmup(self) -> Any:
        return self._get("warmup")

    def parameter_count(self) -> int | None:
        for key in ("parameter_count", "num_parameters", "num_params"):
            value = self._get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        params = self.meta().get("params")
        if isinstance(params, int) and not isinstance(params, bool):
            return params
        return None

    def int_field(self, *names: str) -> int | None:
        value = self._get(*names)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def frames_per_update(self) -> tuple[int | None, str | None]:
        """Frames/samples per update plus the declaration it was derived from."""
        declared = self.int_field("frames_per_update")
        if declared is not None:
            return declared, "declared frames_per_update"
        batch = self.int_field("batch_size")
        sequence = self.int_field("sequence_length")
        if batch is not None and sequence is not None:
            return batch * sequence, "batch_size*sequence_length"
        if batch is not None:
            return batch, "batch_size (target declares no time axis)"
        return None, None

    def throughput_unit(self) -> str | None:
        value = self._get("throughput_unit")
        if value in ("frames", "samples"):
            return str(value)
        return None

    def composed_parts(self) -> tuple[str, ...]:
        value = self._get("composed_parts", "parts")
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return ()


def _device_type(torch_module: Any, device: str) -> str | None:
    """Effective device type ("cpu", "cuda", ...) or ``None`` when torch rejects it."""
    try:
        return str(torch_module.device(device).type)
    except (RuntimeError, ValueError, TypeError, AssertionError):
        return None


def _cuda_call(fn: Any, *args: Any) -> tuple[Any, str | None]:
    """Call a CUDA query, returning (value, error-text) instead of raising."""
    try:
        return fn(*args), None
    except Exception as exc:  # noqa: BLE001 - an unusable CUDA query is a reason, not a crash
        return None, f"{type(exc).__name__}: {exc}"


def _runtime_info(device: str) -> dict:
    """Runtime probe for the *effective* device only.

    A CUDA host must not be queried when the run targets ``cpu``: on such a host
    ``torch.cuda.get_device_name("cpu")`` raises ``ValueError``, and the run's
    device is genuinely CPU. So every ``torch.cuda`` call happens inside the
    ``device_type == "cuda"`` branch and a non-CUDA run touches no CUDA API at
    all (not even ``is_available``); ``cuda_queried`` records which branch ran.

    ``device_name`` is ``"cpu"`` for a CPU device, the queried name for a usable
    CUDA device, and ``None`` with a ``reason`` otherwise. ``cuda_available`` is
    only filled when it was actually queried, so it cannot look like a host fact
    that this run never checked.
    """
    info: dict[str, Any] = {
        "device": device,
        "device_type": None,
        "device_name": None,
        "torch_version": None,
        "cuda_version": None,
        "cuda_available": None,
        "cuda_queried": False,
        "reason": "",
    }
    try:
        import torch  # noqa: PLC0415 - torch is imported only for a real profile run
    except ImportError as exc:
        info["reason"] = f"torch is not importable: {exc}"
        return info
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    device_type = _device_type(torch, device)
    info["device_type"] = device_type

    if device_type != "cuda":
        if device_type == "cpu":
            info["device_name"] = "cpu"
            info["reason"] = (
                f"device {device!r} has type 'cpu'; no CUDA API was queried for this run"
            )
        else:
            info["reason"] = (
                f"device {device!r} has type {device_type!r}; no CUDA device name was queried"
            )
        return info

    info["cuda_queried"] = True
    available, error = _cuda_call(torch.cuda.is_available)
    info["cuda_available"] = bool(available) if error is None else None
    if error is not None:
        info["reason"] = f"torch.cuda.is_available() raised {error}"
        return info
    if not available:
        info["reason"] = "torch.cuda.is_available() is False; no CUDA device name was queried"
        return info
    name, error = _cuda_call(torch.cuda.get_device_name, device)
    if error is not None:
        info["reason"] = f"torch.cuda.get_device_name({device!r}) failed: {error}"
        return info
    info["device_name"] = name
    return info


def _zeros_for(torch_module: Any, raw: Any, shape: tuple, fallback_device: Any) -> Any:
    """Zero tensor in the target's own dtype/device (timing does not need data)."""
    dtype = None
    device = fallback_device
    parameters = getattr(raw, "parameters", None)
    if callable(parameters):
        try:
            first = next(iter(parameters()))
            dtype = first.dtype
            device = first.device
        except (StopIteration, TypeError):
            pass
    return torch_module.zeros(shape, dtype=dtype or torch_module.float32, device=device)


def _target_inputs(
    torch_module: Any, target: _ProfileTarget, device: str
) -> tuple[list, dict, str | None]:
    """Build the inputs one target declares through ``profile_meta``.

    The frozen component contract says a profiler only needs
    ``profile_meta["input_shape"]`` (plus ``batch_size``): targets are plain
    ``nn.Module``s called with a tensor of that shape. Parameters that carry a
    default are left to the module (``fcmae_full(imgs, mask_ratio=None)``). A
    required extra ``mask`` parameter gets the patch mask derived from the same
    shape; any other required parameter is reported as not measured rather than
    guessed.
    """
    raw = target.raw
    shape = target.meta().get("input_shape")
    forward = getattr(raw, "forward", None)
    required: list[tuple[str, str]] = []
    if callable(forward):
        try:
            required = [
                (name, parameter.kind.name)
                for name, parameter in inspect.signature(forward).parameters.items()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (
                    parameter.POSITIONAL_ONLY,
                    parameter.POSITIONAL_OR_KEYWORD,
                    parameter.KEYWORD_ONLY,
                )
            ]
        except (TypeError, ValueError):
            required = []

    if not required:
        if shape is None:
            return [], {}, None
        primary = _zeros_for(torch_module, raw, tuple(int(dim) for dim in shape), device)
        return [primary], {}, None
    if shape is None:
        return (
            [],
            {},
            (
                "target forward takes arguments but profile_meta declares no input_shape; "
                "its inputs cannot be built without guessing"
            ),
        )
    shape = tuple(int(dim) for dim in shape)
    primary = _zeros_for(torch_module, raw, shape, device)
    args: list = []
    kwargs: dict = {}
    for name, kind in required[:1]:
        if kind == "KEYWORD_ONLY":
            kwargs[name] = primary
        else:
            args.append(primary)
    for name, kind in required[1:]:
        if name == "mask" and len(shape) >= 4:
            batch, height, width = shape[0], shape[2], shape[3]
            value = torch_module.zeros((batch, height * width), dtype=torch_module.bool)
        else:
            return (
                [],
                {},
                (
                    f"target forward requires an extra parameter {name!r} that profile_meta "
                    "does not describe"
                ),
            )
        if kind == "KEYWORD_ONLY":
            kwargs[name] = value
        else:
            args.append(value)
    return args, kwargs, None


def _measure_target(
    target: _ProfileTarget,
    device: str,
    updates: int,
    warmup_updates: int,
    runtime: dict,
) -> tuple[dict | None, dict | None]:
    """Run warmup + ``updates`` real updates. Returns (measured, not_measured).

    ``runtime`` is the dict produced by :func:`_runtime_info`; peak-memory queries
    run only when it recorded that a CUDA device really was queried, so a CPU
    device never reaches a ``torch.cuda`` call.
    """
    update = target.update
    if not callable(update):
        return None, {
            "name": target.name,
            "item": "update_callable",
            "reason": "target exposes no update/step/run callable",
        }
    try:
        import torch  # noqa: PLC0415 - real update loop only
    except ImportError as exc:
        return None, {
            "name": target.name,
            "item": "torch",
            "reason": f"torch is not importable: {exc}",
        }

    cuda = bool(runtime.get("cuda_queried"))
    if (
        cuda
        and callable(getattr(update, "to", None))
        and hasattr(update, "state_dict")  # a torch module, whether or not it is the raw entry
    ):
        update.to(device)  # a module target must really run on the requested device
    args, kwargs, input_reason = _target_inputs(torch, target, device)
    if input_reason is not None:
        return None, {"name": target.name, "item": "inputs", "reason": input_reason}

    operation = target._get("operation", default="forward")
    optimizer = None
    if operation == "train_step":
        update.train()
        optimizer = torch.optim.AdamW(update.parameters(), lr=1e-4)
        generator = torch.Generator(device=device).manual_seed(1103)
        args = [
            torch.randn(arg.shape, generator=generator, device=arg.device, dtype=arg.dtype)
            if isinstance(arg, torch.Tensor) and arg.is_floating_point()
            else arg
            for arg in args
        ]

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [tensor for item in value.values() for tensor in tensors(item)]
        if isinstance(value, (list, tuple)):
            return [tensor for item in value for tensor in tensors(item)]
        return []

    losses: list[float] = []

    def one_update() -> Any:
        if optimizer is None:
            with torch.no_grad():
                return update(*args, **kwargs)
        optimizer.zero_grad(set_to_none=True)
        output = update(*args, **kwargs)
        # FCMAE returns its own scalar reconstruction loss. Other component
        # probes use a declared output-square objective, not a research loss.
        if isinstance(output, tuple) and output[0].ndim == 0:
            loss = output[0]
        else:
            loss = sum(t.float().square().mean() for t in tensors(output) if t.requires_grad)
        if not torch.isfinite(loss):
            raise ValueError("non-finite profile loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(update.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        return None

    warmup = target.warmup()
    if callable(warmup):
        try:
            warmup()
        except Exception as exc:  # noqa: BLE001
            return None, {
                "name": target.name,
                "item": "warmup",
                "reason": f"warmup hook raised {type(exc).__name__}: {exc}",
            }

    try:
        for _ in range(warmup_updates):
            one_update()
    except Exception as exc:  # noqa: BLE001
        return None, {
            "name": target.name,
            "item": "warmup_updates",
            "reason": f"warmup update raised {type(exc).__name__}: {exc}",
        }

    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    try:
        for _ in range(updates):
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            one_update()
            if cuda:
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
    except Exception as exc:  # noqa: BLE001
        return None, {
            "name": target.name,
            "item": "measured_updates",
            "reason": f"update raised {type(exc).__name__}: {exc}",
        }

    peak_allocated = None
    peak_reserved = None
    cleanup_note = None
    if cuda:
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        # Hand the device back so the next target's peak is its own residency, not
        # this target's leftovers (targets are measured one after another).
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
            args.clear()
            kwargs.clear()
            update.to("cpu")
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - a failed cleanup is a note, not a lost result
            cleanup_note = f"cuda cleanup after measuring failed: {type(exc).__name__}: {exc}"
    peak_source = (
        "torch.cuda.max_memory_allocated/reserved"
        if cuda
        else f"not measured: {runtime.get('reason') or 'no CUDA device was queried'}"
    )

    frames, throughput_basis = target.frames_per_update()
    mean_seconds = statistics.fmean(durations) if durations else None
    throughput = None
    if frames is not None and mean_seconds:
        throughput = frames / mean_seconds

    batch = target.int_field("batch_size")
    sequence = target.int_field("sequence_length")
    measured = {
        "name": target.name,
        "operation": operation,
        "timing_synchronized": cuda,
        "optimizer": "AdamW" if optimizer is not None else None,
        "optimizer_steps": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "parameter_count": target.parameter_count(),
        "updates": updates,
        "warmup_updates": warmup_updates,
        "seconds_per_update": {
            "mean": mean_seconds,
            "p50": _percentile(durations, 0.5),
            "p95": _percentile(durations, 0.95),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "seconds_total": sum(durations),
        "throughput_per_second": throughput,
        "throughput_unit": target.throughput_unit() or ("samples" if frames is not None else None),
        "throughput_basis": throughput_basis,
        "effective_batch_size": batch,
        "sequence_length": sequence,
        "dtype": None,
        "grad_enabled": optimizer is not None,
        "module_training": getattr(target.raw, "training", None),
        "peak_memory": {
            "allocated_bytes": peak_allocated,
            "reserved_bytes": peak_reserved,
            "source": peak_source,
            "scope": (
                "peak since reset_peak_memory_stats for this target; the target is "
                "returned to cpu before the next target is reset"
                if cuda
                else None
            ),
        },
        "notes": list(target.notes) + ([cleanup_note] if cleanup_note else []),
    }
    return measured, None


def stage_profile(cfg: n02_config.N02FullConfig, args: argparse.Namespace) -> dict:
    output = Path(args.output)
    device = args.device or str(cfg.runtime.get("device", "cpu"))
    payload: dict[str, Any] = {
        "stage": "profile",
        "status": "not_run",
        "reason": "",
        "generated_at": _now(),
        "config_path": str(args.config),
        "config_hash": n02_config.config_hash(cfg),
        "git_sha": git_sha(),
        "runtime": _runtime_info(device),
        "protocol": {
            "updates": args.max_updates,
            "warmup_updates": cfg.profile.get("warmup_updates"),
            "dtype": cfg.runtime.get("dtype"),
            "amp": cfg.runtime.get("amp"),
        },
        "measured": {"targets": []},
        "composed_estimate": {"available": False, "reason": "", "targets": []},
        "not_measured": {"targets": [], "items": []},
    }

    build, missing = _load_attr(PROFILE_MODULE, "build_profile_targets")
    if missing is not None:
        payload["reason"] = missing
        _mark_profile_gaps(payload, device)
        write_json(output / "profile.json", payload)
        return payload

    declared_updates = cfg.profile.get("updates")
    updates = args.max_updates if args.max_updates is not None else declared_updates
    warmup_updates = cfg.profile.get("warmup_updates", 20)
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        payload["reason"] = f"config.profile.updates must be a positive integer, got {updates!r}"
        _mark_profile_gaps(payload, device)
        write_json(output / "profile.json", payload)
        return payload
    if (
        not isinstance(warmup_updates, int)
        or isinstance(warmup_updates, bool)
        or warmup_updates < 0
    ):
        payload["reason"] = f"config.profile.warmup_updates must be >= 0, got {warmup_updates!r}"
        _mark_profile_gaps(payload, device)
        write_json(output / "profile.json", payload)
        return payload
    payload["protocol"]["updates"] = updates
    payload["protocol"]["warmup_updates"] = warmup_updates

    targets, reason, tail = _call(build, cfg)
    if reason is not None:
        payload["reason"] = f"build_profile_targets failed: {reason}"
        payload["error_traceback"] = tail
        _mark_profile_gaps(payload, device)
        write_json(output / "profile.json", payload)
        return payload
    if targets is None:
        payload["reason"] = "build_profile_targets returned None"
        _mark_profile_gaps(payload, device)
        write_json(output / "profile.json", payload)
        return payload
    # A dict container carries the real target names as its keys (the frozen
    # component contract returns dict[str, nn.Module]); keep them as the fallback
    # name for entries that expose no ``name`` of their own.
    if isinstance(targets, dict):
        target_items = [(str(key), value) for key, value in targets.items()]
    else:
        target_items = [(None, value) for value in list(targets)]
    if not target_items:
        payload["reason"] = "build_profile_targets returned no targets"
        _mark_profile_gaps(payload, device)
        write_json(output / "profile.json", payload)
        return payload

    measured_names: set[str] = set()
    for index, (fallback_name, raw) in enumerate(target_items):
        adapter = _ProfileTarget(raw, index, fallback_name)
        measured, failure = _measure_target(
            adapter, device, updates, warmup_updates, payload["runtime"]
        )
        if measured is not None:
            measured["dtype"] = cfg.runtime.get("dtype")
            measured["device_name"] = payload["runtime"].get("device_name")
            payload["measured"]["targets"].append(measured)
            measured_names.add(measured["name"])
        if failure is not None:
            payload["not_measured"]["targets"].append(failure)

    payload["composed_estimate"] = _composed_estimate(target_items, payload["measured"]["targets"])
    _mark_profile_gaps(payload, device)
    measured_count = len(payload["measured"]["targets"])
    if measured_count == 0:
        payload["status"] = "not_run"
        if not payload["reason"]:
            payload["reason"] = "no profile target produced a measured update"
    elif measured_count < len(target_items):
        payload["status"] = "incomplete"
        payload["reason"] = f"{measured_count}/{len(target_items)} targets measured"
    else:
        payload["status"] = "ok"
    write_json(output / "profile.json", payload)
    return payload


def _composed_estimate(targets: list[tuple], measured: list[dict]) -> dict:
    """Sum declared parts of measured targets; never invent a whole-model number."""
    by_name = {entry["name"]: entry for entry in measured}
    rows = []
    for index, (fallback_name, raw) in enumerate(targets):
        adapter = _ProfileTarget(raw, index, fallback_name)
        parts = adapter.composed_parts()
        if not parts:
            continue
        missing = [name for name in parts if name not in by_name]
        if missing:
            rows.append(
                {
                    "name": adapter.name,
                    "status": "not_run",
                    "reason": f"composed part(s) not measured: {', '.join(missing)}",
                    "basis": list(parts),
                }
            )
            continue
        seconds = sum(by_name[name]["seconds_per_update"]["mean"] for name in parts)
        rows.append(
            {
                "name": adapter.name,
                "status": "incomplete",
                "basis": list(parts),
                "estimated_seconds_per_update": seconds,
                "caveat": (
                    "composed estimate: sum of measured per-part means; excludes data movement, "
                    "unmeasured stages and any module that has no profile target"
                ),
            }
        )
    if not rows:
        return {
            "available": False,
            "reason": (
                "no target declared composed_parts; stages without an implemented target are "
                "not estimated"
            ),
            "targets": [],
        }
    return {"available": True, "reason": "", "targets": rows}


def _mark_profile_gaps(payload: dict, device: str) -> None:
    """Record every column this run could not measure, with the reason.

    The peak-memory entries follow ``runtime.cuda_queried``, so they describe the
    actual branch that ran instead of assuming "not cuda" from the device string.
    """
    items = payload["not_measured"]["items"]
    runtime = payload["runtime"]
    if not runtime.get("cuda_queried"):
        reason = runtime.get("reason") or f"device {device!r} did not query a CUDA device"
        for name in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
            items.append({"name": name, "reason": reason})
    runtime_reason = runtime.get("reason")
    if runtime_reason:
        items.append({"name": "runtime_probe", "reason": runtime_reason})
    if not payload["composed_estimate"]["targets"]:
        if not payload["composed_estimate"].get("reason"):
            payload["composed_estimate"]["reason"] = (
                "no target declared composed_parts; stages without an implemented target "
                "are not estimated"
            )
        items.append(
            {
                "name": "composed_estimate",
                "reason": payload["composed_estimate"]["reason"],
            }
        )
    for entry in payload["not_measured"]["targets"]:
        items.append({"name": entry["name"], "reason": entry["reason"]})


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _evidence_candidates(stage: str, output: Path, cfg: n02_config.N02FullConfig) -> list[Path]:
    output_root = Path(str(cfg.paths.get("output_root") or DEFAULT_OUTPUT_ROOT))
    candidates = [
        output.parent / stage / f"{stage}.json",
        output / f"{stage}.json",
        output_root / "p0" / stage / f"{stage}.json",
        Path(DEFAULT_OUTPUT_ROOT) / stage / f"{stage}.json",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _read_evidence(
    stage: str, output: Path, cfg: n02_config.N02FullConfig
) -> tuple[dict | None, Path | None, str | None]:
    for candidate in _evidence_candidates(stage, output, cfg):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return None, candidate, f"{candidate} is not readable JSON: {exc}"
        if not isinstance(payload, dict):
            return None, candidate, f"{candidate} does not contain a JSON object"
        return payload, candidate, None
    return None, None, f"no {stage} evidence found under {output.parent}"


def _verdict_from(payload: dict | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if value in _VERDICTS else None


def _evidence_summary(entry: dict) -> dict:
    """One evidence record for the report: path, status, reason and file hash."""
    payload = entry.get("payload")
    reason = entry.get("reason")
    if not reason and isinstance(payload, dict) and payload.get("status") != "ok":
        reason = payload.get("reason") or None
    path = entry.get("path")
    return {
        "path": str(path) if path is not None else None,
        "status": entry.get("status"),
        "reason": reason,
        "sha256": sha256_file(path) if path is not None else None,
    }


def build_stage_decision(
    evidence: dict[str, dict],
    artifacts: list[Path],
    *,
    git_sha_value: str | None = None,
    config_hash_value: str | None = None,
    data_hash_value: str | None = None,
) -> dict:
    """Four independent verdicts derived only from the evidence actually read.

    The decision carries its own provenance (git SHA, config hash, data hash) so a
    reader does not have to unpack ``report.json`` to see what the verdicts refer
    to. A hash that could not be computed stays ``None``.
    """
    statuses = {stage: entry.get("status") for stage, entry in evidence.items()}
    ran = [
        stage
        for stage, status in statuses.items()
        if status in (*_SUCCESSFUL_STATUSES, "incomplete")
    ]
    failed = [stage for stage, status in statuses.items() if status == "fail"]

    notes: list[str] = []
    if failed:
        implementation = "fail"
        notes.append(f"failing stage evidence: {', '.join(sorted(failed))}")
    elif (
        set(statuses) == {"inventory", "parity", "profile"}
        and len(ran) == len(statuses)
        and all(statuses[stage] in _SUCCESSFUL_STATUSES for stage in ran)
    ):
        implementation = "pass"
    elif ran:
        implementation = "incomplete"
        notes.append(f"stages with real evidence: {', '.join(sorted(ran))}")
    else:
        implementation = "not_run"
        notes.append("no P0 stage produced real evidence")

    parity = evidence.get("parity", {}).get("payload")
    numerical = _verdict_from(parity, "numerical")
    if numerical is not None:
        pass
    elif isinstance(parity, dict):
        declared_status = parity.get("status")
        if declared_status in _VERDICTS:
            # The suite's own top-level status is its verdict (its vocabulary is
            # pass/fail/not_run/incomplete); the per-check counts go in the note so
            # partial coverage stays visible instead of being rounded to pass.
            numerical = declared_status
            counts = ", ".join(
                f"{key}={parity[key]}"
                for key in ("pass_count", "fail_count", "not_run_count", "checks_total")
                if isinstance(parity.get(key), int) and not isinstance(parity.get(key), bool)
            )
            note = f"numerical verdict taken from the parity_suite status {declared_status!r}"
            if counts:
                note += f" ({counts})"
            suite_reason = parity.get("reason")
            if declared_status != "pass" and isinstance(suite_reason, str) and suite_reason:
                note += f": {suite_reason}"
            notes.append(note)
        else:
            numerical = "not_run"
            notes.append(
                str(parity.get("reason") or "parity payload declared no numerical verdict")
            )
    else:
        numerical = "not_run"
        notes.append("parity evidence missing")

    if isinstance(parity, dict) and parity.get("not_run_count", 0) > 0:
        if numerical == "pass":
            numerical = "incomplete"
        if implementation == "pass":
            implementation = "incomplete"
        notes.append("required parity coverage is incomplete; passing checks do not waive not_run")
    if numerical == "fail":
        implementation = "fail"
    if isinstance(parity, dict) and parity.get("fail_count", 0) > 0:
        numerical = implementation = "fail"
        notes.append("nonzero failed checks override any declared pass")
    profile_payload = evidence.get("profile", {}).get("payload")
    if isinstance(profile_payload, dict):
        measured = profile_payload.get("measured", {}).get("targets", [])
        if isinstance(measured, list):
            required = {"convnext_raw4", "convnext_raw3", "fcmae_full", "savi_512"}
            valid = {
                row.get("name")
                for row in measured
                if row.get("operation") == "train_step"
                and row.get("updates", 0) >= 200
                and row.get("optimizer_steps", 0) >= 200
                and row.get("timing_synchronized") is True
            }
            if not required <= valid and implementation == "pass":
                implementation = "incomplete"
                notes.append("missing required synchronized 200-training-update module profiles")
    behavior = None
    for stage in ("parity", "profile", "inventory"):
        candidate = _verdict_from(evidence.get(stage, {}).get("payload"), "behavior")
        if candidate is not None:
            behavior = candidate
            break
    if behavior is None:
        behavior = "not_run"
        notes.append("no behavior evidence in P0 (behavior gates start at P1)")

    closed_loop = "not_run"
    notes.append("P0 runs no closed loop; P7 owns closed-loop evidence")

    evidence_paths = [str(path) for path in artifacts]
    evidence_paths.extend(
        str(entry["path"]) for entry in evidence.values() if entry.get("path") is not None
    )
    return {
        "implementation": implementation,
        "numerical": numerical,
        "behavior": behavior,
        "closed_loop": closed_loop,
        "evidence_paths": evidence_paths,
        "notes": notes,
        "git_sha": git_sha_value,
        "config_hash": config_hash_value,
        "data_hash": data_hash_value,
    }


def _metrics(evidence: dict[str, dict], decision: dict) -> dict:
    inventory = evidence.get("inventory", {}).get("payload")
    parity = evidence.get("parity", {}).get("payload")
    profile = evidence.get("profile", {}).get("payload")

    def pick(payload: dict | None, names: tuple[str, ...]) -> dict:
        if not isinstance(payload, dict):
            return {}
        return {name: payload[name] for name in names if name in payload}

    measured_targets = []
    if isinstance(profile, dict):
        targets = profile.get("measured", {}).get("targets", [])
        if isinstance(targets, list):
            for entry in targets:
                if not isinstance(entry, dict):
                    continue
                timing = entry.get("seconds_per_update") or {}
                memory = entry.get("peak_memory") or {}
                measured_targets.append(
                    {
                        "name": entry.get("name"),
                        "parameter_count": entry.get("parameter_count"),
                        "mean_seconds_per_update": timing.get("mean"),
                        "p95_seconds_per_update": timing.get("p95"),
                        "throughput_per_second": entry.get("throughput_per_second"),
                        "peak_allocated_bytes": memory.get("allocated_bytes"),
                        "peak_reserved_bytes": memory.get("reserved_bytes"),
                    }
                )

    return {
        "stage": "report",
        "generated_at": _now(),
        "stage_decision": decision,
        "inventory": pick(
            inventory,
            (
                "status",
                "num_episodes",
                "episode_count",
                "num_files",
                "file_count",
                "num_frames",
                "frame_count",
                "total_bytes",
            ),
        ),
        "parity": pick(
            parity,
            (
                "status",
                "numerical",
                "checks_total",
                "checks_passed",
                "checks_failed",
                "pass_count",
                "fail_count",
                "not_run_count",
                "max_abs_error",
                "max_rel_error",
                "tolerance",
            ),
        ),
        "profile": {
            "status": profile.get("status") if isinstance(profile, dict) else None,
            "updates": _profile_updates(profile),
            "targets_measured": len(measured_targets),
            "targets": measured_targets,
        },
    }


def _profile_updates(profile: Any) -> Any:
    if not isinstance(profile, dict):
        return None
    protocol = profile.get("protocol") or {}
    return protocol.get("updates") if isinstance(protocol, dict) else None


def stage_report(cfg: n02_config.N02FullConfig, args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, dict] = {}
    for stage in ("inventory", "parity", "profile"):
        payload, path, reason = _read_evidence(stage, output, cfg)
        evidence[stage] = {
            "path": path,
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "reason": reason,
            "payload": payload,
        }

    data_hash = None
    if isinstance(evidence["inventory"]["payload"], dict):
        data_hash = _find_key(evidence["inventory"]["payload"], _DATA_HASH_KEYS)
    git = git_sha()

    planned = [
        output / "stage_decision.json",
        output / "metrics.json",
        output / "report.json",
        output / "run_manifest.json",
    ]
    decision = build_stage_decision(
        evidence,
        planned,
        git_sha_value=git,
        config_hash_value=n02_config.config_hash(cfg),
        data_hash_value=data_hash,
    )
    metrics = _metrics(evidence, decision)
    write_json(output / "stage_decision.json", decision)
    write_json(output / "metrics.json", metrics)

    report = {
        "stage": "report",
        "status": "not_run",
        "reason": "",
        "generated_at": _now(),
        "config_path": str(args.config),
        "config_hash": n02_config.config_hash(cfg),
        "git_sha": git,
        "data_hash": data_hash,
        "evidence": {stage: _evidence_summary(entry) for stage, entry in evidence.items()},
        "stage_decision": decision,
        "metrics": metrics,
    }
    available = [stage for stage, entry in evidence.items() if entry["payload"] is not None]
    not_successful = [
        stage
        for stage, entry in evidence.items()
        if isinstance(entry["payload"], dict)
        and entry["payload"].get("status") not in _SUCCESSFUL_STATUSES
    ]
    if not available:
        report["reason"] = "no evidence JSON found for inventory, parity or profile"
    elif not not_successful and len(available) == 3 and decision["implementation"] == "pass":
        report["status"] = "ok"
    else:
        report["status"] = "incomplete"
        reasons = []
        missing = sorted(set(evidence) - set(available))
        if missing:
            reasons.append(f"evidence missing for: {', '.join(missing)}")
        if not_successful:
            reasons.append(f"evidence not successful for: {', '.join(sorted(not_successful))}")
        report["reason"] = "; ".join(reasons or decision["notes"])
    write_json(output / "report.json", report)

    manifest = {
        "stage": "report",
        "generated_at": _now(),
        "command": " ".join([sys.executable, "-m", "owac.evaluation.n02_full.p0", "report"]),
        "argv": list(sys.argv),
        "interpreter": interpreter_info(),
        "config_path": str(args.config),
        "config_hash": report["config_hash"],
        "git_sha": git,
        "data_hash": data_hash,
        "evidence": report["evidence"],
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in planned
            if path.name != "run_manifest.json"
        ],
    }
    write_json(output / "run_manifest.json", manifest)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, subcommand: str) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config JSON path")
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_OUTPUT_ROOT}/{subcommand}",
        help=f"output directory (default: {DEFAULT_OUTPUT_ROOT}/{subcommand})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for this run (default: config.runtime.device)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m owac.evaluation.n02_full.p0",
        description=(
            "P0 evidence CLI for the N02 full line. Missing data or upstream modules are "
            "reported as status=not_run with a reason; nothing unverified is reported as pass."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{inventory,parity,profile,report}"
    )
    inventory = subparsers.add_parser(
        "inventory", help="audit the configured dataset root and record the runtime/lock summary"
    )
    _add_common(inventory, "inventory")
    inventory.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "read at most this many NPZ files and pass the cap to audit_dataset_root "
            "(default: no cap, i.e. every discovered file is read)"
        ),
    )
    inventory.add_argument(
        "--evidence-kind",
        default=None,
        help=(
            "caller declaration of what the root is, e.g. real_cache; forwarded to "
            "audit_dataset_root (default: the audit's own default)"
        ),
    )
    parity = subparsers.add_parser(
        "parity", help="run the upstream operator parity suite and store its dict verbatim"
    )
    _add_common(parity, "parity")
    profile = subparsers.add_parser(
        "profile", help="measure warmup + N real updates per profile target"
    )
    _add_common(profile, "profile")
    profile.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="override config.profile.updates for this measurement run",
    )
    report = subparsers.add_parser(
        "report", help="compose report/metrics/manifest/stage_decision from the P0 evidence"
    )
    _add_common(report, "report")
    return parser


_STAGE_FUNCTIONS = {
    "inventory": stage_inventory,
    "parity": stage_parity,
    "profile": stage_profile,
    "report": stage_report,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = n02_config.load_config(args.config)
    except n02_config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    payload = _STAGE_FUNCTIONS[args.command](cfg, args)
    status = payload.get("status", "unknown") if isinstance(payload, dict) else "unknown"
    reason = payload.get("reason", "") if isinstance(payload, dict) else ""
    summary = f"{args.command}: status={status} output={args.output}"
    if reason:
        summary += f" reason={reason}"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
