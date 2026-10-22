"""C-full continuation supervisor: single-flight, durable, gated control plane.

This module never constructs a model, imports torch, or touches the GPU.  It
launches one owned child process per phase through the canonical continuation
runner, records the child PID/PGID/``/proc/<pid>/stat`` starttime durably, and
enforces the cumulative new-stage wall and output caps.  ``prepare`` is a
separate GPU load/validation step: ``start`` requires its successful
``prepared.json`` receipt and seeds the cumulative new-stage wall with the
receipt's real ``prepare_wall_seconds``.  Each child phase then adds its own
real wall (never reset, never truncated).  Acceptance of the phase-0051
checkpoint is an explicit operator action; the supervisor never invents a
passing acceptance or retries a failed child.

Reliability rules borrowed from the reviewed pilot supervisor:

* ``subprocess.Popen(..., start_new_session=True)`` so the child owns its own
  session and process group;
* atomic JSON replacement for state and heartbeat;
* ``flock`` on ``harness/supervisor.lock`` for the whole supervisor lifetime;
* signals are sent only to the recorded process group and only after the
  recorded starttime still matches;
* a proactive stop margin (grace + kill-wait + poll) sends SIGTERM early enough
  that the SIGKILL escalation lands strictly before the hard cumulative wall
  cap, while the real measured wall is recorded verbatim.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

STATE_SCHEMA = "owac.raw-stem-temporal.c-full.supervisor-state.v1"
HEARTBEAT_SCHEMA = "owac.raw-stem-temporal.c-full.supervisor-heartbeat.v1"
ACCEPTANCE_SCHEMA = "owac.raw-stem-temporal.c-full.phase-acceptance.v1"
STATUS_SCHEMA = "owac.raw-stem-temporal.c-full.supervisor-status.v1"
RUNNER_MODULE = "owac.runtime.raw_temporal_continuation"

WALL_CAP_SECONDS = 230400  # 64 new GPU-wall hours.
OUTPUT_CAP_BYTES = 20 * 1024**3  # 21474836480 bytes.
CHECKPOINT_RESERVE_BYTES = 2 * 1024**3
PHASE1_STOP_AFTER_UPDATE = 51
PHASE2_TARGET_UPDATES = 5000
PHASE1_ACCEPTANCE_STEP = 51

STATE_NAME = "supervisor-state.json"
HEARTBEAT_NAME = "supervisor-heartbeat.json"
LOCK_NAME = "harness/supervisor.lock"
ACCEPTANCE_NAME = "acceptance/phase-0051.json"
CHILD_LOG_NAME = "supervisor-child.log"

DEFAULT_POLL_SECONDS = 0.2
DEFAULT_HEARTBEAT_SECONDS = 2.0
DEFAULT_STOP_GRACE_SECONDS = 60.0
DEFAULT_KILL_WAIT_SECONDS = 30.0


class SupervisorError(RuntimeError):
    """A refusal or unusable state; callers map this to exit code 2."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically and durably replace ``path`` with ``value`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    try:
        directory = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_proc_stat(pid: int | None) -> tuple[str, int] | None:
    """Read ``(state, field-22 starttime)`` of ``/proc/<pid>/stat``."""
    try:
        numeric = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    try:
        raw = Path(f"/proc/{numeric}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = raw.rfind(")")
    if end < 0:
        return None
    fields = raw[end + 1 :].split()
    # ``fields[0]`` is field 3 (state) because comm is field 2; starttime is 22.
    if len(fields) < 20:
        return None
    try:
        return str(fields[0]), int(fields[19])
    except (TypeError, ValueError):
        return None


def _read_proc_start_ticks(pid: int | None) -> int | None:
    """Read field 22 (starttime) of ``/proc/<pid>/stat``; ``None`` if absent."""
    stat = _read_proc_stat(pid)
    return None if stat is None else stat[1]


def _read_proc_state(pid: int | None) -> str | None:
    """Read the single-letter process state from ``/proc/<pid>/stat``."""
    stat = _read_proc_stat(pid)
    return None if stat is None else stat[0]


def _owned_child_identity(pid: Any, pgid: Any, start_ticks: Any) -> bool:
    """True only if ``pid`` is the exact recorded, living session-leading child."""
    try:
        numeric_pid = int(pid)
        numeric_pgid = int(pgid)
        numeric_ticks = int(start_ticks)
    except (TypeError, ValueError):
        return False
    if numeric_pid <= 1 or numeric_pgid <= 1:
        return False
    if numeric_pid == os.getpid():
        return False
    # ``start_new_session=True`` makes the child both session and group leader.
    if numeric_pgid != numeric_pid:
        return False
    stat = _read_proc_stat(numeric_pid)
    if stat is None:
        return False
    state, ticks = stat
    if state == "Z":
        return False  # A zombie has already exited; the monitor calls it dead.
    if ticks != numeric_ticks:
        return False
    try:
        return os.getpgid(numeric_pid) == numeric_pgid
    except OSError:
        return False


def signal_owned_group(state: Mapping[str, Any], signum: int) -> bool:
    """Signal the recorded child process group if, and only if, it is still ours.

    The recorded starttime is re-read from ``/proc/<pid>/stat`` immediately
    before every signal, so a dead or reused PID/PGID is never signalled.
    """
    pid = state.get("child_pid")
    pgid = state.get("child_pgid")
    start_ticks = state.get("child_start_ticks")
    if not _owned_child_identity(pid, pgid, start_ticks):
        return False
    try:
        os.killpg(int(pgid), int(signum))
    except OSError:
        return False
    return True


def _run_root_output_bytes(run_root: Path) -> int:
    """Count every owned regular byte under the run root.

    Symlink entries (notably the read-only shared ``cache`` link) are excluded
    entirely and never followed; hardlink duplicates are counted once by
    ``(st_dev, st_ino)``.  Harness receipts, state, and logs under the run root
    are new output and are counted, with no partial-only accounting claim.
    """
    root = Path(run_root)
    total = 0
    seen: set[tuple[int, int]] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            iterator = os.scandir(directory)
        except OSError:
            continue
        with iterator:
            for entry in iterator:
                try:
                    if entry.is_symlink():
                        continue  # Shared cache link and any other symlink entry.
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                key = (int(info.st_dev), int(info.st_ino))
                if key in seen:
                    continue  # Hardlink duplicates are one owned payload.
                seen.add(key)
                total += int(info.st_size)
    return total


def _age_seconds(timestamp: Any) -> float | None:
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return max(0.0, (datetime.now().astimezone() - parsed).total_seconds())


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _is_hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def build_runner_command(
    *,
    inputs: Mapping[str, Any],
    run_root: Path,
    stop_after_update: int,
    resume: Path | None = None,
    acceptance: Path | None = None,
) -> list[str]:
    """Build the contract runner command from path-only inputs."""
    command = [
        str(inputs["runner_python"]),
        "-m",
        RUNNER_MODULE,
        "train",
        "--config",
        str(inputs["config"]),
        "--migration",
        str(inputs["migration"]),
        "--authority",
        str(inputs["authority"]),
        "--device",
        str(inputs["device"]),
        "--run-root",
        str(Path(run_root)),
        "--stop-after-update",
        str(int(stop_after_update)),
    ]
    if inputs.get("parity_receipt"):
        command.extend(["--parity-receipt", str(inputs["parity_receipt"])])
    if inputs.get("resource_receipt"):
        command.extend(["--resource-receipt", str(inputs["resource_receipt"])])
    if resume is not None:
        command.extend(["--resume", str(resume)])
    if acceptance is not None:
        command.extend(["--acceptance", str(acceptance)])
    return command


class Supervisor:
    """Single-flight supervisor for the C-full continuation runner."""

    def __init__(
        self,
        run_root: Path | str,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        wall_cap_seconds: int = WALL_CAP_SECONDS,
        output_cap_bytes: int = OUTPUT_CAP_BYTES,
        checkpoint_reserve_bytes: int = CHECKPOINT_RESERVE_BYTES,
        stop_grace_seconds: float = DEFAULT_STOP_GRACE_SECONDS,
        kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.poll_seconds = max(0.01, float(poll_seconds))
        self.heartbeat_seconds = max(0.01, float(heartbeat_seconds))
        self.wall_cap_seconds = int(wall_cap_seconds)
        self.output_cap_bytes = int(output_cap_bytes)
        self.checkpoint_reserve_bytes = int(checkpoint_reserve_bytes)
        self.stop_grace_seconds = max(0.0, float(stop_grace_seconds))
        self.kill_wait_seconds = max(0.0, float(kill_wait_seconds))
        # TERM is sent while at least grace + kill-wait + one poll remains, so
        # the SIGKILL escalation still lands strictly before the hard wall cap.
        self.stop_margin_seconds = (
            self.stop_grace_seconds + self.kill_wait_seconds + self.poll_seconds
        )
        self._lock_handle: Any | None = None
        self._stop_signal: int | None = None
        self._heartbeat_seq = 0
        self._last_heartbeat = 0.0
        self._previous_handlers: dict[int, Any] = {}

    # ------------------------------------------------------------------ paths

    @property
    def state_path(self) -> Path:
        return self.run_root / STATE_NAME

    @property
    def heartbeat_path(self) -> Path:
        return self.run_root / HEARTBEAT_NAME

    @property
    def lock_path(self) -> Path:
        return self.run_root / LOCK_NAME

    @property
    def acceptance_path(self) -> Path:
        return self.run_root / ACCEPTANCE_NAME

    @property
    def child_log_path(self) -> Path:
        return self.run_root / CHILD_LOG_NAME

    def _ensure_run_root(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- lock

    def acquire_lock(self) -> bool:
        self._ensure_run_root()
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self._lock_handle = handle
        return True

    def release_lock(self) -> None:
        handle = self._lock_handle
        self._lock_handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> dict[str, Any]:
        value = _read_json(self.state_path)
        if not isinstance(value, Mapping):
            raise SupervisorError("supervisor state is missing or unreadable")
        if value.get("schema") != STATE_SCHEMA:
            raise SupervisorError("supervisor state has a different schema")
        return dict(value)

    def _save_state(self, state: Mapping[str, Any]) -> None:
        state["updated_at"] = _now_iso()  # type: ignore[index]
        _atomic_json(self.state_path, state)

    def _write_heartbeat(self, state: Mapping[str, Any], *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_heartbeat < self.heartbeat_seconds:
            return False
        self._bootstrap_heartbeat_seq()
        self._heartbeat_seq += 1
        payload = {
            "schema": HEARTBEAT_SCHEMA,
            "pid": os.getpid(),
            "pid_start_ticks": _read_proc_start_ticks(os.getpid()),
            "child_pid": state.get("child_pid"),
            "child_pgid": state.get("child_pgid"),
            "child_start_ticks": state.get("child_start_ticks"),
            "seq": self._heartbeat_seq,
            "updated_at": _now_iso(),
            "status": state.get("status"),
        }
        _atomic_json(self.heartbeat_path, payload)
        self._last_heartbeat = now
        return True

    def _bootstrap_heartbeat_seq(self) -> None:
        if self._heartbeat_seq:
            return
        value = _read_json(self.heartbeat_path)
        if isinstance(value, Mapping):
            try:
                self._heartbeat_seq = max(0, int(value.get("seq") or 0))
            except (TypeError, ValueError):
                self._heartbeat_seq = 0

    def _recorded_child_alive(self, state: Mapping[str, Any]) -> bool:
        return _owned_child_identity(
            state.get("child_pid"),
            state.get("child_pgid"),
            state.get("child_start_ticks"),
        )

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _refusal(status: str, message: str) -> dict[str, Any]:
        return {"status": status, "exit_code": 2, "error": message}

    @staticmethod
    def _require_file(path: Path, label: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise SupervisorError(f"{label} is not a readable file: {resolved}")
        return resolved

    def _resolve_inputs(
        self,
        *,
        config: Path | str,
        migration: Path | str,
        authority: Path | str,
        runner_python: Path | str,
        device: str,
        parity_receipt: Path | str | None,
        resource_receipt: Path | str | None,
    ) -> dict[str, Any]:
        # Preserve the venv executable spelling: resolving bin/python symlinks
        # invokes the base interpreter and discards its site-packages.
        runner = Path(os.path.abspath(Path(runner_python).expanduser()))
        if not runner.is_file():
            raise SupervisorError(f"runner python is not a readable file: {runner}")
        if not os.access(runner, os.X_OK):
            raise SupervisorError(f"runner python is not executable: {runner}")
        if not str(device).strip():
            raise SupervisorError("device must be non-empty")
        return {
            "runner_python": str(runner),
            "config": str(self._require_file(Path(config), "config")),
            "migration": str(self._require_file(Path(migration), "migration manifest")),
            "authority": str(self._require_file(Path(authority), "authority")),
            "device": str(device),
            "parity_receipt": (
                str(self._require_file(Path(parity_receipt), "parity receipt"))
                if parity_receipt is not None
                else None
            ),
            "resource_receipt": (
                str(self._require_file(Path(resource_receipt), "resource receipt"))
                if resource_receipt is not None
                else None
            ),
        }

    def _inputs_from_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        value = state.get("inputs")
        if not isinstance(value, Mapping):
            raise SupervisorError("supervisor state lacks path-only command inputs")
        required = ("runner_python", "config", "migration", "authority", "device")
        inputs: dict[str, Any] = {}
        for key in required:
            if value.get(key) in (None, ""):
                raise SupervisorError(f"supervisor state input {key!r} is missing")
            inputs[key] = str(value[key])
        for key in ("parity_receipt", "resource_receipt"):
            raw = value.get(key)
            inputs[key] = str(raw) if raw not in (None, "") else None
        if not os.access(inputs["runner_python"], os.X_OK):
            raise SupervisorError("recorded runner python is no longer executable")
        return inputs

    def _failed_preflight_ledger(self) -> dict[str, Any]:
        """Sum each actual failed-preflight debit exactly once.

        The successful prepare wall is seeded separately from ``prepared.json``;
        these operator receipts cover failed GPU preflight attempts that never
        reached an optimizer update.  Negative, non-finite, malformed, or
        update-claiming receipts refuse start rather than under-counting cost.
        """
        receipts: list[dict[str, Any]] = []
        total = 0.0
        for path in sorted(self.run_root.glob("prepare-attempt-*-failure.json")):
            if path.is_symlink() or not path.is_file():
                raise SupervisorError(f"failed-preflight receipt is not a regular file: {path}")
            value = _read_json(path)
            if not isinstance(value, Mapping):
                raise SupervisorError(f"failed-preflight receipt is unreadable: {path.name}")
            if value.get("schema") != "owac.sol.c-full.failed-preflight-debit.v1":
                raise SupervisorError(f"failed-preflight receipt schema differs: {path.name}")
            raw = value.get("charged_wall_seconds")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise SupervisorError(
                    f"failed-preflight receipt lacks numeric charged_wall_seconds: {path.name}"
                )
            seconds = float(raw)
            if not math.isfinite(seconds) or seconds < 0:
                raise SupervisorError(
                    f"failed-preflight charged_wall_seconds must be finite and non-negative: "
                    f"{path.name}"
                )
            if value.get("optimizer_updates") not in (0, None):
                raise SupervisorError(
                    f"failed-preflight receipt claims optimizer updates: {path.name}"
                )
            receipts.append(
                {
                    "path": _relative_path(path, self.run_root),
                    "sha256": _sha256_file(path),
                    "charged_wall_seconds": seconds,
                }
            )
            total += seconds
        return {
            "charged_wall_seconds": total,
            "receipts": receipts,
        }

    def _require_prepared_receipt(self) -> dict[str, Any]:
        """Require the separate ``prepare`` receipt and its failed-preflight debits.

        ``prepare`` is not launched by the supervisor; the successful receipt
        seeds cumulative new-stage wall, and each explicit failed-preflight debit
        is added exactly once here.  Resume never recomputes this ledger.
        """
        path = self.run_root / "prepared.json"
        value = _read_json(path)
        if not isinstance(value, Mapping):
            raise SupervisorError(
                "prepared.json is absent or unreadable; run the separate prepare step before start"
            )
        raw = value.get("prepare_wall_seconds")
        try:
            prepare_wall_seconds = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise SupervisorError(
                "prepared.json lacks a numeric prepare_wall_seconds; refusing start"
            ) from None
        if not math.isfinite(prepare_wall_seconds) or prepare_wall_seconds < 0:
            raise SupervisorError("prepared.json prepare_wall_seconds is invalid")
        failed = self._failed_preflight_ledger()
        return {
            "path": "prepared.json",
            "sha256": _sha256_file(path),
            "prepare_wall_seconds": prepare_wall_seconds,
            "failed_preflight_charge_seconds": float(failed["charged_wall_seconds"]),
            "failed_preflight_receipts": list(failed["receipts"]),
        }

    def _initial_state(
        self,
        *,
        inputs: Mapping[str, Any],
        stop_after_update: int,
        prepared: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = _now_iso()
        return {
            "schema": STATE_SCHEMA,
            "status": "created",
            "phase": "created",
            "supervisor_pid": os.getpid(),
            "supervisor_pid_start_ticks": _read_proc_start_ticks(os.getpid()),
            "child_pid": None,
            "child_pgid": None,
            "child_start_ticks": None,
            "child_native_exit": None,
            # Successful prepare wall plus the explicit failed-preflight debits
            # is the cumulative starting point; each child phase adds its own
            # real wall on top and resume never recomputes/re-adds the debits.
            "prepare_wall_seconds": float(prepared["prepare_wall_seconds"]),
            "failed_preflight_charge_seconds": float(
                prepared.get("failed_preflight_charge_seconds", 0.0)
            ),
            "failed_preflight_receipts": list(prepared.get("failed_preflight_receipts", [])),
            "prepared_receipt": dict(prepared),
            "new_stage_wall_seconds": float(prepared["prepare_wall_seconds"])
            + float(prepared.get("failed_preflight_charge_seconds", 0.0)),
            "new_stage_wall_cap_seconds": self.wall_cap_seconds,
            "stop_margin_seconds": self.stop_margin_seconds,
            "wall_stop_deadline": max(0.0, float(self.wall_cap_seconds) - self.stop_margin_seconds),
            "new_stage_output_bytes": 0,
            "new_stage_output_cap_bytes": self.output_cap_bytes,
            "checkpoint_reserve_bytes": self.checkpoint_reserve_bytes,
            "last_error": None,
            "termination_reason": None,
            "acceptance": None,
            "phase1_checkpoint": None,
            "phase_history": [],
            "runner_command": [],
            "inputs": dict(inputs),
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
        }

    # --------------------------------------------------------------- commands

    def start(
        self,
        *,
        config: Path | str,
        migration: Path | str,
        authority: Path | str,
        runner_python: Path | str,
        stop_after_update: int,
        device: str = "cuda:0",
        parity_receipt: Path | str | None = None,
        resource_receipt: Path | str | None = None,
    ) -> dict[str, Any]:
        if not self.acquire_lock():
            return self._refusal(
                "locked",
                f"another supervisor holds {self.lock_path}; duplicate refusal",
            )
        try:
            if self.state_path.exists():
                existing = _read_json(self.state_path)
                if isinstance(existing, Mapping):
                    if self._recorded_child_alive(existing):
                        return self._refusal(
                            "child_alive",
                            "a recorded child process group is still alive; refusing start",
                        )
                    if existing.get("status") in {
                        "waiting_acceptance",
                        "completed",
                        "paused",
                    }:
                        return self._refusal(
                            "state_exists",
                            "supervisor state already exists; use resume instead of start",
                        )
                return self._refusal(
                    "state_exists",
                    "supervisor state already exists (or is unreadable); no automatic start retry",
                )
            if int(stop_after_update) != PHASE1_STOP_AFTER_UPDATE:
                return self._refusal(
                    "invalid_stop_after_update",
                    f"phase 1 must stop at cumulative {PHASE1_STOP_AFTER_UPDATE}",
                )
            inputs = self._resolve_inputs(
                config=config,
                migration=migration,
                authority=authority,
                runner_python=runner_python,
                device=device,
                parity_receipt=parity_receipt,
                resource_receipt=resource_receipt,
            )
            prepared = self._require_prepared_receipt()
            state = self._initial_state(
                inputs=inputs,
                stop_after_update=int(stop_after_update),
                prepared=prepared,
            )
            self._save_state(state)
            self._write_heartbeat(state, force=True)
            return self._launch(
                state,
                phase="phase-1",
                phase_number=1,
                stop_after_update=int(stop_after_update),
                resume_path=None,
                acceptance_path=None,
            )
        except SupervisorError as exc:
            return self._refusal("refused", str(exc))
        finally:
            self.release_lock()

    def resume(
        self,
        *,
        stop_after_update: int,
        acceptance: Path | str | None = None,
    ) -> dict[str, Any]:
        if not self.acquire_lock():
            return self._refusal(
                "locked",
                f"another supervisor holds {self.lock_path}; duplicate refusal",
            )
        try:
            try:
                state = self._load_state()
            except SupervisorError as exc:
                return self._refusal("state_invalid", str(exc))
            if self._recorded_child_alive(state):
                return self._refusal(
                    "child_alive",
                    "a recorded child process group is still alive; refusing duplicate launch",
                )
            status = state.get("status")
            if status not in {"waiting_acceptance", "paused"}:
                return self._refusal(
                    "not_resumable",
                    f"supervisor status {status!r} is not resumable; no automatic retry",
                )
            stop = int(stop_after_update)
            if not PHASE1_STOP_AFTER_UPDATE < stop <= PHASE2_TARGET_UPDATES:
                return self._refusal(
                    "invalid_stop_after_update",
                    f"resume stop_after_update must be in "
                    f"({PHASE1_STOP_AFTER_UPDATE}, {PHASE2_TARGET_UPDATES}]",
                )
            valid, reason, acceptance_value, acceptance_path, checkpoint_path = (
                self._validate_acceptance(state, acceptance)
            )
            if not valid or acceptance_value is None or checkpoint_path is None:
                return self._refusal("acceptance_invalid", reason)
            try:
                inputs = self._inputs_from_state(state)
            except SupervisorError as exc:
                return self._refusal("inputs_invalid", str(exc))
            state["acceptance"] = self._acceptance_summary(
                acceptance_value,
                path=acceptance_path,
                checkpoint_path=checkpoint_path,
            )
            resume_path = checkpoint_path
            resume_step = PHASE1_ACCEPTANCE_STEP
            runner_acceptance = acceptance_path
            resume_source = "phase-0051-acceptance"
            if state.get("status") == "paused":
                # Clean phase-2 pause: continue from the runner's real latest
                # checkpoint.  The 51 acceptance remains a supervisor-side gate,
                # but is not passed as the runner --acceptance for a later step.
                paused = self._discover_paused_checkpoint()
                resume_path = paused["path"]
                resume_step = int(paused["step"])
                runner_acceptance = None
                resume_source = "run-status.last_checkpoint"
            state["resume_checkpoint"] = {
                "source": resume_source,
                "step": resume_step,
                "path": _relative_path(resume_path, self.run_root),
                "sha256": _sha256_file(resume_path),
                "runner_acceptance_passed": runner_acceptance is not None,
            }
            # Base is the recorded cumulative wall; resume never resets it.
            self._save_state(state)
            return self._launch(
                state,
                phase="phase-2",
                phase_number=2,
                stop_after_update=stop,
                resume_path=resume_path,
                acceptance_path=runner_acceptance,
                inputs=inputs,
            )
        except SupervisorError as exc:
            return self._refusal("refused", str(exc))
        finally:
            self.release_lock()

    def _anchor_accepted_checkpoint(self, checkpoint_path: Path, sha: str) -> Path:
        """Hardlink the actual C51 to a permanent owned anchor (no copy/retrain).

        Training retention may prune the ordinary step-51 checkpoint in later
        phases.  The accepted-SHA anchor lives under ``lineage/`` and is never in
        the checkpoint retention set; a hardlink keeps one inode and therefore is
        counted once by output accounting.
        """
        anchor = self.run_root / "lineage" / "accepted-C51.pt"
        anchor.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.is_symlink():
            raise SupervisorError("accepted checkpoint must not be a symlink")
        if anchor.exists():
            if anchor.is_symlink() or not anchor.is_file():
                raise SupervisorError("accepted-C51.pt exists but is not a regular file")
            if _sha256_file(anchor) != sha:
                raise SupervisorError("accepted-C51.pt SHA differs from the accepted checkpoint")
            try:
                same = os.path.samefile(anchor, checkpoint_path)
            except OSError:
                same = False
            if not same:
                raise SupervisorError(
                    "accepted-C51.pt exists but is not a hardlink to the actual checkpoint"
                )
            return anchor
        temporary = anchor.with_name(f".{anchor.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            os.link(checkpoint_path, temporary)
            os.replace(temporary, anchor)
        except OSError as exc:
            raise SupervisorError(
                f"cannot hardlink accepted C51 into lineage (no copy fallback): {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        if _sha256_file(anchor) != sha:
            raise SupervisorError("accepted-C51.pt SHA differs after hardlinking")
        return anchor

    def accept(
        self,
        *,
        step: int,
        checkpoint_sha: str,
        reviewer: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        if not self.acquire_lock():
            return self._refusal(
                "locked",
                f"another supervisor holds {self.lock_path}; refusing acceptance write",
            )
        try:
            try:
                state = self._load_state()
            except SupervisorError as exc:
                return self._refusal("state_invalid", str(exc))
            if state.get("status") != "waiting_acceptance":
                return self._refusal(
                    "not_waiting_acceptance",
                    f"supervisor status {state.get('status')!r} is not waiting_acceptance",
                )
            if int(step) != PHASE1_ACCEPTANCE_STEP:
                return self._refusal(
                    "invalid_step",
                    f"phase-1 acceptance must bind cumulative step {PHASE1_ACCEPTANCE_STEP}",
                )
            sha = str(checkpoint_sha).lower()
            if not _is_hex_sha256(sha):
                return self._refusal("invalid_checkpoint_sha", "checkpoint sha must be 64 hex")
            reviewer_text = str(reviewer).strip()
            if not reviewer_text:
                return self._refusal("invalid_reviewer", "reviewer must be non-empty")
            recorded = state.get("phase1_checkpoint")
            checkpoint_path: Path | None = None
            if isinstance(recorded, Mapping) and recorded.get("path"):
                recorded_path = Path(str(recorded["path"]))
                if not recorded_path.is_absolute():
                    recorded_path = self.run_root / recorded_path
                if not recorded_path.is_file():
                    return self._refusal(
                        "checkpoint_missing",
                        f"recorded phase-1 checkpoint is missing: {recorded_path}",
                    )
                recorded_sha = recorded.get("sha256")
                if isinstance(recorded_sha, str) and recorded_sha != sha:
                    return self._refusal(
                        "checkpoint_sha_mismatch",
                        "reviewed checkpoint sha differs from the recorded phase-1 checkpoint",
                    )
                checkpoint_path = recorded_path
            else:
                checkpoint_path = self._find_checkpoint_by_sha(sha)
                if checkpoint_path is None:
                    return self._refusal(
                        "checkpoint_missing",
                        "no checkpoint file on this run root matches the reviewed sha",
                    )
            actual = _sha256_file(checkpoint_path)
            if actual != sha:
                return self._refusal(
                    "checkpoint_sha_mismatch",
                    "reviewed checkpoint sha differs from the actual checkpoint bytes",
                )
            try:
                checkpoint_path = self._anchor_accepted_checkpoint(checkpoint_path, sha)
            except SupervisorError as exc:
                return self._refusal("anchor_error", str(exc))
            state["phase1_checkpoint"] = {
                "step": PHASE1_ACCEPTANCE_STEP,
                "path": _relative_path(checkpoint_path, self.run_root),
                "sha256": sha,
            }
            acceptance = {
                "schema": ACCEPTANCE_SCHEMA,
                "accepted": True,
                "phase": "phase-0051",
                "step": PHASE1_ACCEPTANCE_STEP,
                "checkpoint_sha256": sha,
                "checkpoint_path": _relative_path(checkpoint_path, self.run_root),
                "reviewer": reviewer_text,
                "accepted_at": _now_iso(),
                "note": None if note is None else str(note),
            }
            _atomic_json(self.acceptance_path, acceptance)
            state["acceptance"] = self._acceptance_summary(
                acceptance,
                path=self.acceptance_path,
                checkpoint_path=checkpoint_path,
            )
            self._save_state(state)
            self._write_heartbeat(state, force=True)
            return {
                "status": "accepted",
                "exit_code": 0,
                "acceptance_path": _relative_path(self.acceptance_path, self.run_root),
                "checkpoint_sha256": sha,
                "checkpoint_path": _relative_path(checkpoint_path, self.run_root),
                "step": PHASE1_ACCEPTANCE_STEP,
            }
        except SupervisorError as exc:
            return self._refusal("refused", str(exc))
        finally:
            self.release_lock()

    def status(self) -> dict[str, Any]:
        value = _read_json(self.state_path)
        if not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA:
            return {
                "schema": STATUS_SCHEMA,
                "run_root": str(self.run_root),
                "status": "missing",
                "exit_code": 2,
                "error": "no valid supervisor state on this run root",
            }
        state = dict(value)
        pid = state.get("child_pid")
        ticks = state.get("child_start_ticks")
        alive = self._recorded_child_alive(state)
        proc_stat = _read_proc_stat(pid) if pid is not None else None
        if pid is None:
            process_state = "absent"
        elif proc_stat is None:
            process_state = "dead"
        elif proc_stat[0] == "Z":
            process_state = "dead"  # A zombie child has already exited.
        elif proc_stat[1] != ticks:
            process_state = "reused"
        elif alive:
            process_state = "alive"
        else:
            process_state = "unknown"
        heartbeat = _read_json(self.heartbeat_path)
        return {
            "schema": STATUS_SCHEMA,
            "run_root": str(self.run_root),
            "status": state.get("status"),
            "phase": state.get("phase"),
            "exit_code": 0,
            "state": state,
            "process": {
                "pid": pid,
                "pgid": state.get("child_pgid"),
                "start_ticks": ticks,
                "alive": alive,
                "process_state": process_state,
            },
            "heartbeat": dict(heartbeat) if isinstance(heartbeat, Mapping) else None,
            "heartbeat_age_seconds": (
                _age_seconds(heartbeat.get("updated_at"))
                if isinstance(heartbeat, Mapping)
                else None
            ),
        }

    # --------------------------------------------------------------- launching

    def _launch(
        self,
        state: dict[str, Any],
        *,
        phase: str,
        phase_number: int,
        stop_after_update: int,
        resume_path: Path | None,
        acceptance_path: Path | None,
        inputs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            command_inputs = dict(inputs) if inputs is not None else self._inputs_from_state(state)
        except SupervisorError as exc:
            return self._prelaunch_error(state, phase, str(exc))

        wall_cap = float(state["new_stage_wall_cap_seconds"])
        wall_base = float(state.get("new_stage_wall_seconds") or 0.0)
        margin = float(state.get("stop_margin_seconds", self.stop_margin_seconds))
        state["stop_margin_seconds"] = margin
        state["wall_stop_deadline"] = max(0.0, wall_cap - margin)
        remaining = wall_cap - wall_base
        if remaining <= 0:
            return self._prelaunch_error(
                state,
                phase,
                "cumulative new-stage wall cap is already exhausted; refusing launch",
            )
        if remaining <= margin:
            return self._prelaunch_error(
                state,
                phase,
                "remaining cumulative wall is inside the proactive stop margin "
                f"({remaining:.3f}s <= {margin:.3f}s); refusing launch",
            )
        output_now = _run_root_output_bytes(self.run_root)
        state["new_stage_output_bytes"] = output_now
        reserve = int(state["checkpoint_reserve_bytes"])
        output_cap = int(state["new_stage_output_cap_bytes"])
        if output_now + reserve > output_cap:
            return self._prelaunch_error(
                state,
                phase,
                "run-root new output plus checkpoint reserve would exceed the cap",
            )

        command = build_runner_command(
            inputs=command_inputs,
            run_root=self.run_root,
            stop_after_update=stop_after_update,
            resume=resume_path,
            acceptance=acceptance_path,
        )
        now = _now_iso()
        state.update(
            {
                "status": "running",
                "phase": phase,
                "supervisor_pid": os.getpid(),
                "supervisor_pid_start_ticks": _read_proc_start_ticks(os.getpid()),
                "runner_command": list(command),
                "termination_reason": None,
                "last_error": None,
                "started_at": state.get("started_at") or now,
                "finished_at": None,
            }
        )
        entry: dict[str, Any] = {
            "phase": phase,
            "phase_number": int(phase_number),
            "attempt": len(state.get("phase_history") or []) + 1,
            "status": "launching",
            "started_at": now,
            "ended_at": None,
            "child_pid": None,
            "child_pgid": None,
            "child_start_ticks": None,
            "native_exit": None,
            "stop_after_update": int(stop_after_update),
            "wall_seconds_before": wall_base,
            "new_stage_wall_seconds": wall_base,
            "stop_margin_seconds": margin,
            "wall_stop_deadline": state["wall_stop_deadline"],
            "termination_reason": None,
        }
        state.setdefault("phase_history", []).append(entry)
        self._save_state(state)
        self._write_heartbeat(state, force=True)

        child_env = dict(os.environ)
        source = str(REPO_ROOT / "src")
        existing = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = f"{source}:{existing}" if existing else source

        try:
            log_handle = self.child_log_path.open("a", encoding="utf-8")
        except OSError as exc:
            return self._prelaunch_error(state, phase, f"cannot open child log: {exc}")
        try:
            with log_handle:
                try:
                    process = subprocess.Popen(
                        list(command),
                        cwd=str(REPO_ROOT),
                        env=child_env,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as exc:
                    return self._prelaunch_error(state, phase, f"cannot launch runner child: {exc}")
                child_pid = int(process.pid)
                try:
                    child_pgid = int(os.getpgid(child_pid))
                except OSError:
                    child_pgid = child_pid
                child_start_ticks = _read_proc_start_ticks(child_pid)
                state["child_pid"] = child_pid
                state["child_pgid"] = child_pgid
                state["child_start_ticks"] = child_start_ticks
                entry["child_pid"] = child_pid
                entry["child_pgid"] = child_pgid
                entry["child_start_ticks"] = child_start_ticks
                entry["status"] = "running"
                self._save_state(state)
                self._write_heartbeat(state, force=True)
                if child_start_ticks is None or child_pgid != child_pid:
                    # Without a recorded starttime the group could never be
                    # signalled safely, so fail closed on our own direct child.
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=5.0)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                    state["status"] = "error"
                    state["phase"] = "error"
                    state["child_native_exit"] = process.poll()
                    state["last_error"] = "could not record a session-leading child starttime"
                    state["finished_at"] = _now_iso()
                    entry["status"] = "error"
                    entry["termination_reason"] = "unknown_child_identity"
                    entry["ended_at"] = _now_iso()
                    self._save_state(state)
                    self._write_heartbeat(state, force=True)
                    return self._result(state)
                self._install_signal_handlers()
                try:
                    native_exit, termination_reason = self._monitor(
                        state, entry, process, wall_base
                    )
                finally:
                    self._restore_signal_handlers()
        finally:
            # A final state/heartbeat write happens inside _finalize.
            pass
        return self._finalize(
            state,
            entry,
            phase=phase,
            stop_after_update=stop_after_update,
            native_exit=native_exit,
            termination_reason=termination_reason,
        )

    def _monitor(
        self,
        state: dict[str, Any],
        entry: dict[str, Any],
        process: subprocess.Popen[Any],
        wall_base: float,
    ) -> tuple[int | None, str | None]:
        wall_cap = float(state["new_stage_wall_cap_seconds"])
        output_cap = int(state["new_stage_output_cap_bytes"])
        reserve = int(state["checkpoint_reserve_bytes"])
        margin = float(state.get("stop_margin_seconds", self.stop_margin_seconds))
        stop_deadline = float(state.get("wall_stop_deadline", max(0.0, wall_cap - margin)))
        started = time.monotonic()
        reason: str | None = None
        term_at: float | None = None
        kill_at: float | None = None
        code: int | None = None
        while True:
            code = process.poll()
            now = time.monotonic()
            # Real cumulative new-stage wall; overruns are recorded verbatim.
            wall = wall_base + (now - started)
            state["new_stage_wall_seconds"] = wall
            state["new_stage_output_bytes"] = _run_root_output_bytes(self.run_root)
            entry["new_stage_wall_seconds"] = wall
            entry["stop_margin_seconds"] = margin
            entry["wall_stop_deadline"] = stop_deadline
            if code is not None:
                break
            if reason is None:
                if self._stop_signal is not None:
                    reason = "supervisor_sigterm"
                elif wall >= wall_cap:
                    reason = "wall_cap_hard"
                elif wall >= stop_deadline:
                    reason = "wall_stop_margin"
                elif state["new_stage_output_bytes"] + reserve > output_cap:
                    reason = "output_cap"
                if reason is not None:
                    state["status"] = "stopping"
                    state["termination_reason"] = reason
                    self._save_state(state)
                    self._write_heartbeat(state, force=True)
                    # Only the recorded group, only after a starttime recheck.
                    signal_owned_group(state, signal.SIGTERM)
                    term_at = time.monotonic()
                    if wall >= wall_cap:
                        # Absolute hard stop: never linger past the hard cap.
                        signal_owned_group(state, signal.SIGKILL)
                        kill_at = term_at
            elif term_at is not None and kill_at is None:
                if time.monotonic() - term_at >= self.stop_grace_seconds:
                    signal_owned_group(state, signal.SIGKILL)
                    kill_at = time.monotonic()
            elif kill_at is not None:
                if time.monotonic() - kill_at >= self.kill_wait_seconds:
                    state["last_error"] = (
                        "child process group did not exit after the bounded SIGKILL wait"
                    )
                    break
            if self._write_heartbeat(state):
                self._save_state(state)
            time.sleep(self.poll_seconds)
        native_exit = None if code is None else int(code)
        elapsed = time.monotonic() - started
        state["new_stage_wall_seconds"] = wall_base + elapsed  # real, untruncated
        state["new_stage_output_bytes"] = _run_root_output_bytes(self.run_root)
        state["child_native_exit"] = native_exit
        entry["new_stage_wall_seconds"] = state["new_stage_wall_seconds"]
        entry["native_exit"] = native_exit
        entry["termination_reason"] = reason
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return native_exit, reason

    def _finalize(
        self,
        state: dict[str, Any],
        entry: dict[str, Any],
        *,
        phase: str,
        stop_after_update: int,
        native_exit: int | None,
        termination_reason: str | None,
    ) -> dict[str, Any]:
        state["child_native_exit"] = native_exit
        state["new_stage_output_bytes"] = _run_root_output_bytes(self.run_root)
        entry["ended_at"] = _now_iso()
        entry["native_exit"] = native_exit
        entry["termination_reason"] = termination_reason
        entry["new_stage_wall_seconds"] = state.get("new_stage_wall_seconds")
        if termination_reason in {"wall_stop_margin", "wall_cap_hard", "output_cap"}:
            state["status"] = "error"
            state["phase"] = "error"
            state["last_error"] = (
                f"{termination_reason}: supervisor terminated only the recorded child "
                f"process group (native_exit={native_exit}, "
                f"new_stage_wall_seconds={state.get('new_stage_wall_seconds')})"
            )
            entry["status"] = "terminated"
        elif termination_reason == "supervisor_sigterm":
            state["status"] = "stopped"
            state["phase"] = "stopped"
            state["last_error"] = None
            entry["status"] = "stopped"
        elif native_exit != 0:
            state["status"] = "error"
            state["phase"] = "error"
            state["last_error"] = f"runner child exited nonzero ({native_exit}); no retry"
            entry["status"] = "error"
        elif phase == "phase-1":
            state["status"] = "waiting_acceptance"
            state["phase"] = "waiting_acceptance"
            state["acceptance"] = None
            checkpoint = self._discover_phase1_checkpoint()
            state["phase1_checkpoint"] = checkpoint
            if checkpoint is None:
                state["last_error"] = (
                    "phase-1 child exited cleanly but no checkpoint could be "
                    "discovered; acceptance cannot proceed"
                )
            entry["status"] = "completed"
        else:
            if int(stop_after_update) >= PHASE2_TARGET_UPDATES:
                state["status"] = "completed"
                state["phase"] = "completed"
            else:
                state["status"] = "paused"
                state["phase"] = "paused"
            entry["status"] = "completed"
        state["finished_at"] = _now_iso()
        self._save_state(state)
        self._write_heartbeat(state, force=True)
        return self._result(state)

    def _result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        status = state.get("status")
        exit_code = 0 if status in {"waiting_acceptance", "completed", "paused", "stopped"} else 2
        return {
            "status": status,
            "phase": state.get("phase"),
            "child_native_exit": state.get("child_native_exit"),
            "new_stage_wall_seconds": state.get("new_stage_wall_seconds"),
            "new_stage_output_bytes": state.get("new_stage_output_bytes"),
            "last_error": state.get("last_error"),
            "phase1_checkpoint": state.get("phase1_checkpoint"),
            "exit_code": exit_code,
            "state": dict(state),
        }

    def _prelaunch_error(self, state: dict[str, Any], phase: str, message: str) -> dict[str, Any]:
        state["status"] = "error"
        state["phase"] = "error"
        state["last_error"] = f"{phase}: {message}"
        state["finished_at"] = _now_iso()
        self._save_state(state)
        self._write_heartbeat(state, force=True)
        return self._result(state)

    # ------------------------------------------------------------- acceptance

    def _acceptance_summary(
        self,
        value: Mapping[str, Any],
        *,
        path: Path,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        return {
            "schema": ACCEPTANCE_SCHEMA,
            "step": PHASE1_ACCEPTANCE_STEP,
            "checkpoint_sha256": str(value.get("checkpoint_sha256")),
            "checkpoint_path": _relative_path(checkpoint_path, self.run_root),
            "reviewer": value.get("reviewer"),
            "accepted_at": value.get("accepted_at"),
            "note": value.get("note"),
            "path": _relative_path(path, self.run_root),
        }

    def _validate_acceptance(
        self,
        state: Mapping[str, Any],
        acceptance: Path | str | None,
    ) -> tuple[bool, str, dict[str, Any] | None, Path | None, Path | None]:
        def fail(message: str) -> tuple[bool, str, None, None, None]:
            return False, message, None, None, None

        path = (
            Path(acceptance).expanduser().resolve()
            if acceptance is not None
            else self.acceptance_path
        )
        if not path.is_file():
            return fail(f"acceptance file is absent: {path}")
        value = _read_json(path)
        if not isinstance(value, Mapping):
            return fail(f"acceptance file is unreadable: {path}")
        if value.get("schema") != ACCEPTANCE_SCHEMA:
            return fail("acceptance schema differs")
        if value.get("accepted") is not True:
            return fail("acceptance is not marked accepted")
        try:
            step = int(value.get("step"))
        except (TypeError, ValueError):
            return fail("acceptance step is not an integer")
        if step != PHASE1_ACCEPTANCE_STEP:
            return fail(f"acceptance step is {step}, expected {PHASE1_ACCEPTANCE_STEP}")
        sha = value.get("checkpoint_sha256")
        if not _is_hex_sha256(sha):
            return fail("acceptance checkpoint sha is not 64 hex")
        reviewer = value.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            return fail("acceptance reviewer is missing")
        sha = str(sha).lower()
        recorded = state.get("phase1_checkpoint")
        checkpoint_path: Path | None = None
        if isinstance(recorded, Mapping) and recorded.get("path"):
            candidate = Path(str(recorded["path"]))
            if not candidate.is_absolute():
                candidate = self.run_root / candidate
            if not candidate.is_file():
                return fail("recorded accepted phase-1 checkpoint/anchor is missing")
            checkpoint_path = candidate
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint_by_sha(sha)
        if checkpoint_path is None:
            return fail("no checkpoint on this run root matches the accepted sha")
        if _sha256_file(checkpoint_path) != sha:
            return fail("accepted checkpoint sha no longer matches the checkpoint file")
        recorded_sha = recorded.get("sha256") if isinstance(recorded, Mapping) else None
        if isinstance(recorded_sha, str) and recorded_sha.lower() != sha:
            return fail("acceptance sha differs from the recorded phase-1 checkpoint")
        return True, "", dict(value), path, checkpoint_path

    # -------------------------------------------------------------- discovery

    def _checkpoint_files(self) -> list[Path]:
        directory = self.run_root / "checkpoints"
        if not directory.is_dir():
            return []
        files: list[Path] = []
        try:
            entries = list(directory.iterdir())
        except OSError:
            return []
        for path in entries:
            try:
                if path.is_symlink() or not path.is_file() or path.suffix != ".pt":
                    continue
            except OSError:
                continue
            files.append(path)
        return files

    def _preferred_checkpoints(self) -> list[Path]:
        def rank(path: Path) -> tuple[int, float]:
            if "000051" in path.name or "final_000051" in path.name:
                return (0, 0.0)
            if path.name == "latest.pt":
                return (1, 0.0)
            try:
                mtime = -path.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (2, mtime)

        return sorted(self._checkpoint_files(), key=rank)

    def _find_checkpoint_by_sha(self, sha: str) -> Path | None:
        for path in self._preferred_checkpoints():
            if _sha256_file(path) == sha:
                return path
        return None

    def _discover_paused_checkpoint(self) -> dict[str, Any]:
        """Strictly resolve the latest owned checkpoint for a clean paused phase-2.

        This is deliberately not the phase-51 acceptance checkpoint: a clean pause
        at 250/1000 must continue from the runner's actual latest checkpoint.  The
        path is restricted to the owned ``checkpoints/`` tree, the recorded SHA and
        step must match the actual bytes, and the runner will still strict-load the
        payload.  A malformed or missing record refuses resume rather than silently
        falling back to step 51.
        """
        status = _read_json(self.run_root / "run-status.json")
        if not isinstance(status, Mapping):
            raise SupervisorError("paused resume requires run-status.json")
        run_status = status.get("status")
        if run_status not in {"stopped_at_update_limit", "stopped", "paused"}:
            raise SupervisorError(
                f"paused resume requires a clean terminal run-status, got {run_status!r}"
            )
        if status.get("native_exit") not in (0, None):
            raise SupervisorError("paused resume requires native_exit 0")
        last = status.get("last_checkpoint")
        if not isinstance(last, Mapping):
            raise SupervisorError("paused resume requires run-status.last_checkpoint")
        step = last.get("step")
        try:
            step = int(step)
        except (TypeError, ValueError):
            raise SupervisorError("paused checkpoint step is not an integer") from None
        if step <= PHASE1_ACCEPTANCE_STEP or step > PHASE2_TARGET_UPDATES:
            raise SupervisorError(
                f"paused checkpoint step {step} is outside "
                f"({PHASE1_ACCEPTANCE_STEP}, {PHASE2_TARGET_UPDATES}]"
            )
        cumulative = status.get("cumulative_updates")
        if int(cumulative) != step:
            raise SupervisorError("paused checkpoint step differs from cumulative_updates")
        raw_path = last.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise SupervisorError("paused checkpoint path is missing")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.run_root / candidate
        if candidate.is_symlink():
            raise SupervisorError("paused checkpoint path must not be a symlink")
        checkpoints = (self.run_root / "checkpoints").resolve()
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(checkpoints)
        except (OSError, ValueError):
            raise SupervisorError(
                "paused checkpoint path is outside the owned checkpoints directory"
            ) from None
        match = re.fullmatch(r"checkpoint(?:_final)?_(\d{6})\.pt", resolved.name)
        if match is None or int(match.group(1)) != step:
            raise SupervisorError("paused checkpoint filename does not match its step")
        if not resolved.is_file():
            raise SupervisorError("paused checkpoint file is absent")
        actual = _sha256_file(resolved)
        recorded = last.get("sha256")
        if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
            raise SupervisorError("paused checkpoint requires its actual SHA256")
        if recorded != actual:
            raise SupervisorError("paused checkpoint SHA differs from the actual file")
        return {
            "step": step,
            "path": resolved,
            "relative_path": _relative_path(resolved, self.run_root),
            "sha256": actual,
        }

    def _discover_phase1_checkpoint(self) -> dict[str, Any] | None:
        status = _read_json(self.run_root / "run-status.json")
        if isinstance(status, Mapping):
            last = status.get("last_checkpoint")
            if isinstance(last, Mapping) and isinstance(last.get("path"), str):
                candidate = Path(str(last["path"]))
                if not candidate.is_absolute():
                    candidate = self.run_root / candidate
                try:
                    inside = candidate.resolve().relative_to(self.run_root).as_posix()
                except (OSError, ValueError):
                    inside = None
                if inside is not None and candidate.is_file() and not candidate.is_symlink():
                    actual = _sha256_file(candidate)
                    recorded_sha = last.get("sha256")
                    step = last.get("step")
                    step_ok = step is None
                    if not step_ok:
                        try:
                            step_ok = int(step) == PHASE1_STOP_AFTER_UPDATE
                        except (TypeError, ValueError):
                            step_ok = False
                    if step_ok and (
                        not isinstance(recorded_sha, str) or recorded_sha.lower() == actual
                    ):
                        return {
                            "step": PHASE1_STOP_AFTER_UPDATE,
                            "path": inside,
                            "sha256": actual,
                            "source": "run-status",
                        }
        for path in self._preferred_checkpoints():
            try:
                actual = _sha256_file(path)
            except OSError:
                continue
            return {
                "step": PHASE1_STOP_AFTER_UPDATE,
                "path": _relative_path(path, self.run_root),
                "sha256": actual,
                "source": "checkpoints-scan",
            }
        return None

    # ---------------------------------------------------------------- signals

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self._stop_signal = signum

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous_handlers[signum] = signal.signal(signum, handler)
            except (ValueError, OSError):
                continue

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError, TypeError):
                continue
        self._previous_handlers.clear()


# ---------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-root", required=True)

    start = subparsers.add_parser("start", parents=[common])
    start.add_argument("--config", required=True)
    start.add_argument("--migration", required=True)
    start.add_argument("--authority", required=True)
    start.add_argument("--runner-python", required=True)
    start.add_argument("--stop-after-update", type=int, required=True)
    start.add_argument("--device", default="cuda:0")
    start.add_argument("--parity-receipt", default=None)
    start.add_argument("--resource-receipt", default=None)

    resume = subparsers.add_parser("resume", parents=[common])
    resume.add_argument("--stop-after-update", type=int, required=True)
    resume.add_argument("--acceptance", default=None)

    accept = subparsers.add_parser("accept", parents=[common])
    accept.add_argument("--step", type=int, required=True)
    accept.add_argument("--checkpoint-sha", required=True)
    accept.add_argument("--reviewer", required=True)
    accept.add_argument("--note", default=None)

    subparsers.add_parser("status", parents=[common])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    supervisor = Supervisor(Path(arguments.run_root))
    if arguments.command == "start":
        result = supervisor.start(
            config=Path(arguments.config),
            migration=Path(arguments.migration),
            authority=Path(arguments.authority),
            runner_python=Path(arguments.runner_python),
            stop_after_update=arguments.stop_after_update,
            device=arguments.device,
            parity_receipt=arguments.parity_receipt,
            resource_receipt=arguments.resource_receipt,
        )
    elif arguments.command == "resume":
        result = supervisor.resume(
            stop_after_update=arguments.stop_after_update,
            acceptance=arguments.acceptance,
        )
    elif arguments.command == "accept":
        result = supervisor.accept(
            step=arguments.step,
            checkpoint_sha=arguments.checkpoint_sha,
            reviewer=arguments.reviewer,
            note=arguments.note,
        )
    else:
        result = supervisor.status()
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
