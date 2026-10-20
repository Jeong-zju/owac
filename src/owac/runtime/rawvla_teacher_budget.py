"""Process-safe phase budget ledger for the authorized teacher run.

The ledger is rooted at the authority run root, not at an individual trainer
object.  It charges GPU-reserved wall time, update/anchor/replay/policy counts,
and new disk bytes across processes and resume boundaries.  Import and status
operations must not create a GPU lease.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_GPU_SECONDS = 12 * 3600
MAX_UPDATES = 2000
MAX_ANCHORS = 16000
MAX_REPLAYS = 50
MAX_POLICY_EPISODES = 50
MAX_NEW_BYTES = 50 * 1024**3


class BudgetExceeded(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@dataclass
class PhaseBudget:
    run_root: Path
    authorized_gpu_seconds: float = MAX_GPU_SECONDS
    authorized_updates: int = MAX_UPDATES
    authorized_anchors: int = MAX_ANCHORS
    authorized_replays: int = MAX_REPLAYS
    authorized_policy_episodes: int = MAX_POLICY_EPISODES
    authorized_new_bytes: int = MAX_NEW_BYTES
    clock: Callable[[], float] = _now
    ledger_path: Path | None = None
    _state: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root).resolve()
        if self.ledger_path is None:
            self.ledger_path = self.run_root / "budget" / "ledger.json"
        if self.authorized_gpu_seconds <= 0 or self.authorized_gpu_seconds > MAX_GPU_SECONDS:
            raise BudgetExceeded("GPU budget must be positive and <=12h")
        if self.authorized_updates < 0 or self.authorized_updates > MAX_UPDATES:
            raise BudgetExceeded("update budget exceeds 2000")
        if self.authorized_anchors < 0 or self.authorized_anchors > MAX_ANCHORS:
            raise BudgetExceeded("anchor budget exceeds 16000")
        if self.authorized_replays < 0 or self.authorized_replays > MAX_REPLAYS:
            raise BudgetExceeded("replay budget exceeds 50")
        if (
            self.authorized_policy_episodes < 0
            or self.authorized_policy_episodes > MAX_POLICY_EPISODES
        ):
            raise BudgetExceeded("policy episode budget exceeds 50")
        if self.authorized_new_bytes < 0 or self.authorized_new_bytes > MAX_NEW_BYTES:
            raise BudgetExceeded("new-disk budget exceeds 50 GiB")
        self._authority_binding = None
        authority_path = self.run_root / "authority.json"
        if authority_path.is_file():
            payload = json.loads(authority_path.read_text())
            if payload.get("schema") != "owac.raw-teacher.authorization.v1":
                raise BudgetExceeded("phase authority schema differs")
            if Path(payload["run_root"]).resolve() != self.run_root:
                raise BudgetExceeded("phase authority root differs")
            canonical = self.run_root / "budget/ledger.json"
            if Path(self.ledger_path).resolve() != canonical:
                raise BudgetExceeded("authority phase requires its one canonical budget ledger")
            self._authority_binding = hashlib.sha256(authority_path.read_bytes()).hexdigest()
        self._state = self._read()
        recorded = self._state.get("authority_sha256")
        if recorded is not None and recorded != self._authority_binding:
            raise BudgetExceeded("budget authority identity changed")

    def _read(self) -> dict[str, Any]:
        path = Path(self.ledger_path)
        if not path.is_file():
            return {
                "gpu_seconds": 0.0,
                "updates": 0,
                "anchors": 0,
                "replays": 0,
                "policy_episodes": 0,
                "new_bytes": 0,
                "completed_scopes": [],
                "last_stop_reason": None,
            }
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") not in (None, "owac.raw-teacher.budget.v1"):
            raise BudgetExceeded("budget ledger schema differs")
        # Legacy ledger migration: preserve all existing charges and add new
        # counters without resetting or re-charging earlier work.
        defaults = {
            "gpu_seconds": 0.0,
            "updates": 0,
            "anchors": 0,
            "replays": 0,
            "policy_episodes": 0,
            "validation_episodes": 0,
            "test_episodes": 0,
            "new_bytes": 0,
            "pending_updates": 0,
            "pending_anchors": 0,
            "completed_scopes": [],
            "last_stop_reason": None,
        }
        for key, value in defaults.items():
            payload.setdefault(key, value)
        payload["schema"] = "owac.raw-teacher.budget.v1"
        return payload

    def _write(self) -> None:
        path = Path(self.ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "owac.raw-teacher.budget.v1", **self._state}
        if self._authority_binding is not None:
            payload["authority_sha256"] = self._authority_binding
            payload["run_root"] = str(self.run_root)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    @property
    def remaining(self) -> dict[str, float]:
        return {
            "gpu_seconds": self.authorized_gpu_seconds - float(self._state["gpu_seconds"]),
            "updates": self.authorized_updates - int(self._state["updates"]),
            "anchors": self.authorized_anchors - int(self._state["anchors"]),
            "replays": self.authorized_replays - int(self._state["replays"]),
            "policy_episodes": self.authorized_policy_episodes
            - int(self._state["policy_episodes"]),
            "new_bytes": self.authorized_new_bytes - int(self._state["new_bytes"]),
        }

    def _lease_path(self) -> Path:
        return Path(self.ledger_path).parent / "lease.json"

    @contextmanager
    def _locked(self):
        path = Path(self.ledger_path).parent / "budget.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def check_deadline(self, *, reserved_gpu_seconds: float | None = None) -> None:
        elapsed = (
            max(0.0, float(self.clock()) - float(self._active_started))
            if getattr(self, "_active_started", None) is not None
            else 0.0
        )
        limit = self.authorized_gpu_seconds - float(self._state["gpu_seconds"])
        if reserved_gpu_seconds is not None:
            limit = min(limit, float(reserved_gpu_seconds))
        active_limit = getattr(self, "_active_reserve", None)
        if active_limit is not None:
            limit = min(limit, active_limit)
        if elapsed >= limit:
            raise BudgetExceeded("GPU-reserved wall deadline exceeded")

    def _require_owner(self) -> None:
        path = self._lease_path()
        if not path.is_file() or int(json.loads(path.read_text()).get("pid", 0)) != os.getpid():
            raise BudgetExceeded("budget mutation requires the owned active phase lease")

    def check_disk(self, *, reserve_bytes: int = 0) -> int:
        actual = sum(path.stat().st_size for path in self.run_root.rglob("*") if path.is_file())
        if actual + reserve_bytes > self.authorized_new_bytes:
            raise BudgetExceeded("phase-wide disk budget exhausted")
        with self._locked():
            self._state = self._read()
            self._state["new_bytes"] = actual
            self._write()
        return actual

    def charge_update(self, *, anchors: int) -> None:
        self._require_owner()
        self.check_deadline()
        if anchors < 1:
            raise BudgetExceeded("anchors must be positive")
        with self._locked():
            self._state = self._read()
            pending = int(self._state.get("pending_updates", 0))
            pending_anchors = int(self._state.get("pending_anchors", 0))
            if int(self._state["updates"]) + pending + 1 > self.authorized_updates:
                raise BudgetExceeded("cumulative update budget exhausted")
            if int(self._state["anchors"]) + pending_anchors + anchors > self.authorized_anchors:
                raise BudgetExceeded("cumulative anchor budget exhausted")
            self._state["pending_updates"] = pending + 1
            self._state["pending_anchors"] = pending_anchors + anchors
            self._write()

    def commit_update(self) -> None:
        self._require_owner()
        with self._locked():
            self._state = self._read()
            pending = int(self._state.get("pending_updates", 0))
            pending_anchors = int(self._state.get("pending_anchors", 0))
            if pending < 1:
                raise BudgetExceeded("no pending update intent to commit")
            self._state["updates"] = int(self._state["updates"]) + pending
            self._state["anchors"] = int(self._state["anchors"]) + pending_anchors
            self._state["pending_updates"] = 0
            self._state["pending_anchors"] = 0
            self._write()

    def charge_policy_episode(self, split: str) -> None:
        self._require_owner()
        if split not in {"validation", "test"}:
            raise BudgetExceeded("policy episode split must be validation or test")
        with self._locked():
            self._state = self._read()
            if self._state.get("pending_updates", 0) or self._state.get("pending_anchors", 0):
                raise BudgetExceeded(
                    "ambiguous pending update requires audited checkpoint recovery"
                )
            key = f"{split}_episodes"
            if int(self._state["policy_episodes"]) + 1 > self.authorized_policy_episodes:
                raise BudgetExceeded("cumulative policy episode budget exhausted")
            if int(self._state.get(key, 0)) + 1 > (10 if split == "validation" else 40):
                raise BudgetExceeded(f"{split} policy episode budget exhausted")
            self._state["policy_episodes"] = int(self._state["policy_episodes"]) + 1
            self._state[key] = int(self._state.get(key, 0)) + 1
            self._write()

    @contextmanager
    def lease(
        self,
        scope: str,
        *,
        gpu: bool = True,
        updates: int = 0,
        anchors: int = 0,
        replays: int = 0,
        policy_episodes: int = 0,
        new_bytes: int = 0,
        reserved_gpu_seconds: float = 0.0,
    ):
        if min(updates, anchors, replays, policy_episodes, new_bytes) < 0:
            raise BudgetExceeded("negative budget request")
        lease_path = self._lease_path()
        started = float(self.clock())
        with self._locked():
            if lease_path.is_file():
                current = json.loads(lease_path.read_text(encoding="utf-8"))
                pid = int(current.get("pid", 0))
                if _pid_alive(pid):
                    raise BudgetExceeded(f"active teacher budget lease held by live pid {pid}")
                raise BudgetExceeded(
                    "stale teacher budget lease exists; explicit operator recovery is required"
                )
            self._state = self._read()
            if self._state.get("pending_updates", 0) or self._state.get("pending_anchors", 0):
                raise BudgetExceeded(
                    "ambiguous pending update requires audited checkpoint recovery"
                )
            requested = {
                "updates": int(self._state["updates"]) + updates,
                "anchors": int(self._state["anchors"]) + anchors,
                "replays": int(self._state["replays"]) + replays,
                "policy_episodes": int(self._state["policy_episodes"]) + policy_episodes,
                "new_bytes": int(self._state["new_bytes"]) + new_bytes,
            }
            if requested["updates"] > self.authorized_updates:
                raise BudgetExceeded("cumulative update budget exhausted")
            if requested["anchors"] > self.authorized_anchors:
                raise BudgetExceeded("cumulative anchor budget exhausted")
            if requested["replays"] > self.authorized_replays:
                raise BudgetExceeded("cumulative replay budget exhausted")
            if requested["policy_episodes"] > self.authorized_policy_episodes:
                raise BudgetExceeded("cumulative policy episode budget exhausted")
            if requested["new_bytes"] > self.authorized_new_bytes:
                raise BudgetExceeded("cumulative new-disk budget exhausted")
            if gpu and self.authorized_gpu_seconds - float(self._state["gpu_seconds"]) <= 0:
                raise BudgetExceeded("GPU-reserved wall budget exhausted")
            if gpu and float(reserved_gpu_seconds) > (
                self.authorized_gpu_seconds - float(self._state["gpu_seconds"])
            ):
                raise BudgetExceeded("requested GPU reserve exceeds remaining phase budget")
            lease = {
                "schema": "owac.raw-teacher.budget-lease.v1",
                "pid": os.getpid(),
                "scope": str(scope),
                "started": started,
                "gpu": bool(gpu),
                "reserved_updates": int(updates),
                "reserved_anchors": int(anchors),
            }
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            lease_path.write_text(json.dumps(lease, sort_keys=True) + "\n", encoding="utf-8")
            self._active_started = started if gpu else None
            self._active_reserve = float(reserved_gpu_seconds) if reserved_gpu_seconds > 0 else None
        try:
            yield self
        finally:
            elapsed = max(0.0, float(self.clock()) - started) if gpu else 0.0
            with self._locked():
                self._state = self._read()
                self._state["gpu_seconds"] = float(self._state["gpu_seconds"]) + elapsed
                self._state["updates"] = int(self._state["updates"]) + int(updates)
                self._state["anchors"] = int(self._state["anchors"]) + int(anchors)
                self._state["replays"] = int(self._state["replays"]) + int(replays)
                self._state["policy_episodes"] = int(self._state["policy_episodes"]) + int(
                    policy_episodes
                )
                self._state["new_bytes"] = int(self._state["new_bytes"]) + int(new_bytes)
                self._state["completed_scopes"].append({"scope": str(scope), "seconds": elapsed})
                self._write()
                if lease_path.is_file():
                    current = json.loads(lease_path.read_text(encoding="utf-8"))
                    if int(current.get("pid", 0)) == os.getpid():
                        lease_path.unlink()
            self._active_started = None
            self._active_reserve = None

    def record_new_bytes(self, byte_count: int) -> None:
        if byte_count < 0:
            raise BudgetExceeded("new byte count must be nonnegative")
        with self._locked():
            self._state = self._read()
            self._state["new_bytes"] = int(self._state["new_bytes"]) + int(byte_count)
            if self._state["new_bytes"] > self.authorized_new_bytes:
                raise BudgetExceeded("new-disk budget exhausted")
            self._write()


__all__ = ["BudgetExceeded", "PhaseBudget"]
