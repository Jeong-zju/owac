"""Persistent B/C pilot supervisor; gated, resumable, and CPU-only control plane.

This module never constructs a model.  It launches owned child processes from
canonical CLI files, records their PID/PGID, and persists enough state that a
new supervisor invocation resumes rather than duplicating work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_SCHEMA = "owac.raw-distillation.pilot-acceptance.v1"
STATE_SCHEMA = "owac.raw-distillation.pilot-supervisor.v1"
GRACE_SECONDS = 1800
DEFAULT_PYTHON = REPO_ROOT / "outputs/rawvla-teacher-20261020/runtime/bin/python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _runroot_bytes(run_root: Path) -> int:
    """Count owned run bytes, excluding the verified shared cache link/tree."""
    cache = run_root / "cache"
    total = 0
    for item in run_root.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        try:
            item.relative_to(cache)
            continue
        except ValueError:
            pass
        total += item.stat().st_size
    return total


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def acceptance_example(*, arm: str = "B", checkpoint_sha256: str = "") -> dict[str, Any]:
    """Schema only; authorized is deliberately false and is not an acceptance."""
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "authorized": False,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha256,
        "accepted": False,
        "acceptance_id": "",
        "note": "example only; Sol must replace with a signed actual acceptance",
    }


@dataclass
class PilotState:
    schema: str = STATE_SCHEMA
    phase: str = "created"
    started_at: float | None = None
    b_first_checkpoint: str | None = None
    b_first_sha256: str | None = None
    b_resume_done: bool = False
    c_first_checkpoint: str | None = None
    c_first_sha256: str | None = None
    c_resume_done: bool = False
    last_exit: int | None = None
    last_error: str | None = None
    jobs: list[dict[str, Any]] = field(default_factory=list)
    arm_wall_seconds: dict[str, float] = field(default_factory=dict)
    phase_exit_codes: dict[str, int] = field(default_factory=dict)
    active_job: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: Path) -> PilotState:
        if not path.is_file():
            return cls()
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("pilot supervisor state schema differs")
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in value.items() if key in allowed})

    def save(self, path: Path) -> None:
        _atomic_json(path, self.__dict__)


def _config_digest(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("authorization", None)
    assets = value.get("assets")
    if isinstance(assets, Mapping):
        assets.pop("output_root", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _owned_pid_alive(pid: int, pgid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return os.getpgid(int(pid)) == int(pgid)
    except OSError:
        return False


def _readiness_ok(run_root: Path, config: Path) -> bool:
    expected = _config_digest(config)
    for name in ("parity-receipt.json", "resource-receipt-B.json", "resource-receipt-C.json"):
        path = run_root / name
        if not path.is_file():
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("passed") is not True:
            return False
        if value.get("research_config_sha256") != expected:
            return False
        if int(value.get("optimizer_updates", 0)) != 0:
            return False
    return True


def _freeze_ok(run_root: Path) -> bool:
    freeze = run_root / "source-freeze.json"
    if not freeze.is_file():
        return False
    value = json.loads(freeze.read_text(encoding="utf-8"))
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        return False
    for relative, expected in files.items():
        source = REPO_ROOT / str(relative)
        if not source.is_file() or _sha256(source) != str(expected):
            return False
    return True


def build_command(
    *,
    python: Path,
    config: Path,
    authority: Path,
    cache_dir: Path,
    output_dir: Path,
    arm: str,
    max_updates: int,
    parity_receipt: Path | None = None,
    resource_receipt: Path | None = None,
    phase1_receipt: Path | None = None,
    matched_cd_receipt: Path | None = None,
    resume: Path | None = None,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "owac.runtime.raw_temporal_distillation",
        "train",
        "--config",
        str(config),
        "--authority",
        str(authority),
        "--arm",
        arm,
        "--max-updates",
        str(int(max_updates)),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
    ]
    for flag, value in (
        ("--parity-receipt", parity_receipt),
        ("--resource-receipt", resource_receipt),
        ("--phase1-receipt", phase1_receipt),
        ("--matched-cd-receipt", matched_cd_receipt),
        ("--resume", resume),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    return command


class PilotSupervisor:
    def __init__(
        self,
        run_root: Path,
        *,
        config: Path | None = None,
        python: Path = DEFAULT_PYTHON,
        grace_seconds: int = GRACE_SECONDS,
        runner: Any = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.config = Path(config) if config else self.run_root / "authorized-config.json"
        self.python = Path(python)
        self.grace_seconds = int(grace_seconds)
        self.state_path = self.run_root / "supervisor-state.json"
        self.lock_path = self.run_root / "supervisor.lock"
        self.state = PilotState.load(self.state_path)
        self._lock_handle = None
        self._runner = runner
        self._stop_requested = False

    def lock(self) -> bool:
        import fcntl

        handle = self.lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self._lock_handle = handle
        return True

    def _budget(self) -> tuple[int, int, dict[str, Any]]:
        authorization = json.loads(
            (self.run_root / "user-authorization.json").read_text(encoding="utf-8")
        )
        combined = int(authorization["combined_wall_cap_seconds"])
        output_cap = int(authorization["new_output_cap_bytes"])
        return combined, output_cap, authorization

    def _authority(self, arm: str) -> dict[str, Any]:
        path = self.run_root / f"authority-{arm}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("authorized"):
            raise ValueError(f"{arm} authority is not authorized")
        if int(value.get("max_wall_seconds", 0)) <= 0:
            raise ValueError(f"{arm} authority lacks max_wall_seconds")
        if int(value.get("max_output_bytes", 0)) <= 0:
            raise ValueError(f"{arm} authority lacks max_output_bytes")
        user_path = self.run_root / "user-authorization.json"
        if value.get("receipt_sha256") != _sha256(user_path):
            raise ValueError(f"{arm} authority does not bind user authorization")
        if int(value.get("max_updates", 0)) > 50:
            raise ValueError(f"{arm} authority exceeds pilot cap")
        return value

    def _base_command(self, arm: str, max_updates: int, resume: Path | None = None) -> list[str]:
        return build_command(
            python=self.python,
            config=self.config,
            authority=self.run_root / f"authority-{arm}.json",
            cache_dir=self.run_root / "cache",
            output_dir=self.run_root / f"pilot-{arm}",
            arm=arm,
            max_updates=max_updates,
            parity_receipt=self.run_root / "parity-receipt.json",
            resource_receipt=self.run_root / f"resource-receipt-{arm}.json",
            phase1_receipt=self.run_root / "phase1-receipt.json",
            matched_cd_receipt=self.run_root / "matched-cd-receipt.json",
            resume=resume,
        )

    def plan(self) -> list[dict[str, Any]]:
        return [
            {"phase": "b_first1", "launch": False, "command": self._base_command("B", 1)},
            {"phase": "wait_for_b_acceptance"},
            {"phase": "b_resume50", "resume": "b_first_checkpoint"},
            {"phase": "c_first1"},
            {"phase": "wait_for_c_acceptance"},
            {"phase": "c_resume50"},
        ]

    def _checkpoint_for(self, arm: str, updates: int) -> Path:
        return self.run_root / f"pilot-{arm}" / f"checkpoint_final_{updates:06d}.pt"

    def _acceptance(self, arm: str, checkpoint: Path) -> bool:
        path = self.run_root / f"acceptance-{arm}.json"
        if not path.is_file() or not checkpoint.is_file():
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            value.get("schema") == ACCEPTANCE_SCHEMA
            and value.get("authorized") is True
            and value.get("accepted") is True
            and value.get("arm") == arm
            and value.get("checkpoint_sha256") == _sha256(checkpoint)
        )

    def run(self) -> dict[str, Any]:
        if not self.lock():
            return {"status": "locked", "state": self.state.__dict__}
        if self.state.active_job and self.state.active_job.get("status") in {
            "prepared",
            "running",
        }:
            job = self.state.active_job
            alive = (
                job.get("pid") is not None
                and job.get("pgid") is not None
                and _owned_pid_alive(int(job["pid"]), int(job["pgid"]))
            )
            return {
                "status": "attention_active_job",
                "active_job_alive": bool(alive),
                "state": self.state.__dict__,
            }
        if self.state.phase.endswith("_failed"):
            return {"status": "failed", "state": self.state.__dict__}
        if not (self.run_root / "START_ALLOWED").is_file():
            self.state.phase = "waiting_for_sol_start"
            self.state.save(self.state_path)
            return {"status": "waiting_for_sol_start", "state": self.state.__dict__}
        if not _freeze_ok(self.run_root):
            self.state.phase = "waiting_for_source_freeze"
            self.state.save(self.state_path)
            return {"status": "waiting_for_source_freeze", "state": self.state.__dict__}
        if not _readiness_ok(self.run_root, self.config):
            self.state.phase = "waiting_for_sol_readiness"
            self.state.save(self.state_path)
            return {"status": "waiting_for_sol_readiness", "state": self.state.__dict__}
        combined, output_cap, _authorization = self._budget()
        if self.state.started_at is None:
            self.state.started_at = time.time()
        elapsed = time.time() - self.state.started_at
        if elapsed >= combined:
            self.state.phase = "combined_wall_exhausted"
            self.state.save(self.state_path)
            return {"status": "combined_wall_exhausted", "state": self.state.__dict__}
        if self.state.phase in {"created", "waiting_for_sol_start"}:
            return self._run_phase("B", 1, "b_first1", combined - elapsed)
        if self.state.phase == "waiting_for_b_acceptance":
            checkpoint = self._checkpoint_for("B", 1)
            if not self._acceptance("B", checkpoint):
                return {"status": "waiting_for_b_acceptance", "state": self.state.__dict__}
            self.state.phase = "b_accepted"
        if self.state.phase in {"b_accepted", "b_first1"}:
            return self._run_phase("B", 50, "b_resume50", combined - elapsed)
        if self.state.phase == "b_resume50":
            return self._run_phase("C", 1, "c_first1", combined - elapsed)
        if self.state.phase == "waiting_for_c_acceptance":
            checkpoint = self._checkpoint_for("C", 1)
            if not self._acceptance("C", checkpoint):
                return {"status": "waiting_for_c_acceptance", "state": self.state.__dict__}
            self.state.phase = "c_accepted"
        if self.state.phase in {"c_accepted", "c_first1"}:
            return self._run_phase("C", 50, "c_resume50", combined - elapsed)
        if self.state.phase == "c_resume50":
            return {"status": "complete", "state": self.state.__dict__}
        return {"status": self.state.phase, "state": self.state.__dict__}

    def _run_phase(self, arm: str, updates: int, phase: str, remaining: float) -> dict[str, Any]:
        if self.state.active_job and self.state.active_job.get("status") in {
            "prepared",
            "running",
        }:
            job = self.state.active_job
            alive = (
                job.get("pid") is not None
                and job.get("pgid") is not None
                and _owned_pid_alive(int(job["pid"]), int(job["pgid"]))
            )
            return {
                "status": "attention_active_job",
                "active_job_alive": bool(alive),
                "state": self.state.__dict__,
            }
        combined, output_cap, _authorization = self._budget()
        reserve = 2 * 1024**3
        if _runroot_bytes(self.run_root) + reserve > output_cap:
            self.state.phase = "disk_cap_waiting"
            self.state.last_error = "runroot bytes plus checkpoint reserve exceed cap"
            self.state.save(self.state_path)
            return {"status": "disk_cap_waiting", "state": self.state.__dict__}
        try:
            authority = self._authority(arm)
            arm_output = _bytes(self.run_root / f"pilot-{arm}") + _bytes(self.run_root / "cache")
            if arm_output + reserve > int(authority["max_output_bytes"]):
                self.state.phase = "arm_output_cap_waiting"
                self.state.last_error = "per-arm output plus cache reserve exceeds cap"
                self.state.save(self.state_path)
                return {"status": "arm_output_cap_waiting", "state": self.state.__dict__}
            now = time.time()
            global_hard = float(self.state.started_at or now) + combined
            if now >= global_hard:
                self.state.phase = "combined_wall_exhausted"
                self.state.save(self.state_path)
                return {"status": "combined_wall_exhausted", "state": self.state.__dict__}
            per_arm_used = float(self.state.arm_wall_seconds.get(arm, 0.0))
            per_arm_left = float(authority["max_wall_seconds"]) - per_arm_used
            if per_arm_left <= 0:
                self.state.phase = f"{arm.lower()}_wall_exhausted"
                self.state.save(self.state_path)
                return {"status": "arm_wall_exhausted", "state": self.state.__dict__}
            soft_deadline = min(now + max(0.0, per_arm_left - 120.0), global_hard - GRACE_SECONDS)
            hard_deadline = min(now + per_arm_left, global_hard)
            resume = None
            if phase == "b_resume50":
                resume = Path(self.state.b_first_checkpoint or "")
            elif phase == "c_resume50":
                resume = Path(self.state.c_first_checkpoint or "")
            command = self._base_command(arm, updates, resume)
            code = self._run_child(
                arm,
                command,
                phase=phase,
                soft_deadline=soft_deadline,
                hard_deadline=hard_deadline,
                output_cap=output_cap,
                per_arm_output_cap=int(authority["max_output_bytes"]),
                reserve=reserve,
            )
            self.state.last_exit = code
            self.state.phase_exit_codes[phase] = int(code)
            if code != 0:
                self.state.phase = f"{arm.lower()}_failed"
                self.state.last_error = f"child exit {code}"
                self.state.save(self.state_path)
                return {"status": "failed", "exit": code, "state": self.state.__dict__}
            checkpoint = self._checkpoint_for(arm, updates)
            if not checkpoint.is_file():
                self.state.phase = f"{arm.lower()}_failed"
                self.state.last_error = "expected final checkpoint absent"
                self.state.save(self.state_path)
                return {"status": "failed", "state": self.state.__dict__}
            checkpoint_sha = _sha256(checkpoint)
            if phase == "b_first1":
                self.state.b_first_checkpoint = str(checkpoint)
                self.state.b_first_sha256 = checkpoint_sha
                self.state.phase = "waiting_for_b_acceptance"
            elif phase == "b_resume50":
                self.state.b_resume_done = True
                self.state.phase = "b_resume50"
            elif phase == "c_first1":
                self.state.c_first_checkpoint = str(checkpoint)
                self.state.c_first_sha256 = checkpoint_sha
                self.state.phase = "waiting_for_c_acceptance"
            elif phase == "c_resume50":
                self.state.c_resume_done = True
                self.state.phase = "c_resume50"
            self.state.save(self.state_path)
            return {
                "status": self.state.phase,
                "checkpoint": str(checkpoint),
                "sha256": checkpoint_sha,
                "phase_exit_code": int(code),
            }
        except Exception as exc:
            self.state.phase = f"{arm.lower()}_failed"
            self.state.last_error = f"{type(exc).__name__}: {exc}"
            self.state.save(self.state_path)
            return {
                "status": "failed",
                "error": self.state.last_error,
                "state": self.state.__dict__,
            }

    def _run_child(
        self,
        arm: str,
        command: Sequence[str],
        *,
        phase: str,
        soft_deadline: float,
        hard_deadline: float,
        output_cap: int,
        per_arm_output_cap: int,
        reserve: int,
    ) -> int:
        attempt = f"{phase}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.state.active_job = {
            "attempt_id": attempt,
            "arm": arm,
            "phase": phase,
            "status": "prepared",
            "pid": None,
            "pgid": None,
        }
        self.state.save(self.state_path)
        if self._runner is not None:
            try:
                return int(self._runner(command))
            finally:
                self.state.active_job["status"] = "done"
                self.state.save(self.state_path)
        receipt_path = self.run_root / f"launch-{arm}-{attempt}.json"
        log_path = self.run_root / f"pilot-{arm}-{attempt}.log"
        receipt = {
            "arm": arm,
            "phase": phase,
            "attempt_id": attempt,
            "pid": None,
            "pgid": None,
            "command": list(command),
            "cwd": str(REPO_ROOT),
            "started_at": time.time(),
            "exit": None,
        }
        env = dict(os.environ)
        for key in list(env):
            if key.startswith("DEEPSEEK") or key.startswith("API_KEY") or key.endswith("_TOKEN"):
                env.pop(key, None)
        env.update(
            {
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "PYTHONPATH": f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt["pid"] = process.pid
            receipt["pgid"] = os.getpgid(process.pid)
            self.state.active_job.update(
                {"status": "running", "pid": process.pid, "pgid": receipt["pgid"]}
            )
            self.state.save(self.state_path)
            _atomic_json(receipt_path, receipt)
            grace_hard = None
            disk_soft = False
            try:
                while True:
                    code = process.poll()
                    if code is not None:
                        receipt["exit"] = int(code)
                        self.state.arm_wall_seconds[arm] = self.state.arm_wall_seconds.get(
                            arm, 0.0
                        ) + (time.time() - float(receipt["started_at"]))
                        self.state.phase_exit_codes[phase] = int(code)
                        self.state.active_job["status"] = "done"
                        self.state.active_job["exit"] = int(code)
                        self.state.save(self.state_path)
                        return int(code)
                    now = time.time()
                    arm_output = _bytes(self.run_root / f"pilot-{arm}") + _bytes(
                        self.run_root / "cache"
                    )
                    if (
                        _runroot_bytes(self.run_root) + reserve > output_cap
                        or arm_output + reserve > per_arm_output_cap
                    ) and not disk_soft:
                        disk_soft = True
                        os.killpg(receipt["pgid"], signal.SIGTERM)
                        grace_hard = min(now + 60.0, hard_deadline)
                    if now > soft_deadline or disk_soft:
                        if grace_hard is None:
                            os.killpg(receipt["pgid"], signal.SIGTERM)
                            grace_hard = min(now + self.grace_seconds, hard_deadline)
                    if now > hard_deadline or (grace_hard is not None and now > grace_hard):
                        os.killpg(receipt["pgid"], signal.SIGKILL)
                        code = int(process.wait())
                        receipt["exit"] = code
                        self.state.active_job["status"] = "hard_killed"
                        self.state.active_job["exit"] = code
                        self.state.arm_wall_seconds[arm] = self.state.arm_wall_seconds.get(
                            arm, 0.0
                        ) + (time.time() - float(receipt["started_at"]))
                        self.state.save(self.state_path)
                        return code
                    time.sleep(0.5)
            finally:
                receipt["ended_at"] = time.time()
                receipt["exit"] = receipt.get("exit")
                _atomic_json(receipt_path, receipt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    supervisor = PilotSupervisor(
        Path(args.run_root),
        config=Path(args.config) if args.config else None,
        python=Path(args.python),
    )
    if args.dry_run:
        print(
            json.dumps(
                {"plan": supervisor.plan(), "acceptance_example": acceptance_example()}, indent=2
            )
        )
        return 0
    result = supervisor.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
