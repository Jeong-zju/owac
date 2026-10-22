"""CPU-only tests for the C-full supervisor; no GPU, model, or torch training.

Every launch goes through a generated fake ``runner-python`` that parses the
same ``-m owac.runtime.raw_temporal_continuation train ...`` command line as the
real runner, records its own session/PID/starttime, and exits or traps signals
as each scenario needs.  The supervisor itself is always the real module.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from owac.runtime.raw_temporal_c_supervisor import (
    ACCEPTANCE_SCHEMA,
    STATE_SCHEMA,
    Supervisor,
    _read_proc_start_ticks,
    _read_proc_state,
    _run_root_output_bytes,
    build_runner_command,
    main,
    signal_owned_group,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Shared fake-runner preamble.  The body appended by each test starts with
# ``run_root`` and ``stop_after`` already defined, and can use ``_record``,
# ``_write_status``, ``_arg``, ``_train`` and ``sys``.
_FAKE_PRELUDE = """\
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path


def _arg(argv, name, default=None):
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


def _start_ticks(pid):
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(raw[raw.rfind(")") + 2 :].split()[19])


def _record(run_root, **values):
    run_root.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "start_ticks": _start_ticks(os.getpid()),
        "argv": sys.argv[1:],
        **values,
    }
    with (run_root / "fake-launches.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\\n")
        handle.flush()
    return record


def _write_status(run_root, step, name=None):
    checkpoints = run_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = name or f"checkpoint_final_{step:06d}.pt"
    path = checkpoints / name
    path.write_bytes(f"fake-checkpoint-{step}".encode())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    status = {
        "schema": "owac.raw-stem-temporal.c-full.run-status.v1",
        "status": "stopped_at_update_limit" if step == 51 else "completed",
        "cumulative_updates": step,
        "last_checkpoint": {
            "step": step,
            "path": f"checkpoints/{name}",
            "sha256": digest,
            "kind": "final",
        },
    }
    (run_root / "run-status.json").write_text(
        json.dumps(status, sort_keys=True) + "\\n", encoding="utf-8"
    )
    return path, digest


_argv = sys.argv[1:]
assert _argv[:3] == ["-m", "owac.runtime.raw_temporal_continuation", "train"], _argv
_train = _argv[3:]
run_root = Path(_arg(_train, "--run-root"))
stop_after = int(_arg(_train, "--stop-after-update"))
"""


def _write_runner(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_runner.py"
    script.write_text(_FAKE_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    wrapper = tmp_path / "fake-python"
    wrapper.write_text(
        f'#!/bin/sh\nexec {shlex.quote(PYTHON)} {shlex.quote(str(script))} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _clean_body(*, sleep_seconds: float = 0.15, exit_code: int = 0) -> str:
    return f"""
_record(run_root, stop_after=stop_after, resume=_arg(_train, "--resume"),
        acceptance=_arg(_train, "--acceptance"))
_write_status(run_root, 51 if stop_after == 51 else 5000)
time.sleep({sleep_seconds})
sys.exit({exit_code})
"""


def _make_case(
    tmp_path: Path, *, prepare_wall_seconds: float | None = 0.0
) -> tuple[Path, Path, Path, Path]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = tmp_path / "config.json"
    migration = tmp_path / "migration.json"
    authority = tmp_path / "authority.json"
    for path in (config, migration, authority):
        path.write_text("{}\n", encoding="utf-8")
    if prepare_wall_seconds is not None:
        (run_root / "prepared.json").write_text(
            json.dumps(
                {
                    "schema": "owac.raw-stem-temporal.c-full.prepared.v1",
                    "identity_ok": True,
                    "prepare_wall_seconds": prepare_wall_seconds,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return run_root, config, migration, authority


def _write_failure_debit(
    run_root: Path,
    seconds,
    *,
    name: str = "prepare-attempt-01-failure.json",
    optimizer_updates=0,
    schema: str = "owac.sol.c-full.failed-preflight-debit.v1",
) -> Path:
    path = run_root / name
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "attempt": 1,
                "charged_wall_seconds": seconds,
                "optimizer_updates": optimizer_updates,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _start_args(
    run_root: Path,
    config: Path,
    migration: Path,
    authority: Path,
    runner: Path,
    *,
    stop_after_update: int = 51,
) -> list[str]:
    return [
        "start",
        "--run-root",
        str(run_root),
        "--config",
        str(config),
        "--migration",
        str(migration),
        "--authority",
        str(authority),
        "--runner-python",
        str(runner),
        "--stop-after-update",
        str(stop_after_update),
    ]


def _resume_args(run_root: Path, *, stop_after_update: int = 5000) -> list[str]:
    return [
        "resume",
        "--run-root",
        str(run_root),
        "--stop-after-update",
        str(stop_after_update),
    ]


def _load_state(run_root: Path) -> dict:
    return json.loads((run_root / "supervisor-state.json").read_text(encoding="utf-8"))


def _load_heartbeat(run_root: Path) -> dict:
    return json.loads((run_root / "supervisor-heartbeat.json").read_text(encoding="utf-8"))


def _launch_records(run_root: Path) -> list[dict]:
    path = run_root / "fake-launches.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _phase1_sha(run_root: Path) -> tuple[Path, str]:
    state = _load_state(run_root)
    checkpoint = state["phase1_checkpoint"]
    assert checkpoint is not None
    path = run_root / checkpoint["path"]
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _accept_args(run_root: Path, sha: str, *, reviewer: str = "sol") -> list[str]:
    return [
        "accept",
        "--run-root",
        str(run_root),
        "--step",
        "51",
        "--checkpoint-sha",
        sha,
        "--reviewer",
        reviewer,
        "--note",
        "independent phase-0051 audit",
    ]


def test_venv_runner_symlink_preserved_but_other_inputs_strictly_resolved(
    tmp_path: Path,
) -> None:
    run_root, config, migration, authority = _make_case(tmp_path)
    runner_target = _write_runner(tmp_path, _clean_body())
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    runner_link = venv_bin / "python"
    runner_link.symlink_to(runner_target)
    config_link = tmp_path / "config-link.json"
    config_link.symlink_to(config)
    migration_link = tmp_path / "migration-link.json"
    migration_link.symlink_to(migration)
    authority_link = tmp_path / "authority-link.json"
    authority_link.symlink_to(authority)

    supervisor = Supervisor(run_root)
    inputs = supervisor._resolve_inputs(
        config=config_link,
        migration=migration_link,
        authority=authority_link,
        runner_python=runner_link,
        device="cpu",
        parity_receipt=None,
        resource_receipt=None,
    )

    # The venv interpreter spelling is preserved lexically: resolving the
    # symlink would run the bare base interpreter without the venv site-packages.
    assert Path(inputs["runner_python"]) == Path(os.path.abspath(runner_link))
    assert Path(inputs["runner_python"]) != runner_target.resolve()
    assert Path(inputs["runner_python"]).is_symlink()
    command = build_runner_command(
        inputs=inputs,
        run_root=run_root,
        stop_after_update=51,
    )
    assert Path(command[0]) == Path(os.path.abspath(runner_link))
    assert command[0] != str(runner_target.resolve())

    # All non-interpreter inputs keep strict symlink resolution.
    assert Path(inputs["config"]) == config.resolve()
    assert Path(inputs["migration"]) == migration.resolve()
    assert Path(inputs["authority"]) == authority.resolve()
    assert inputs["config"] != str(config_link)


def test_start_records_session_pid_pgid_starttime_and_native_exit(tmp_path, capsys):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=2.5)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.2))

    assert main(_start_args(run_root, config, migration, authority, runner)) == 0
    capsys.readouterr()

    state = _load_state(run_root)
    heartbeat = _load_heartbeat(run_root)
    launches = _launch_records(run_root)
    assert len(launches) == 1
    launch = launches[0]

    # start_new_session=True: child is its own session and process-group leader.
    assert launch["sid"] == launch["pid"]
    assert launch["pgid"] == launch["pid"]
    assert launch["stop_after"] == 51

    assert state["schema"] == STATE_SCHEMA
    assert state["child_pid"] == launch["pid"]
    assert state["child_pgid"] == launch["pgid"]
    assert state["child_pid"] == state["child_pgid"]
    assert state["child_start_ticks"] == launch["start_ticks"]
    assert state["child_native_exit"] == 0
    assert state["status"] == "waiting_acceptance"
    assert state["phase"] == "waiting_acceptance"
    assert state["supervisor_pid"] == os.getpid()
    assert state["supervisor_pid_start_ticks"] == _read_proc_start_ticks(os.getpid())
    # The successful prepare receipt seeds cumulative new-stage wall; the real
    # child duration is added on top without resetting or truncating it.
    assert state["prepare_wall_seconds"] == pytest.approx(2.5)
    assert state["prepared_receipt"]["path"] == "prepared.json"
    assert state["new_stage_wall_seconds"] >= 2.5 + 0.15
    assert state["phase_history"][0]["wall_seconds_before"] == pytest.approx(2.5)
    assert state["new_stage_wall_cap_seconds"] == 230400
    assert state["new_stage_output_cap_bytes"] == 20 * 1024**3
    # TERM happens one full grace+kill-wait+poll margin before the hard cap.
    assert state["stop_margin_seconds"] == pytest.approx(60.0 + 30.0 + 0.2)
    assert state["wall_stop_deadline"] == pytest.approx(230400 - (60.0 + 30.0 + 0.2))
    assert state["new_stage_wall_seconds"] < state["wall_stop_deadline"]

    # Durable heartbeat mirrors the recorded child identity and advances.
    assert heartbeat["pid"] == os.getpid()
    assert heartbeat["child_pid"] == launch["pid"]
    assert heartbeat["child_start_ticks"] == launch["start_ticks"]
    assert heartbeat["seq"] >= 1
    assert list(run_root.glob(".supervisor-state.json.*.tmp")) == []

    # Command is paths/flags only; no environment or secret material in state.
    command = state["runner_command"]
    assert command[1:4] == ["-m", "owac.runtime.raw_temporal_continuation", "train"]
    assert command[command.index("--run-root") + 1] == str(run_root)
    assert command[command.index("--device") + 1] == "cuda:0"
    state_text = (run_root / "supervisor-state.json").read_text(encoding="utf-8")
    for marker in ("PYTHONPATH", "API_KEY", "SECRET", "TOKEN="):
        assert marker not in state_text

    # The phase-1 checkpoint is discovered from real runner status, and no
    # acceptance is invented by the supervisor.
    checkpoint = state["phase1_checkpoint"]
    assert checkpoint["path"] == "checkpoints/checkpoint_final_000051.pt"
    assert (
        checkpoint["sha256"]
        == hashlib.sha256((run_root / checkpoint["path"]).read_bytes()).hexdigest()
    )
    assert state["acceptance"] is None
    assert not (run_root / "acceptance/phase-0051.json").exists()


def test_duplicate_supervisor_lock_refusal_exits_2_without_touching_child(tmp_path, capsys):
    run_root, config, migration, authority = _make_case(tmp_path)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.15))
    assert main(_start_args(run_root, config, migration, authority, runner)) == 0
    capsys.readouterr()
    _checkpoint, sha = _phase1_sha(run_root)
    assert main(_accept_args(run_root, sha)) == 0
    capsys.readouterr()
    launches_before = len(_launch_records(run_root))

    lock_path = run_root / "harness" / "supervisor.lock"
    assert lock_path.parent.is_dir()
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert main(_start_args(run_root, config, migration, authority, runner)) == 2
        start_result = json.loads(capsys.readouterr().out)
        assert start_result["status"] == "locked"
        assert main(_resume_args(run_root)) == 2
        resume_result = json.loads(capsys.readouterr().out)
        assert resume_result["status"] == "locked"
        assert len(_launch_records(run_root)) == launches_before


def test_acceptance_gate_requires_matching_valid_acceptance(tmp_path, capsys):
    run_root, config, migration, authority = _make_case(tmp_path)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.15))
    assert main(_start_args(run_root, config, migration, authority, runner)) == 0
    capsys.readouterr()
    assert _load_state(run_root)["status"] == "waiting_acceptance"
    checkpoint, sha = _phase1_sha(run_root)
    assert checkpoint.is_file()
    assert len(_launch_records(run_root)) == 1

    # No acceptance yet: resume refuses and launches no second child.
    assert main(_resume_args(run_root)) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "acceptance_invalid"
    assert len(_launch_records(run_root)) == 1

    # A reviewed sha that does not match an actual checkpoint is refused.
    acceptance_path = run_root / "acceptance/phase-0051.json"
    assert main(_accept_args(run_root, "0" * 64)) == 2
    capsys.readouterr()
    assert not acceptance_path.exists()
    assert len(_launch_records(run_root)) == 1

    # The matching acceptance is written atomically with reviewer provenance.
    assert main(_accept_args(run_root, sha)) == 0
    capsys.readouterr()
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    assert acceptance["schema"] == ACCEPTANCE_SCHEMA
    assert acceptance["accepted"] is True
    assert acceptance["step"] == 51
    assert acceptance["checkpoint_sha256"] == sha
    assert acceptance["reviewer"] == "sol"
    anchor = run_root / "lineage" / "accepted-C51.pt"
    assert anchor.is_file()
    assert os.path.samefile(anchor, checkpoint)
    assert acceptance["checkpoint_path"] == "lineage/accepted-C51.pt"
    assert list(acceptance_path.parent.glob(".*.tmp")) == []
    state_after_accept = _load_state(run_root)
    assert state_after_accept["acceptance"]["checkpoint_sha256"] == sha
    assert state_after_accept["phase1_checkpoint"]["path"] == "lineage/accepted-C51.pt"
    assert state_after_accept["phase1_checkpoint"]["sha256"] == sha

    # A tampered acceptance cannot unlock resume.
    acceptance["checkpoint_sha256"] = "1" * 64
    acceptance_path.write_text(json.dumps(acceptance) + "\n", encoding="utf-8")
    assert main(_resume_args(run_root)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "acceptance_invalid"
    assert len(_launch_records(run_root)) == 1
    assert _load_state(run_root)["status"] == "waiting_acceptance"

    # Restoring the real acceptance allows exactly one explicit phase-2 launch.
    assert main(_accept_args(run_root, sha)) == 0
    capsys.readouterr()
    assert main(_resume_args(run_root)) == 0
    capsys.readouterr()

    launches = _launch_records(run_root)
    assert len(launches) == 2
    phase2 = launches[1]
    assert phase2["stop_after"] == 5000
    assert phase2["resume"] == str(anchor)
    assert phase2["acceptance"] == str(acceptance_path)
    final = _load_state(run_root)
    assert final["status"] == "completed"
    assert final["phase"] == "completed"
    assert final["acceptance"]["checkpoint_sha256"] == sha
    assert final["child_native_exit"] == 0


def test_cumulative_new_stage_wall_across_phases_is_never_reset(tmp_path):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=1.0)
    body = """
_record(run_root, stop_after=stop_after, resume=_arg(_train, "--resume"))
_write_status(run_root, 51 if stop_after == 51 else 5000)
time.sleep(0.35)
sys.exit(0)
"""
    runner = _write_runner(tmp_path, body)
    supervisor = Supervisor(run_root, poll_seconds=0.02, heartbeat_seconds=0.05)

    first = supervisor.start(
        config=config,
        migration=migration,
        authority=authority,
        runner_python=runner,
        stop_after_update=51,
    )
    assert first["status"] == "waiting_acceptance"
    state_one = _load_state(run_root)
    wall_one = state_one["new_stage_wall_seconds"]
    # The prepare wall is the base, and the real phase-1 child wall is added.
    assert wall_one >= 1.0 + 0.25
    assert _load_heartbeat(run_root)["seq"] >= 1

    _checkpoint, sha = _phase1_sha(run_root)
    assert supervisor.accept(step=51, checkpoint_sha=sha, reviewer="sol")["status"] == "accepted"
    second = supervisor.resume(stop_after_update=5000)
    assert second["status"] == "completed"

    final = _load_state(run_root)
    wall_two = final["new_stage_wall_seconds"]
    assert len(final["phase_history"]) == 2
    assert final["phase_history"][0]["new_stage_wall_seconds"] == pytest.approx(wall_one, abs=0.05)
    # The phase-2 wall includes phase-1 wall; resume did not reset accounting.
    assert wall_two >= wall_one + 0.25
    assert wall_two <= final["new_stage_wall_cap_seconds"]
    assert final["phase_history"][0]["status"] == "completed"
    assert final["phase_history"][1]["status"] == "completed"
    assert [entry["child_pid"] for entry in final["phase_history"]] == [
        record["pid"] for record in _launch_records(run_root)
    ]


def test_nonzero_child_is_terminal_error_and_never_retried(tmp_path):
    run_root, config, migration, authority = _make_case(tmp_path)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.1, exit_code=3))
    supervisor = Supervisor(run_root, poll_seconds=0.02, heartbeat_seconds=0.05)

    result = supervisor.start(
        config=config,
        migration=migration,
        authority=authority,
        runner_python=runner,
        stop_after_update=51,
    )
    assert result["status"] == "error"
    assert result["exit_code"] == 2
    state = _load_state(run_root)
    assert state["status"] == "error"
    assert state["phase"] == "error"
    assert state["child_native_exit"] == 3
    assert "nonzero" in state["last_error"]
    assert state["acceptance"] is None
    launches = _launch_records(run_root)
    assert len(launches) == 1

    # No automatic retry: explicit resume refuses a terminal error state.
    retry = supervisor.resume(stop_after_update=5000)
    assert retry["status"] == "not_resumable"
    assert retry["exit_code"] == 2
    # A second start refuses even though the child is gone.
    again = supervisor.start(
        config=config,
        migration=migration,
        authority=authority,
        runner_python=runner,
        stop_after_update=51,
    )
    assert again["status"] == "state_exists"
    assert len(_launch_records(run_root)) == 1


def test_signal_owned_group_refuses_reused_wrong_group_dead_and_zombie_pids():
    decoy_code = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    )
    decoy = subprocess.Popen([PYTHON, "-c", decoy_code], start_new_session=True)
    try:
        ticks = _read_proc_start_ticks(decoy.pid)
        assert ticks is not None
        assert decoy.poll() is None

        # Same PID but a different recorded starttime => a reused PID, no signal.
        reused = {
            "child_pid": decoy.pid,
            "child_pgid": decoy.pid,
            "child_start_ticks": ticks + 1,
        }
        assert signal_owned_group(reused, signal.SIGTERM) is False
        assert decoy.poll() is None

        # Same PID and starttime but a different group is never signalled.
        wrong_group = {
            "child_pid": decoy.pid,
            "child_pgid": decoy.pid + 1,
            "child_start_ticks": ticks,
        }
        assert signal_owned_group(wrong_group, signal.SIGTERM) is False
        assert decoy.poll() is None

        # The supervisor never signals itself.
        own = {
            "child_pid": os.getpid(),
            "child_pgid": os.getpid(),
            "child_start_ticks": _read_proc_start_ticks(os.getpid()),
        }
        assert signal_owned_group(own, signal.SIGTERM) is False
    finally:
        if decoy.poll() is None:
            os.killpg(decoy.pid, signal.SIGKILL)
        decoy.wait(timeout=5)

    dead = {
        "child_pid": decoy.pid,
        "child_pgid": decoy.pid,
        "child_start_ticks": ticks,
    }
    assert signal_owned_group(dead, signal.SIGTERM) is False

    # A zombie child has already exited: like the monitor, treat it as dead.
    zombie = subprocess.Popen(
        [PYTHON, "-c", "import time; time.sleep(0.05)"], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _read_proc_state(zombie.pid) != "Z":
            time.sleep(0.01)
        assert _read_proc_state(zombie.pid) == "Z"
        zombie_ticks = _read_proc_start_ticks(zombie.pid)
        assert zombie_ticks is not None
        zombie_state = {
            "child_pid": zombie.pid,
            "child_pgid": zombie.pid,
            "child_start_ticks": zombie_ticks,
        }
        assert signal_owned_group(zombie_state, signal.SIGTERM) is False
    finally:
        zombie.wait(timeout=5)


def test_output_cap_signals_only_own_recorded_group_and_decoy_survives(tmp_path):
    run_root, config, migration, authority = _make_case(tmp_path)
    body = """
_record(run_root, stop_after=stop_after)


def _handle(signum, frame):
    (run_root / "fake-sigterm.json").write_text(
        json.dumps({"pid": os.getpid(), "signum": signum}), encoding="utf-8"
    )
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle)
(run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
(run_root / "checkpoints" / "junk.bin").write_bytes(b"x" * (256 * 1024))
time.sleep(30)
"""
    runner = _write_runner(tmp_path, body)
    decoy = subprocess.Popen(
        [PYTHON, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        supervisor = Supervisor(
            run_root,
            poll_seconds=0.02,
            heartbeat_seconds=0.05,
            output_cap_bytes=64 * 1024,
            checkpoint_reserve_bytes=0,
        )
        result = supervisor.start(
            config=config,
            migration=migration,
            authority=authority,
            runner_python=runner,
            stop_after_update=51,
        )
        assert result["status"] == "error"
        assert result["exit_code"] == 2
        state = _load_state(run_root)
        assert state["termination_reason"] == "output_cap"
        assert "output_cap" in state["last_error"]
        assert state["child_native_exit"] == 0
        assert state["new_stage_output_bytes"] > 64 * 1024

        marker = json.loads((run_root / "fake-sigterm.json").read_text(encoding="utf-8"))
        assert marker["pid"] == state["child_pid"]
        assert marker["signum"] == signal.SIGTERM
        # The decoy sits in a different session/process group and survives.
        assert decoy.poll() is None
        assert len(_launch_records(run_root)) == 1
    finally:
        if decoy.poll() is None:
            os.killpg(decoy.pid, signal.SIGKILL)
        decoy.wait(timeout=5)


def test_wall_stop_margin_terminates_own_group_before_hard_cap(tmp_path):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=0.1)
    body = """
_record(run_root, stop_after=stop_after)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""
    runner = _write_runner(tmp_path, body)
    supervisor = Supervisor(
        run_root,
        poll_seconds=0.02,
        heartbeat_seconds=0.05,
        wall_cap_seconds=1.0,
        stop_grace_seconds=0.2,
        kill_wait_seconds=0.2,
    )
    started = time.monotonic()
    result = supervisor.start(
        config=config,
        migration=migration,
        authority=authority,
        runner_python=runner,
        stop_after_update=51,
    )
    elapsed = time.monotonic() - started
    state = _load_state(run_root)

    stop_margin = 0.2 + 0.2 + 0.02
    assert result["status"] == "error"
    assert state["stop_margin_seconds"] == pytest.approx(stop_margin)
    assert state["wall_stop_deadline"] == pytest.approx(1.0 - stop_margin)
    assert state["termination_reason"] == "wall_stop_margin"
    assert state["child_native_exit"] == -signal.SIGKILL
    # The real wall overruns the stop deadline but stays below the hard cap,
    # and the overrun is persisted rather than truncated to either boundary.
    recorded = state["new_stage_wall_seconds"]
    assert recorded > state["wall_stop_deadline"]
    assert recorded < state["new_stage_wall_cap_seconds"]
    assert recorded == pytest.approx(0.1 + elapsed, abs=0.5)
    assert state["prepare_wall_seconds"] == pytest.approx(0.1)
    # Bounded: stop margin plus short grace, not an unbounded wait.
    assert elapsed < 10.0
    assert len(_launch_records(run_root)) == 1


def test_start_requires_prepared_receipt_and_seeds_real_wall(tmp_path, capsys):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=None)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.1))

    assert main(_start_args(run_root, config, migration, authority, runner)) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "refused"
    assert "prepare" in refusal["error"]
    assert not (run_root / "supervisor-state.json").exists()
    assert not (run_root / "fake-launches.jsonl").exists()

    (run_root / "prepared.json").write_text(
        json.dumps({"schema": "prepared", "prepare_wall_seconds": 0.75}) + "\n",
        encoding="utf-8",
    )
    assert main(_start_args(run_root, config, migration, authority, runner)) == 0
    capsys.readouterr()
    state = _load_state(run_root)
    assert state["prepare_wall_seconds"] == pytest.approx(0.75)
    assert state["prepared_receipt"]["prepare_wall_seconds"] == pytest.approx(0.75)
    assert state["new_stage_wall_seconds"] >= 0.75 + 0.05


def test_prepare_wall_inside_stop_margin_refuses_launch(tmp_path):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=0.8)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.1))
    supervisor = Supervisor(
        run_root,
        poll_seconds=0.02,
        heartbeat_seconds=0.05,
        wall_cap_seconds=1.0,
        stop_grace_seconds=0.2,
        kill_wait_seconds=0.2,
    )
    result = supervisor.start(
        config=config,
        migration=migration,
        authority=authority,
        runner_python=runner,
        stop_after_update=51,
    )
    state = _load_state(run_root)
    assert result["status"] == "error"
    assert result["exit_code"] == 2
    assert state["status"] == "error"
    assert "stop margin" in state["last_error"]
    assert state["prepare_wall_seconds"] == pytest.approx(0.8)
    assert not (run_root / "fake-launches.jsonl").exists()


def test_start_refused_when_cap_cannot_cover_stop_margin(tmp_path):
    # A 1 s cap cannot even cover grace+kill-wait+poll, so the proactive stop
    # margin makes any launch unsafe; refuse before starting a child.
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=0.0)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.1))
    supervisor = Supervisor(
        run_root,
        poll_seconds=0.02,
        wall_cap_seconds=1.0,
        stop_grace_seconds=2.0,
        kill_wait_seconds=5.0,
    )
    result = supervisor.start(
        config=config,
        migration=migration,
        authority=authority,
        runner_python=runner,
        stop_after_update=51,
    )
    state = _load_state(run_root)
    assert result["status"] == "error"
    assert result["exit_code"] == 2
    assert state["status"] == "error"
    assert state["stop_margin_seconds"] == pytest.approx(2.0 + 5.0 + 0.02)
    assert state["wall_stop_deadline"] == 0.0
    assert "stop margin" in state["last_error"]
    assert state["child_pid"] is None
    assert not (run_root / "fake-launches.jsonl").exists()


def test_resume_does_not_reset_cumulative_wall_cap(tmp_path):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=0.1)
    body = """
_record(run_root, stop_after=stop_after)
if stop_after == 51:
    _write_status(run_root, 51)
    time.sleep(0.7)
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""
    runner = _write_runner(tmp_path, body)
    supervisor = Supervisor(
        run_root,
        poll_seconds=0.02,
        heartbeat_seconds=0.05,
        wall_cap_seconds=2.5,
        stop_grace_seconds=0.2,
        kill_wait_seconds=0.2,
    )
    first = supervisor.start(
        config=config,
        migration=migration,
        authority=authority,
        runner_python=runner,
        stop_after_update=51,
    )
    assert first["status"] == "waiting_acceptance"
    wall_one = _load_state(run_root)["new_stage_wall_seconds"]
    assert wall_one >= 0.1 + 0.6

    _checkpoint, sha = _phase1_sha(run_root)
    assert supervisor.accept(step=51, checkpoint_sha=sha, reviewer="sol")["status"] == "accepted"
    started = time.monotonic()
    second = supervisor.resume(stop_after_update=5000)
    elapsed = time.monotonic() - started
    state = _load_state(run_root)

    assert second["status"] == "error"
    assert state["termination_reason"] == "wall_stop_margin"
    # The cap is cumulative: phase 2 consumed only the remaining budget, so its
    # own duration is well under a fresh margin; real overrun is still recorded.
    recorded = state["new_stage_wall_seconds"]
    assert recorded > state["wall_stop_deadline"]
    assert recorded < state["new_stage_wall_cap_seconds"]
    assert elapsed < 1.9
    assert state["phase_history"][1]["wall_seconds_before"] == pytest.approx(wall_one)
    assert len(_launch_records(run_root)) == 2


def test_supervisor_sigterm_stops_only_its_own_child_group(tmp_path):
    run_root, config, migration, authority = _make_case(tmp_path)
    body = """
def _handle(signum, frame):
    (run_root / "fake-sigterm.json").write_text(
        json.dumps({"pid": os.getpid(), "signum": signum}), encoding="utf-8"
    )
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle)
_record(run_root, stop_after=stop_after)
time.sleep(30)
"""
    runner = _write_runner(tmp_path, body)
    env = dict(os.environ)
    source = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = f"{source}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else source
    process = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "owac.runtime.raw_temporal_c_supervisor",
            *_start_args(run_root, config, migration, authority, runner),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            state = (
                json.loads((run_root / "supervisor-state.json").read_text(encoding="utf-8"))
                if (run_root / "supervisor-state.json").is_file()
                else None
            )
            if (
                state
                and state.get("status") == "running"
                and state.get("child_pid") is not None
                and _launch_records(run_root)
            ):
                break
            time.sleep(0.02)
        else:
            pytest.fail("supervisor never reached a running child state")
        assert process.poll() is None
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == 0, output
    state = _load_state(run_root)
    assert state["status"] == "stopped"
    assert state["termination_reason"] == "supervisor_sigterm"
    assert state["child_native_exit"] == 0
    assert state["last_error"] is None
    marker = json.loads((run_root / "fake-sigterm.json").read_text(encoding="utf-8"))
    assert marker["pid"] == state["child_pid"]
    assert len(_launch_records(run_root)) == 1


def test_output_accounting_counts_harness_excludes_symlinks_and_dedupes_inodes(tmp_path):
    run_root = tmp_path / "run"
    checkpoints = run_root / "checkpoints"
    checkpoints.mkdir(parents=True)
    payload = b"p" * 1024
    first = checkpoints / "checkpoint_000051.pt"
    first.write_bytes(payload)
    os.link(first, checkpoints / "latest.pt")
    harness = run_root / "harness"
    harness.mkdir()
    (harness / "task.md").write_bytes(b"h" * 5000)
    (harness / "receipt.json").write_bytes(b"r" * 200)
    shared_cache = tmp_path / "shared-cache"
    shared_cache.mkdir()
    (shared_cache / "cached.bin").write_bytes(b"c" * 7000)
    (run_root / "cache").symlink_to(shared_cache, target_is_directory=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"s" * 9000)
    (run_root / "linked-outside.bin").symlink_to(outside)

    expected = len(payload) + 5000 + 200  # checkpoint hardlinks count once.
    assert _run_root_output_bytes(run_root) == expected
    # Harness receipts under the run root are new output: no partial-only claim.
    assert _run_root_output_bytes(run_root) > len(payload)
    assert _run_root_output_bytes(run_root) < expected + 7000 + 9000


def test_status_reports_process_state_and_missing_state(tmp_path, capsys):
    run_root, config, migration, authority = _make_case(tmp_path)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.1))
    assert main(_start_args(run_root, config, migration, authority, runner)) == 0
    capsys.readouterr()

    assert main(["status", "--run-root", str(run_root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"]["status"] == "waiting_acceptance"
    assert status["process"]["alive"] is False
    assert status["process"]["process_state"] in {"dead", "absent"}
    assert status["heartbeat"]["schema"].endswith("supervisor-heartbeat.v1")
    assert status["heartbeat_age_seconds"] is not None

    missing_root = tmp_path / "missing"
    assert main(["status", "--run-root", str(missing_root)]) == 2
    missing = json.loads(capsys.readouterr().out)
    assert missing["status"] == "missing"


def test_start_adds_failed_preflight_debit_once_and_resume_does_not_double(tmp_path, capsys):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=0.5)
    charge = 114.33517408370972
    _write_failure_debit(run_root, charge)
    runner = _write_runner(tmp_path, _clean_body(sleep_seconds=0.1))
    assert main(_start_args(run_root, config, migration, authority, runner)) == 0
    capsys.readouterr()
    state = _load_state(run_root)
    assert state["failed_preflight_charge_seconds"] == pytest.approx(charge)
    assert state["prepare_wall_seconds"] == pytest.approx(0.5)
    assert state["new_stage_wall_seconds"] >= 0.5 + charge
    assert len(state["failed_preflight_receipts"]) == 1
    assert state["failed_preflight_receipts"][0]["charged_wall_seconds"] == pytest.approx(charge)
    wall_before_resume = state["new_stage_wall_seconds"]

    _checkpoint, sha = _phase1_sha(run_root)
    assert main(_accept_args(run_root, sha)) == 0
    capsys.readouterr()
    assert main(_resume_args(run_root)) == 0
    capsys.readouterr()
    final = _load_state(run_root)
    assert final["status"] == "completed"
    phase2 = [item for item in final["phase_history"] if item["phase"] == "phase-2"][-1]
    # Resume starts from the persisted wall; the failed-preflight debit is not
    # added a second time.
    assert phase2["wall_seconds_before"] == pytest.approx(wall_before_resume, abs=1.0)
    assert final["new_stage_wall_seconds"] < wall_before_resume + 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seconds": -1.0},
        {"seconds": "114.3"},
        {"seconds": 1.0, "optimizer_updates": 1},
        {"seconds": 1.0, "schema": "wrong.schema"},
    ],
)
def test_start_refuses_invalid_failed_preflight_ledger(tmp_path, capsys, kwargs):
    run_root, config, migration, authority = _make_case(tmp_path, prepare_wall_seconds=0.5)
    _write_failure_debit(run_root, **kwargs)
    runner = _write_runner(tmp_path, _clean_body())
    assert main(_start_args(run_root, config, migration, authority, runner)) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "refused"
    assert not (run_root / "fake-launches.jsonl").exists()


def test_paused_phase2_resume_uses_latest_checkpoint_and_keeps_c51_anchor(tmp_path, capsys):
    run_root, config, migration, authority = _make_case(tmp_path)
    body = """
_record(run_root, stop_after=stop_after, resume=_arg(_train, "--resume"),
        acceptance=_arg(_train, "--acceptance"))
_write_status(run_root, stop_after)
_status_path = run_root / "run-status.json"
_status = json.loads(_status_path.read_text())
_status["status"] = "stopped_at_update_limit"
_status["native_exit"] = 0
_status_path.write_text(json.dumps(_status))
time.sleep(0.1)
sys.exit(0)
"""
    runner = _write_runner(tmp_path, body)
    acceptance_path = run_root / "acceptance/phase-0051.json"

    assert main(_start_args(run_root, config, migration, authority, runner)) == 0
    capsys.readouterr()
    assert _load_state(run_root)["status"] == "waiting_acceptance"
    original_c51, sha = _phase1_sha(run_root)
    assert main(_accept_args(run_root, sha)) == 0
    capsys.readouterr()
    anchor = run_root / "lineage" / "accepted-C51.pt"
    assert anchor.is_file() and os.path.samefile(anchor, original_c51)
    state = _load_state(run_root)
    assert state["phase1_checkpoint"]["path"] == "lineage/accepted-C51.pt"
    assert state["phase1_checkpoint"]["sha256"] == sha

    # Simulate later checkpoint retention pruning the ordinary C51 file.  The
    # accepted hardlink anchor must keep the originally approved SHA available.
    original_c51.unlink()
    assert anchor.is_file()
    assert acceptance_path.is_file()

    assert main(_resume_args(run_root, stop_after_update=250)) == 0
    capsys.readouterr()
    launches = _launch_records(run_root)
    assert len(launches) == 2
    phase2 = launches[1]
    assert phase2["stop_after"] == 250
    assert phase2["resume"] == str(anchor)
    assert phase2["acceptance"] == str(acceptance_path)
    paused = _load_state(run_root)
    assert paused["status"] == "paused"
    assert paused["new_stage_wall_seconds"] >= 0.0
    wall_before_phase3 = paused["new_stage_wall_seconds"]
    latest_250 = run_root / "checkpoints" / "checkpoint_final_000250.pt"
    assert latest_250.is_file()

    assert main(_resume_args(run_root, stop_after_update=1000)) == 0
    capsys.readouterr()
    launches = _launch_records(run_root)
    assert len(launches) == 3
    phase3 = launches[2]
    assert phase3["stop_after"] == 1000
    # Clean paused phase-2 resume continues from the actual latest checkpoint,
    # not silently from phase 51 again.
    assert phase3["resume"] == str(latest_250)
    assert phase3["acceptance"] is None
    final = _load_state(run_root)
    assert final["status"] == "paused"
    assert final["resume_checkpoint"]["source"] == "run-status.last_checkpoint"
    assert final["resume_checkpoint"]["step"] == 250
    assert (
        final["resume_checkpoint"]["sha256"] == hashlib.sha256(latest_250.read_bytes()).hexdigest()
    )
    phase3_entry = [item for item in final["phase_history"] if item["phase"] == "phase-2"][-1]
    assert phase3_entry["wall_seconds_before"] == pytest.approx(wall_before_phase3, abs=1.0)
    assert final["new_stage_wall_seconds"] > wall_before_phase3
    assert anchor.is_file()
    assert acceptance_path.is_file()
