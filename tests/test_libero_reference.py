import json
import logging
import socket
from types import SimpleNamespace

import pytest

from owac.evaluation.libero_reference import EpisodeObserver
from owac.runtime.libero_reference import run


def emit(observer, message):
    observer.emit(logging.LogRecord("test", logging.INFO, "", 0, message, (), None))


def test_infrastructure_failure_cannot_be_reported_as_policy_success(tmp_path):
    observer = EpisodeObserver(tmp_path, 1)
    observer.task = {"task_id": 0, "task_name": "example"}
    emit(observer, "\nTask: example")
    emit(observer, "Caught exception: connection lost")
    # An upstream stale done flag must not turn a transport exception into success.
    emit(observer, "Success: True")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["infrastructure_errors"] == 1
    assert summary["successes"] == 0
    assert summary["success_rate"] is None
    assert observer.rows[0]["status"] == "error"


def test_initial_stabilization_is_excluded_from_control_steps(tmp_path):
    observer = EpisodeObserver(tmp_path, 1)
    observer.task = {"task_id": 0, "task_name": "example"}
    emit(observer, "\nTask: example")
    observer.current["environment_steps"] = 24
    emit(observer, "Success: True")
    assert observer.rows[0]["control_steps"] == 14
    assert observer.rows[0]["episode_index"] == 0


def test_runner_rejects_an_existing_policy_port_before_creating_a_run(tmp_path):
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen()
        args = SimpleNamespace(trials=1, port=existing.getsockname()[1], output=tmp_path / "run")
        with pytest.raises(OSError):
            run(args)
    assert not args.output.exists()
