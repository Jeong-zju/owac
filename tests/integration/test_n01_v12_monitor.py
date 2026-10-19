"""Integration tests for the N01 v1.2 read-only monitor service and writer library.

These tests exercise the real HTTP surface (``http.client``, not mocks) plus the disk
contract between :class:`RunStateWriter`, :func:`reconcile` and the page endpoints.
No GPU is required: the GPU path is exercised through an injected ``nvidia-smi`` command.
"""

from __future__ import annotations

import http.client
import json
import threading
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from owac.runtime import n01_v12_monitor as monitor

pytestmark = pytest.mark.integration

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(92))
ROLE = "libero-raw_n01-seed101"


def _request(
    server: monitor.ThreadingHTTPServer,
    path: str,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = server.server_address[0], server.server_address[1]
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _request_json(
    server: monitor.ThreadingHTTPServer, path: str
) -> tuple[int, dict[str, str], Any]:
    status, headers, body = _request(server, path)
    return status, headers, json.loads(body.decode("utf-8"))


def _write_worker_file(run_root: Path, role: str, **fields: Any) -> Path:
    payload = {
        "role": role,
        "method": "raw_n01",
        "input_domain": "RAW",
        "data_kind": "hdr_derived",
        "platform": "libero",
        "seed": 101,
        "planned_steps": 50_000,
        "step": 5000,
        "phase": "training",
        "state": "training",
        "alive": True,
        "finished": False,
        "started_at": monitor.iso_now(),
        "updated_at": monitor.iso_now(),
        "rollouts": {
            "completed_rollouts": 50,
            "expected_rollouts": 50,
            "successes": 23,
            "failures": 24,
            "infra_pending": 3,
            "note": "P3 开发闭环",
        },
        "recent_log": [],
    }
    payload.update(fields)
    path = monitor.workers_dir(run_root) / f"{role}.json"
    monitor.atomic_write_json(path, payload)
    return path


def _write_cases(run_root: Path) -> None:
    monitor.artifacts_dir(run_root).mkdir(parents=True, exist_ok=True)
    (monitor.artifacts_dir(run_root) / "case1.png").write_bytes(PNG_BYTES)
    monitor.atomic_write_json(
        monitor.monitor_dir(run_root) / monitor.CASES_RELATIVE,
        {
            "artifacts": [
                {
                    "id": "case1",
                    "kind": "image",
                    "url": "artifacts/case1.png",
                    "caption": "同源RAW固定预览",
                    "camera": "agent",
                    "episode": 7,
                    "frame": 123,
                    "legend": "平均融合权重",
                    "source_time": "t=1.5s",
                    "unit": "raw counts",
                }
            ]
        },
    )


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run-root"
    root.mkdir()
    return root


@pytest.fixture()
def server(run_root: Path):
    state = monitor.MonitorState(run_root)
    httpd = monitor.build_server(state, "127.0.0.1", 0)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_module_import_starts_no_background_threads() -> None:
    threads = [
        thread.name for thread in threading.enumerate() if thread.name.startswith("n01-monitor")
    ]
    assert threads == []
    default_page = monitor.MonitorState(Path("/nonexistent-run")).html_file
    assert default_page.name == monitor.MONITOR_HTML_NAME


def test_writer_roundtrip_and_public_endpoints(server, run_root: Path) -> None:
    run = monitor.RunStateWriter(
        run_root,
        ROLE,
        method="raw_n01",
        input_domain="RAW",
        data_kind="hdr_derived",
        platform="libero",
        seed=101,
        planned_steps=50_000,
        gpu_index=0,
        strict=True,
    )
    run.update(step=5000, phase="training", activity="训练中")
    run.metric("train_action_loss", 5000, 0.83, unit="loss", label="训练 action loss")
    run.rollouts(50, 50, 23, 24, 3, note="P3 开发闭环")
    run.log("开始训练", level="info")
    _write_cases(run_root)
    monitor.atomic_write_json(
        monitor.monitor_dir(run_root) / monitor.GPUS_NAME,
        {
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA H100",
                    "memory_used_mib": 12_000,
                    "memory_total_mib": 97_887,
                    "utilization_pct": 88,
                }
            ],
            "sampled_at": monitor.iso_now(),
            "error": None,
        },
    )

    status, headers, payload = _request_json(server, "/api/status")
    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert payload["stage"] == "P3"
    assert payload["state"] == "training"
    assert payload["current_run"]["method"] == "raw_n01"
    assert payload["current_run"]["step"] == 5000
    assert payload["runs"][0]["alive"] is True
    assert payload["runs"][0]["rollouts"]["successes"] == 23
    assert payload["resources"]["gpus"][0]["queue"] == "RAW"
    assert payload["resources"]["gpus"][0]["job"] == ROLE
    assert payload["resources"]["dev_loop"]["budget_rollouts"] == 100
    assert payload["resources"]["dev_loop"]["by_method"][0]["completed_rollouts"] == 50
    assert payload["artifacts"][0]["url"] == "artifacts/case1.png"
    assert payload["recent_log"][-1]["message"] == "开始训练"

    status, _, metrics = _request_json(server, "/api/metrics")
    assert status == 200
    series = {item["name"]: item for item in metrics["series"]}
    point = series["train_action_loss"]["points"][0]
    assert (point["step"], point["value"], isinstance(point["t"], int)) == (5000, 0.83, True)
    # The label carries both the caller's label and the arm.  Two arms write the same
    # metric names into one file, so a bare label leaves the legend with two identical
    # entries -- the visual form of merging the arms into one curve.
    label = series["train_action_loss"]["label"]
    assert label.startswith("训练 action loss")
    assert "libero-raw_n01-seed101" in label, f"the legend must name the arm, got {label!r}"
    assert series["train_action_loss"]["role"] == "libero-raw_n01-seed101"
    assert series["train_action_loss"]["unit"] == "loss"

    status, _, artifacts = _request_json(server, "/api/artifacts")
    assert status == 200
    assert [item["id"] for item in artifacts["artifacts"]] == ["case1"]

    status, _, logs = _request_json(server, "/api/logs?tail=5")
    assert status == 200
    assert logs["logs"][-1]["message"] == "开始训练"

    status, headers, body = _request(server, "/artifacts/case1.png")
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == PNG_BYTES

    status, headers, body = _request(server, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"n01" in body.lower()

    run.finish()
    assert monitor.reconcile(run_root)["state"] == "stage_end"
    _, _, finished = _request_json(server, "/api/status")
    assert finished["runs"][0]["alive"] is False
    assert finished["runs"][0]["exit_reason"] == "completed"


def test_status_json_is_written_and_reused_without_memory(tmp_path: Path, run_root: Path) -> None:
    _write_worker_file(run_root, ROLE)
    payload = monitor.reconcile(run_root)
    on_disk = json.loads((monitor.monitor_dir(run_root) / monitor.STATUS_NAME).read_text("utf-8"))
    assert on_disk == payload
    fresh_state = monitor.MonitorState(run_root)
    assert fresh_state.status()["runs"][0]["role"] == ROLE


def test_empty_run_root_status(run_root: Path) -> None:
    payload = monitor.reconcile(run_root)
    assert payload["runs"] == []
    assert payload["current_run"] is None
    assert payload["state"] is None
    assert payload["stage"] == "P3"
    assert payload["artifacts"] == []
    assert payload["recent_log"] == []
    assert payload["resources"]["dev_loop"]["by_method"] == []
    assert payload["resources"]["gpus"] == []
    assert "尚未收到任何 worker 心跳" in payload["activity"]
    assert json.loads((monitor.monitor_dir(run_root) / monitor.STATUS_NAME).read_text("utf-8"))


def test_heartbeat_expiry_marks_run_suspicious(run_root: Path) -> None:
    # Age the fixture beyond the *default* bound rather than a fixed five minutes: the
    # bound has to sit above the writer's cadence (log_every x slowest step time), and a
    # fixed small age silently stopped being stale once that was corrected.
    age = monitor.DEFAULT_HEARTBEAT_TIMEOUT_S + 600.0
    stale = (datetime.now(UTC) - timedelta(seconds=age)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_worker_file(run_root, ROLE, updated_at=stale)
    payload = monitor.reconcile(run_root)
    run = payload["runs"][0]
    assert run["alive"] is False
    assert run["suspicious"] is True
    assert run["exit_reason"] and "心跳过期" in run["exit_reason"]
    assert payload["state"] == "failed"
    assert "心跳均已过期" in payload["activity"]

    _write_worker_file(run_root, ROLE)
    payload = monitor.reconcile(run_root)
    assert payload["runs"][0]["alive"] is True
    assert payload["runs"][0]["suspicious"] is False
    assert payload["runs"][0]["exit_reason"] is None


def test_writer_recovers_started_at_and_leaves_no_temp_files(run_root: Path) -> None:
    writer = monitor.RunStateWriter(run_root, ROLE, method="raw_n01", strict=True)
    started_at = writer.snapshot()["started_at"]
    writer.update(step=10)
    writer.finish()
    again = monitor.RunStateWriter(run_root, ROLE, method="raw_n01", strict=True)
    assert again.snapshot()["started_at"] == started_at
    assert again.snapshot()["finished"] is False
    assert list(monitor.workers_dir(run_root).glob("*.tmp")) == []
    assert len(list(monitor.workers_dir(run_root).glob("*.json"))) == 1


def test_missing_html_returns_503_plain_chinese(run_root: Path, tmp_path: Path) -> None:
    state = monitor.MonitorState(run_root, html_path=tmp_path / "absent.html")
    httpd = monitor.build_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, body = _request(httpd, "/")
        assert status == 503
        assert headers["Content-Type"].startswith("text/plain")
        assert "监控页面文件尚未就绪" in body.decode("utf-8")
        assert "Traceback" not in body.decode("utf-8")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_path_traversal_is_rejected(server, run_root: Path, tmp_path: Path) -> None:
    (run_root / "secret.txt").write_text("run-root-secret", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside-secret", encoding="utf-8")
    _write_cases(run_root)
    for path in (
        "/artifacts/../secret.txt",
        "/artifacts/../../outside.txt",
        "/artifacts/%2e%2e/secret.txt",
        "/artifacts/..%2f..%2foutside.txt",
        "/artifacts//etc/passwd",
    ):
        status, headers, body = _request(server, path)
        text = body.decode("utf-8", errors="replace")
        assert status in {403, 404}, (path, status, text)
        assert "secret" not in text and "root:" not in text, (path, text)


def test_range_requests_are_supported(server, run_root: Path) -> None:
    _write_cases(run_root)
    status, headers, body = _request(server, "/artifacts/case1.png", {"Range": "bytes=10-19"})
    assert status == 206
    assert headers["Content-Range"] == f"bytes 10-19/{len(PNG_BYTES)}"
    assert headers["Accept-Ranges"] == "bytes"
    assert body == PNG_BYTES[10:20]

    status, headers, body = _request(server, "/artifacts/case1.png", {"Range": "bytes=999-1000"})
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{len(PNG_BYTES)}"


def test_metrics_downsampling_is_bounded(run_root: Path) -> None:
    metrics_path = monitor.monitor_dir(run_root) / monitor.METRICS_NAME
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "".join(
            json.dumps(
                {
                    "name": "train_action_loss",
                    "label": "训练 action loss",
                    "unit": "loss",
                    "step": step,
                    "value": 1.0 / (step + 1),
                    "t": 1_750_000_000 + step,
                }
            )
            + "\n"
            for step in range(5000)
        ),
        encoding="utf-8",
    )
    payload = monitor.read_metrics(run_root)
    points = payload["series"][0]["points"]
    assert len(points) == monitor.MAX_POINTS_PER_SERIES
    assert points[0]["step"] == 0
    assert points[-1]["step"] == 4999
    assert all(left["step"] < right["step"] for left, right in pairwise(points))
    assert monitor.downsample([{"step": index} for index in range(10)], 3)[0] == {"step": 0}
    assert len(monitor.downsample([{"step": index} for index in range(10)], 3)) == 3


def test_log_tail_reads_only_the_tail(run_root: Path, server) -> None:
    log_path = monitor.logs_dir(run_root) / f"{ROLE}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "".join(
            json.dumps({"ts": monitor.iso_now(), "level": "info", "message": f"line {index}"})
            + "\n"
            for index in range(5000)
        ),
        encoding="utf-8",
    )
    lines = monitor.tail_lines(log_path, 100)
    assert len(lines) == 100
    assert json.loads(lines[-1])["message"] == "line 4999"
    status, _, payload = _request_json(server, "/api/logs?tail=10")
    assert status == 200
    assert len(payload["logs"]) == 10
    assert payload["logs"][-1]["message"] == "line 4999"


def test_gpu_sampling_degrades_gracefully(run_root: Path) -> None:
    missing = monitor.sample_gpus(command=["n01-monitor-missing-binary"])
    assert missing["gpus"] == []
    assert missing["unavailable"] is True
    assert missing["error"]

    sampler = monitor.GpuSampler(run_root, interval_s=60.0, command=["n01-monitor-missing-binary"])
    payload = sampler.sample_once()
    assert payload["gpus"] == []
    cached = json.loads((monitor.monitor_dir(run_root) / monitor.GPUS_NAME).read_text("utf-8"))
    assert cached["unavailable"] is True
    assert monitor.reconcile(run_root)["resources"]["gpus"] == []


def test_parses_real_nvidia_smi_csv_shape(run_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> Any:
        class Result:
            returncode = 0
            stdout = (
                "0, NVIDIA H100 80GB HBM3, 12000, 97887, 88\n"
                "1, NVIDIA H100 80GB HBM3, 3, 97887, 0\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)
    payload = monitor.sample_gpus()
    assert [gpu["index"] for gpu in payload["gpus"]] == [0, 1]
    assert payload["gpus"][0]["memory_used_mib"] == 12_000
    assert payload["gpus"][0]["utilization_pct"] == 88
    assert payload["gpus"][1]["utilization_pct"] == 0


def test_once_mode_and_non_loopback_refusal(
    run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert monitor.main(["--run-root", str(run_root), "--once"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["runs"] == []
    assert (monitor.monitor_dir(run_root) / monitor.STATUS_NAME).is_file()

    exit_code = monitor.main(["--run-root", str(run_root), "--host", "0.0.0.0", "--port", "0"])
    assert exit_code == 2
    assert "拒绝启动" in capsys.readouterr().err

    with pytest.raises(ValueError, match="拒绝启动"):
        monitor.build_server(monitor.MonitorState(run_root), "192.168.0.10", 0)


def test_overrides_drive_orchestrator_state(run_root: Path) -> None:
    monitor.write_overrides(
        run_root, stage="P3", state="waiting_resource", activity="等待 GPU 空闲", p4_total=53_000
    )
    payload = monitor.reconcile(run_root)
    assert payload["state"] == "waiting_resource"
    assert payload["activity"] == "等待 GPU 空闲"
    assert payload["p4_total"] == 53_000


def test_unknown_route_returns_json_error(server) -> None:
    status, headers, payload = _request_json(server, "/api/definitely-missing")
    assert status == 404
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert "error" in payload and payload["status"] == 404


def test_heartbeat_bound_exceeds_the_writer_cadence() -> None:
    """The staleness bound must be longer than the gap between worker-file writes.

    The writer updates once per logged step, so the bound has to cover
    ``log_every x slowest step time``.  Measured: LIBERO ~110 s between writes and
    RoboTwin ~540 s.  At the previous 90 s default the page reported a running LIBERO arm
    as "heartbeat expired, possibly abnormal exit" while it was training at 86% GPU
    utilisation -- a false positive in a delivered artifact, which a reader would believe.
    """

    assert monitor.DEFAULT_HEARTBEAT_TIMEOUT_S >= 900.0, (
        "a bound below (log_every x slowest step time) reports healthy runs as dead"
    )


def test_metric_rows_carry_the_role_and_series_are_keyed_by_it(run_root: Path) -> None:
    """Two arms publish the same metric names from two processes into one file.

    Keying the chart series by name alone merged them: a `val_action_loss` curve alternated
    between two different models, and nothing in the payload said so (measured on the live
    run: 12 validation points, every row without a role, two values per step).
    """

    for role, values in (
        ("libero-raw_n01-seed101", [0.9, 0.8]),
        ("libero-rgb_n01-seed101", [0.5, 0.4]),
    ):
        writer = monitor.RunStateWriter(run_root, role)
        for value in values:
            writer.metric("val_action_loss", 1000, value, unit="loss")

    rows = [
        json.loads(line)
        for line in (monitor.monitor_dir(run_root) / monitor.METRICS_NAME).read_text().splitlines()
        if line.strip()
    ]
    assert [row["role"] for row in rows] == [
        "libero-raw_n01-seed101",
        "libero-raw_n01-seed101",
        "libero-rgb_n01-seed101",
        "libero-rgb_n01-seed101",
    ], "every metric row must name the arm that produced it"

    payload = monitor.read_metrics(run_root)
    assert len(payload["series"]) == 2, "the two arms must not be merged into one series"
    assert {series["role"] for series in payload["series"]} == {
        "libero-raw_n01-seed101",
        "libero-rgb_n01-seed101",
    }
    for series in payload["series"]:
        assert len(series["points"]) == 2
        values = sorted(point["value"] for point in series["points"])
        assert values in ([0.8, 0.9], [0.4, 0.5]), (
            f"points from different arms were mixed: {values}"
        )


def test_unattributed_metric_rows_are_labelled_not_merged(run_root: Path) -> None:
    """Rows written before the role existed must not be guessed into an arm."""

    path = monitor.monitor_dir(run_root) / monitor.METRICS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"name": "val_action_loss", "step": 1000, "value": 0.07}) + "\n",
        encoding="utf-8",
    )
    payload = monitor.read_metrics(run_root)
    assert len(payload["series"]) == 1
    assert payload["series"][0]["role"] is None
    assert "未标注" in payload["series"][0]["label"]
