"""CPU-only tests for the independent SwanLab sidecar."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from owac.runtime.raw_full_visual_swanlab import (
    GpuSampler,
    Sidecar,
    SwanLabUploader,
)


class FakeSwanLab:
    def __init__(self) -> None:
        self.login_calls = []
        self.init_kwargs = None
        self.logs = []
        self.finishes = 0
        self.url = "https://swanlab.example/OWAC/owac-fullvisual-shared-stage"

    def login(self, **kwargs):
        self.login_calls.append(kwargs)

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return SimpleNamespace(url=self.url)

    def log(self, metrics, step):
        self.logs.append((int(step), dict(metrics)))

    def finish(self):
        self.finishes += 1


def _write_run(tmp_path: Path, *, paused: bool = False, completed: bool = False) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    updates = [
        {
            "new_global_updates": 0,
            "new_anchor_presentations": 16,
            "loss_components": {"total": 2.5, "mse": 2.4, "cosine": 0.1},
            "lr_used": 1e-5,
            "host_wall_phases": {"total_seconds": 2.0},
            "effective_batch": 16,
            "global_token_denominator": 8192,
            "gpu_allocated_bytes": 123,
            "remote_gpu_wall_used_seconds": 90.0,
        },
        {
            "new_global_updates": 1,
            "new_anchor_presentations": 32,
            "loss_components": {"total": 2.4, "mse": 2.3, "cosine": 0.1},
            "lr_used": 2e-5,
            "host_wall_phases": {"total_seconds": 1.5},
            "effective_batch": 16,
            "global_token_denominator": 8192,
        },
    ]
    (root / "updates.jsonl").write_text(
        "\n".join(json.dumps(item) for item in updates) + "\n", encoding="utf-8"
    )
    validations = [
        {
            "step": 0,
            "feature_total": 2.2,
            "feature_mse": 2.1,
            "feature_cosine": 0.1,
        },
        {
            "step": 1,
            "feature_total": 2.1,
            "feature_mse": 2.0,
            "feature_cosine": 0.1,
        },
    ]
    (root / "validation.jsonl").write_text(
        "\n".join(json.dumps(item) for item in validations) + "\n", encoding="utf-8"
    )
    status = {
        "status": "completed" if completed else "paused" if paused else "running",
        "remote_gpu_wall_used_seconds": 100.0,
    }
    (root / "run-status.json").write_text(json.dumps(status), encoding="utf-8")
    return root


def _sidecar(root: Path, sdk: FakeSwanLab) -> Sidecar:
    uploader = SwanLabUploader(sdk=sdk, logdir=root / "swanlab")
    return Sidecar(run_root=root, uploader=uploader)


def test_sidecar_backfills_train_validation_and_uploads_real_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SWANLAB_API_KEY", "secret-not-uploaded")
    root = _write_run(tmp_path)
    sdk = FakeSwanLab()
    sidecar = _sidecar(root, sdk)
    sidecar.gpu = SimpleNamespace(sample=lambda **kwargs: {})
    assert sidecar.poll_once() is False
    assert sdk.init_kwargs["project"] == "OWAC"
    assert sdk.init_kwargs["mode"] == "online"
    assert sdk.init_kwargs["public"] is False
    assert sdk.init_kwargs["resume"] == "allow"
    assert sdk.logs[0][0] == 0
    assert sdk.logs[0][1]["train/loss_total"] == 2.5
    assert sdk.logs[0][1]["val/loss_total"] == 2.2
    assert sdk.logs[0][1]["train/lr"] == 1e-5
    assert sdk.logs[0][1]["train/anchors_per_second"] == 8.0
    assert sdk.logs[0][1]["budget/remote_gpu_wall_seconds"] == 90.0
    assert sdk.logs[1][1]["budget/remote_gpu_wall_seconds"] == 100.0
    state = Path(root / "swanlab-sidecar-state.json").read_text()
    assert "secret-not-uploaded" not in state
    status = json.loads(Path(root / "swanlab-sidecar-status.json").read_text())
    assert status["url"] == sdk.url
    assert status["last_step"] == 1


def test_late_validation_is_merged_and_restart_does_not_repeat_history(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SWANLAB_API_KEY", "secret")
    root = _write_run(tmp_path)
    # Remove step 1 validation, then append it after the train record was logged.
    (root / "validation.jsonl").write_text(
        json.dumps({"step": 0, "feature_total": 2.2, "feature_mse": 2.1, "feature_cosine": 0.1})
        + "\n",
        encoding="utf-8",
    )
    sdk = FakeSwanLab()
    sidecar = _sidecar(root, sdk)
    sidecar.gpu = SimpleNamespace(sample=lambda **kwargs: {})
    sidecar.poll_once()
    assert [step for step, _ in sdk.logs] == [0, 1]
    with (root / "validation.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"step": 1, "feature_total": 2.1, "feature_mse": 2.0, "feature_cosine": 0.1})
            + "\n"
        )
    sidecar.poll_once()
    assert sdk.logs[-1][0] == 1
    assert sdk.logs[-1][1]["val/loss_total"] == 2.1
    # Fresh process with the same cursor must not re-upload old train steps.
    sdk2 = FakeSwanLab()
    restarted = _sidecar(root, sdk2)
    restarted.gpu = SimpleNamespace(sample=lambda **kwargs: {})
    restarted.poll_once()
    assert sdk2.logs == []
    assert restarted.cursor.validation_steps == {0, 1}


def test_paused_keeps_waiting_and_completed_finishes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWANLAB_API_KEY", "secret")
    paused = _write_run(tmp_path / "paused", paused=True)
    sdk = FakeSwanLab()
    sidecar = _sidecar(paused, sdk)
    sidecar.gpu = SimpleNamespace(sample=lambda **kwargs: {})
    assert sidecar.poll_once() is False
    assert sidecar.cursor.status == "waiting_paused"
    assert sdk.finishes == 0

    completed = _write_run(tmp_path / "completed", completed=True)
    sdk2 = FakeSwanLab()
    sidecar2 = _sidecar(completed, sdk2)
    sidecar2.gpu = SimpleNamespace(sample=lambda **kwargs: {})
    assert sidecar2.poll_once() is True
    assert sidecar2.cursor.finished is True
    assert sdk2.finishes == 1


def test_gpu_sampler_is_low_frequency_and_does_not_invent_metrics(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="0, 44, 1024, 24576\n1, 47, 2048, 24576\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampler = GpuSampler(interval=60)
    first = sampler.sample()
    assert first["gpu/0/utilization_percent"] == 44
    assert first["gpu/1/memory_used_mb"] == 2048
    assert len(calls) == 1
    assert sampler.sample() == {}
    assert len(calls) == 1


def test_uploader_uses_get_url_fallback(monkeypatch) -> None:
    class NoUrlRun:
        def get_url(self):
            return "https://swanlab.example/@Jeong/OWAC/runs/stable"

    class SDK:
        def login(self, **kwargs):
            pass

        def init(self, **kwargs):
            return NoUrlRun()

    monkeypatch.setenv("SWANLAB_API_KEY", "secret")
    from owac.runtime.raw_full_visual_swanlab import SwanLabUploader

    uploader = SwanLabUploader(sdk=SDK(), logdir=Path("/tmp/swanlab-test"))
    assert uploader.initialize() == "https://swanlab.example/@Jeong/OWAC/runs/stable"


def test_validation_metrics_include_nested_startup_eligible_domain_lighting() -> None:
    from owac.runtime.raw_full_visual_swanlab import _validation_metrics

    metrics = _validation_metrics(
        {
            "feature_total": 1.2,
            "feature_mse": 1.0,
            "feature_cosine": 2.0,
            "valid_tokens": 512,
            "windows": 46,
            "startup": {"feature_mse": 0.8},
            "eligible_ge_25": {"feature_mse": 1.1},
            "per_domain": {"libero": {"feature_mse": 0.9}, "robotwin2": {"feature_mse": 1.1}},
            "per_lighting": {"Normal": {"feature_mse": 0.7}, "Low": {"feature_mse": 1.3}},
        }
    )
    assert metrics["val/startup_mse"] == 0.8
    assert metrics["val/eligible_mse"] == 1.1
    assert metrics["val/domain_libero/mse"] == 0.9
    assert metrics["val/light_Low/mse"] == 1.3
    assert metrics["val/mse"] == 1.0


def test_restarted_sdk_session_finishes_after_previous_session_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("SWANLAB_API_KEY", "test-key")
    root = _write_run(tmp_path)
    first = _sidecar(root, FakeSwanLab())
    first.gpu = SimpleNamespace(sample=lambda: {})
    first.poll_once()
    first._finish()
    first._save()
    (root / "run-status.json").write_text(json.dumps({"status": "completed"}))
    sdk = FakeSwanLab()
    restarted = _sidecar(root, sdk)
    restarted.gpu = SimpleNamespace(sample=lambda: {})
    assert restarted.poll_once() is True
    assert sdk.finishes == 1
