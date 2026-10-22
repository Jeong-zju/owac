"""Fixed-route read-only monitor tests for the full-visual stage."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from owac.runtime.raw_full_visual_monitor import EXPECTED_TRAINABLE, make_server, snapshot


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _current_processes() -> list[dict[str, int]]:
    from owac.runtime.raw_full_visual_monitor import _proc_stat

    stat = _proc_stat(os.getpid())
    assert stat is not None
    return [
        {
            "rank": rank,
            "pid": int(os.getpid()),
            "pid_start_ticks": int(stat["start_ticks"]),
            "pgid": int(stat["pgid"]),
        }
        for rank in (0, 1)
    ]


def _run_root(tmp_path: Path) -> Path:
    root = tmp_path / "full-visual-run"
    root.mkdir()
    _write(
        root / "run-status.json",
        {
            "status": "running",
            "phase": "full_visual_training",
            "new_global_updates": 1,
            "target_cumulative_updates": 3116,
            "new_anchor_presentations": 16,
            "target_anchor_presentations": 49856,
            "budget_update_equivalents": 1885,
            "full_visual_parameters": EXPECTED_TRAINABLE,
            "remote_gpu_wall_used_seconds": 123.0,
            "remote_gpu_wall_cap_seconds": 140841.2631939833,
            "total_gpu_wall_cap_seconds": 230400,
            "output_bytes": 1000,
            "output_cap_bytes": 15396584412,
            "finalization_reserve_seconds": 600,
            "native_exit": None,
            "last_checkpoint": {
                "step": 1,
                "kind": "final",
                "path": "checkpoints/checkpoint_new_000001",
                "sha256": "a" * 64,
            },
            "rank_processes": _current_processes(),
            "gpu_by_rank": [
                {
                    "rank": 0,
                    "allocated_bytes": 1024**3,
                    "reserved_bytes": 2 * 1024**3,
                    "max_allocated_bytes": 3 * 1024**3,
                    "max_reserved_bytes": 4 * 1024**3,
                },
                {
                    "rank": 1,
                    "allocated_bytes": 2048**3,
                    "reserved_bytes": 3 * 1024**3,
                    "max_allocated_bytes": 4 * 1024**3,
                    "max_reserved_bytes": 5 * 1024**3,
                },
            ],
        },
    )
    _write(
        root / "run-meta.json",
        {
            "data_contract": {"protocol": "legacy-c-full-40-5-5.v1"},
            "split": {
                "train": [f"train-{i}" for i in range(40)],
                "validation": [f"val-{i}" for i in range(5)],
                "test": [f"test-{i}" for i in range(5)],
            },
        },
    )
    _write(
        root / "latest.json",
        {"new_global_updates": 1, "kind": "final", "checkpoint_sha256": "a" * 64},
    )
    _write(
        root / "selection.json",
        {
            "schema": "owac.raw-stem-temporal.feature-selection.v1",
            "selection_metric": "validation_token_total",
            "best_step": 1,
            "best_value": 0.5,
            "checkpoint_sha256": "a" * 64,
        },
    )
    (root / "updates.jsonl").write_text(
        json.dumps(
            {
                "new_global_updates": 1,
                "loss_components": {"mse": 0.2, "cosine": 0.1, "total": 0.21},
                "lr_used": 1e-5,
                "timestamp": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                "gpu_by_rank": [
                    {"rank": 0, "allocated_bytes": 1024, "reserved_bytes": 2048},
                    {"rank": 1, "allocated_bytes": 4096, "reserved_bytes": 8192},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "validation.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "feature_mse": 0.2,
                "feature_cosine": 0.1,
                "feature_total": 0.21,
                "relative_feature_mse": 0.02,
                "valid_tokens": 52224,
                "sources": 5,
                "windows": 102,
                "episode_resets": 5,
                "checkpoint_sha256": "a" * 64,
                "is_best": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_snapshot_is_bounded_and_path_free(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    data = snapshot(root)
    assert data["new_global_updates"] == 1
    assert data["target_new_global_updates"] == 3116
    assert data["actual_scope"]["trainable_parameters"] == EXPECTED_TRAINABLE
    assert data["actual_scope"]["teacher_frozen"] is True
    assert data["source"]["source_count"] == 50
    assert data["source"]["split"] == {"train": 40, "validation": 5, "test": 5}
    assert data["source"]["public_shared_dataset_used"] is False
    assert data["validation"]["feature_mse"] == 0.2
    assert data["validation"]["relative_feature_mse"] == 0.02
    assert data["selection"]["best_step"] == 1
    assert data["status"] == "running"
    assert data["process_health"]["all_alive"] is True
    assert all(item["classification"] == "alive" for item in data["processes"])
    assert data["native_exit"] is None
    assert len(data["gpu"]) == 2
    assert data["last_checkpoint"]["step"] == 1
    encoded = json.dumps(data, ensure_ascii=False)
    assert "checkpoint_new_000001" not in encoded
    assert "checkpoints/" not in encoded


def test_fixed_routes_only_and_no_checkpoint_exposure(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    server = make_server(root, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:

        def get(path: str) -> tuple[int, str, str]:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{path}", timeout=5
                ) as response:
                    return (
                        response.status,
                        response.headers.get("Content-Type", ""),
                        response.read().decode("utf-8"),
                    )
            except urllib.error.HTTPError as exc:
                return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode("utf-8")

        index_status, index_type, index_body = get("/")
        assert index_status == 200
        assert "text/html" in index_type
        assert "学生整视觉头" in index_body
        assert "428,579,952" in index_body
        assert "102 windows / 5 sources" in index_body
        assert "unknown" in index_body

        api_status, api_type, api_body = get("/api/status")
        assert api_status == 200
        assert "application/json" in api_type
        assert json.loads(api_body)["new_global_updates"] == 1

        health_status, _, health_body = get("/healthz")
        assert health_status == 200 and json.loads(health_body)["ok"] is True

        for forbidden in (
            "/checkpoints/checkpoint_new_000001/model.pt",
            "/run-meta.json",
            "/../run-status.json",
        ):
            status, _, _ = get(forbidden)
            assert status == 404

        assert "model.pt" not in index_body
        assert "checkpoint_new_000001" not in index_body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _mutate_status(root: Path, **changes: object) -> None:
    path = root / "run-status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_fake_pid_is_missing_and_dead_effective_status(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _mutate_status(
        root,
        rank_processes=[
            {"rank": 0, "pid": 999_999_999, "pid_start_ticks": 1, "pgid": 1},
            {"rank": 1, "pid": 999_999_998, "pid_start_ticks": 1, "pgid": 1},
        ],
    )
    data = snapshot(root)
    assert data["status_claimed"] == "running"
    assert data["status"] == "stale_dead"
    assert data["process_health"]["all_alive"] is False
    assert all(item["classification"] == "missing" for item in data["processes"])
    assert data["native_exit"] is None


def test_reused_startticks_is_not_alive_and_dead_status(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    current = _current_processes()
    current[0]["pid_start_ticks"] += 1
    _mutate_status(root, rank_processes=current)
    data = snapshot(root)
    assert data["processes"][0]["classification"] == "reused"
    assert data["processes"][0]["alive"] is False
    assert data["status"] == "stale_dead"
    assert data["native_exit"] is None


def test_live_process_with_stale_heartbeat_is_stale_not_running(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    (root / "updates.jsonl").write_text(
        json.dumps(
            {
                "new_global_updates": 1,
                "loss_components": {"mse": 0.2, "cosine": 0.1, "total": 0.21},
                "timestamp": "2020-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data = snapshot(root)
    assert data["process_health"]["all_alive"] is True
    assert data["process_health"]["stale_heartbeat"] is True
    assert data["status"] == "stale"
    assert data["native_exit"] is None


def test_missing_preparation_metadata_never_defaults_to_50(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    (root / "run-meta.json").unlink()
    data = snapshot(root)
    assert data["source"]["source_count"] is None
    assert data["source"]["split"] == {"train": None, "validation": None, "test": None}
    html_body = __import__(
        "owac.runtime.raw_full_visual_monitor", fromlist=["_render_html"]
    )._render_html(data)
    assert "待核验" in html_body
    assert "50 源" not in html_body


def test_public_shared_usage_is_derived_from_data_contract(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    meta_path = root / "run-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["data_contract"] = {"protocol": "owac.shared-rawvla.task6-normal.8-2-2.v1"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert snapshot(root)["source"]["public_shared_dataset_used"] is True

    meta.pop("data_contract")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert snapshot(root)["source"]["public_shared_dataset_used"] is None

    meta["data_contract"] = {"protocol": "legacy-c-full-40-5-5.v1"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert snapshot(root)["source"]["public_shared_dataset_used"] is False
