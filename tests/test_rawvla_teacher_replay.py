"""Fixture tests for the bounded author replay adapter and CLI (no simulator)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from owac.runtime.rawvla_teacher import main
from owac.simulation.rawvla_teacher_replay import (
    MAX_REPLAY_SOURCES,
    ReplayConfig,
    TeacherReplayBlocked,
    inspect_source_hdf5,
    prepare_task6_normal,
)


def _authority(path: Path, *, replay_max: int = 50) -> Path:
    payload = {
        "schema": "owac.raw-teacher.authorization.v1",
        "authorization": {"demonstration_replay": True},
        "limits": {
            "demonstrations_replayed_max": replay_max,
            "local_gpu_hours": 12,
            "additional_disk_bytes": 50 * 1024**3,
        },
        "training": {"max_cumulative_updates": 2000},
    }
    path.write_text(json.dumps(payload))
    return path


def _hdf5_root(tmp_path: Path, demos: int = 50) -> Path:
    root = tmp_path / "LIBERO-datasets" / "libero_goal"
    root.mkdir(parents=True)
    path = root / "put_the_cream_cheese_in_the_bowl_demo.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index in range(demos):
            group = data.create_group(f"demo_{index}")
            group.create_dataset("actions", data=np.zeros((6, 7), dtype=np.float32))
            group.create_dataset("states", data=np.zeros((6, 17), dtype=np.float64))
            group.create_dataset("robot_states", data=np.zeros((6, 8), dtype=np.float32))
            group.attrs["model_file"] = "<xml/>"
    return root.parent


def test_inspect_source_is_metadata_only_and_counts_ids(tmp_path):
    root = _hdf5_root(tmp_path)
    report = inspect_source_hdf5(root)
    assert report["demo_count"] == 50
    assert len(report["source_demo_ids"]) == 50
    assert report["source_demo_ids"][0].startswith(
        "libero_goal/put_the_cream_cheese_in_the_bowl/demo_"
    )
    assert report["simulator_imported"] is False
    assert "libero" not in sys.modules or "libero.libero" not in sys.modules


def test_prepare_dry_run_requires_authority_normal_and_exact_cap(tmp_path):
    root = _hdf5_root(tmp_path)
    authority = _authority(tmp_path / "authority.json")
    config = ReplayConfig(
        author_root=tmp_path / "author-main",
        hdf5_root=root,
        out_root=tmp_path / "raw",
        rgb_out_root=tmp_path / "rgb",
    )
    report = prepare_task6_normal(
        config,
        authority=json.loads(authority.read_text()),
        repo_root=tmp_path,
        dry_run=True,
    )
    assert report["executed"] is False
    assert len(report["selected_source_demo_ids"]) == MAX_REPLAY_SOURCES
    assert report["split_counts"] == {"train": 40, "validation": 5, "test": 5}
    assert report["normal_domain"] == "Normal"
    with pytest.raises(TeacherReplayBlocked, match="requires exactly 50"):
        prepare_task6_normal(
            config,
            authority=json.loads(authority.read_text()),
            repo_root=tmp_path,
            max_attempts=49,
            dry_run=True,
        )


def test_prepare_blocks_bad_cap_before_author_import(tmp_path):
    root = _hdf5_root(tmp_path)
    authority = _authority(tmp_path / "authority.json", replay_max=49)
    config = ReplayConfig(
        author_root=tmp_path / "missing-author",
        hdf5_root=root,
        out_root=tmp_path / "raw",
        rgb_out_root=tmp_path / "rgb",
    )
    with pytest.raises(TeacherReplayBlocked, match="permits only 49"):
        prepare_task6_normal(
            config,
            authority=json.loads(authority.read_text()),
            repo_root=tmp_path,
            dry_run=True,
        )


def test_cli_inspect_and_status_and_import_are_safe(tmp_path, capsys):
    root = _hdf5_root(tmp_path)
    code = main(["inspect-source", "--hdf5-root", str(root)])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["demo_count"] == 50
    code = main(["status", "--manifest", str(tmp_path / "missing.json")])
    status = json.loads(capsys.readouterr().out)
    assert code == 2
    assert status["status"] == "unavailable"
    assert "simulator" not in "".join(sys.modules)


def test_prepare_cli_dry_run_writes_no_replay_outputs(tmp_path, capsys):
    root = _hdf5_root(tmp_path)
    authority = _authority(tmp_path / "authority.json")
    code = main(
        [
            "--repo-root",
            str(tmp_path),
            "prepare-data",
            "--authority",
            str(authority),
            "--author-source",
            str(tmp_path / "missing-author"),
            "--hdf5-root",
            str(root),
            "--out-root",
            str(tmp_path / "raw"),
            "--rgb-out-root",
            str(tmp_path / "rgb"),
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["executed"] is False
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "rgb").exists()
