"""Full evaluation manifest integrity and paired stochastic-policy boundaries."""

import json
import random
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from owac.runtime.rawvla_eval_audit import audit, completed_rows
from owac.runtime.rawvla_full_eval import (
    paired_seed,
    seed_instruction_generators,
    validate_manifest,
    verify_frozen,
)
from owac.runtime.rawvla_paper_eval import get_model
from owac.runtime.rawvla_robotwin import ModelClient, reset_model


def robotwin_manifest():
    domains = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")
    return {
        "entries": [
            {"task_name": f"task{task}", "dataset_seed": seed, "lighting_domain": domain}
            for task in range(13)
            for seed in range(50)
            for domain in domains
        ]
    }


def test_fixed_manifest_rejects_duplicate_replacing_a_case():
    manifest = robotwin_manifest()
    validate_manifest(manifest, "robotwin")
    manifest["entries"][-1] = dict(manifest["entries"][0])
    with pytest.raises(ValueError, match="Duplicate or incomplete"):
        validate_manifest(manifest, "robotwin")


def test_fixed_manifest_rejects_missing_domain_and_missing_episode():
    manifest = robotwin_manifest()
    manifest["entries"][-1]["lighting_domain"] = "unlisted"
    with pytest.raises(ValueError, match="Duplicate or incomplete"):
        validate_manifest(manifest, "robotwin")
    manifest["entries"].pop()
    with pytest.raises(ValueError, match="3249 != 3250"):
        validate_manifest(manifest, "robotwin")


def test_frozen_code_tamper_is_rejected(tmp_path):
    code = tmp_path / "runtime.py"
    code.write_text("changed code")
    (tmp_path / "protocol.json").write_text(
        json.dumps({"code_sha256": {str(code): "different"}, "manifests": {}})
    )
    with pytest.raises(ValueError, match="Frozen code changed"):
        verify_frozen(tmp_path)


def test_live_ledger_tolerates_only_unfinished_tail(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"success":false}\n{"suc')
    assert completed_rows(path) == [{"success": False}]
    path.write_text('{"success":false}\n{"suc\n')
    with pytest.raises(json.JSONDecodeError):
        completed_rows(path)


def test_denominator_audit_rejects_duplicate_success_without_hiding_failure(tmp_path):
    entry = {
        "suite": "libero_spatial",
        "task_id": 0,
        "init_state_index": 0,
        "lighting_domain": "Normal",
    }
    robot_entry = {"task_name": "grab_roller", "dataset_seed": 0, "lighting_domain": "Normal"}
    (tmp_path / "protocol.json").write_text(json.dumps({"total_episodes": 2}))
    (tmp_path / "libero-manifest.json").write_text(json.dumps({"entries": [entry]}))
    (tmp_path / "robotwin-manifest.json").write_text(json.dumps({"entries": [robot_entry]}))
    libero = tmp_path / "libero/libero_spatial"
    libero.mkdir(parents=True)
    robotwin = tmp_path / "robotwin"
    robotwin.mkdir()
    success = {
        "rawvla_bench_entry": entry,
        "success": True,
        "num_steps": 20,
        "task_description": "pick",
        "wall_seconds": 1,
    }
    failure = {
        "entry": robot_entry,
        "case_id": "robotwin-0",
        "success": False,
        "eval_seed": 0,
        "num_steps": 400,
        "language": "grab",
        "wall_seconds": 2,
    }
    (libero / "task_0.episodes.jsonl").write_text(json.dumps(success) + "\n")
    (robotwin / "episodes.jsonl").write_text(json.dumps(failure) + "\n")
    report = audit(tmp_path)
    assert report["state"] == "complete"
    assert report["platforms"]["robotwin"]["final_success_rate_percent"] == 0
    with (libero / "task_0.episodes.jsonl").open("a") as stream:
        stream.write(json.dumps(success) + "\n")
    report = audit(tmp_path)
    assert report["state"] != "complete"
    assert "libero: Duplicate completed episode" in report["errors"]


def test_action_noise_identity_is_independent_of_prior_episodes(monkeypatch):
    seed = paired_seed("robotwin", "grab_roller", 0)
    assert seed != paired_seed("robotwin", "grab_roller", 1)
    assert seed != paired_seed("robotwin", "lift_pot", 0)
    monkeypatch.setenv("RAWVLA_FLOW_SEED", str(seed))
    monkeypatch.setenv("RAWVLA_CASE_ID", "case-identity")

    class Client:
        seeds = []

        def predict_action(self, query):
            example = query["examples"][0]
            self.seeds.append(example["flow_seed"])
            return {"ok": True, "data": {"actions": [[0] * 14] * 50}}

    model = ModelClient.__new__(ModelClient)
    model.client = Client()
    reset_model(model)
    model.step({}, 0)
    model.step({}, 50)
    reset_model(model)
    model.step({}, 0)
    assert model.client.seeds == [seed, (seed + 50) % (2**32), seed]


def test_original_condition_plugin_changes_only_sensor_noise_seed(monkeypatch):
    from owac.runtime import rawvla_robotwin

    calls = []

    def apply_lighting(task, *, seed):
        calls.append(seed)
        return {"lighting_ev": -6.5, "domain": "Low"}

    base = SimpleNamespace(apply_lighting_from_env=apply_lighting)
    monkeypatch.setitem(sys.modules, "envs", SimpleNamespace(_base_task=base))
    client = object()
    monkeypatch.setattr(rawvla_robotwin, "get_model", lambda args: client)
    monkeypatch.setenv("RAWVLA_EXPECTED_ENTRY", json.dumps({"noise_seed": 1234}))
    assert get_model({}) is client
    task = SimpleNamespace(
        cameras=SimpleNamespace(raw_sensor_noise_seed=0, raw_sensor_noise_frame_index=99),
        physical_pose=(1, 2, 3),
    )
    actual = base.apply_lighting_from_env(task, seed=18)
    assert calls == [18]
    assert actual == {"lighting_ev": -6.5, "domain": "Low", "noise_seed": 1234}
    assert task.cameras.raw_sensor_noise_frame_index == 0
    assert task.physical_pose == (1, 2, 3)


def test_description_and_choice_rng_reset_together():
    import numpy as np

    def instruction():
        candidates = list(range(20))
        random.shuffle(candidates)
        description = random.choice(candidates)
        return description, int(np.random.choice(candidates))

    seed_instruction_generators(1234)
    first = instruction()
    for _ in range(10):
        instruction()
    seed_instruction_generators(1234)
    assert instruction() == first


def test_supervisor_signal_stops_owned_simulation_and_preserves_peer():
    child_code = "import time; time.sleep(30)"
    peer = subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True)
    parent_code = """
import signal, subprocess, sys, time
from owac.runtime.rawvla_job_guard import terminate_children
signal.signal(signal.SIGTERM, terminate_children)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                         start_new_session=True)
print(child.pid, flush=True)
time.sleep(30)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True
    )
    try:
        owned = int(parent.stdout.readline())
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=8) == 143
        assert peer.poll() is None
        status = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(owned)], capture_output=True, text=True, check=False
        ).stdout.strip()
        assert not status or status.startswith("Z")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        peer.terminate()
        peer.wait()
