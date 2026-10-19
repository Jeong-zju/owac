"""The simple view must never substitute historical probes for the current experiment."""

import json
import subprocess
import sys
from pathlib import Path

from owac.runtime import n02_v12_monitor as monitor
from owac.runtime.n02_v12_status import append_history, update_status


def test_task_chain_status_is_separate_and_does_not_expose_extra_fields(tmp_path):
    path = tmp_path / "task-chain-r14/small/status.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                branch="small",
                phase="frozen",
                state="running",
                step=8,
                updated_at=0,
                secret="never expose",
            )
        )
    )
    result = monitor.task_chain_payload(tmp_path)
    assert result["runs"][0]["step"] == 8
    assert result["runs"][0]["stale"] is True
    assert "secret" not in result["runs"][0]
    assert result["runs"][1] == dict(branch="pi", state="waiting")
    (tmp_path / "task-chain-r14/comparison.json").write_text(
        json.dumps(
            dict(
                rows=[dict(branch="small", phase="joint", test_joint_deg=18.0, secret="hidden")],
                state_only_test_deg=15.0,
                secret="hidden",
            )
        )
    )
    result = monitor.task_chain_payload(tmp_path)
    assert result["comparison"]["rows"][0]["test_joint_deg"] == 18
    assert "secret" not in json.dumps(result)


def test_learning_monitor_isolates_r15_and_blocks_symlink_status(tmp_path):
    old = tmp_path / "task-chain-r14/small/status.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(dict(state="running", step=999)))
    result = monitor.learning_payload(tmp_path)
    assert result["runs"] == [] and result["completed_runs"] == 0
    run = tmp_path / "learning-r15/raw-3-s15019"
    run.mkdir(parents=True)
    (run / "status.json").write_text(
        json.dumps(dict(run_id=run.name, state="running", step=64, secret="hidden"))
    )
    result = monitor.learning_payload(tmp_path)
    assert result["runs"][0]["step"] == 64 and "secret" not in json.dumps(result)
    external = tmp_path.parent / (tmp_path.name + "-outside-status.json")
    external.write_text(json.dumps(dict(run_id="pi-3-s15019", state="running", step=888)))
    other = tmp_path / "learning-r15/pi-3-s15019"
    other.mkdir()
    (other / "status.json").symlink_to(external)
    result = monitor.learning_payload(tmp_path)
    assert len(result["runs"]) == 1


def test_fewshot_fit_gate_is_separate_from_relative_dev_improvement(tmp_path):
    run = tmp_path / "P2" / "fewshot"
    update_status(
        run, run_id=run.name, stage="P2", state="running", step=64, metric_kind="sampled_control"
    )
    (tmp_path / "active_training.json").write_text(
        json.dumps(dict(phase="fewshot_fit", runs=[dict(stage="P2", run_id=run.name)]))
    )
    values = dict(
        joint_rms_deg=12.0, first_joint_max_deg=4.0, gripper_mae=0.04, gripper_max_error=0.1
    )
    limits = dict(
        joint_rms_deg_max=3.0,
        first_joint_max_deg_max=5.0,
        gripper_mae_max=0.05,
        gripper_max_error_max=0.15,
    )
    record = dict(step=64, passed=True, maxima=values, limits=limits, rows=[dict(seed=1)])
    (run / "fit_latest.json").write_text(json.dumps(record))
    (run / "fit_0000064.json").write_text(json.dumps(record))
    (run / "fit_0000000.json").write_text(
        json.dumps({**record, "step": 0, "maxima": {**values, "joint_rms_deg": 35.0}})
    )
    append_history(run, step=64, wall_time=1, train_loss=None, sampled_control_mse=0.001)
    payload = monitor.training_payload(tmp_path)
    assert payload["phase"] == "fewshot_fit"
    card = payload["runs"][0]
    assert card["training_fit"]["passed"] is False
    assert card["curve"] == [dict(step=64, error=0.001)]
    assert card["training_fit_curve"] == [dict(step=0, error=35.0), dict(step=64, error=12.0)]
    assert card["action_validation"] is None
    values["joint_rms_deg"] = float("nan")
    (run / "fit_latest.json").write_text(json.dumps(record))
    assert monitor.training_payload(tmp_path)["runs"][0]["training_fit"] is None


def test_isolated_standalone_r11_monitor_before_and_after_first_validation(tmp_path):
    run = tmp_path / "P2/r11-standalone"
    update_status(
        run, run_id=run.name, stage="P2", state="running", step=3, metric_kind="sampled_control"
    )
    (tmp_path / "active_training.json").write_text(
        json.dumps(dict(phase="task_adaptation", runs=[dict(stage="P2", run_id=run.name)]))
    )
    protocol = dict(
        version="n02-control-r11-v1",
        horizon=50,
        replan_interval=50,
        max_joint_delta=None,
        max_gripper_delta=None,
        clip_grippers=False,
    )
    (run / "run_config.json").write_text(
        json.dumps(
            dict(
                task_adaptation=dict(protocol="n02-task-repair-r11-v1"),
                execution=protocol,
                online_checks=dict(enabled=True),
            )
        )
    )

    def metrics(score):
        opening = dict(
            mse=score,
            first10_mse=score,
            prefix_joint_mse=score,
            prefix_gripper_mse=score,
            joint_mse=score,
            gripper_mse=score,
        )
        return dict(
            sampled_control_mse=score,
            sampled_metric="balanced_sampled_control14_mse",
            task_control=dict(strata={"Normal/opening": opening}, **opening),
            executed_prefix_proxy=dict(balanced_mse=score, balanced_prefix_mse=score),
            execution_contract=protocol,
            task_dev_manifest_sha256="fixture",
            calibration_weights_sha256="fixture",
            online_checks=dict(valid=15, expected_causal=15),
        )

    baseline = metrics(1.0)
    (run / "dev_baseline.json").write_text(json.dumps(baseline))
    (run / "dev_sampled_latest.json").write_text(json.dumps(baseline))
    append_history(run, step=0, wall_time=0, train_loss=None, sampled_control_mse=1.0)
    program = (
        "import runpy,json,sys; from pathlib import Path; "
        f"m=runpy.run_path({str(Path(monitor.__file__).resolve())!r}); "
        f"p=m['training_payload'](Path({str(tmp_path)!r})); "
        "assert 'owac' not in sys.modules; print(json.dumps(p))"
    )

    def isolated_payload():
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", program],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return json.loads(result.stdout)["runs"][0]

    pending = isolated_payload()["action_validation"]
    assert pending["passed"] is None and pending["pending"] is True
    trained = {"step": 50, **metrics(0.9)}
    (run / "dev_sampled_latest.json").write_text(json.dumps(trained))
    (run / "checkpoint_0000050.metrics.json").write_text(json.dumps(trained))
    (run / "checkpoint_0000050.pt").touch()
    append_history(run, step=50, wall_time=50, train_loss=0.5, sampled_control_mse=0.9)
    assert isolated_payload()["action_validation"]["passed"] is True


def test_training_speed_uses_actual_short_block_exposures(tmp_path):
    run = tmp_path / "P2/short-blocks"
    update_status(
        run,
        run_id=run.name,
        stage="P2",
        state="running",
        step=1,
        step_seconds=20.0,
        anchors_per_second=0.5,
    )
    (tmp_path / "active_training.json").write_text(
        json.dumps(dict(runs=[dict(stage="P2", run_id=run.name, effective_batch_size=12)]))
    )
    card = monitor.training_payload(tmp_path)["runs"][0]
    assert card["samples_per_minute"] == 30.0


def test_native_execution_witnesses_do_not_become_raw_successes(tmp_path):
    case = tmp_path / "repair-r11-20261017/native-original-a1"
    case.mkdir(parents=True)
    (case / "result.json").write_text(
        json.dumps(
            dict(
                returncode=0,
                accepted_success=True,
                physical=dict(
                    physical_accepted=True,
                    diagnostic_only=False,
                    control_steps=89,
                    max_hold_seconds=2.98,
                ),
            )
        )
    )
    (case / "protocol.json").write_text(json.dumps(dict(scope="native RGB execution witness")))
    (case / "episode0.mp4").write_bytes(b"video")
    result = monitor.execution_cases(tmp_path)
    assert len(result) == 1 and result[0]["policy"] == "native_rgb_reference"
    assert result[0]["accepted_success"] is True
    assert monitor.repair_cases(tmp_path) == [] and monitor.representation_cases(tmp_path) == []
    assert monitor.execution_video(tmp_path, "../native-original-a1") is None
    (case / "episode0.mp4").unlink()
    (case / "episode0.mp4").symlink_to(tmp_path / "outside.mp4")
    (tmp_path / "outside.mp4").write_bytes(b"outside")
    assert monitor.execution_video(tmp_path, "native-original-a1") is None
    (case / "protocol.json").write_text(json.dumps(dict(scope="RAW rollout")))
    assert monitor.execution_cases(tmp_path)[0]["accepted_success"] is None


def test_task_monitor_scores_the_checkpoint_actually_selected_for_rollout(tmp_path):
    run = tmp_path / "P2" / "task-selection"
    update_status(
        run,
        run_id=run.name,
        stage="P2",
        state="evaluating",
        step=32,
        step_seconds=4.0,
        metric_kind="sampled_control",
    )
    for step, score in ((0, 1.0), (16, 0.9), (32, 1.2)):
        append_history(run, step=step, wall_time=step, train_loss=None, sampled_control_mse=score)
    baseline = dict(
        sampled_control_mse=1.0, task_control={"strata": {"Normal/opening": {"mse": 1.0}}}
    )
    latest = dict(
        sampled_control_mse=1.2,
        task_control={"strata": {"Normal/opening": {"mse": 1.2}}},
        task_dev_manifest_sha256="fixed",
    )
    saved = dict(
        step=16,
        sampled_control_mse=0.9,
        task_control={"strata": {"Normal/opening": {"mse": 1.0}}},
        task_dev_manifest_sha256="fixed",
    )
    for name, data in (
        ("dev_baseline.json", baseline),
        ("dev_sampled_latest.json", latest),
        ("checkpoint_0000016.metrics.json", saved),
        ("selection.json", dict(selected=dict(step=16, eligible=True))),
    ):
        (run / name).write_text(json.dumps(data))
    (tmp_path / "active_training.json").write_text(
        json.dumps(
            dict(
                phase="task_adaptation",
                runs=[
                    dict(stage="P2", run_id=run.name, label="冻结视觉对照", effective_batch_size=8)
                ],
            )
        )
    )
    card = monitor.training_payload(tmp_path)["runs"][0]
    assert card["latest_dev"] == dict(step=16, error=0.9)
    assert card["latest_training_dev"] == dict(step=32, error=1.2)
    assert card["selected_checkpoint_step"] == 16
    assert card["action_validation"]["passed"] is True
    assert card["label"] == "冻结视觉对照"
    assert card["samples_per_minute"] == 120


def test_task_monitor_does_not_show_overall_gain_as_pass_when_opening_regresses(tmp_path):
    run = tmp_path / "P2" / "task-adaptation"
    update_status(
        run, run_id=run.name, stage="P2", state="running", step=16, metric_kind="sampled_control"
    )
    baseline = dict(
        sampled_control_mse=1.0, task_control={"strata": {"Normal/opening": {"mse": 1.0}}}
    )
    latest = dict(
        sampled_control_mse=0.8, task_control={"strata": {"Normal/opening": {"mse": 1.2}}}
    )
    (run / "dev_baseline.json").write_text(json.dumps(baseline))
    (run / "dev_sampled_latest.json").write_text(json.dumps(latest))
    (tmp_path / "active_training.json").write_text(
        json.dumps(dict(phase="task_adaptation", runs=[dict(stage="P2", run_id=run.name)]))
    )
    result = monitor.training_payload(tmp_path)
    assert result["phase"] == "task_adaptation"
    validation = result["runs"][0]["action_validation"]
    assert validation["passed"] is False
    assert "不晋级" in validation["label"]


def test_video_http_ranges_seek_head_and_reject_invalid_requests(tmp_path):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    directory = tmp_path / "representation-r8-rollouts" / "range-probe"
    directory.mkdir(parents=True)
    video = directory / "episode0.mp4"
    video.write_bytes(b"0123456789")

    class Handler(monitor.Handler):
        run_root_override = tmp_path

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for requested, code, expected, content_range in (
            (None, 200, b"0123456789", None),
            ("bytes=2-5", 206, b"2345", "bytes 2-5/10"),
            ("bytes=7-", 206, b"789", "bytes 7-9/10"),
            ("bytes=-3", 206, b"789", "bytes 7-9/10"),
            ("bytes=8-200", 206, b"89", "bytes 8-9/10"),
            ("bytes=20-", 416, b"", "bytes */10"),
            ("bytes=4-2", 416, b"", "bytes */10"),
            ("bytes=-0", 416, b"", "bytes */10"),
            ("bytes=0-1,3-4", 416, b"", "bytes */10"),
        ):
            connection = http.client.HTTPConnection(*server.server_address, timeout=2)
            connection.request(
                "GET",
                "/api/representation-video?case=range-probe",
                headers={"Range": requested} if requested else {},
            )
            response = connection.getresponse()
            assert response.status == code
            assert response.getheader("Content-Range") == content_range
            assert response.read() == expected
            connection.close()
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request(
            "HEAD", "/api/representation-video?case=range-probe", headers={"Range": "bytes=3-4"}
        )
        response = connection.getresponse()
        assert response.status == 206 and response.getheader("Content-Length") == "2"
        assert response.read() == b""
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_current_training_selection_and_persistent_validation(tmp_path):
    root = tmp_path
    old = root / "P2" / "old-profile"
    update_status(old, run_id="old-profile", stage="P2", state="completed", step=1000)
    active = root / "P2" / "current-raw"
    update_status(
        active,
        run_id="current-raw",
        stage="P2",
        state="running",
        step=12,
        target_steps=1000,
        step_seconds=4.0,
        dev_action_error=None,
    )
    append_history(active, step=0, wall_time=0, train_loss=None, dev_action_error=2.0)
    append_history(active, step=10, wall_time=40, train_loss=1.0, dev_action_error=0.5)
    append_history(active, step=12, wall_time=48, train_loss=0.9, dev_action_error=None)
    (root / "active_training.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "current-raw",
                        "stage": "P2",
                        "label": "RAW 模型",
                        "effective_batch_size": 2,
                    },
                    {"run_id": "current-rgb", "stage": "P2", "label": "RGB 对照"},
                ]
            }
        )
    )
    result = monitor.training_payload(root)
    assert [r["run_id"] for r in result["runs"]] == ["current-raw", "current-rgb"]
    raw, rgb = result["runs"]
    assert raw["latest_dev"] == {"step": 10, "error": 0.5}
    assert raw["improvement_percent"] == 75.0
    assert raw["samples_per_minute"] == 30.0
    assert rgb["state"] == "waiting" and rgb["latest_dev"] is None


def test_bad_run_reference_does_not_read_outside_root(tmp_path):
    (tmp_path / "active_training.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"run_id": "../outside", "stage": "P2"},
                    {"run_id": "raw", "stage": "../../outside"},
                ]
            }
        )
    )
    assert monitor.training_payload(tmp_path)["runs"] == []


def test_representation_curve_is_never_reported_as_action_quality(tmp_path):
    run = tmp_path / "P2" / "raw-representation"
    update_status(
        run,
        run_id=run.name,
        stage="P2",
        state="running",
        step=50,
        metric_kind="raw_representation",
        metric_label="RAW局部重建误差",
        representation_metrics={"crop_match_accuracy": 0.3, "representation_gate_passed": False},
    )
    append_history(
        run,
        step=0,
        wall_time=0,
        train_loss=None,
        dev_representation_error=1.2,
        dev_action_error=9.0,
    )
    append_history(run, step=50, wall_time=50, train_loss=1.0, dev_representation_error=0.8)
    (tmp_path / "active_training.json").write_text(
        json.dumps({"runs": [{"run_id": run.name, "stage": "P2"}]})
    )
    payload = monitor.training_payload(tmp_path)
    card = payload["runs"][0]
    assert card["latest_dev"] == {"step": 50, "error": 0.8}
    assert card["baseline_dev"] == 1.2
    assert card["sampled_control_mse"] is None
    assert card["representation_metrics"]["representation_gate_passed"] is False
    assert payload["closed_loop"]["completed"] == 0


def test_complete_action_sampling_curve_never_uses_velocity_loss(tmp_path):
    run = tmp_path / "P2" / "backend-current"
    update_status(run, run_id=run.name, stage="P2", metric_kind="sampled_control")
    for step, velocity, sampled in [(0, 0.2, 2.0), (25, 0.01, 1.0)]:
        append_history(
            run,
            step=step,
            wall_time=step,
            train_loss=None,
            dev_action_error=velocity,
            sampled_control_mse=sampled,
        )
    (tmp_path / "active_training.json").write_text(
        json.dumps({"runs": [{"stage": "P2", "run_id": run.name}]})
    )
    card = monitor.training_payload(tmp_path)["runs"][0]
    assert card["latest_dev"] == {"step": 25, "error": 1.0}
    assert card["baseline_dev"] == 2.0 and card["improvement_percent"] == 50.0


def test_demo_directory_rejects_traversal_and_symlink_escape(tmp_path):
    outside = tmp_path / "private"
    outside.mkdir()
    demos = tmp_path / "demos"
    demos.mkdir()
    (demos / "escape").symlink_to(outside)
    selected = tmp_path / "active_closed_loop.json"
    for reference in ("../private", str(outside), "demos/escape"):
        selected.write_text(json.dumps({"directory": reference}))
        assert monitor.demo_directory(tmp_path) is None
    (demos / "current").mkdir()
    selected.write_text(json.dumps({"directory": "demos/current"}))
    assert monitor.demo_directory(tmp_path) == (demos / "current").resolve()


def test_demo_uses_completed_episode_ledger_while_next_case_runs(tmp_path):
    demo = tmp_path / "demos" / "current"
    demo.mkdir(parents=True)
    (tmp_path / "active_closed_loop.json").write_text(json.dumps({"directory": "demos/current"}))
    (demo / "protocol.json").write_text(
        json.dumps(
            {
                "scope": "small demonstration",
                "entries": [
                    {"lighting_domain": "Normal", "lighting_ev": 0, "task_name": "grab_roller"}
                ],
            }
        )
    )
    (demo / "raw").mkdir()
    (demo / "raw/status.json").write_text(json.dumps({"state": "running", "completed": 0}))
    (demo / "raw/episodes.jsonl").write_text(
        json.dumps(
            {
                "case_id": "grab-roller-s0-Normal",
                "success": True,
                "num_steps": 100,
                "video": "some.mp4",
            }
        )
        + '\n{"partial":'
    )
    result = monitor.demo_payload(tmp_path)
    assert result["arms"]["raw"]["completed"] == 1
    assert result["arms"]["raw"]["success"] == 1
    assert result["arms"]["rgb"]["completed"] == 0
    assert result["cases"][0]["arms"]["raw"]["video"].startswith("/api/demo-video?")


def test_policy_audit_video_is_whitelisted_and_rejects_symlink_escape(tmp_path):
    audit = tmp_path / "policy-audit-20261016"
    case = audit / "original-normal"
    case.mkdir(parents=True)
    video = case / "episode0.mp4"
    video.write_bytes(b"video")
    assert monitor.policy_audit_video(tmp_path, "original") == video.resolve()
    assert monitor.policy_audit_video(tmp_path, "../../private") is None
    private = tmp_path / "private.mp4"
    private.write_bytes(b"private")
    video.unlink()
    video.symlink_to(private)
    assert monitor.policy_audit_video(tmp_path, "original") is None


def test_physics_video_rejects_traversal_and_symlink_escape(tmp_path):
    case = tmp_path / "physics-audit-20261016" / "raw-time-continued"
    case.mkdir(parents=True)
    video = case / "episode0.mp4"
    video.write_bytes(b"video")
    assert monitor.physics_video(tmp_path, "raw-time-continued") == video.resolve()
    assert monitor.physics_video(tmp_path, "../../private") is None
    private = tmp_path / "private.mp4"
    private.write_bytes(b"private")
    video.unlink()
    video.symlink_to(private)
    assert monitor.physics_video(tmp_path, "raw-time-continued") is None


def test_repair_monitor_counts_real_completed_results_and_limits_video_paths(tmp_path):
    case = tmp_path / "repair-r7-20261016" / "fresh-step0050"
    case.mkdir(parents=True)
    video = case / "episode0.mp4"
    video.write_bytes(b"video")
    assert monitor.repair_cases(tmp_path) == []
    (case / "result.json").write_text(
        json.dumps(
            {
                "kind": "fresh",
                "step": 50,
                "returncode": 0,
                "results": [{"success": False, "num_steps": 400}],
            }
        )
    )
    result = monitor.repair_cases(tmp_path)
    assert len(result) == 1 and result[0]["success"] is False
    assert monitor.repair_video(tmp_path, "fresh-step0050") == video.resolve()
    assert monitor.repair_video(tmp_path, "../private") is None
    private = tmp_path / "private.mp4"
    private.write_bytes(b"private")
    video.unlink()
    video.symlink_to(private)
    assert monitor.repair_video(tmp_path, "fresh-step0050") is None


def test_simulator_success_flag_does_not_count_as_verified_grasp(tmp_path):
    case = tmp_path / "repair-r7-20261016" / "warm-step0100"
    case.mkdir(parents=True)
    (case / "result.json").write_text(
        json.dumps(
            {
                "kind": "warm",
                "step": 100,
                "returncode": 0,
                "results": [{"success": True, "num_steps": 205}],
            }
        )
    )
    result = monitor.repair_cases(tmp_path)[0]
    assert result["success"] is True
    assert result["accepted_success"] is None
    (case / "physical_assessment.json").write_text(
        json.dumps({"accepted_success": False, "reason": "transient contact, no sustained lift"})
    )
    result = monitor.repair_cases(tmp_path)[0]
    assert result["success"] is True and result["accepted_success"] is False


def test_representation_video_and_physical_receipt_reject_fake_success_and_escape(tmp_path):
    case = tmp_path / "representation-r8-rollouts" / "fusion-rms0100-a1"
    case.mkdir(parents=True)
    (case / "protocol.json").write_text(json.dumps({"replan_interval": 50}))
    video = case / "episode0.mp4"
    video.write_bytes(b"video")
    receipt = case / "result.json"
    record = {"returncode": 0, "accepted_success": True, "results": [{"success": True}]}
    receipt.write_text(json.dumps(record))
    assert monitor.representation_cases(tmp_path)[0]["accepted_success"] is None
    record["physical"] = {
        "action_source": "live_raw_policy",
        "diagnostic_only": False,
        "physical_accepted": False,
        "max_hold_seconds": 0,
    }
    receipt.write_text(json.dumps(record))
    assert monitor.representation_cases(tmp_path)[0]["accepted_success"] is False
    assert monitor.representation_video(tmp_path, case.name) == video.resolve()
    assert monitor.representation_video(tmp_path, "../private") is None
    private = tmp_path / "private.mp4"
    private.write_bytes(b"private")
    video.unlink()
    video.symlink_to(private)
    assert monitor.representation_video(tmp_path, case.name) is None


def test_residual_payload_is_separate_sanitized_and_rejects_symlink_escape(tmp_path):
    root = tmp_path / "outputs"
    directory = root / "residual-r16"
    (directory / "raw-s15019").mkdir(parents=True)
    (directory / "raw-s15019/status.json").write_text(
        json.dumps(dict(state="running", step=16, updated_at=0, private_token="do-not-export"))
    )
    (directory / "progress.json").write_text(
        json.dumps(
            dict(
                audited=True,
                reports=[
                    dict(
                        run_id="raw-s15019",
                        modality="raw",
                        seed=15019,
                        private_token="secret",
                        baseline={"test": {"joint_rms_deg": 11, "private_token": "secret"}},
                        rows=[],
                    ),
                    dict(run_id="raw-5-s15019", rows=[]),
                ],
            )
        )
    )
    value = monitor.residual_payload(root)
    assert value["planned_runs"] == 6 and value["completed_runs"] == 1 and value["audited"]
    assert value["reports"][0]["baseline"]["test"] == {"joint_rms_deg": 11}
    assert "secret" not in json.dumps(value) and "private_token" not in json.dumps(value)
    assert next(r for r in value["runs"] if r["run_id"] == "raw-s15019")["stale"]
    outside = tmp_path / "outside.json"
    outside.write_text('{"reports": [{"run_id": "native-s15019", "rows": []}]}')
    (directory / "progress.json").unlink()
    (directory / "progress.json").symlink_to(outside)
    assert monitor.residual_payload(root)["reports"] == []


def test_joint_monitor_keeps_collection_denominator_and_separates_previous_experiment(tmp_path):
    root = tmp_path / "outputs"
    directory = root / "residual-joint-r17"
    directory.mkdir(parents=True)
    (directory / "capture_progress.json").write_text(
        json.dumps(dict(planned=20, attempted=5, accepted=4, secret="hidden"))
    )
    (directory / "progress.json").write_text(
        json.dumps(
            dict(
                reports=[
                    dict(run_id="raw-s17019", modality="raw", seed=17019, rows=[]),
                    dict(run_id="raw-s15019", rows=[]),
                ]
            )
        )
    )
    result = monitor.joint_payload(root)
    assert result["planned_runs"] == 10 and result["completed_runs"] == 1
    assert result["collection"] == dict(planned=20, attempted=5, accepted=4)
    assert "hidden" not in json.dumps(result)
    outside = tmp_path / "secret.json"
    outside.write_text('{"reports":[{"run_id":"raw-s17020","rows":[]}]}')
    (directory / "progress.json").unlink()
    (directory / "progress.json").symlink_to(outside)
    assert monitor.joint_payload(root)["completed_runs"] == 0
