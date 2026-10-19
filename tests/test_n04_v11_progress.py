"""N04 v1.1 监控采样器与只读服务的真实行为测试。

覆盖：只读采集（progress.json / metrics.jsonl / eval_results.jsonl / artifacts / 日志）、
与 ``training.n04_v11`` 同口径的闭环计数（status 语义、按身份去重、基础设施错误单独计数）、
跨平台的 100 次合计、空状态不造数、原子写出，以及 HTTP 只读路由与路径越界防护。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from owac.runtime.n04_v11_progress import Handler, Server, sample_once


def test_pid_file_requires_matching_live_process(tmp_path):
    from owac.runtime.n04_v11_progress import _run_pids

    run = tmp_path / "stopped-run"
    run.mkdir()
    (run / "train.pid").write_text("12345\n")
    assert _run_pids(run, [])[0] == []
    assert _run_pids(run, [{"pid": 12345, "cmdline": "python unrelated.py"}])[0] == []
    assert _run_pids(
        run, [{"pid": 12345, "cmdline": f"python diagnose.py --checkpoint {run}/checkpoint.pt"}]
    ) == ([], None, [])
    assert _run_pids(run, [{"pid": 12345, "cmdline": f"python train.py --output {run}"}])[0] == [
        12345
    ]


RUN_KEY = "s3/libero-D-seed101"


def _write_run(run_dir: Path, *, with_artifacts: bool = True) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "train_loss": 1.0 / step,
                    "dev_loss": 0.5,
                    "terms": {
                        "action": 1.0 / step / 2,
                        "correspondence": 0.1,
                        "forecast": 0.2,
                        "representation": 0.99,
                    },
                }
            )
            for step in (100, 200, 200)
        )
        + "\n"
    )
    (run_dir / "eval_results.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "identity": "libero:t1",
                    "platform": "libero",
                    "status": "success",
                    "success": True,
                },
                {
                    "identity": "libero:t2",
                    "platform": "libero",
                    "status": "failure",
                    "success": False,
                },
                # 同身份重跑取最后一条（成功）
                {
                    "identity": "libero:t2",
                    "platform": "libero",
                    "status": "success",
                    "success": True,
                },
                {
                    "identity": "libero:t3",
                    "platform": "libero",
                    "status": "infrastructure_error",
                    "success": None,
                },
            )
        )
        + "\n"
    )
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "stage": "s3",
                "platform": "libero",
                "arm": "D",
                "seed": 101,
                "phase": "train",
                "updated_at": "2026-10-15T09:00:00Z",
                "step": 300,
                "budget_steps": 100000,
                "train_loss": 0.25,
                "dev_loss": None,
                "step_seconds": 1.5,
                # 单平台 run 目录：planned 就是该平台的计划数（50）
                "eval": {"completed": 4, "planned": 50, "successes": 2, "errors": 1},
                "log_tail": ["[t] from progress.json"],
            },
            ensure_ascii=False,
        )
    )
    (run_dir / "logs" / "train.log").write_text(
        "\n".join([f"[t] step {i}" for i in range(30)] + ["[t] RuntimeError: boom"]) + "\n"
    )
    if with_artifacts:
        artifacts = run_dir / "artifacts"
        artifacts.mkdir()
        (artifacts / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n0")
        (artifacts / "b.mp4").write_bytes(b"\x00" * 8)
        (artifacts / "index.json").write_text(
            json.dumps([{"file": "a.png", "caption": "注意力叠加", "step": 200}])
        )


def test_sample_once_aggregates_readonly_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "v1.1" / "s3" / "libero-D-seed101"
    _write_run(run_dir)
    out_dir = tmp_path / "state"
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))

    payload = sample_once(tmp_path / "runs" / "v1.1", out_dir)

    assert payload["schema"] == "n04-v11-monitor/1"
    assert payload["run_root"].endswith("runs/v1.1")
    assert [run["run_id"] for run in payload["runs"]] == [RUN_KEY]
    run = payload["runs"][0]
    # 元信息与进度
    assert (run["stage"], run["platform"], run["arm"], run["seed"], run["phase"]) == (
        "s3",
        "libero",
        "D",
        101,
        "train",
    )
    assert run["step"] == 300 and run["budget_steps"] == 100000
    assert run["train_step"] == 300 and run["train_budget_steps"] == 100000
    assert run["remaining_steps"] == 99700
    assert run["train_loss"] == 0.25
    assert run["dev_loss"] == 0.5  # metrics.jsonl 提供
    assert run["updated_at"] == "2026-10-15T09:00:00Z" and run["data_age_s"] is not None
    # 曲线：训练合计 / 动作项 / 开发 / 辅助，重复 step 保留
    assert [point[0] for point in run["series"]["train"]] == [100, 200, 200, 300]
    assert run["series"]["train"][-1][1] == 0.25  # 心跳点补在曲线末尾，与卡片数字一致
    assert set(run["series"]) == {"train", "action", "dev", "corr", "forecast", "repr"}
    assert run["series_keys"]["action"] == "terms.action"
    assert run["series_keys"]["corr"] == "terms.correspondence"
    assert run["series_keys"]["repr"] == "terms.representation"
    assert run["series"]["forecast"][-1] == [200, 0.2]
    assert run["series"]["repr"][-1] == [200, 0.99]
    assert run["series"]["action"][-1][1] == 0.0025  # 动作项来自嵌套的 terms.action
    # 评估：训练侧 progress.json 的口径优先（完成 4、成功 2、基础设施错误 1）
    assert run["eval"]["completed"] == 4
    assert run["eval"]["successes"] == 2
    assert run["eval"]["errors"] == 1
    assert run["eval"]["planned"] == 50  # 单平台 run 目录的计划数
    # 平台分项来自逐条结果（身份去重后 libero 完成 2），与训练侧合计不一致时给出提示
    assert run["eval"]["libero"]["completed"] == 2
    assert run["eval"]["counts_note"] and "不一致" in run["eval"]["counts_note"]
    # 同一模型跨平台合计：LIBERO 有 run，RoboTwin 尚无
    model = run["model_eval"]
    assert model["completed"] == 2 and model["planned"] == 100
    assert model["libero"]["completed"] == 2 and model["libero"]["source_run"] == RUN_KEY
    assert model["robotwin"]["completed"] is None
    assert model["planned_source"] == "protocol-default"
    # 可视化与日志
    assert sorted(art["name"] for art in run["artifacts"]) == ["a.png", "b.mp4"]
    assert next(a for a in run["artifacts"] if a["name"] == "a.png")["caption"] == "注意力叠加"
    assert len(run["log"]["tail"]) == 20
    assert "RuntimeError" in (run["log"]["last_error"] or "")
    # 原子写出且在 run 目录之外
    saved = json.loads((out_dir / "progress.json").read_text())
    assert saved["runs"][0]["run_id"] == RUN_KEY
    assert (out_dir / "state.json").exists()
    # 只读保证：采样后 run 目录内容未变，也没有留下临时文件
    after = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    assert before == after
    assert not list(run_dir.glob(".progress.json.tmp"))


def test_eval_results_fallback_deduplicates_by_episode_identity(tmp_path: Path) -> None:
    """训练侧未写 eval 汇总时，从 eval_results.jsonl 统计并按身份去重。"""
    run_dir = tmp_path / "runs" / "v1.1" / "s3" / "libero-D-seed101"
    _write_run(run_dir)
    summary = json.loads((run_dir / "progress.json").read_text())
    summary.pop("eval")
    (run_dir / "progress.json").write_text(json.dumps(summary, ensure_ascii=False))

    payload = sample_once(tmp_path / "runs" / "v1.1", tmp_path / "state2")
    run = payload["runs"][0]
    # t2 重跑一次取最后一条（成功），t3 是基础设施错误不进完成数 → 完成 2、成功 2、错误 1
    assert run["eval"]["completed"] == 2
    assert run["eval"]["successes"] == 2
    assert run["eval"]["errors"] == 1
    # 无列表文件时按协议默认值（LIBERO 50 / RoboTwin 50），并标注来源
    assert run["eval"]["libero"]["planned"] == 50
    assert run["eval"]["planned_source"] == "protocol-default"
    assert run["eval"]["counts_note"] is None  # 无训练侧合计时不提示口径差异
    # episode 2 的重跑把失败覆盖成成功，且基础设施错误不计入失败清单
    assert run["recent_failures"] == []


def test_eval_phase_reports_eval_step_separately(tmp_path: Path) -> None:
    """phase=eval 时 progress.json 的 step/budget 是评估进度，训练 step 取曲线最后一点。"""
    run_dir = tmp_path / "runs" / "v1.1" / "s4" / "libero-D-seed101"
    _write_run(run_dir)
    data = json.loads((run_dir / "progress.json").read_text())
    data["phase"] = "eval"
    data["step"] = 12  # 评估完成 12 条
    data["budget_steps"] = 50  # 评估计划
    data["eval"] = {"completed": 12, "planned": 50, "successes": 4, "errors": 0}
    (run_dir / "progress.json").write_text(json.dumps(data, ensure_ascii=False))

    payload = sample_once(tmp_path / "runs" / "v1.1", tmp_path / "state3")
    run = payload["runs"][0]
    assert run["phase_step"] == 12 and run["budget_steps"] == 50
    assert run["train_step"] == 200  # 无历史时取曲线最后一点（metrics.jsonl）
    assert run["train_budget_steps"] is None
    assert run["remaining_steps"] is None  # 不拿评估预算算训练剩余
    assert run["eval"]["completed"] == 12  # 评估进度仍如实显示


def test_eval_phase_keeps_last_seen_train_step(tmp_path: Path) -> None:
    """训练阶段采样过一次 300 步后进入评估：训练 step 沿用 300，不退回评估计数。"""
    run_dir = tmp_path / "runs" / "v1.1" / "s4" / "libero-D-seed101"
    _write_run(run_dir)
    out_dir = tmp_path / "state4"
    first = sample_once(tmp_path / "runs" / "v1.1", out_dir)
    assert first["runs"][0]["train_step"] == 300
    assert first["train_steps"] == {"s4/libero-D-seed101": 300}

    data = json.loads((run_dir / "progress.json").read_text())
    data.update({"phase": "eval", "step": 12, "budget_steps": 50})
    (run_dir / "progress.json").write_text(json.dumps(data, ensure_ascii=False))

    second = sample_once(tmp_path / "runs" / "v1.1", out_dir, first)["runs"][0]
    assert second["phase_step"] == 12 and second["train_step"] == 300
    assert second["train_budget_steps"] == 100000  # 沿用上次训练预算记忆
    assert second["remaining_steps"] == 99700

    # 评估进度写在 budget_steps=50 上，不被当成训练预算
    assert second["budget_steps"] == 50


def test_missing_run_root_reports_empty_state_without_data(tmp_path: Path) -> None:
    payload = sample_once(tmp_path / "runs" / "v1.1", tmp_path / "state")
    assert payload["runs"] == []
    assert any("不存在" in warning for warning in payload["warnings"])
    assert isinstance(payload["gpu"], list)


def test_run_without_progress_json_is_still_sampled(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "v1.1" / "s2" / "robotwin-C-seed101"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 5, "train_loss": 9.0}) + "\n")
    payload = sample_once(tmp_path / "runs" / "v1.1", tmp_path / "state")
    run = payload["runs"][0]
    assert (run["stage"], run["platform"], run["arm"]) == ("s2", "robotwin", "C")
    assert run["step"] == 5 and run["train_loss"] == 9.0
    # 没有训练侧上报的预算时不能编造 0
    assert run["budget_steps"] is None and run["remaining_steps"] is None
    assert run["eval"]["completed"] == 0 and run["eval"]["planned"] is None
    # 跨平台合计的缺失平台用空值，不用 0 冒充
    assert run["model_eval"]["libero"]["completed"] is None
    assert run["model_eval"]["planned"] == 100  # 协议默认 50 + 50，来源已标注


@pytest.fixture()
def monitor_server(tmp_path: Path):
    run_dir = tmp_path / "runs" / "v1.1" / "s3" / "libero-D-seed101"
    _write_run(run_dir)
    out_dir = tmp_path / "state"
    payload = sample_once(tmp_path / "runs" / "v1.1", out_dir)
    server = Server(
        ("127.0.0.1", 0),
        Handler,
        run_root=tmp_path / "runs" / "v1.1",
        out_dir=out_dir,
        html_path=Path(__file__).resolve().parents[1]
        / "src"
        / "owac"
        / "runtime"
        / "n04_v11_progress.html",
        snapshot=type("S", (), {"get": lambda self: (payload, None)})(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", run_dir
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - 本机测试地址
        return response.status, dict(response.headers), response.read()


def test_http_readonly_routes_and_path_guard(monitor_server) -> None:
    base, run_dir = monitor_server
    status, headers, body = _get(f"{base}/api/progress")
    assert status == 200 and json.loads(body)["runs"][0]["run_id"] == RUN_KEY
    etag = headers["ETag"]
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base}/api/progress", {"If-None-Match": etag})
    assert excinfo.value.code == 304

    status, headers, body = _get(f"{base}/file/s3/libero-D-seed101/artifacts/a.png")
    assert status == 200 and headers["Content-Type"] == "image/png" and body.startswith(b"\x89PNG")
    status, headers, body = _get(
        f"{base}/file/s3/libero-D-seed101/artifacts/b.mp4", {"Range": "bytes=0-3"}
    )
    assert status == 206 and headers["Content-Range"] == "bytes 0-3/8" and body == b"\x00" * 4

    # 越界与非法路径一律拒绝，且不泄露 run 目录外的文件
    for path in (
        "/file/s3/libero-D-seed101/../progress.json",
        "/file/s3/libero-D-seed101/artifactz/a.png",
        "/file/s3/libero-D-seed101/logs/train.log",
        "/file/../etc/passwd",
    ):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + path)
        assert err.value.code in {403, 404}
    assert (run_dir / "artifacts" / "a.png").exists()


def test_healthz_reports_snapshot(monitor_server) -> None:
    base, _ = monitor_server
    status, _, body = _get(f"{base}/healthz")
    assert status == 200
    assert json.loads(body)["ok"] is True


def test_server_serves_page_without_external_dependencies(monitor_server) -> None:
    base, _ = monitor_server
    status, headers, body = _get(f"{base}/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    html = body.decode()
    assert "N04 v1.1" in html
    for external in ("http://", "https://", "cdn."):
        assert external not in html.replace("http://www.w3.org", "")  # 无外部资源引用


def test_codex_cached_dev_values_are_not_new_measurements(tmp_path):
    root = tmp_path / "runs"
    run = root / "libero-D-seed101"
    _write_run(run, with_artifacts=False)
    (run / "config.used.json").write_text(
        json.dumps(
            {
                "owner": "Codex",
                "monitor": {"early_until": 100, "eval_every_early": 5, "eval_every": 25},
            }
        )
    )
    (run / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": s,
                    "phase": "validate" if s == 0 else "train",
                    "dev_loss": 0.5 if s < 5 else 0.4,
                    "train_loss": 1.0,
                }
            )
            for s in range(7)
        )
        + "\n"
    )
    payload = sample_once(root, tmp_path / "out")
    selected = payload["runs"][0]
    assert selected["series"]["dev"] == [[0, 0.5], [5, 0.4]]
    assert selected["activity"]["last_validation_step"] == 5


def test_codex_checkpoint_counter_restore_is_not_training_speed():
    from owac.runtime.n04_v11_progress import _speed_info

    run = {
        "run_id": "libero-D-seed101",
        "owner": "Codex",
        "step": 10,
        "train_step": 10,
        "step_seconds_reported": None,
    }
    history = {"history": {run["run_id"]: [{"step": 0, "t": 100}]}}
    assert _speed_info(run, history, 110) == (None, None)
    run["step_seconds_reported"] = 140.0
    assert _speed_info(run, history, 110) == (140.0, "reported")


@pytest.mark.parametrize("historical_best,expected_step", [(0.295, 35), (0.2, None)])
def test_assessment_keeps_best_score_and_qualifies_convergence(
    tmp_path, historical_best, expected_step
):
    root = tmp_path / "runs"
    run = root / "libero-D-seed101"
    _write_run(run, with_artifacts=False)
    progress = json.loads((run / "progress.json").read_text())
    progress.update(step=45, dev_loss=0.385, activity={"best_dev_loss": historical_best})
    (run / "progress.json").write_text(json.dumps(progress))
    (run / "config.used.json").write_text(
        json.dumps(
            {
                "owner": "Codex",
                "optim": {"warmup_steps": 200},
                "performance": {"compile_prediction": True, "compile_banded": False},
                "data": {"dev_windows": 2},
                "monitor": {"early_until": 100, "eval_every_early": 5},
            }
        )
    )
    (run / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "phase": "train",
                    "dev_loss": value,
                    "train_loss": 0.7,
                }
            )
            for step, value in [(35, 0.295), (40, 0.383), (41, 0.383), (45, 0.385)]
        )
    )
    assessment = sample_once(root, tmp_path / "out")["runs"][0]["assessment"]
    assert assessment == {
        "warmup_steps": 200,
        "warmup_complete": False,
        "dev_windows": 2,
        "best_dev_loss": historical_best,
        "best_dev_step": expected_step,
        "convergence": "not_assessed",
        "representation_mode": None,
        "representation_coefficients": {},
        "loss_weights": {},
        "compiled_modules": ["prediction"],
    }


def test_repair_trial_progress_stays_out_of_formal_counts(tmp_path):
    root = tmp_path / "runs" / "v1.1" / "codex-s2"
    _write_run(root / "libero-D-seed101")
    trial = root.parent / "codex-diagnosis" / "libero-pilot"
    trial.mkdir(parents=True)
    (trial / "progress.json").write_text(
        json.dumps({"step": 5, "budget_steps": 12, "phase": "train"})
    )
    state = tmp_path / "monitor" / "state"
    state.parent.mkdir()
    (state.parent / "repair-runs.json").write_text(
        json.dumps(
            {
                "active": True,
                "runs": [
                    {"platform": "libero", "run_dir": str(trial)},
                    {"platform": "private", "run_dir": str(tmp_path / "outside")},
                ],
            }
        )
    )
    payload = sample_once(root, state)
    assert len(payload["runs"]) == 1 and payload["runs"][0]["step"] == 300
    assert payload["repair"]["formal_progress"] is False
    assert len(payload["repair"]["runs"]) == 1
    assert payload["repair"]["runs"][0]["step"] == 5
