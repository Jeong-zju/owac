"""P1 evaluation-CLI tests: metrics, thresholds, receipts and the not_run policy.

The metric layer is exercised on explicit per-scenario records; the CLI tests pin
the honest failure policy (missing modules or assets are ``not_run`` with a
reason and exit code 0, a malformed config exits non-zero) and that a short run
can never be reported as a satisfied acceptance criterion.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from owac.evaluation.n02_full import p1 as ev
from owac.frameworks.n02_full import config as n02_config

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/encoders/n02_full/p1_eval.json"


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def test_wilson_interval_brackets_the_estimate_and_narrows_with_n() -> None:
    low, high = ev.wilson_interval(950, 1000)
    assert low < 0.95 < high
    assert high - low < 0.05
    narrow_low, narrow_high = ev.wilson_interval(9500, 10000)
    assert narrow_high - narrow_low < high - low
    assert ev.wilson_interval(0, 0) == (0.0, 1.0)


def test_posterior_drift_is_the_maximum_absolute_change() -> None:
    assert ev.posterior_drift([1.0, 0.0], [0.96, 0.04]) == pytest.approx(0.04)
    assert ev.posterior_drift([0.2, 0.8], [0.2, 0.8]) == 0.0


@pytest.mark.parametrize(
    ("prior", "posterior"),
    [
        ([0.5, 0.4], [0.5, 0.5]),  # prior does not sum to 1
        ([0.5, 0.5], [0.6, 0.5]),  # posterior does not sum to 1
        ([1.0], [1.0, 0.0]),  # length mismatch
    ],
)
def test_posterior_drift_rejects_unusable_vectors(prior, posterior) -> None:
    with pytest.raises(ValueError):
        ev.posterior_drift(prior, posterior)


def test_threshold_defaults_match_the_frozen_p1_acceptance() -> None:
    thresholds = ev.BehaviorThresholds()
    assert thresholds.cue_swap_both_correct == 0.90
    assert thresholds.no_information_drift_max == 0.05
    assert thresholds.no_information_pass_rate == 0.95
    assert thresholds.wrong_prior_correction == 0.80
    assert thresholds.normal_false_positive_max == 0.05
    assert thresholds.insufficiency_detection == 0.90
    assert thresholds.min_scenarios_per_class == 1000
    assert thresholds.seeds == (1103, 2207, 3301)


def _records(per_seed: int = 1000, *, seeds=(1103,), quality: str = "good") -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        for index in range(per_seed):
            good = quality == "good"
            rows.append(
                {
                    "family": "cue_swap",
                    "scenario_id": f"s{index}",
                    "seed": seed,
                    "direction": "a",
                    "correct": good or index % 100 < 70,
                }
            )
            rows.append(
                {
                    "family": "cue_swap",
                    "scenario_id": f"s{index}",
                    "seed": seed,
                    "direction": "b",
                    "correct": good or index % 100 < 70,
                }
            )
            rows.append(
                {
                    "family": "no_information",
                    "scenario_id": f"s{index}",
                    "seed": seed,
                    "drift": 0.01 if good else 0.5,
                }
            )
            rows.append(
                {
                    "family": "wrong_prior",
                    "scenario_id": f"s{index}",
                    "seed": seed,
                    "direction": "a",
                    "corrected": good or index % 100 < 70,
                }
            )
            rows.append(
                {
                    "family": "wrong_prior",
                    "scenario_id": f"s{index}",
                    "seed": seed,
                    "direction": "b",
                    "corrected": good or index % 100 < 70,
                }
            )
            for group in ("exposure", "noise", "shift"):
                rows.append(
                    {
                        "family": "normal",
                        "scenario_id": f"s{index}",
                        "seed": seed,
                        "group": group,
                        "flagged": not good and index % 100 < 40,
                    }
                )
                rows.append(
                    {
                        "family": "insufficiency",
                        "scenario_id": f"s{index}",
                        "seed": seed,
                        "group": group,
                        "detected": good or index % 100 < 40,
                    }
                )
    return rows


def test_summarise_behavior_passes_a_clean_record_set() -> None:
    summary = ev.summarise_behavior(_records(), ev.BehaviorThresholds())
    assert summary["status"] == "pass"
    assert summary["cue_swap"]["status"] == "pass"
    assert summary["cue_swap"]["both_correct"]["rate"] == 1.0
    assert summary["no_information"]["status"] == "pass"
    assert summary["wrong_prior"]["status"] == "pass"
    assert summary["checks"]["status"] == "pass"
    assert summary["checks"]["by_group"]["exposure"]["status"] == "pass"


def test_summarise_behavior_fails_a_degraded_record_set() -> None:
    thresholds = ev.BehaviorThresholds(min_scenarios_per_class=10)
    summary = ev.summarise_behavior(_records(per_seed=100, quality="bad"), thresholds)
    assert summary["cue_swap"]["status"] == "fail"
    assert summary["no_information"]["status"] == "fail"
    assert summary["wrong_prior"]["status"] == "fail"
    assert summary["checks"]["status"] == "fail"
    assert summary["status"] == "fail"


def test_summarise_behavior_is_incomplete_below_the_scenario_floor() -> None:
    thresholds = ev.BehaviorThresholds(min_scenarios_per_class=1000)
    summary = ev.summarise_behavior(_records(per_seed=10), thresholds)
    assert summary["status"] == "incomplete"
    assert summary["cue_swap"]["status"] == "incomplete"


def test_summarise_behavior_requires_every_family() -> None:
    with pytest.raises(ev.DeficitError):
        ev.summarise_behavior(
            [{"family": "cue_swap", "scenario_id": "s", "seed": 1}], ev.BehaviorThresholds()
        )


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


def test_eval_config_loads_the_declared_arms_and_thresholds() -> None:
    cfg = n02_config.load_config(CONFIG)
    assert n02_config.validate_config(cfg) == []
    spec = ev.load_eval_spec(cfg)
    assert spec.scenarios == 1000
    assert spec.seeds == (1103, 2207, 3301)
    assert spec.thresholds.cue_swap_both_correct == 0.90
    assert spec.thresholds.min_scenarios_per_class == 1000
    names = [arm.get("name") for arm in spec.ablation_arms]
    assert names == [
        "nre",
        "bnre",
        "candidate_discrimination",
        "conditional_likelihood",
        "raw_frame",
        "conditional_innovation",
        "registration",
    ]
    arms = {arm["name"]: arm for arm in spec.ablation_arms}
    assert arms["bnre"]["hyperparameters"]["lambda"] == 100
    assert arms["nre"]["pinned_source"] == "cbf500c09c538a014e72477d166acc62a6d8409a"
    assert arms["nre"]["budget_updates"] == arms["bnre"]["budget_updates"] == 50000


@pytest.mark.parametrize("name", ["nre", "bnre"])
def test_declared_official_ratio_losses_resolve(name: str) -> None:
    """The declared NRE/BNRE loss sources must be the vendored official classes."""
    cfg = n02_config.load_config(CONFIG)
    spec = ev.load_eval_spec(cfg)
    arm = next(item for item in spec.ablation_arms if item["name"] == name)
    module_name, _, attr = str(arm["loss_source"]).partition(":")
    module = pytest.importorskip(module_name)
    assert hasattr(module, attr)
    loss = getattr(module, attr)
    assert loss.__module__.startswith("owac.baselines")


def test_eval_spec_rejects_unknown_threshold_keys(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["evaluation"]["scenarios"]["thresholds"]["too_strict"] = 0.99
    broken = tmp_path / "eval.json"
    broken.write_text(json.dumps(payload))
    cfg = n02_config.load_config(broken)
    with pytest.raises(n02_config.ConfigError) as excinfo:
        ev.load_eval_spec(cfg)
    assert "too_strict" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_behavior_reports_not_run_when_the_stack_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _absent():
        raise ev.EvalNotAvailable(
            "model.py not integrated", missing=("owac.frameworks.n02_full.model",)
        )

    monkeypatch.setattr(ev, "load_observation_stack", _absent)
    monkeypatch.setattr(ev, "load_scenario_runner", _absent)
    code = ev.main(["behavior", "--config", str(CONFIG), "--output", str(tmp_path / "out")])
    assert code == 0
    receipt = json.loads((tmp_path / "out" / "behavior.json").read_text())
    assert receipt["status"] == "not_run"
    assert "not integrated" in receipt["reason"]
    assert "verdict" not in receipt


def test_oracle_stage_is_wired_to_the_analytic_oracle() -> None:
    """The oracle stage is wired and never fabricates a rate.

    The stage runs the analytic oracle through the public query builders; the capability
    probe and the hook hash stay pinned here, while the (slow) scenario run itself is
    exercised by the acceptance run instead of by this test.
    """
    assert ev.ORACLE_HOOK_HASH.startswith("synthetic.py ")
    assert len(ev.ORACLE_REQUIRED_APIS) == 2
    assert any("render_states" in item for item in ev.ORACLE_REQUIRED_APIS)
    assert any("hypothesis" in item for item in ev.ORACLE_REQUIRED_APIS)
    source = inspect.getsource(ev.stage_oracle)
    assert "oracle_scenario_records" in source
    assert "scenario_records.jsonl" in source
    # no fabrication path: the routine either returns records or raises EvalNotAvailable
    source = inspect.getsource(ev.oracle_scenario_records)
    assert "EvalNotAvailable" in source


def test_cli_behavior_keeps_per_scenario_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "load_observation_stack", lambda: {})
    monkeypatch.setattr(ev, "load_scenario_runner", lambda: object())
    monkeypatch.setattr(
        ev, "_scenario_records", lambda runner, spec, *, output: _records(per_seed=1000)
    )
    output = tmp_path / "out"
    code = ev.main(["behavior", "--config", str(CONFIG), "--output", str(output)])
    assert code == 0
    receipt = json.loads((output / "behavior.json").read_text())
    assert receipt["status"] == "ok"
    assert receipt["verdict"] == "pass"
    lines = (output / "scenario_records.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(_records())


def test_cli_report_composes_artifacts_from_receipts(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "oracle.json").write_text(
        json.dumps({"status": "ok", "verdict": "pass", "correct": {"rate": 0.99}})
    )
    (output / "behavior.json").write_text(
        json.dumps({"status": "not_run", "reason": "stack absent", "verdict": None})
    )
    code = ev.main(["report", "--config", str(CONFIG), "--output", str(output)])
    assert code == 0
    decision = json.loads((output / "stage_decision.json").read_text())
    assert decision["numerical"] == "pass"
    assert decision["behavior"] == "not_run"
    assert decision["closed_loop"] == "not_run"
    assert (output / "metrics.json").is_file()
    assert (output / "run_manifest.json").is_file()


def test_cli_rejects_a_malformed_config(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"evaluation": {"thresholds": {"nope": 1}}}))
    assert ev.main(["report", "--config", str(broken), "--output", str(tmp_path / "out")]) == 2


def test_cli_rejects_bad_argument_values(tmp_path: Path) -> None:
    assert ev.main(["behavior", "--config", str(CONFIG), "--scenarios", "0"]) == 2
    assert ev.main(["behavior", "--config", str(CONFIG), "--seeds", "abc"]) == 2
    with pytest.raises(SystemExit):
        ev.main(["unknown-command", "--config", str(CONFIG)])


def test_oracle_hypothesis_maps_candidate_geometry_to_object_ids() -> None:
    """The oracle's hypothesis mapping must line up with the frame's object order."""
    synth = pytest.importorskip("owac.frameworks.n02_full.synthetic")
    factors = pytest.importorskip("owac.frameworks.n02_full.factors")
    sensors = (
        synth.SensorSpec(
            camera_id="cam0",
            height=16,
            width=16,
            input_mode="bayer_cfa",
            cfa_pattern="RGGB",
            black_level=64.0,
            white_level=4095.0,
            bit_depth=12,
            read_noise_e=2.0,
            gain=1.0,
            exposure_ms=1.0,
        ),
    )
    dataset = synth.generate_dataset(
        sensors=sensors,
        measurement_model=synth.MeasurementModel(kind="cfa"),
        partitions=("train",),
        scenes=2,
        frames=3,
        views_per_scene=1,
        seed=11,
        canvas=(16, 16),
        action_horizon=2,
    )
    sequence = next(iter(dataset))
    task = factors.build_p1_task(sequence, frame_index=1)
    spec = task.dictionary.specs[0]
    ids = tuple(sequence.truth[1].object_ids)
    mapping = ev.oracle_hypothesis(spec, ids)
    assert tuple(mapping) == ids
    assert all(set(entry) <= {"world_px", "scale_px", "depth"} for entry in mapping.values())
    assert all(len(entry["world_px"]) == 2 for entry in mapping.values())
    assert all(math.isfinite(entry["world_px"][0]) for entry in mapping.values())
    with pytest.raises(ev.EvalNotAvailable):
        ev.oracle_hypothesis(spec, ids[:-1])
