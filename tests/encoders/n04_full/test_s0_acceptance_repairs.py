"""Independent adversarial S0 acceptance cases, separate from producer fixtures."""

from dataclasses import replace

import numpy as np
import pytest

from owac.raw.telemetry import ExecutedControl, FrameTiming, TelemetrySidecar, interval_report


def control(**changes):
    args = dict(
        command_id="c",
        value=np.ones(2),
        status="executed",
        evidence_level="executed_log",
        command_time=0.1,
        start_time=0.1,
        end_time=0.9,
        time_unit="s",
        source="executor.jsonl:1",
    )
    args.update(changes)
    return ExecutedControl(**args)


def sidecar(**changes):
    args = dict(
        episode_id="e",
        time_unit="s",
        timing=(
            FrameTiming(
                frame=0,
                clock_id="shared",
                time_unit="s",
                exposure_start=0.0,
                exposure_mid=0.01,
                exposure_end=0.02,
                arrival_time=0.04,
            ),
        ),
    )
    args.update(changes)
    return TelemetrySidecar(**args)


@pytest.mark.parametrize("level", ["unknown", "inferred", "producer_code_order"])
def test_unverified_control_never_reaches_online_replay(level):
    context = sidecar(controls=(control(evidence_level=level),)).context_for(
        exposure_time=0.01, decision_time=0.5
    )
    assert context.replay_controls == ()
    assert context.pending_control_ids == ("c",)


def test_interval_returns_only_executed_fragment():
    original = control()
    result = interval_report([original], t0=0.2, t1=0.5, time_unit="s")
    assert result.controls[0].start_time == 0.2
    assert result.controls[0].end_time == 0.5
    assert original.start_time == 0.1 and original.end_time == 0.9
    assert result.coverage == pytest.approx(1.0)


def test_mixed_time_units_cannot_enter_context():
    with pytest.raises(ValueError, match="time_unit"):
        sidecar(controls=(control(time_unit="ms"),)).context_for(
            exposure_time=0.01, decision_time=0.5
        )


def test_mixed_clocks_cannot_enter_context():
    first = sidecar().timing[0]
    with pytest.raises(ValueError, match="clock"):
        sidecar(timing=(first, replace(first, frame=1, clock_id="other"))).context_for(
            exposure_time=0.01, decision_time=0.5
        )


def test_missing_value_and_frame_only_control_are_not_replayable():
    for c in (
        control(value=None),
        control(
            command_time=None, start_time=None, end_time=None, frame=0, dispatch_phase="unknown"
        ),
    ):
        result = interval_report([c], t0=0.0, t1=1.0, time_unit="s", exposure_lookup={0: 0.1})
        assert not result.controls
        assert result.coverage == 0.0


def test_delays_have_distinct_physical_meanings():
    context = sidecar().context_for(exposure_time=0.01, decision_time=0.5)
    assert context.arrival_delay == pytest.approx(0.02)
    assert context.processing_delay == pytest.approx(0.46)
    assert context.age_since_exposure == pytest.approx(0.49)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_time_is_rejected(value):
    with pytest.raises(ValueError):
        FrameTiming(frame=0, exposure_mid=value)
    with pytest.raises(ValueError):
        control(command_time=value)


def test_state_roundtrip_is_not_measurement_correspondence(tmp_path, monkeypatch):
    from owac.raw import n04_mechanism_data as m

    states = np.arange(12, dtype=float).reshape(3, 4)
    path = tmp_path / "demo.npz"
    np.savez(path, states=states)

    class Env:
        def __init__(self):
            self.sim = self

        def set_init_state(self, state):
            self.state = state

        def get_state(self):
            return self.state

        def close(self):
            pass

    monkeypatch.setattr(m, "_libero_env_from_npz", lambda *a, **kw: (Env(), {}, path))
    verified, notes = m._probe_libero_correspondence(path, limit=2)
    assert not verified, "writing state and reading it back does not verify camera/action alignment"
    assert any("correspondence" in note for note in notes)


def test_trace_collects_continuous_steps_and_applies_cold_start(tmp_path, monkeypatch):
    from owac.raw import n04_mechanism_data as m

    path = tmp_path / "demo.npz"
    np.savez(
        path,
        states=np.array([[0.0], [100.0], [200.0], [300.0]]),
        actions=np.ones((3, 1), dtype=np.float32),
    )
    environments = []

    class Env:
        def __init__(self):
            self.sim = self
            self.state = np.zeros(1)
            self.warm = 7.0
            self.resets = 0

        def reset(self):
            self.state = np.zeros(1)

        def set_init_state(self, state):
            self.resets += 1
            self.state = state.copy()

        def get_state(self):
            return self.state.copy()

        def step(self, action):
            self.state += action[0] + self.warm
            self.warm = 5.0

        def close(self):
            pass

    def build(*args, **kwargs):
        env = Env()
        environments.append(env)
        return env, {}, path

    def policy(env, mode, **kw):
        assert mode == "zeroed_each_step"
        env.warm = 0.0

    monkeypatch.setattr(m, "_libero_env_from_npz", build)
    monkeypatch.setattr(m, "_apply_warmstart_policy", policy)
    record = m.d1_collect_libero_trace(
        path, output_dir=tmp_path, name="extension", steps=3, warmstart_mode="zeroed_each_step"
    )
    with np.load(tmp_path / record["npz"]) as data:
        np.testing.assert_array_equal(data["pre_state"].ravel(), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(data["post_state"].ravel(), [1.0, 2.0, 3.0])
    assert [env.resets for env in environments] == [1, 1]
    assert record["recomputability"]["bit_exact_recomputable"]


def test_replay_boolean_cannot_override_nonfinite_error():
    from owac.raw.n04_mechanism_data import D1_TRACE_STEP_FIELDS, d1_recomputability_report

    report = d1_recomputability_report(
        dict(
            fields_present=list(D1_TRACE_STEP_FIELDS),
            warmstart_mode="zeroed_each_step",
            controller_buffer_policy="recorded_initial_conditions",
            recipe_recorded=True,
            verification=dict(bit_exact=True, max_abs_diff=float("nan"), steps=3, tolerance=1e-12),
        )
    )
    assert not report["bit_exact_recomputable"]


def test_unverified_source_identity_cannot_split_variants():
    from owac.raw.n04_manifest import build_split

    records = [
        dict(
            source_trajectory_id=f"path:episode/variant_{i}.npz",
            identity_source="path",
            platform="libero",
            task_key="task",
            relpath=f"episode/variant_{i}.npz",
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="source identity"):
        build_split(records, seed=20261014, ratios=dict(train=0.8, dev=0.1, final=0.1), name="main")


def test_collection_plan_cannot_report_success_without_data(tmp_path, monkeypatch):
    from owac.raw import n04_mechanism_data as m

    monkeypatch.setattr(
        m, "d1_npz_field_probe", lambda path: {"state_dim": 4, "actions_per_frame": 7}
    )
    monkeypatch.setattr(m, "_probe_libero_correspondence", lambda *a, **kw: (True, ["fixture"]))
    code = m.d1_main(
        ["--platform", "libero", "--npz", "demo.npz", "--sim-probe", "--output", str(tmp_path)]
    )
    assert code != 0
    assert not list(tmp_path.glob("*.npz"))
