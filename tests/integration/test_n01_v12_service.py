"""N01 v1.2 P3 deadline readout semantics.

These checks are about correctness of the *contract*, not throughput: a legal
snapshot or an explicit unavailable answer, never a fabricated zero action, and a
stale answer always carrying its source time and being labelled stale.

The model is stubbed on purpose.  Running the real DBSR here would only test
PyTorch; what needs testing is the turn-taking rule around it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from owac.frameworks.n01_full.model import N01V12Config
from owac.runtime.n01_v12_service import (
    LATENCY_STAGES,
    DeadlinePolicy,
    LatencyLedger,
    N01V12Service,
    Snapshot,
)


class _Protocol:
    action_horizon = 10
    model_action_dim = 7
    replan_steps = 8


class _Policy:
    action_horizon = 10
    action_dim = 7


class _ActionCache:
    def reset(self, batch_mask=None):
        return None


class _Bridge:
    protocol = _Protocol()
    action_cache = _ActionCache()


class _Model:
    support = None

    def __init__(self) -> None:
        self.config = N01V12Config(arm="raw_n01", platform="libero")

    def reset(self, state, batch_mask=None):  # pragma: no cover - unused by these checks
        return state


class _ScriptedService(N01V12Service):
    """Service whose snapshot computation follows a script, one entry per call."""

    def __init__(self, script, **kwargs):
        super().__init__(_Model(), _Policy(), _Bridge(), device="cpu", **kwargs)
        self.script = list(script)
        self.calls = 0

    def _fresh_snapshot(self, data, noise=None):
        self.calls += 1
        self.last_noise = noise
        if not self.script:
            return None
        outcome = self.script.pop(0)
        if outcome is None:
            return None
        return Snapshot(
            actions=np.full((1, 10, 7), outcome, dtype=np.float32),
            kind="fresh",
            created_at=__import__("time").perf_counter(),
            source_time=float(outcome),
            unique_frames={"agentview": 6},
        )


def _payload() -> dict:
    return {"frames": {"agentview": {"values": np.zeros((1, 4, 3, 8, 8), dtype=np.float32)}}}


def test_fresh_snapshot_is_returned_and_labelled() -> None:
    service = _ScriptedService([0.5])
    response = service.infer(_payload())
    assert response["kind"] == "fresh"
    assert response["snapshot"]["kind"] == "fresh"
    assert response["normalized_actions"].shape == (1, 10, 7)
    assert float(response["normalized_actions"][0, 0, 0]) == pytest.approx(0.5)
    assert service.counters["fresh"] == 1


def test_no_legal_input_yields_unavailable_and_never_a_zero_action() -> None:
    service = _ScriptedService([None])
    response = service.infer(_payload())
    assert "unavailable" in response
    assert "normalized_actions" not in response, (
        "a fabricated zero action would look like a successful inference"
    )
    assert response["legal_snapshot"] is None
    assert service.counters["unavailable"] == 1


def test_stale_answer_is_labelled_and_carries_its_source_time() -> None:
    service = _ScriptedService([0.25, None])
    first = service.infer(_payload())
    assert first["kind"] == "fresh"
    second = service.infer(_payload())
    assert second["kind"] == "stale"
    assert second["snapshot"]["kind"] == "stale"
    assert second["snapshot"]["source_time"] == pytest.approx(0.25), (
        "a stale answer must expose which source time it is based on"
    )
    assert float(second["normalized_actions"][0, 0, 0]) == pytest.approx(0.25)
    assert service.counters == {
        "requests": 2,
        "fresh": 1,
        "stale": 1,
        "single_frame_degraded": 0,
        "deadline_missed": 0,
        "unavailable": 0,
        "reset": 0,
    }


def test_stale_can_be_refused_by_policy() -> None:
    policy = DeadlinePolicy(deadline_ms=10.0, allow_stale_snapshot=False)
    service = _ScriptedService([0.25, None], deadline=policy)
    service.infer(_payload())
    second = service.infer(_payload())
    assert "unavailable" in second
    assert "normalized_actions" not in second


def test_stale_age_is_reported_and_can_be_declared_expired() -> None:
    service = _ScriptedService([0.25, None], deadline=DeadlinePolicy(max_snapshot_age_ms=1e-6))
    service.infer(_payload())
    snapshot = service.snapshots["latest"]
    # ``age_ms >= 0.0`` cannot fail for any real input, so it checked nothing.  The
    # honest assertions are that the age reflects the recorded instant and that a
    # snapshot older than the declared bound is reported as expired.
    now = __import__("time").perf_counter()
    description = snapshot.describe(now, service.deadline)
    assert description["age_ms"] == pytest.approx(0.0, abs=1.0)

    snapshot.created_at = now - 5.0
    aged = snapshot.describe(now, service.deadline)
    assert aged["age_ms"] == pytest.approx(5000.0, rel=0.05)
    assert aged["expired_by_age"] is True, (
        "a snapshot older than max_snapshot_age_ms must be reported as expired"
    )

    generous = snapshot.describe(now, DeadlinePolicy(max_snapshot_age_ms=60000.0))
    assert generous["expired_by_age"] is False
    assert description["expired_by_age"] is True


def test_health_reports_arm_cameras_and_no_hard_realtime_claim() -> None:
    service = _ScriptedService([0.5])
    service.infer(_payload())
    health = service.health()
    assert health["arm"] == "raw_n01"
    assert health["platform"] == "libero"
    assert health["cameras"] == ["agentview", "wrist"]
    assert "hard real-time" in health["deadline"]["guarantee"]
    assert health["counters"]["requests"] == 1
    for stage in LATENCY_STAGES:
        assert stage in health["latency"]["stages"]
    assert "No composite" in health["latency"]["note"]


def test_latency_percentiles_are_ordered_and_per_stage() -> None:
    ledger = LatencyLedger()
    for value in (0.001, 0.002, 0.003, 0.004, 0.010):
        ledger.record("queue", value)
    summary = ledger.stage_summary("queue")
    assert summary["n"] == 5
    assert summary["p50_ms"] <= summary["p95_ms"] <= summary["p99_ms"] <= summary["max_ms"]
    assert summary["max_ms"] == pytest.approx(10.0, abs=0.01)
    # Unrecorded stages stay explicitly empty instead of reporting a fake zero.
    assert ledger.stage_summary("backend")["p50_ms"] is None


def test_latency_ledger_rejects_an_unknown_stage() -> None:
    with pytest.raises(KeyError):
        LatencyLedger().record("not_a_stage", 0.001)


def test_deadline_policy_validation() -> None:
    with pytest.raises(ValueError):
        DeadlinePolicy(deadline_ms=0.0)
    with pytest.raises(ValueError):
        DeadlinePolicy(max_snapshot_age_ms=-1.0)
    assert DeadlinePolicy().to_dict()["deadline_ms"] is None


def test_reset_clears_snapshots_and_counts() -> None:
    service = _ScriptedService([0.5, 0.5])
    service.infer(_payload())
    assert service.snapshots
    service.reset()
    assert service.snapshots == {}
    assert service.states == {}
    assert service.counters["reset"] == 1


# ---------------------------------------------------------------------------
# openpi payload adapter: the author LIBERO client's contract
# ---------------------------------------------------------------------------


class _StubService:
    """Minimal stand-in exposing only what the adapter is allowed to use."""

    camera_order = ("agentview", "wrist")

    def __init__(self, response):
        self.response = response
        self.queries = []
        self.noises = []
        self.resets = 0

    def infer(self, data, noise=None):
        self.queries.append(data)
        self.noises.append(noise)
        return self.response

    def reset(self, batch_mask=None):
        self.resets += 1
        return {"reset": True}

    def health(self):
        return {"arm": "raw_n01", "cameras": list(self.camera_order)}


def _ok_response(shape=(1, 10, 32)):
    return {
        "normalized_actions": np.zeros(shape, dtype=np.float32),
        "kind": "fresh",
        "snapshot": {"kind": "fresh"},
        "unique_frames": {"agentview": 3, "wrist": 3},
        "latency_ms": {"total": 12.0},
    }


def _example(burst_frames=3, size=8):
    # Real burst entries are [T, C, H, W]; the adapter must take the newest frame
    # and keep the channel axis.
    burst = [
        np.zeros((burst_frames, 3, size, size), dtype=np.float32) + camera_index
        for camera_index in (1.0, 2.0)
    ]
    return {
        "examples": [
            {
                "raw_burst_float32": burst,
                "lang": "pick up the black bowl",
                "state": np.zeros((1, 8), dtype=np.float32),
            }
        ],
        "noise": np.zeros((10, 32), dtype=np.float32),
    }


def _adapter(response=None, **overrides):
    from owac.runtime.n01_v12_service import N01V12OpenPIAdapter, OpenPIAdapterConfig

    service = _StubService(response or _ok_response())
    config = OpenPIAdapterConfig(cameras=("agentview", "wrist"), **overrides)
    return N01V12OpenPIAdapter(service, config), service


def test_adapter_maps_the_raw_burst_newest_frame_to_bchw() -> None:
    adapter, service = _adapter()
    payload = _example(burst_frames=3, size=8)
    data = adapter.predict_action(**payload)
    # The author server adds the {"ok","data"} envelope; the adapter must not add a
    # second one or the client's response["data"]["normalized_actions"] lookup fails.
    assert "ok" not in data and "data" not in data
    assert list(data["normalized_actions"].shape) == [1, 10, 32]

    query = service.queries[0]
    assert sorted(query["frames"]) == ["agentview", "wrist"]
    values = query["frames"]["agentview"]["values"]
    assert values.shape == (1, 3, 8, 8), "one arrival must be BCHW: batch, channels, H, W"
    assert float(values.mean()) == pytest.approx(1.0)
    assert float(query["frames"]["wrist"]["values"].mean()) == pytest.approx(2.0)
    assert query["frames"]["agentview"]["value_units"] == "normalized_raw_measurement"
    assert query["language"] == ["pick up the black bowl"]


def test_adapter_forwards_the_client_noise_to_the_flow() -> None:
    adapter, service = _adapter()
    payload = _example()
    adapter.predict_action(**payload)
    assert service.noises[0] is not None
    assert np.asarray(service.noises[0]).shape == (10, 32)


def test_adapter_refuses_to_guess_cameras() -> None:
    from owac.runtime.n01_v12_service import N01V12OpenPIAdapter, OpenPIAdapterConfig

    service = _StubService(_ok_response())
    with pytest.raises(ValueError, match="refusing to guess"):
        N01V12OpenPIAdapter(service, OpenPIAdapterConfig(cameras=("wrist", "agentview")))


def test_adapter_fails_loudly_when_the_raw_burst_is_absent() -> None:
    adapter, service = _adapter()
    payload = _example()
    for example in payload["examples"]:
        example.pop("raw_burst_float32")
    with pytest.raises(ValueError, match="raw_burst"):
        adapter.predict_action(**payload)
    assert service.queries == [], (
        "a missing RAW burst must not silently fall back to the resized RGB image"
    )


def test_adapter_can_read_the_image_field_and_converts_uint8_hwc() -> None:
    adapter, service = _adapter(image_source="image")
    images = [np.full((8, 8, 3), 255, dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
    payload = {"examples": [{"image": images, "lang": "task", "state": np.zeros((1, 8))}]}
    adapter.predict_action(**payload)
    values = service.queries[0]["frames"]["agentview"]["values"]
    assert values.shape == (1, 3, 8, 8), "HWC uint8 must become BCHW float"
    assert float(values.max()) == pytest.approx(1.0)


def test_adapter_reports_unavailable_as_not_ok_without_fabricating_actions() -> None:
    adapter, _ = _adapter(
        {
            "unavailable": "no_legal_frame",
            "reason": "no legal frame arrived",
            "legal_snapshot": None,
        }
    )
    with pytest.raises(RuntimeError, match="no legal snapshot"):
        adapter.predict_action(**_example())


def test_adapter_requires_exactly_one_example() -> None:
    adapter, _ = _adapter()
    with pytest.raises(ValueError, match="batch 1"):
        adapter.predict_action(examples=[{"image": [], "lang": "t", "state": np.zeros((1, 8))}] * 2)


def test_adapter_reset_clears_the_sequence_state() -> None:
    adapter, service = _adapter()
    adapter.set_episode("task_0/episode_0")
    adapter.reset()
    assert service.resets == 2
    assert adapter._frame_index == 0
    adapter.predict_action(**_example())
    assert service.queries[0]["frames"]["agentview"]["frame_ids"].tolist() == [0]


def test_newest_frame_reads_channel_last_history_stacks() -> None:
    """The author client sends [rnn, burst, H, W, C]; that must not become BCHW."""

    from owac.runtime.n01_v12_service import N01V12OpenPIAdapter

    height = width = 6
    newest = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
    older = np.full((1, 1, height, width, 3), -1.0, dtype=np.float32)
    stack = np.concatenate([older, newest[None, None]], axis=0)  # [2, 1, H, W, 3]
    assert stack.shape == (2, 1, height, width, 3)
    frame = N01V12OpenPIAdapter._newest_frame(stack)
    assert frame.shape == (height, width, 3)
    assert np.array_equal(frame, newest)

    # A channel-first [T, C, H, W] stack must land on [C, H, W].
    channel_first = np.zeros((4, 3, height, width), dtype=np.float32)
    channel_first[-1] = 7.0
    reduced = N01V12OpenPIAdapter._newest_frame(channel_first)
    assert reduced.shape == (3, height, width)
    assert float(reduced.mean()) == pytest.approx(7.0)


def test_adapter_maps_a_real_channel_last_burst_without_confusing_channels() -> None:
    """A 224-wide frame must not be read as 224 channels."""

    size = 224
    newest = np.full((1, 1, size, size, 3), 0.5, dtype=np.float32)
    adapter, service = _adapter()
    payload = {
        "examples": [
            {
                "raw_burst_float32": [newest, newest],
                "lang": "task",
                "state": np.zeros((1, 8), dtype=np.float32),
            }
        ]
    }
    adapter.predict_action(**payload)
    values = service.queries[0]["frames"]["agentview"]["values"]
    assert values.shape == (1, 3, size, size), f"got {values.shape}"


def test_snapshot_diagnostics_survive_the_reply() -> None:
    """The timeline figure needs the support/fusion state the action was built from.

    Those tensors are computed on the way to the action and otherwise discarded, so if
    they do not survive the reply a figure could only replay them - and a replay may not
    match the weights that actually produced the action.
    """

    from owac.runtime.n01_v12_service import Snapshot, _small

    snapshot = Snapshot(
        actions=np.zeros((1, 8, 7), dtype=np.float32),
        kind="fresh",
        created_at=0.0,
        source_time=1.0,
        unique_frames={"agentview": {"filled": 3}},
        diagnostics={"support": {"source_usable": [[True, False]]}},
    )
    described = snapshot.describe(0.5, _deadline())
    assert described["unique_frames"] == {"agentview": {"filled": 3}}
    assert snapshot.diagnostics["support"]["source_usable"] == [[True, False]]

    # _small must bound the payload: a full weight matrix cannot ride every reply.
    strided = _small(torch.arange(8192, dtype=torch.float32), limit=64)
    assert len(strided) == 64
    assert strided[0] == 0.0
    assert _small(None) is None
    assert _small(torch.zeros(0)) == []
    assert _small(torch.ones(4)) == [1.0, 1.0, 1.0, 1.0]


def _deadline():
    from owac.runtime.n01_v12_service import DeadlinePolicy

    return DeadlinePolicy(deadline_ms=100.0)


def test_adapter_forwards_every_response_field() -> None:
    """The adapter must not re-pack a hand-listed subset of the reply.

    An earlier version listed four keys, so anything added to the service's response
    afterwards -- the synchronized-timeline diagnostics in particular -- was dropped
    while the code that produced it looked correct on its own.
    """

    response = _ok_response()
    response["diagnostics"] = {"support": {"usable_cells": 1, "total_cells": 2}}
    response["cost"] = {"agentview": 1.5}
    response["deadline"] = {"deadline_ms": 100.0}
    adapter, _ = _adapter(response=response)
    moved = adapter.predict_action(**_example())
    assert "normalized_actions" in moved
    assert moved["diagnostics"] == {"support": {"usable_cells": 1, "total_cells": 2}}
    for field in ("snapshot", "unique_frames", "latency_ms", "kind", "cost", "deadline"):
        assert field in moved, f"{field} was dropped by the adapter"


def test_last_diagnostics_reads_the_adapter_not_the_top_level_model() -> None:
    """The support/fusion state lives on the DBSR adapter, not on the model wrapper.

    Reading it off the top-level model returned an empty block on every reply, and a
    plain getattr cannot distinguish "missing" from "not here".
    """

    class _Encoder:
        last_support = "SUPPORT"
        last_fusion = {"camera_id": "agentview", "frame_ids": [7, 6], "weights": None}

    class _Model:
        encoder = _Encoder()

    class _Service:
        model = _Model()
        _last_diagnostics = N01V12Service._last_diagnostics

    summary = _Service()._last_diagnostics()
    assert summary["support"]["usable_cells"] is None
    assert summary["support"]["change_logodds"] is None
    assert summary["fusion"]["camera_id"] == "agentview"
    assert summary["fusion"]["frame_ids"] == [7, 6]


class _SlowService(_ScriptedService):
    """Scripted service that can also spend time, so the deadline can be exercised."""

    def __init__(self, script, delays, **kwargs):
        super().__init__(script, **kwargs)
        self.delays = list(delays)

    def _fresh_snapshot(self, data, noise=None):
        delay = self.delays.pop(0) if self.delays else 0.0
        if delay:
            time.sleep(delay)
        return super()._fresh_snapshot(data, noise=noise)


def test_a_fresh_answer_within_budget_says_the_deadline_was_met() -> None:
    service = _SlowService([0.5], [0.0], deadline=DeadlinePolicy(deadline_ms=1000.0))
    response = service.infer(_payload())
    assert response["kind"] == "fresh"
    assert response["deadline_met"] is True
    assert response["deadline_elapsed_ms"] >= 0.0
    assert service.counters["deadline_missed"] == 0


def test_a_missed_deadline_prefers_a_permitted_non_expired_stale_snapshot() -> None:
    """Past the budget the service must return the timely answer, not the late one."""

    service = _SlowService([0.25, 0.75], [0.0, 0.05], deadline=DeadlinePolicy(deadline_ms=10.0))
    first = service.infer(_payload())
    assert first["kind"] == "fresh"
    assert first["deadline_met"] is True

    late = service.infer(_payload())
    assert late["kind"] == "stale", "a late fresh answer was returned while a timely one existed"
    assert late["deadline_met"] is False
    assert "took" in (late["note"] or "")
    assert float(late["normalized_actions"][0, 0, 0]) == pytest.approx(0.25)
    assert service.counters["deadline_missed"] == 1


def test_a_missed_deadline_degrades_to_one_frame_when_stale_is_disallowed() -> None:
    service = _SlowService(
        [0.25, 0.75, 0.9],
        [0.0, 0.05, 0.0],
        deadline=DeadlinePolicy(
            deadline_ms=10.0, allow_stale_snapshot=False, allow_single_frame_degradation=True
        ),
    )
    service.infer(_payload())
    degraded = service.infer(_payload())
    assert degraded["kind"] == "single_frame_degraded"
    assert degraded["deadline_met"] is False
    assert service.counters["single_frame_degraded"] == 1


def test_a_missed_deadline_is_admitted_when_no_timely_fallback_exists() -> None:
    """With every fallback disabled it must say the deadline was missed, not imply it held."""

    service = _SlowService(
        [0.75],
        [0.05],
        deadline=DeadlinePolicy(
            deadline_ms=10.0, allow_stale_snapshot=False, allow_single_frame_degradation=False
        ),
    )
    response = service.infer(_payload())
    assert response["kind"] == "fresh"
    assert response["deadline_met"] is False
    assert "no timely fallback" in (response["note"] or "")
    assert service.counters["deadline_missed"] == 1


def test_an_expired_snapshot_is_not_reused_under_its_age_bound() -> None:
    """``max_snapshot_age_ms`` used to be reported and then ignored."""

    service = _ScriptedService(
        [0.25, None],
        deadline=DeadlinePolicy(allow_stale_snapshot=True, max_snapshot_age_ms=1.0),
    )
    service.infer(_payload())
    service.snapshots["latest"].created_at -= 10.0
    response = service.infer(_payload())
    assert "unavailable" in response, "an out-of-age snapshot was served despite the bound"
    assert service.counters["stale"] == 0
