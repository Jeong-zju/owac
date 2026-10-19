"""Causal online inputs and episode isolation of the real N02 policy adapter."""

import numpy as np
import pytest

from owac.frameworks.n02_full.sequence_v12 import window_inputs
from owac.runtime.n02_v12_robotwin import ModelClient, online_window, reset_model


def example(step):
    return {
        "step": step,
        "case_id": "scene-a",
        "lang": "grab roller",
        "normalized_state": np.arange(14, dtype=np.float32),
        "history": [
            {
                "step": index,
                "image": [np.full((240, 320, 3), index / 100, dtype=np.float32) for _ in range(3)],
            }
            for index in range(max(0, step - 7), step + 1)
        ],
    }


def test_online_prefix_padding_is_not_counted_as_new_measurement():
    batch = online_window(example(0), ("head", "left", "right"))
    assert batch.frame_index.tolist() == [0] * 8
    assert len(window_inputs(batch)) == 1
    batch = online_window(example(50), ("head", "left", "right"))
    assert batch.frame_index.tolist() == list(range(43, 51))
    assert len(window_inputs(batch)) == 8
    assert tuple(batch.observation.frames["head"].values.shape) == (8, 3, 240, 320)
    assert float(batch.observation.frames["head"].values[-1].mean()) == pytest.approx(0.5)
    assert batch.observation.context.executed_action_available is False


def test_online_input_rejects_future_rows_and_integer_display_images():
    data = example(2)
    data["history"][-1]["step"] = 3
    with pytest.raises(ValueError, match="decision step"):
        online_window(data, ("head", "left", "right"))
    data = example(0)
    data["history"][0]["image"][0] = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="float32 RAW3"):
        online_window(data, ("head", "left", "right"))


def test_client_collects_every_control_frame_and_resets_between_episodes(monkeypatch):
    class RPC:
        def __init__(self):
            self.requests = []

        def predict_action(self, request):
            self.requests.append(request)
            return {"ok": True, "data": {"actions": np.zeros((50, 14))}}

    monkeypatch.setenv("RAWVLA_CASE_ID", "scene-a")
    monkeypatch.setenv("RAWVLA_FLOW_SEED", "1103")
    rpc = RPC()
    client = ModelClient(rpc)
    for step in range(51):
        client.step({"image": [step] * 3}, step)
    assert len(rpc.requests) == 2
    assert [r["step"] for r in rpc.requests[-1]["examples"][0]["history"]] == list(range(43, 51))
    reset_model(client)
    monkeypatch.setenv("RAWVLA_CASE_ID", "scene-b")
    client.step({"image": [0] * 3}, 0)
    assert rpc.requests[-1]["examples"][0]["case_id"] == "scene-b"
    assert len(rpc.requests[-1]["examples"][0]["history"]) == 1


def test_fixed_flow_noise_changes_sampling_without_reusing_actions(monkeypatch):
    monkeypatch.setenv("RAWVLA_CASE_ID", "fixed-flow")
    monkeypatch.setenv("RAWVLA_FLOW_SEED", "1103")
    monkeypatch.setenv("N02_FLOW_NOISE_MODE", "fixed")

    class RPC:
        def __init__(self):
            self.seeds = []

        def predict_action(self, request):
            self.seeds.append(request["examples"][0]["flow_seed"])
            return {"ok": True, "data": {"actions": np.full((50, 14), len(self.seeds))}}

    rpc = RPC()
    client = ModelClient(rpc)
    first = client.step({"image": [0] * 3}, 0)
    second = client.step({"image": [50] * 3}, 50)
    assert rpc.seeds == [1103, 1103]
    assert not np.array_equal(first, second)
    assert len(client.history) == 2


def test_client_history_owns_frames_when_a_camera_reuses_its_buffer(monkeypatch):
    monkeypatch.setenv("RAWVLA_CASE_ID", "buffer-reuse")
    monkeypatch.setenv("RAWVLA_FLOW_SEED", "1103")

    class RPC:
        def predict_action(self, request):
            self.history = request["examples"][0]["history"]
            return {"ok": True, "data": {"actions": np.zeros((50, 14))}}

    rpc = RPC()
    client = ModelClient(rpc)
    buffer = np.zeros((2, 3, 3), dtype=np.float32)
    client.step({"image": [buffer] * 3}, 0)
    buffer.fill(1)
    client.step({"image": [buffer] * 3}, 50)
    assert np.array_equal(rpc.history[0]["image"][0], np.zeros_like(buffer))
    assert np.array_equal(rpc.history[1]["image"][0], np.ones_like(buffer))
    buffer.fill(2)
    assert np.array_equal(client.history[-1]["image"][0], np.ones_like(buffer))


def test_bounded_execution_preserves_history_and_limits_chunk_switch(monkeypatch):
    monkeypatch.setenv("RAWVLA_CASE_ID", "bounded-test")
    monkeypatch.setenv("RAWVLA_FLOW_SEED", "1103")

    class RPC:
        def __init__(self):
            self.calls = 0

        def predict_action(self, request):
            self.calls += 1
            return {
                "ok": True,
                "data": {"actions": np.full((50, 14), 3.0 if self.calls % 2 else -3.0)},
            }

    rpc = RPC()
    client = ModelClient(rpc, replan_interval=10, max_joint_delta=0.05)
    previous = np.zeros(14, dtype=np.float32)
    for step in range(21):
        action = client.step({"image": [step] * 3, "state": previous.copy()}, step)
        assert np.abs(action - previous).max() <= 0.050001
        assert ((action[[6, 13]] >= 0) & (action[[6, 13]] <= 1)).all()
        previous = action
    assert rpc.calls == 3
    reset_model(client)
    assert client.last_action is None and not client.history
