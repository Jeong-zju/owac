from pathlib import Path

import numpy as np
import torch

from owac.evaluation.raw_all_bench_actions import (
    _window_metrics,
    select_libero_sources,
    selected_windows,
)
from owac.raw.all_bench_distillation import BenchSource


def _source(source_id, suite, lighting, split="validation", domain="libero", frames=60):
    return BenchSource(
        source_id, domain, suite, suite, split, Path("unused.npz"), frames, lighting, ("a", "b")
    )


def test_source_selection_one_per_suite_lighting_deterministic():
    groups = [
        _source("a", "s1", "Normal"),
        _source("b", "s1", "Normal"),
        _source("c", "s1", "Low"),
        _source("d", "s2", "Normal"),
        _source("rt", "r", "Normal", domain="robotwin2"),
        _source("train", "s1", "Normal", split="train"),
    ]
    selected = select_libero_sources(groups, max_sources=20)
    assert [item.group_id for item in selected] == ["c", "a", "d"]


def test_window_selection_endpoint_startup_and_last_valid():
    source = _source("a", "s", "Normal", frames=70)
    actions = np.zeros((70, 7), dtype=np.float32)
    windows = selected_windows(source, actions=actions)
    assert 0 in windows and 25 in windows
    assert max(windows) <= 60


def test_metrics_use_all_10_rows_and_fixed_dimension_denominators():
    expert = np.zeros((10, 7), dtype=np.float32)
    prediction = np.zeros((10, 7), dtype=np.float32)
    prediction[:, 0] = 1.0
    prediction[:, 1] = 2.0
    metrics = _window_metrics(prediction, expert)
    expected = 3.0 / 7.0
    assert abs(metrics["mae_first1"] - expected) < 1e-6
    assert abs(metrics["mae_first5"] - expected) < 1e-6
    assert abs(metrics["mae_first10"] - expected) < 1e-6
    assert abs(metrics["mae_translation"] - 1.0) < 1e-6
    assert abs(metrics["mae_rotation"] - 0.0) < 1e-6
    assert abs(metrics["mae_gripper"] - 0.0) < 1e-6


def test_noise_seed_101_clone_is_identical():
    a = torch.randn((1, 10, 32), generator=torch.Generator().manual_seed(101))
    b = torch.randn((1, 10, 32), generator=torch.Generator().manual_seed(101))
    torch.testing.assert_close(a, b)


def test_deployment_state_uses_q01_q99_affine_formula():
    from types import SimpleNamespace

    from owac.evaluation.raw_all_bench_actions import deployment_state8

    stats = SimpleNamespace(state={"q01": [0.0] * 8, "q99": [1.0] * 8})
    raw = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.2, 0.8, 0.1], dtype=np.float32)
    expected = ((raw - 0.0) / (1.0 - 0.0 + 1e-6) * 2 - 1).astype(np.float32)
    np.testing.assert_allclose(deployment_state8(raw, stats), expected)


def test_aggregate_rmse_is_rms_of_window_rmse_not_mean():
    from owac.evaluation.raw_all_bench_actions import _aggregate_rows

    rows = [{"rmse_first1": 1.0, "mae_first1": 1.0}, {"rmse_first1": 3.0, "mae_first1": 3.0}]
    result = _aggregate_rows(rows)
    assert abs(result["rmse_first1"] - (5**0.5)) < 1e-6
    assert abs(result["mae_first1"] - 2.0) < 1e-6


def test_checkpoint_ddp_prefix_strip_and_shared_prepared_action_inputs(monkeypatch):
    from types import SimpleNamespace

    from owac.evaluation import raw_all_bench_actions as module

    tensor = torch.zeros(1)
    normalized = module.normalize_checkpoint_state(
        {"model": {"module.base.x": tensor, "base.y": tensor}}
    )
    assert set(normalized) == {"base.x", "base.y"}

    lang = torch.tensor([[1, 2]])
    masks = torch.tensor([[True, True]])
    state = torch.zeros(1, 32)
    monkeypatch.setattr(module, "_prepare_action_inputs", lambda *a, **k: (lang, masks, state))
    got = module.prepare_shared_inputs(None, np.zeros(8), "prompt", torch.device("cpu"))
    assert got[0] is lang and got[1] is masks and got[2] is state
    assert got[3].shape == (1, 10, 32)

    class Bridge:
        def __init__(self):
            self.calls = []

        def predict_action_from_visual_tokens(
            self, tokens, lang_tokens, lang_masks, state32, *, noise, num_steps
        ):
            self.calls.append((tokens, lang_tokens, lang_masks, state32, noise, num_steps))
            return {"normalized_actions": np.zeros((1, 10, 7), dtype=np.float32)}

    bridge = Bridge()
    stats = SimpleNamespace(action={"q01": [0.0] * 7, "q99": [1.0] * 7})
    tokens_t = SimpleNamespace(values=torch.zeros(1, 1, 7))
    tokens_s = SimpleNamespace(values=torch.zeros(1, 1, 7))
    noise = got[3]
    module._action_from_prepared(bridge, stats, tokens_t, lang, masks, state, noise)
    module._action_from_prepared(bridge, stats, tokens_s, lang, masks, state, noise.clone())
    assert bridge.calls[0][1] is bridge.calls[1][1]
    assert bridge.calls[0][3] is bridge.calls[1][3]
    assert torch.equal(bridge.calls[0][4], bridge.calls[1][4])
    assert bridge.calls[0][4] is not bridge.calls[1][4]
    assert bridge.calls[0][5] == 10
