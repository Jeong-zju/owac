"""Validation isolation and matched-capacity/sample control for r15 learning curves."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owac.evaluation.n02_full.action_learning import (  # noqa: E402
    fit_action_head,
    full_targets,
    sample_order,
    selected_splits,
)


def fixture():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 2, 8)).astype(np.float32)
    state = rng.normal(size=(24, 14)).astype(np.float32)
    actions = np.broadcast_to(state[:, None], (24, 50, 14)).copy()
    mask = np.ones((24, 50), dtype=bool)
    rows = [
        dict(group=str(i // 4), split="train" if i < 16 else "val" if i < 20 else "test")
        for i in range(24)
    ]
    return x, state, actions, mask, rows


def test_sample_order_is_paired_and_only_uses_declared_training_groups():
    *_, rows = fixture()
    ix = selected_splits(rows, ["0", "1"])
    assert ix["train"] == list(range(8))
    order = sample_order(ix["train"], 16, 15019)
    assert order == sample_order(ix["train"], 16, 15019)
    assert all(set(batch) <= set(range(8)) for batch in order)
    with pytest.raises(ValueError, match="subset"):
        selected_splits(rows, ["5"])


def test_test_targets_cannot_change_head_or_validation_selection():
    torch.set_num_threads(1)
    x, state, y, mask, rows = fixture()
    kwargs = dict(
        train_groups=["0", "1", "2", "3"],
        std=np.ones(14),
        seed=1,
        budgets=(4, 8),
        validate_every=4,
        device="cpu",
    )
    a, pa = fit_action_head(x, state, y, mask, rows, **kwargs)
    changed = y.copy()
    changed[20:] += 10000
    b, pb = fit_action_head(x, state, changed, mask, rows, **kwargs)
    assert [r["selected_sha"] for r in a["rows"]] == [r["selected_sha"] for r in b["rows"]]
    assert [r["selected_step"] for r in a["rows"]] == [r["selected_step"] for r in b["rows"]]
    for k in pa:
        np.testing.assert_array_equal(pa[k], pb[k])


def test_state_control_has_identical_initial_head_and_is_independent_of_visual_input():
    x, state, y, mask, rows = fixture()
    kwargs = dict(
        train_groups=["0", "1", "2", "3"],
        std=np.ones(14),
        seed=2,
        budgets=(4,),
        validate_every=4,
        device="cpu",
        state_only=True,
    )
    a, pa = fit_action_head(x, state, y, mask, rows, **kwargs)
    b, pb = fit_action_head(x * 1000, state, y, mask, rows, **kwargs)
    c, _ = fit_action_head(x, state, y, mask, rows, **{**kwargs, "state_only": False})
    assert a["initial_sha"] == b["initial_sha"] == c["initial_sha"]
    assert a["parameters"] == b["parameters"] == c["parameters"]
    for k in pa:
        np.testing.assert_array_equal(pa[k], pb[k])


def test_feature_producer_is_frozen_and_future_padding_is_native32():
    x, state, y, mask, rows = fixture()
    x = torch.tensor(x, requires_grad=True)
    fit_action_head(
        x,
        state,
        y,
        mask,
        rows,
        ["0", "1", "2", "3"],
        np.ones(14),
        seed=3,
        budgets=(2,),
        validate_every=2,
        device="cpu",
    )
    assert x.grad is None
    target = full_targets(torch.tensor(y))
    assert target.shape == (24, 50, 32)
    assert not target[..., 14:].any()
