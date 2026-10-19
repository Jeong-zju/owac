"""Residual bounds, baseline identity, frozen gradients and selection isolation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owac.evaluation.n02_full.action_residual import (  # noqa: E402
    BoundedActionResidual,
    fit_action_residual,
)
from owac.frameworks.n02_full.task_chain import SmallActionHead, state_digest  # noqa: E402


def inputs():
    torch.set_num_threads(1)
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 2, 8)).astype(np.float32)
    state = rng.normal(size=(12, 14)).astype(np.float32)
    target = rng.normal(size=(12, 50, 14)).astype(np.float32)
    mask = np.ones((12, 50), dtype=bool)
    rows = [dict(group=str(i // 4), split=("train", "val", "test")[i // 4]) for i in range(12)]
    torch.manual_seed(20)
    baseline = SmallActionHead(width=8, tokens=2)
    return baseline, x, state, target, mask, rows


def fit(baseline, x, state, y, mask, rows, **kwargs):
    return fit_action_residual(
        baseline,
        x,
        state,
        y,
        mask,
        rows,
        ["0"],
        np.ones(14),
        seed=16,
        budgets=(4, 8),
        validate_every=4,
        device="cpu",
        **kwargs,
    )


def test_zero_initialization_and_bound_are_exact():
    baseline, x, state, *_ = inputs()
    head = BoundedActionResidual(width=8, tokens=2, cap=0.5)
    x, state = torch.from_numpy(x), torch.from_numpy(state)
    d = head(x, state)
    assert torch.equal(d, torch.zeros_like(d))
    with torch.no_grad():
        head.predict[-1].bias.fill_(1e6)
    assert head(x, state).abs().max() <= 0.5
    with pytest.raises(ValueError):
        BoundedActionResidual(width=8, tokens=2, cap=-1)


def test_baseline_is_frozen_and_zero_gate_is_exact_fallback():
    baseline, x, state, y, mask, rows = inputs()
    original = state_digest(baseline)
    features = torch.tensor(x, requires_grad=True)
    # Targets equal the exact baseline: no correction can beat MSE0 on validation.
    with torch.no_grad():
        y = baseline(torch.zeros_like(features), torch.from_numpy(state))[..., :14].numpy()
    report, predictions = fit(baseline, features, state, y, mask, rows)
    assert original == state_digest(baseline) == report["baseline_sha_after"]
    assert features.grad is None and all(p.grad is None for p in baseline.parameters())
    assert all(r["selected_step"] == 0 and r["selected_gate"] == 0 for r in report["rows"])
    np.testing.assert_array_equal(predictions["8_selected_test"], predictions["baseline_test"])


def test_test_targets_cannot_affect_gate_or_checkpoint():
    args = inputs()
    r1, p1 = fit(*args)
    baseline, x, state, y, mask, rows = args
    changed = y.copy()
    changed[8:] += 10000
    r2, p2 = fit(baseline, x, state, changed, mask, rows)
    assert [(r["selected_sha"], r["selected_gate"]) for r in r1["rows"]] == [
        (r["selected_sha"], r["selected_gate"]) for r in r2["rows"]
    ]
    for name in p1:
        np.testing.assert_array_equal(p1[name], p2[name])
    assert all(
        r["selected_validation_mse"] <= r1["baseline"]["val"]["normalized_control_mse"]
        for r in r1["rows"]
    )


def test_state_residual_control_is_independent_of_images_and_matches_initial_capacity():
    args = inputs()
    r1, p1 = fit(*args, state_only=True)
    baseline, x, *rest = args
    r2, p2 = fit(baseline, x * 1000, *rest, state_only=True)
    r3, _ = fit(*args)
    assert r1["parameters"] == r3["parameters"]
    assert r1["initial_sha"] == r3["initial_sha"]
    assert r1["order"] == r2["order"] == r3["order"]
    for name in p1:
        np.testing.assert_array_equal(p1[name], p2[name])


def test_joint_provider_updates_only_declared_front_and_restores_selected_features():
    from owac.evaluation.n02_full.action_learning import cpu_state

    baseline, x, state, y, mask, rows = inputs()
    front = torch.nn.Linear(8, 8)
    start = state_digest(front)
    frozen = torch.tensor(x)
    snapshots = {}
    r, p = fit(
        baseline,
        x,
        state,
        y,
        mask,
        rows,
        feature_provider=lambda indices: front(frozen[indices]),
        feature_parameters=front.parameters(),
        feature_snapshot=lambda front=front: cpu_state(front),
        feature_restore=lambda values: front.load_state_dict(values),
        checkpoint=lambda budget, saved: snapshots.update({budget: saved}),
    )
    assert state_digest(front) != start
    assert snapshots[8]["selected_features"] and snapshots[8]["last_features"]
    assert r["baseline_sha"] == r["baseline_sha_after"]
    assert all(
        row["selected_validation_mse"] <= r0["val"]["normalized_control_mse"]
        for r0 in [r["baseline"]]
        for row in r["rows"]
    )


def test_joint_validation_selection_is_independent_of_test_targets():
    import copy

    from owac.evaluation.n02_full.action_learning import cpu_state

    baseline, x, state, y, mask, rows = inputs()
    torch.manual_seed(17019)
    initial = torch.nn.Linear(8, 8)
    outputs = []
    for alter in (False, True):
        front = copy.deepcopy(initial)
        target = y.copy()
        if alter:
            target[8:] += 1e4
        r, p = fit(
            baseline,
            x,
            state,
            target,
            mask,
            rows,
            feature_provider=lambda indices, front=front: front(torch.from_numpy(x[indices])),
            feature_parameters=front.parameters(),
            feature_snapshot=lambda front=front: cpu_state(front),
            feature_restore=lambda saved, front=front: front.load_state_dict(saved),
        )
        outputs.append((r, p, state_digest(front)))
    a, b = outputs
    assert a[2] == b[2]
    assert [(r["selected_gate"], r["selected_sha"]) for r in a[0]["rows"]] == [
        (r["selected_gate"], r["selected_sha"]) for r in b[0]["rows"]
    ]
    np.testing.assert_array_equal(a[1]["8_selected_test"], b[1]["8_selected_test"])


def test_external_precision_changes_are_restored_even_after_failure():
    import pytest

    from owac.evaluation.n02_full.action_residual import matmul_precision_scope

    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with matmul_precision_scope():
            torch.set_float32_matmul_precision("highest")
        assert torch.get_float32_matmul_precision() == "high"
        with pytest.raises(RuntimeError), matmul_precision_scope("highest"):
            assert torch.get_float32_matmul_precision() == "highest"
            raise RuntimeError("external loader failed")
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(original)
