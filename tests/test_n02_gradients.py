"""Numerical and checkpoint-isolation tests for the read-only gradient audit."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owac.evaluation.n02_gradients import (  # noqa: E402
    batch_digest,
    clone_with_optimizer,
    decompose_gradients,
    disposable_update,
    training_objective,
    tree_digest,
)
from owac.frameworks.n02.config import N02Config  # noqa: E402
from owac.frameworks.n02.model import build_encoder  # noqa: E402
from owac.training.n02 import synthetic_batch  # noqa: E402


def test_gradient_decomposition_distinguishes_zero_disconnection_and_conflict():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    zero = torch.nn.Parameter(torch.tensor([3.0]))
    absent = torch.nn.Parameter(torch.tensor([4.0]))
    terms = {"one": p[0] + zero.sum() * 0, "two": -2 * p[0] + 3 * p[1]}
    report, gradients, total = decompose_gradients(
        terms,
        sum(terms.values()),
        [
            ("measurement.head.geometry.p", p),
            ("measurement.head.contrast.zero", zero),
            ("measurement.head.log_variance.absent", absent),
        ],
        clip=1,
    )
    assert report["total_global_norm"] == pytest.approx(10**0.5)
    assert report["groups"]["geometry"]["cosines"]["one|two"] == pytest.approx(-2 / 13**0.5)
    assert report["groups"]["contrast"]["terms"]["one"]["connected_tensors"] == 1
    assert report["groups"]["contrast"]["terms"]["one"]["exact_zero_connected_tensors"] == 1
    assert report["groups"]["contrast"]["terms"]["two"]["connected_tensors"] == 0
    assert report["groups"]["variance"]["cosines"]["one|two"] is None
    torch.testing.assert_close(total[0], gradients["one"][0] + gradients["two"][0])


def tiny_payload():
    torch.manual_seed(123)
    config = N02Config.from_json("configs/encoders/n02/tiny_contracts.json")
    config.cycles = 1
    model = build_encoder(config)
    batch = synthetic_batch(
        config,
        np.random.default_rng(42),
        batch_size=1,
        num_frames=4,
        frame_size=64,
        seed=42,
        split="train",
    )
    with torch.no_grad():
        training_objective(model, batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter) * 0.1
    optimizer.step()
    return {
        "config": config.as_dict(),
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "run": {"lr": 3e-4, "grad_clip": 1.0, "diversity_margin": 0.1},
    }


def test_adam_clone_isolation_and_momentum_control():
    torch.set_num_threads(1)
    payload = tiny_payload()
    before = tree_digest(payload)
    model, optimizer = clone_with_optimizer(payload)
    for key, state in optimizer.state.items():
        del key
        if "exp_avg" in state:
            state["exp_avg"].add_(123)
            break
    assert tree_digest(payload) == before
    gradients = tuple(torch.zeros_like(parameter) for parameter in model.parameters())
    updated, result = disposable_update(payload, gradients, clip=1.0)
    assert any(value > 0 for value in result["branch_parameter_update_norm"].values())
    # With None instead of explicit zero, AdamW skips both momentum and decay.
    unchanged, null_result = disposable_update(payload, (None,) * len(gradients), clip=1.0)
    assert all(value == 0 for value in null_result["branch_parameter_update_norm"].values())
    assert tree_digest(unchanged.state_dict()) == tree_digest(payload["model"])
    assert tree_digest(updated.state_dict()) != tree_digest(payload["model"])
    assert tree_digest(payload) == before


def test_production_objective_gradient_sum_and_detached_evaluation():
    torch.set_num_threads(1)
    payload = tiny_payload()
    model, _ = clone_with_optimizer(payload)
    batch = synthetic_batch(
        model.config,
        np.random.default_rng(900),
        batch_size=2,
        num_frames=8,
        frame_size=64,
        seed=900,
        split="train",
    )
    before = batch_digest(batch)
    total, weighted, report = training_objective(model, batch)
    decomposition, _, gradients = decompose_gradients(
        weighted, total, list(model.named_parameters()), clip=1.0
    )
    assert decomposition["gradient_sum_relative_error"] < 2e-5
    assert report["active_steps"]["geometry_set"] == 8
    assert report["active_steps"]["counterfactual"] == 8
    assert decomposition["groups"]["contrast"]["terms"]["counterfactual"]["grad_norm"] > 0
    updated, _ = disposable_update(payload, gradients, clip=1.0)
    with torch.no_grad():
        no_grad_total, no_grad_terms, _ = training_objective(updated, batch)
    assert not no_grad_total.requires_grad
    assert all(not value.requires_grad for value in no_grad_terms.values())
    assert batch_digest(batch) == before
