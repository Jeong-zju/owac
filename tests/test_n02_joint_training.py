"""Exact loss identity, gradient intervention, and original-AdamW isolation."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from owac.evaluation.n02_gradients import (  # noqa: E402
    clone_with_optimizer,
    training_objective,
    tree_digest,
)
from owac.evaluation.n02_joint_training import (  # noqa: E402
    compute_gradients,
    make_batch,
    objective,
    optimizer_step,
    paired_cf_components,
    project_geometry,
)
from owac.frameworks.n02.config import N02Config  # noqa: E402
from owac.frameworks.n02.model import build_encoder  # noqa: E402
from owac.training.n02 import counterfactual_measurement_loss  # noqa: E402


def setup_model():
    torch.set_num_threads(1)
    torch.manual_seed(318)
    config = N02Config.from_json("configs/encoders/n02/tiny_contracts.json")
    config.cycles = 1
    model = build_encoder(config)
    batch = make_batch(config, 920, 920)
    with torch.no_grad():
        objective(model, batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter) * 0.01
    optimizer.step()
    return (
        model,
        batch,
        {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "config": config.as_dict(),
            "run": {"lr": 3e-4},
        },
    )


def test_adjacent_pair_identity_includes_variance_and_all_parameter_gradients():
    model, _, _ = setup_model()
    # Unequal actual geometries and nonzero learned contrast: a class-mean
    # shortcut would erase within-parity differences and fail this identity.
    model.measurement.head.contrast.weight.data.normal_(0, 0.1)
    h = torch.randn(2, 1, 4, 32, 5, 7, requires_grad=True)
    targets = torch.randn(2, 2, 3, 20, 28)
    rep = SimpleNamespace(h=h)
    sensor = torch.rand(2, 3, 5, 7) * 0.01 + 0.0001
    original = counterfactual_measurement_loss(model, rep, targets, sensor_variance=sensor)
    parts = paired_cf_components(model, rep, targets, sensor_variance=sensor)
    decomposed = parts["common"] + parts["difference"] + parts["log_variance"]
    torch.testing.assert_close(decomposed, original, rtol=2e-6, atol=2e-6)
    params = (*model.parameters(), h)
    first = torch.autograd.grad(original, params, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(decomposed, params, allow_unused=True)
    for left, right in zip(first, second, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-5)


def test_projection_is_one_sided_blockwise_and_preserves_none():
    geometry = (torch.tensor([1.0, 2.0]), torch.tensor([2.0]), None, None)
    mixture = (torch.tensor([-3.0, 1.0]), torch.tensor([-4.0]), torch.tensor([7.0]), None)
    total = (mixture[0] + geometry[0], mixture[1] + geometry[1], mixture[2], None)
    projected, report = project_geometry(total, mixture, geometry, [0, 1])
    assert report["applied"]
    assert abs(report["mixture_geometry_dot_after"]) < 2e-6
    assert projected[2] is total[2] and projected[3] is None
    unchanged, neutral = project_geometry(total, geometry, geometry, [0, 1])
    assert not neutral["applied"]
    assert all(a is b for a, b in zip(unchanged, total, strict=True))


def test_baseline_matches_production_gradients_and_adam_update_without_payload_changes():
    _, batch, payload = setup_model()
    original = tree_digest(payload)
    model, optimizer = clone_with_optimizer(payload)
    reference, reference_optimizer = clone_with_optimizer(payload)
    total, weighted, _ = objective(model, batch, balanced=False)
    production, production_weighted, _ = training_objective(reference, batch)
    torch.testing.assert_close(total, production, rtol=0, atol=0)
    gradients, geometry, _ = compute_gradients(model, total, weighted, project=False)
    reference_geometry = torch.autograd.grad(
        production_weighted["geometry_set"],
        list(reference.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    for (name, _), left, right in zip(
        model.named_parameters(), geometry, reference_geometry, strict=True
    ):
        if ".head.geometry." in name:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    production.backward()
    for gradient, parameter in zip(gradients, reference.parameters(), strict=True):
        assert (gradient is None) == (parameter.grad is None)
        if gradient is not None:
            torch.testing.assert_close(gradient, parameter.grad, rtol=0, atol=0)
    optimizer_step(model, optimizer, gradients, geometry)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_optimizer.step()
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert tree_digest(payload) == original


def test_baseline_clip_factor_control_preserves_non_geometry_update():
    _, batch, payload = setup_model()
    model, optimizer = clone_with_optimizer(payload)
    reference, reference_optimizer = clone_with_optimizer(payload)
    total, weighted, _ = objective(model, batch)
    projected, geometry, report = compute_gradients(model, total, weighted, project=True)
    production, weights, _ = objective(reference, batch)
    original, original_geometry, _ = compute_gradients(
        reference, production, weights, project=False
    )
    factor = min(1.0, 1.0 / (report["baseline_global_norm"] + 1e-6))
    optimizer_step(model, optimizer, projected, geometry, fixed_factor=factor)
    optimizer_step(reference, reference_optimizer, original, original_geometry, fixed_factor=factor)
    for (name, a), (_, b) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if ".head.geometry." not in name:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
