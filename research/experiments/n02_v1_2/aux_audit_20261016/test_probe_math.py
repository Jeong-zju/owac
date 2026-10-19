"""Check the virtual Adam calculation against a disposable three-value optimizer."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch


def test_virtual_adam_matches_actual_disposable_adamw_step():
    path = Path(__file__).with_name("gradient_probe.py")
    spec = importlib.util.spec_from_file_location("aux_gradient_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=0.0003, betas=(0.9, 0.95), weight_decay=0.01)
    parameter.grad = torch.tensor([0.2, 0.8, -0.3], dtype=torch.float64)
    optimizer.step()  # Only the disposable three-value fixture, never policy weights.
    state = copy.deepcopy(optimizer.state_dict())
    values = parameter.detach().clone()
    grads = {
        "action": (torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64),),
        "masked_raw": (torch.tensor([-2.0, 1.0, 0.0], dtype=torch.float64),),
        "causal_raw": (torch.tensor([0.0, 0.5, -3.0], dtype=torch.float64),),
    }
    predicted = module.virtual_adam([parameter], grads, state)
    scenarios = {
        "action_only": (1, 0, 0),
        "action_masked": (1, 1, 0),
        "action_causal": (1, 0, 1),
        "full": (1, 1, 1),
    }
    for label, weights in scenarios.items():
        actual = torch.nn.Parameter(values.clone())
        opt = torch.optim.AdamW([actual], lr=0.0003)
        opt.load_state_dict(copy.deepcopy(state))
        actual.grad = sum(g[0] * w for g, w in zip(grads.values(), weights, strict=True))
        torch.nn.utils.clip_grad_norm_([actual], 1.0)
        opt.step()
        delta = actual.detach() - values
        dot = float((grads["action"][0] * delta).sum())
        assert predicted[label]["first_order_action_loss_change"] == pytest.approx(dot, abs=1e-12)
        assert predicted[label]["delta_l2"] == pytest.approx(float(delta.norm()), abs=1e-12)
    assert torch.equal(parameter.detach(), values)
    assert torch.equal(optimizer.state_dict()["state"][0]["exp_avg"], state["state"][0]["exp_avg"])
