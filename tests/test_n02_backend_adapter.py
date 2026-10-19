"""The low-rank path must affect direct weight calls, preserve base weights and reload."""

import torch
from torch import nn
from torch.nn import functional as F

from owac.backends.low_rank import adapter_state, install_low_rank, restore_adapters


def test_original_tower_is_excluded_from_matched_backend_budget():
    base = torch.nn.Module()
    base.vision = torch.nn.Module()
    base.vision.q_proj = torch.nn.Linear(4, 4)
    base.language = torch.nn.Module()
    base.language.q_proj = torch.nn.Linear(4, 4)
    base.language.v_proj = torch.nn.Linear(4, 4)
    base.requires_grad_(False)
    vision_weight = base.vision.q_proj.weight.detach().clone()
    adapters, report = install_low_rank(
        base, {"kind": "lora", "rank": 2, "alpha": 4, "exclude_prefixes": ["vision"]}
    )
    assert set(adapters) == {"language.q_proj", "language.v_proj"}
    assert report["trainable_parameters"] == 32
    base.language.q_proj(torch.ones(1, 4)).sum().backward()
    assert adapters["language.q_proj"].B.grad.abs().sum() > 0
    assert base.vision.q_proj.weight.grad is None
    assert torch.equal(base.vision.q_proj.weight, vision_weight)


def test_parameter_digest_includes_scalar_and_bfloat16_values():
    from owac.training.n02_v12_profile import parameter_digest

    scalar = torch.nn.Parameter(torch.tensor(0.1))
    vector = torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.bfloat16))
    initial = parameter_digest([scalar, vector])
    with torch.no_grad():
        scalar.add_(0.1)
    assert parameter_digest([scalar, vector]) != initial


def test_lora_direct_weight_gradient_update_and_restore():
    torch.manual_seed(1)
    base = nn.Module()
    base.q_proj = nn.Linear(12, 8)
    base.v_proj = nn.Linear(12, 8)
    base.requires_grad_(False)
    x = torch.randn(3, 12)
    expected = F.linear(x, base.q_proj.weight, base.q_proj.bias).detach()
    original = base.q_proj.weight.detach().clone()
    adapters, report = install_low_rank(base, {"kind": "lora", "rank": 2, "alpha": 4})
    torch.testing.assert_close(
        F.linear(x, base.q_proj.weight, base.q_proj.bias), expected, rtol=0, atol=0
    )
    optimizer = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=0.01)
    loss = F.linear(x, base.q_proj.weight, base.q_proj.bias).square().mean()
    loss.backward()
    assert adapters["q_proj"].B.grad.abs().sum() > 0
    optimizer.step()
    assert base.q_proj.parametrizations.weight.original.grad is None
    torch.testing.assert_close(
        base.q_proj.parametrizations.weight.original, original, rtol=0, atol=0
    )
    changed = F.linear(x, base.q_proj.weight, base.q_proj.bias).detach().clone()
    assert not torch.equal(changed, expected)
    saved = adapter_state(adapters)
    with torch.no_grad():
        adapters["q_proj"].B.zero_()
    restore_adapters(adapters, saved)
    torch.testing.assert_close(
        F.linear(x, base.q_proj.weight, base.q_proj.bias), changed, rtol=0, atol=0
    )
    assert report["original_backend_frozen"]


def test_accepted_raw_transfer_resets_readout_without_changing_data_split(tmp_path):
    import hashlib
    from types import SimpleNamespace

    import pytest

    from owac.frameworks.n02_full.sequence_v12 import EXECUTION_REVISION
    from owac.training.n02_v12 import N02V12Trainer
    from owac.training.n02_v12_representation import FORMAT

    model = nn.Module()
    model.inference = nn.Module()
    model.inference.encoder = nn.Linear(2, 2)
    model.inference.readout = nn.Linear(2, 2)
    initial_readout = {k: v.clone() for k, v in model.inference.readout.state_dict().items()}
    source = tmp_path / "source.pt"
    torch.save(
        {
            "execution_revision": EXECUTION_REVISION,
            "cursor": 17,
            "training_generator": torch.Generator().manual_seed(4113).get_state(),
            "model": {k: torch.full_like(v, 5) for k, v in model.state_dict().items()},
        },
        source,
    )
    identity = {
        name: {"batch_size": 1, "manifest": "fixed", "anchors": "fixed", "history": 8}
        for name in ("train", "dev")
    }
    representation = tmp_path / "representation.pt"
    torch.save(
        {
            "format": FORMAT,
            "step": 400,
            "dev": {"representation_gate_passed": True},
            "data_identity": identity,
            "encoder": {
                k: torch.full_like(v, 7) for k, v in model.inference.encoder.state_dict().items()
            },
        },
        representation,
    )
    settings = {
        "checkpoint": str(source),
        "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "representation_checkpoint": str(representation),
        "representation_sha256": hashlib.sha256(representation.read_bytes()).hexdigest(),
        "reset_readout": True,
    }
    trainer = object.__new__(N02V12Trainer)
    trainer.model = model
    trainer.identity = {n: {**i, "batch_size": 12} for n, i in identity.items()}
    trainer.plan = SimpleNamespace(config={"policy": {"freeze_encoder": True}})
    trainer.optimizer = torch.optim.AdamW(model.parameters())
    trainer.backend_adapter_report = {"kind": "none"}
    trainer._write_json = lambda *args: None
    trainer.initialise_representation_weights(settings)
    for p in model.inference.encoder.parameters():
        assert torch.equal(p, torch.full_like(p, 7))
    for name, value in model.inference.readout.state_dict().items():
        torch.testing.assert_close(value, initial_readout[name], rtol=0, atol=0)
    trainer.initialise_representation_weights({**settings, "continue_data_order": True})
    assert trainer.initial_data_cursor == 17
    assert torch.equal(
        trainer.initial_training_generator, torch.Generator().manual_seed(4113).get_state()
    )
    trainer.identity["dev"]["manifest"] = "different"
    with pytest.raises(ValueError, match="dev partitions differ"):
        trainer.initialise_representation_weights(settings)
    trainer.identity["dev"]["manifest"] = "fixed"
    with pytest.raises(ValueError, match="source checkpoint hash changed"):
        trainer.initialise_representation_weights({**settings, "checkpoint_sha256": "wrong"})


def test_dev_best_selection_requires_saved_weights_and_sampled_actions(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path("research/experiments/n02_v1_2/representation_r8_20261016/train_then_rollout.py")
    spec = importlib.util.spec_from_file_location("r8_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for step, velocity, sampled, saved in [
        (25, 0.001, 2.0, True),
        (50, 0.1, 0.5, True),
        (75, 0.001, 0.01, False),
        (100, 0.0, -1.0, True),
    ]:
        (tmp_path / f"checkpoint_{step:07d}.metrics.json").write_text(
            json.dumps({"step": step, "action_error": velocity, "sampled_control_mse": sampled})
        )
        if saved:
            (tmp_path / f"checkpoint_{step:07d}.pt").write_bytes(b"weights")
    score, step, checkpoint = module.select_saved_checkpoint(tmp_path)
    assert score == 0.5 and step == 50 and checkpoint.name == "checkpoint_0000050.pt"
