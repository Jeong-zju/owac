"""Performance changes must preserve masking, gradients and consumed sample order."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from owac.frameworks.n01_full.learned_motion import DomainWindowMotion, MotionConfig
from owac.frameworks.n01_full.model import N01V12Config, build_model
from owac.frameworks.n01_full.training_plan import BackendSchedule, TrainingPlan
from owac.runtime.n01_v12_monitor import _parse_gpu_line
from owac.training.n01_v12_input import PrefetchBatches


def test_motion_pair_batching_preserves_source_order_flow_cycle_and_gradients():
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(6, 2, 1)

        def forward(self, reference, source, **kwargs):
            return {"final": self.layer(torch.cat([reference, source], 1))}

    provider = DomainWindowMotion(MotionConfig(activation_checkpointing=False), "RAW", Network())
    batched = copy.deepcopy(provider)
    batched.pair_chunk = 8
    values = torch.rand(4, 2, 3, 12, 12)
    state = SimpleNamespace(
        values=values,
        valid=torch.ones_like(values, dtype=torch.bool),
        observed=torch.ones(4, 2, dtype=torch.bool),
        filled=torch.tensor([4, 4]),
        frame_ids=torch.arange(4)[:, None].expand(-1, 2),
    )
    first = provider(frame=None, state=state)
    second = batched(frame=None, state=state)
    assert torch.allclose(first.offsets_xy, second.offsets_xy, atol=1e-6)
    assert torch.allclose(first.risk, second.risk, atol=1e-6)
    assert torch.equal(first.visibility, second.visibility)
    assert torch.equal(first.valid, second.valid)
    first.offsets_xy.sum().backward()
    second.offsets_xy.sum().backward()
    for a, b in zip(provider.parameters(), batched.parameters(), strict=True):
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("reference_withdrawal", [False, True])
def test_pruning_empty_sources_preserves_reference_fallback_and_gradients(reference_withdrawal):
    model = build_model(
        N01V12Config(
            arm="raw_base", feature_width=8, stem_width=4, residual_blocks=1, window_frames=4
        )
    )
    full, pruned = model.encoder, copy.deepcopy(model.encoder)
    pruned.skip_empty_sources = True
    values = torch.rand(2, 4, 3, 12, 12, requires_grad=True)
    mask = torch.ones(2, 4, 1, 12, 12, dtype=torch.bool)
    mask[:, 2] = False
    mask[:, 0, :, :5] = False
    offset = torch.zeros(2, 4, 2, 12, 12)
    offset[:, 1, 0] = 0.25
    reference = mask.clone() if reference_withdrawal else None
    if reference is not None:
        reference[:, 3] = False
    a, available, weights = full.encode_window(values, mask, offset, reference_mask=reference)
    b, present, compact = pruned.encode_window(
        values, mask, offset, reference_mask=reference, compact_weights=True
    )
    assert torch.equal(available, present)
    assert torch.allclose(a, b, atol=1e-7, rtol=1e-5)
    assert torch.allclose(weights.mean(2, keepdim=True), compact, atol=1e-7)
    a.sum().backward()
    b.sum().backward()
    for first, second in zip(full.parameters(), pruned.parameters(), strict=True):
        assert torch.allclose(first.grad, second.grad, atol=1e-5, rtol=1e-4)


def test_small_backend_updates_accumulate_in_master_weights():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = nn.Linear(2, 2, dtype=torch.bfloat16)
            self.language = nn.Linear(2, 2, dtype=torch.bfloat16)

    policy = Policy()
    scope = BackendSchedule(nn.Linear(2, 2), policy, TrainingPlan(), master_weights=True)
    scope.apply(2000)
    weight = policy.action_head.weight
    with torch.no_grad():
        weight.fill_(0.5)
    assert weight.dtype == torch.float32
    assert policy.language.weight.dtype == torch.bfloat16
    weight.grad = torch.ones_like(weight)
    optimizer = torch.optim.SGD([weight], lr=1e-6)
    optimizer.step()
    assert bool((weight < 0.5).all())
    restored = Policy()
    target = BackendSchedule(nn.Linear(2, 2), restored, TrainingPlan(), master_weights=True)
    target.restore(scope.checkpoint())
    assert torch.equal(restored.action_head.weight, weight)


def test_prefetch_and_resume_keep_consumed_order_across_epochs(monkeypatch):
    from owac.raw import rawvla_sequence

    monkeypatch.setattr(rawvla_sequence, "collate_sequence_samples", lambda items: items)

    class Dataset:
        def __len__(self):
            return 17

        def __getitem__(self, i):
            return {"index": i}

    source = Dataset()
    serial = PrefetchBatches(source, 101, 0, 2, 0)
    parallel = PrefetchBatches(source, 101, 0, 2, 3)
    try:
        for _ in range(10):
            assert next(serial) == next(parallel)
        resumed = PrefetchBatches(source, 101, 20, 2, 2)
        try:
            assert next(resumed) == next(parallel)
        finally:
            resumed.close()
    finally:
        serial.close()
        parallel.close()


def test_gpu_power_sample_and_legacy_csv_both_parse():
    old = _parse_gpu_line("0, NVIDIA Pro 6000, 12000, 97887, 99")
    new = _parse_gpu_line("0, NVIDIA Pro 6000, 12000, 97887, 99, 432.1, 600")
    assert old["utilization_pct"] == new["utilization_pct"] == 99
    assert new["memory_used_mib"] == 12000
    assert new["power_w"] == pytest.approx(432.1)
