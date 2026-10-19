"""Regression tests for misleading window weighting and missing task coverage."""

import numpy as np
import pytest
import torch

from owac.training.n02_v12_task import (
    BalancedBlocks,
    aggregate_control,
    control_flow_loss,
    control_metrics,
    dev_selection,
)


def records():
    return [
        dict(
            index=e * 96 + t,
            path=f"episode{e}",
            t=t,
            frames=96,
            light=light,
            phase="opening" if t < 10 else "approach" if t < 58 else "late",
            valid=min(50, 95 - t),
        )
        for e, light in enumerate(("Normal", "Over"))
        for t in range(96)
    ]


def test_dev_selection_covers_every_episode_start_and_nontrivial_tail():
    manifest = dev_selection(records())
    assert len(manifest["rows"]) == 8
    assert [(r["light"], r["t"]) for r in manifest["rows"] if r["phase"] == "opening"] == [
        ("Normal", 0),
        ("Over", 0),
    ]
    assert min(r["valid"] for r in manifest["rows"]) == 10
    assert set(manifest["missing_lighting"]) == {"Low", "ExtremeLow", "ExtremeOver"}
    assert manifest == dev_selection(records())


def test_sampler_balances_strata_deterministically_and_never_crosses_episodes():
    sampler = BalancedBlocks(records(), 8, 1103)
    same = BalancedBlocks(records(), 8, 1103)
    groups = set()
    for i in range(len(sampler.keys)):
        start, stop = sampler.bounds(i)
        assert (start, stop) == same.bounds(i)
        r = records()[start:stop]
        assert len({x["path"] for x in r}) == 1
        groups.add((r[0]["light"], r[0]["phase"]))
    assert groups == set(sampler.keys)
    for i in range(200):
        start, stop = sampler.bounds(i)
        assert any(r["valid"] for r in records()[start:stop])


def test_real_control_loss_ignores_padded_dimensions_and_nan_masked_horizons():
    v = torch.randn(2, 50, 32, requires_grad=True)
    u = torch.randn_like(v)
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[0, :50] = True
    mask[1, 0] = True
    u[1, 1:] = float("nan")
    loss = control_flow_loss(v, u, mask, 14)
    torch.testing.assert_close(loss, (v[mask][:, :14] - u[mask][:, :14]).square().mean())
    loss.backward()
    assert torch.isfinite(v.grad).all()
    assert v.grad[..., 14:].abs().sum() == 0
    assert v.grad[~mask].abs().sum() == 0
    with pytest.raises(ValueError):
        control_flow_loss(v, u, torch.zeros_like(mask), 14)


def test_metrics_do_not_weight_one_action_as_fifty_and_keep_strata_visible():
    rows = []
    for i, (n, scale, light) in enumerate(((50, 1, "Normal"), (1, 10, "Over"))):
        mask = np.arange(50)[None, :] < n
        pred = np.ones((1, 50, 14)) * scale
        pred[~mask] = float("nan")
        rows.append(
            control_metrics(
                pred, np.zeros_like(pred), mask, dict(light=light, phase="opening", index=i)
            )
        )
    result = aggregate_control(rows)
    assert result["mse"] == pytest.approx(150 / 51)
    assert result["balanced_mse"] == pytest.approx(50.5)
    assert result["strata"]["Normal/opening"]["mse"] == 1


def test_checkpoint_selection_rejects_tail_gain_with_worse_normal_opening(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path("research/experiments/n02_v1_2/task_r9_20261017/run.py")
    spec = importlib.util.spec_from_file_location("task_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = dict(
        sampled_control_mse=1.0,
        task_dev_manifest_sha256="fixed",
        task_control={"strata": {"Normal/opening": {"mse": 1.0}}},
    )
    (tmp_path / "dev_baseline.json").write_text(json.dumps(baseline))
    for step, score, opening, manifest in (
        (16, 0.1, 1.2, "fixed"),
        (32, 0.9, 1.04, "fixed"),
        (48, 0.01, 0.5, "other"),
    ):
        record = dict(
            step=step,
            sampled_control_mse=score,
            task_dev_manifest_sha256=manifest,
            task_control={"strata": {"Normal/opening": {"mse": opening}}},
        )
        (tmp_path / f"checkpoint_{step:07d}.metrics.json").write_text(json.dumps(record))
        (tmp_path / f"checkpoint_{step:07d}.pt").touch()
    decision = module.selection(tmp_path)
    assert decision["selected"]["step"] == 32
    (tmp_path / "checkpoint_0000032.pt").unlink()
    assert module.selection(tmp_path)["selected"] is None
