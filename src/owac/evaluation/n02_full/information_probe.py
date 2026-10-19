"""Small supervised readouts of detached features; no policy or simulator dependency."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def split_indices(records):
    """Require whole source episodes/scenes to belong to exactly one diagnostic split."""
    groups = {}
    result = {name: [] for name in ("train", "val", "test")}
    for i, row in enumerate(records):
        split, group = row["split"], row["group"]
        if split not in result:
            raise ValueError("unknown diagnostic split")
        if group in groups and groups[group] != split:
            raise ValueError("source group leaks across diagnostic splits")
        groups[group] = split
        result[split].append(i)
    if any(not values for values in result.values()):
        raise ValueError("all three independent splits are required")
    return {key: np.asarray(value) for key, value in result.items()}


class SmallReadout(nn.Module):
    """Shared channel bottleneck, preserved camera/grid positions, optional 64-unit MLP."""

    def __init__(self, shape, targets, nonlinear=True):
        super().__init__()
        if len(shape) == 4:
            cameras, channels, h, w = shape
            self.channel = nn.Linear(channels, 8)
            width = cameras * h * w * 8
        else:
            self.channel = None
            width = int(np.prod(shape))
        self.head = (
            nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, targets))
            if nonlinear
            else nn.Linear(width, targets)
        )

    def forward(self, features):
        if self.channel is not None:
            features = self.channel(features.permute(0, 1, 3, 4, 2))
        return self.head(features.flatten(1))


def metrics(prediction, target, panel):
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite predictions/targets")
    slices = (
        {
            "object": (0, 3),
            "left_gripper": (3, 6),
            "right_gripper": (6, 9),
            "object_minus_left": (9, 12),
            "object_minus_right": (12, 15),
        }
        if panel == "controlled"
        else {"left_gripper": (0, 3), "right_gripper": (3, 6)}
    )
    output = {}
    for name, (a, b) in slices.items():
        distance = np.linalg.norm(prediction[:, a:b] - target[:, a:b], axis=1) * 100
        output[name] = dict(
            mean_cm=float(distance.mean()),
            p90_cm=float(np.quantile(distance, 0.9)),
            within_2cm=float((distance <= 2).mean()),
            within_5cm=float((distance <= 5).mean()),
        )
    delta = prediction[:, -2:] - target[:, -2:]
    binary = (target[:, -2:] <= 0.2) | (target[:, -2:] >= 0.8)
    output["openness"] = dict(
        mae=float(np.abs(delta).mean()), binary_count=int(binary.sum()), balanced_accuracy=None
    )
    if binary.any():
        truth = target[:, -2:][binary] >= 0.5
        pred = prediction[:, -2:][binary] >= 0.5
        recalls = [
            float((pred[truth == label] == label).mean())
            for label in (False, True)
            if (truth == label).any()
        ]
        output["openness"]["balanced_accuracy"] = float(np.mean(recalls))
    return output


def fit_readout(
    features,
    targets,
    records,
    *,
    seed=13019,
    nonlinear=True,
    shuffle=False,
    updates=600,
    device="cpu",
    checkpoint_path=None,
):
    """Fit only this head; all normalizers fit train, checkpoint selected only on val."""
    indices = split_indices(records)
    x = torch.as_tensor(features, dtype=torch.float32, device=device).detach()
    y = torch.as_tensor(targets, dtype=torch.float32, device=device).detach()
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("nonfinite diagnostic dataset")
    train, val, test = [
        torch.as_tensor(indices[k], device=device) for k in ("train", "val", "test")
    ]
    torch.manual_seed(seed)
    # Preserve fixed spatial layout. One normalizer per camera/channel, not per position.
    axes = (0, 3, 4) if x.ndim == 5 else (0,)
    mean = x[train].mean(axes, keepdim=True)
    std = x[train].std(axes, keepdim=True, unbiased=False).clamp_min(1e-4)
    x = (x - mean) / std
    ym, ys = y[train].mean(0), y[train].std(0, unbiased=False).clamp_min(0.01)
    normalized = (y - ym) / ys
    train_y = normalized[train].clone()
    if shuffle:
        train_y = train_y[torch.randperm(len(train), device=device)]
    model = SmallReadout(tuple(x.shape[1:]), y.shape[1], nonlinear).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_step, state, curve = float("inf"), 0, None, []
    for step in range(1, updates + 1):
        batch = torch.randperm(len(train), device=device)[:96]
        loss = (model(x[train[batch]]) - train_y[batch]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == updates:
            with torch.no_grad():
                error = float((model(x[val]) - normalized[val]).square().mean())
            curve.append(dict(step=step, train_loss=float(loss.detach()), val_mse=error))
            if error < best:
                best, best_step, state = error, step, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    if checkpoint_path is not None:
        torch.save(
            dict(
                model=model.state_dict(),
                feature_mean=mean,
                feature_std=std,
                target_mean=ym,
                target_std=ys,
                selected_step=best_step,
                seed=seed,
                nonlinear=nonlinear,
            ),
            checkpoint_path,
        )
    with torch.no_grad():
        prediction = (model(x) * ys + ym).cpu().numpy()
    return dict(
        seed=seed,
        selected_step=best_step,
        val_normalized_mse=best,
        parameters=sum(p.numel() for p in model.parameters()),
        updates=updates,
        curve=curve,
        shuffled_train_labels=shuffle,
    ), prediction
