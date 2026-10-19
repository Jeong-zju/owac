"""Matched full-action readouts and fixed-budget, validation-only learning curves."""

import copy
import time

import numpy as np
import torch

from owac.frameworks.n02_full.task_chain import (
    SmallActionHead,
    action_metrics,
    masked_action_loss,
    split_groups,
    state_digest,
)


def selected_splits(records, train_groups):
    splits = split_groups(records)
    allowed = set(train_groups)
    actual = {records[i]["group"] for i in splits["train"]}
    if not allowed or not allowed <= actual:
        raise ValueError("training groups must be a nonempty subset of training partition")
    splits["train"] = [i for i in splits["train"] if records[i]["group"] in allowed]
    return splits


def sample_order(indices, updates, seed, batch_size=8):
    if not indices or updates < 1:
        raise ValueError("nonempty training indices and positive budget required")
    rng = np.random.default_rng(seed)
    return [
        rng.choice(indices, min(batch_size, len(indices)), replace=False).tolist()
        for _ in range(updates)
    ]


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def full_targets(actions):
    if actions.ndim != 3 or actions.shape[1:] != (50, 14):
        raise ValueError("expected B,50,14 normalized controls")
    target = actions.new_zeros((len(actions), 50, 32))
    target[..., :14] = actions
    return target


def fit_action_head(
    features,
    states,
    actions,
    masks,
    records,
    train_groups,
    std,
    *,
    seed,
    state_only=False,
    budgets=(256, 512, 1024),
    validate_every=64,
    device="cuda",
    callback=None,
    checkpoint=None,
):
    """Same architecture/init/order for each modality; test read only after optimization.

    State-only uses zero visual slots throughout fitting/evaluation. Parameter
    shapes match exactly, though weights linked to absent information naturally
    have fewer effective degrees of freedom. No fitted feature normalization.
    """
    if tuple(sorted(set(budgets))) != tuple(budgets) or not 1 <= budgets[0] <= budgets[-1] <= 1024:
        raise ValueError("unique increasing budgets within1..1024 required")
    splits = selected_splits(records, train_groups)
    torch.manual_seed(seed)
    x = torch.as_tensor(features).detach().float().to(device)
    if state_only:
        x = torch.zeros_like(x)
    state = torch.as_tensor(states).detach().float().to(device)
    y = torch.as_tensor(actions).detach().float().to(device)
    valid = torch.as_tensor(masks, dtype=torch.bool, device=device)
    target = full_targets(y)
    head = SmallActionHead(width=x.shape[-1], tokens=x.shape[1], state_dim=state.shape[-1]).to(
        device
    )
    initial_sha = state_digest(head)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=1e-3, weight_decay=1e-3, fused=str(device).startswith("cuda")
    )
    order = sample_order(splits["train"], budgets[-1], seed)

    @torch.no_grad()
    def predict(indices):
        return torch.cat(
            [
                head(x[chunk], state[chunk])[..., :14].cpu()
                for chunk in [indices[i : i + 8] for i in range(0, len(indices), 8)]
            ]
        ).numpy()

    @torch.no_grad()
    def measure(split):
        indices = splits[split]
        p = predict(indices)
        return action_metrics(p, np.asarray(actions)[indices], np.asarray(masks)[indices], std), p

    best = float("inf")
    best_step = 0
    best_state = cpu_state(head)
    val, _ = measure("val")
    best = val["normalized_control_mse"]
    history = [dict(step=0, val=val)]
    snapshots = {}
    start = time.perf_counter()
    last_status = None
    for step, indices in enumerate(order, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_action_loss(head(x[indices], state[indices]), target[indices], valid[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite direct action loss")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        if not np.isfinite(grad):
            raise FloatingPointError("nonfinite direct action gradient")
        optimizer.step()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        row = dict(
            step=step,
            loss=float(loss.detach()),
            gradient_norm=grad,
            seconds_per_update=(time.perf_counter() - start) / step,
        )
        if step % validate_every == 0 or step in budgets:
            val, _ = measure("val")
            row["val"] = val
            history.append(copy.deepcopy(row))
            if val["normalized_control_mse"] < best:
                best, best_step, best_state = val["normalized_control_mse"], step, cpu_state(head)
        if step in budgets:
            snapshots[step] = dict(
                selected=copy.deepcopy(best_state),
                last=cpu_state(head),
                selected_step=best_step,
                selected_validation_mse=best,
            )
        if callback:
            callback(
                dict(**row, max_updates=budgets[-1], selected_step=best_step, selected_val_mse=best)
            )
        last_status = row
    report = dict(
        seed=seed,
        initial_sha=initial_sha,
        parameters=sum(p.numel() for p in head.parameters()),
        state_only=state_only,
        actual_updates=budgets[-1],
        train_indices=splits["train"],
        order=order,
        history=history,
        elapsed_s=time.perf_counter() - start,
        rows=[],
    )
    predictions = {}
    # Fixed continuation is complete before reading any test targets/metrics.
    for budget in budgets:
        snapshot = snapshots[budget]
        row = dict(
            budget=budget,
            selected_step=snapshot["selected_step"],
            selected_validation_mse=snapshot["selected_validation_mse"],
        )
        for version in ("selected", "last"):
            head.load_state_dict(snapshot[version], strict=True)
            values = {}
            for split in ("train", "val", "test"):
                metrics, p = measure(split)
                values[split] = metrics
                predictions[f"{budget}_{version}_{split}"] = p[None]
            row[version] = values
            row[version + "_sha"] = state_digest(head)
        report["rows"].append(row)
        if checkpoint:
            checkpoint(budget, snapshot)
    report["last_status"] = last_status
    return report, predictions
