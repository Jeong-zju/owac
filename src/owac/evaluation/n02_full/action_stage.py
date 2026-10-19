"""Action-stage proxies and paired residual direction diagnostics, never policy inputs."""

import numpy as np


def action_stages(actions, left_ee, right_ee):
    """Proxy stages from commanded closure and measured end-effector rise.

    These labels do not establish contact, object lift or physical success.
    No relative-time bins: a trajectory without closure/rise has no lift stage.
    """
    actions, left_ee, right_ee = map(np.asarray, (actions, left_ee, right_ee))
    if (
        actions.ndim != 2
        or actions.shape[1] != 14
        or len(left_ee) != len(actions)
        or len(right_ee) != len(actions)
    ):
        raise ValueError("matched full14-action and end-effector trajectories required")
    stages = np.full(len(actions), "approach", dtype="U8")
    closing = np.flatnonzero((actions[:, [6, 13]] < 0.95).any(1))
    if not len(closing):
        return stages
    stages[closing[0] :] = "grasp"
    closed = np.flatnonzero((actions[:, [6, 13]] <= 0.2).all(1))
    if len(closed):
        t = closed[0]
        raised = (
            (left_ee[:, 2] - left_ee[t, 2] >= 0.02)
            & (right_ee[:, 2] - right_ee[t, 2] >= 0.02)
            & (np.arange(len(actions)) >= t)
        )
        lift = np.flatnonzero(raised)
        if len(lift):
            stages[lift[0] :] = "lift"
    return stages


def correction_direction(base, prediction, target, mask, std):
    """Joint-only physical direction toward the demonstrated target, no FK claim."""
    joints = [*range(6), *range(7, 13)]
    scale = np.asarray(std)[joints]
    valid = np.asarray(mask, dtype=bool)
    desired = ((target - base)[..., joints] * scale)[valid]
    correction = ((prediction - base)[..., joints] * scale)[valid]
    dot = np.sum(desired * correction, -1)
    den = np.linalg.norm(desired, axis=-1) * np.linalg.norm(correction, axis=-1)
    nonzero = den > 1e-10
    cosine = dot[nonzero] / den[nonzero]
    before = np.sum(desired**2, -1)
    after = np.sum((desired - correction) ** 2, -1)
    return dict(
        valid_steps=len(dot),
        nonzero_direction_steps=int(nonzero.sum()),
        direction_toward_target_fraction=float((dot[nonzero] > 0).mean())
        if nonzero.any()
        else None,
        mean_direction_cosine=float(cosine.mean()) if len(cosine) else None,
        joint_error_reduced_fraction=float((after < before).mean()),
        correction_rms_deg=float(np.mean(correction**2) ** 0.5 * 180 / np.pi),
    )
