"""Exact banded evaluation of the author's RKN equations, without dense zero blocks."""

from __future__ import annotations

import torch

from owac.baselines._vendor.rkn.cell import var_activation


def band_layout(cell):
    width, bandwidth = cell._lod, int(cell.c.bandwidth)
    rows = torch.arange(width)[:, None]
    columns = rows + torch.arange(-bandwidth, bandwidth + 1)[None]
    valid = (columns >= 0) & (columns < width)
    table = torch.zeros(width, width, dtype=torch.long)
    table[cell._idx0.cpu(), cell._idx1.cpu()] = torch.arange(cell._idx0.numel())
    columns = columns.clamp(0, width - 1)
    return table[rows, columns], columns, valid


def transition_scale(a, b, c, d, limit, columns=None):
    """A differentiable bound: ||A||2 <= sqrt(||A||1 * ||A||inf).

    Applies to the assembled observation/hidden transition, including its identity
    blocks. Mean and covariance always use the exact same scaled transition.
    """
    row_max = torch.maximum(
        (a.abs() + b.abs()).sum(-1).amax(-1), (c.abs() + d.abs()).sum(-1).amax(-1)
    )
    obs, hidden = a.abs() + c.abs(), b.abs() + d.abs()
    if columns is None:
        obs, hidden = obs.sum(-2), hidden.sum(-2)
    else:
        indices = columns.expand(a.shape[0], -1, -1).flatten(1)
        blank = a.new_zeros((a.shape[0], a.shape[1]))
        obs = blank.scatter_add(1, indices, obs.flatten(1))
        hidden = blank.scatter_add(1, indices, hidden.flatten(1))
    col_max = torch.maximum(obs.amax(-1), hidden.amax(-1))
    bound = (row_max * col_max).clamp_min(1e-12).sqrt()
    return (float(limit) / bound).clamp_max(1)[:, None, None]


def predict_dense_bounded(cell, mean, upper, lower, side, limit):
    from owac.baselines._vendor.rkn.cell import bmv, dadat, dadbt

    (a, b, c, d), noise = cell.get_transition_model(mean)
    scale = transition_scale(a, b, c, d, limit)
    a, b, c, d = (block * scale for block in (a, b, c, d))
    mu, ml = mean.chunk(2, -1)
    return (
        torch.cat((bmv(a, mu) + bmv(b, ml), bmv(c, mu) + bmv(d, ml)), -1),
        dadat(a, upper) + 2 * dadbt(a, side, b) + dadat(b, lower) + noise[..., : cell._lod],
        dadat(c, upper) + 2 * dadbt(c, side, d) + dadat(d, lower) + noise[..., cell._lod :],
        dadbt(c, upper, a) + dadbt(d, side, a) + dadbt(c, side, b) + dadbt(d, lower, b),
    )


def predict_banded(cell, mean, upper, lower, side, lookup, columns, valid, max_operator_norm=None):
    """Same mean and diagonal-block covariance as RKNCell._predict, O(B*d*band)."""
    coefficients = cell._coefficient_net(mean)
    basis = torch.stack(
        (cell._tm_11_basis, cell._tm_12_basis, cell._tm_21_basis, cell._tm_22_basis)
    )
    flat = torch.einsum("bk,vke->bve", coefficients, basis)
    bands = flat[:, :, lookup] * valid
    identity = (columns == torch.arange(cell._lod, device=mean.device)[:, None]) * valid
    a, b, c, d = bands.unbind(1)
    a, d = a + identity, d + identity
    if max_operator_norm is not None:
        scale = transition_scale(a, b, c, d, max_operator_norm, columns)
        a, b, c, d = (block * scale for block in (a, b, c, d))
    mu, ml = mean[:, : cell._lod][:, columns], mean[:, cell._lod :][:, columns]
    cu, cl, cs = upper[:, columns], lower[:, columns], side[:, columns]
    nmu = (a * mu + b * ml).sum(-1)
    nml = (c * mu + d * ml).sum(-1)
    noise = var_activation(cell._log_transition_noise)
    ncu = (a.square() * cu + 2 * a * b * cs + b.square() * cl).sum(-1)
    ncl = (c.square() * cu + 2 * c * d * cs + d.square() * cl).sum(-1)
    ncs = (c * a * cu + d * a * cs + c * b * cs + d * b * cl).sum(-1)
    return (
        torch.cat((nmu, nml), -1),
        ncu + noise[..., : cell._lod],
        ncl + noise[..., cell._lod :],
        ncs,
    )
