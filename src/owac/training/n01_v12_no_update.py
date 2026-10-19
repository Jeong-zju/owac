"""Read-only code acceptance helpers. No optimizer is created or stepped."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def forbid_optimizer_updates():
    originals = {}

    def forbidden(*args, **kwargs):
        raise RuntimeError("optimizer updates are forbidden by the current no-training request")

    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            originals[cls] = cls.step
            cls.step = forbidden
    try:
        yield
    finally:
        for cls, step in originals.items():
            cls.step = step
