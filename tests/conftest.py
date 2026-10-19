"""Optional acceptance mode that cannot perform a neural optimizer update."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-optimizer-updates",
        action="store_true",
        default=False,
        help="skip any test at its first attempted torch optimizer update",
    )


@pytest.fixture(autouse=True)
def _no_optimizer_updates(request, monkeypatch):
    if not request.config.getoption("--no-optimizer-updates"):
        return
    import torch

    def forbidden(*args, **kwargs):
        pytest.skip("optimizer update forbidden: user requested code work without training")

    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            monkeypatch.setattr(cls, "step", forbidden)
