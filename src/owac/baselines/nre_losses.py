"""LAMPE NRE/BNRE losses, vendored at cbf500c09c538a014e72477d166acc62a6d8409a.

Source: https://github.com/probabilists/lampe/tree/cbf500c09c538a014e72477d166acc62a6d8409a
Only imports/formatting are consolidated. Class ASTs are unchanged.

MIT License

Copyright (c) 2021 The Probabilists

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NRELoss(nn.Module):
    r"""Creates a module that calculates the cross-entropy loss for a NRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{2N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i))

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return (l1 + l0) / 2


class BNRELoss(nn.Module):
    r"""Creates a module that calculates the balanced cross-entropy loss for NRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{2N} \sum_{i = 1}^N
            \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i)) \\
          & + \lambda \left(1 - \frac{1}{N} \sum_{i = 1}^N
            d_\phi(\theta_i, x_i) + d_\phi(\theta_{i+1}, x_i) \right)^2

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
        lmbda: The weight :math:`\lambda` controlling the strength of the balancing
            condition.
    """

    def __init__(self, estimator: nn.Module, lmbda: float = 100.0):
        super().__init__()

        self.estimator = estimator
        self.lmbda = lmbda

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()
        lb = (torch.sigmoid(log_r) + torch.sigmoid(log_r_prime) - 1).mean().square()

        return (l1 + l0) / 2 + self.lmbda * lb
