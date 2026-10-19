# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# REDUCED VARIANT of upstream savi/lib/utils.py (repository google-research/slot-attention-video
# @ ba8f15ee19472c6f9425c9647daf87910f17b605, sha256 2f879ef89ccc527f...).
#
# Only the five helpers below are kept, copied VERBATIM from the upstream file, because the
# cached module set uses exactly these and no others.  The upstream module imports tensorflow,
# clu, matplotlib, ml_collections and skimage at module scope for logging/plotting helpers that
# the reference modules never call; keeping them would force a multi-gigabyte dependency into the
# reference environment.  Nothing in the kept functions is modified.

import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Optional, Sequence, Union

import jax

Array = jax.Array


def spatial_broadcast(x: Array, resolution: Sequence[int]) -> Array:
  """Broadcast flat inputs to a 2D grid of a given resolution."""
  # x.shape = (batch_size, features).
  x = x[:, jnp.newaxis, jnp.newaxis, :]
  return jnp.tile(x, [1, resolution[0], resolution[1], 1])

def time_distributed(cls, in_axes=1, axis=1):
  """Wrapper for time-distributed (vmapped) application of a module."""
  return nn.vmap(
      cls, in_axes=in_axes, out_axes=axis, axis_name="time",
      # Stack debug vars along sequence dim and broadcast params.
      variable_axes={
          "params": None, "intermediates": axis, "batch_stats": None},
      split_rngs={"params": False, "dropout": True, "state_init": True})

def broadcast_across_batch(inputs: Array, batch_size: int) -> Array:
  """Broadcasts inputs across a batch of examples (creates new axis)."""
  return jnp.broadcast_to(
      arr=jnp.expand_dims(inputs, axis=0),
      shape=(batch_size,) + inputs.shape)

def create_gradient_grid(
    samples_per_dim: Sequence[int], value_range: Sequence[float] = (-1.0, 1.0)
    ) -> Array:
  """Creates a tensor with equidistant entries from -1 to +1 in each dim.

  Args:
    samples_per_dim: Number of points to have along each dimension.
    value_range: In each dimension, points will go from range[0] to range[1]

  Returns:
    A tensor of shape [samples_per_dim] + [len(samples_per_dim)].
  """
  s = [jnp.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
  pe = jnp.stack(jnp.meshgrid(*s, sparse=False, indexing="ij"), axis=-1)
  return jnp.array(pe)

def convert_to_fourier_features(inputs: Array, basis_degree: int) -> Array:
  """Convert inputs to Fourier features, e.g. for positional encoding."""

  # inputs.shape = (..., n_dims).
  # inputs should be in range [-pi, pi] or [0, 2pi].
  n_dims = inputs.shape[-1]

  # Generate frequency basis.
  freq_basis = jnp.concatenate(  # shape = (n_dims, n_dims * basis_degree)
      [2**i * jnp.eye(n_dims) for i in range(basis_degree)], 1)

  # x.shape = (..., n_dims * basis_degree)
  x = inputs @ freq_basis  # Project inputs onto frequency basis.

  # Obtain Fourier features as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
  return jnp.sin(jnp.concatenate([x, x + 0.5 * jnp.pi], axis=-1))
