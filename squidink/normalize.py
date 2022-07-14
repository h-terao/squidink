from __future__ import annotations

import jax.numpy as jnp
import chex

from .utils import flatten


def normalize(
    x: chex.Array,
    mean: chex.Array | chex.Scalar | tuple[chex.Scalar, ...] = 0.0,
    std: chex.Array | chex.Scalar | tuple[chex.Scalar, ...] = 1.0,
):
    mean = jnp.array(mean).reshape(1, 1, 1, -1)
    std = jnp.array(std).reshape(1, 1, 1, -1)
    x, unflatten = flatten(x)
    x = (x - mean) / std
    return unflatten(x)


def de_normalize(
    x: chex.Array,
    mean: chex.Array | chex.Scalar | tuple[chex.Scalar, ...] = 0.0,
    std: chex.Array | chex.Scalar | tuple[chex.Scalar, ...] = 1.0,
):
    mean = jnp.array(mean).reshape(1, 1, 1, -1)
    std = jnp.array(std).reshape(1, 1, 1, -1)
    x, unflatten = flatten(x)
    x = x * std + mean
    return unflatten(x)
