# Reference: https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/intensity.py
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import chex

from .utils import flatten


def salt(rng: chex.PRNGKey, x: chex.Array, ratio: float = 0.01) -> chex.Array:
    """Set a certain fraction of pixel intensities to one, hence they become white.

    Args:
        rng: JAX PRNG key.
        x: Input array.
        ratio (float): Probability of whether each pixel becomes one.

    Returns:
        Transformed array.
    """
    x, unflatten = flatten(x)
    _, H, W, _ = x.shape
    mask = jr.uniform(rng, (1, H, W, 1)) < ratio
    x = jnp.where(mask, 1, x)
    return unflatten(x)


def pepper(rng: chex.PRNGKey, x: chex.Array, ratio: float = 0.01) -> chex.Array:
    """Set a certain fraction of pixel intensities to zero, hence they become black.

    Args:
        rng: JAX PRNG key.
        x: Input array.
        ratio (float): Probability of whether each pixel becomes zero.

    Returns:
        Transformed array.
    """
    x, unflatten = flatten(x)
    _, H, W, _ = x.shape
    mask = jr.uniform(rng, (1, H, W, 1)) < ratio
    x = jnp.where(mask, 0, x)
    return unflatten(x)
