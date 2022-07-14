from __future__ import annotations
import jax
import jax.numpy as jnp
import chex
from .utils import flatten, to_tuple


def resize(x: chex.Array, size: int | tuple[int, int], order: int, antialias: bool = True):
    """Resize input array.

    Args:
        x: Input array.
        size: Desired size.
        order: Interpolation order.
        antialias (bool): Antialiasing or not.

    Returns:
        Resized array.
    """
    method = ["nearest", "linear", "cubic"][order]
    x, unflatten = flatten(x)
    N, _, _, C = x.shape
    shape = jnp.array([N, *to_tuple(size), C])
    x = jax.image.resize(x, shape, method, antialias)
    return unflatten(x)
