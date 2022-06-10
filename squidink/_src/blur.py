from __future__ import annotations
import functools

import jax.numpy as jnp
import chex

from .utils import convolve, blend, to_tuple


def mean_blur(
    x: chex.Array, factor: float = 1.0, kernel_size: int | tuple[int, int] = 9
) -> chex.Array:
    kernel_size = to_tuple(kernel_size)
    degenerate = convolve(x, functools.partial(jnp.mean, axis=(1, 2)), kernel_size)
    return blend(degenerate, x, factor)


def median_blur(
    x: chex.Array, factor: float = 1.0, kernel_size: int | tuple[int, int] = 9
) -> chex.Array:
    kernel_size = to_tuple(kernel_size)
    degenerate = convolve(x, functools.partial(jnp.median, axis=(1, 2)), kernel_size)
    return blend(degenerate, x, factor)
