from __future__ import annotations
import jax.numpy as jnp
import chex

from .utils import blend, convolve


def sharpness(x: chex.Array, factor: float) -> chex.Array:
    # Smooth PIL kernel.
    kernel = jnp.array(
        [
            [1, 1, 1],
            [1, 5, 1],
            [1, 1, 1],
        ],
        dtype=x.dtype,
    )
    kernel /= 13.0
    degenerate = convolve(x, kernel)
    return blend(degenerate, x, factor)
