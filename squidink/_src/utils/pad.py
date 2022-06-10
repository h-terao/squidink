from __future__ import annotations

import jax.numpy as jnp
import chex

from . import flatten


def pad(
    x: chex.Array, pad_size: int | tuple[int, int], mode: str = "constant", **kwargs
) -> chex.Array:
    """Padding operation."""
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size)

    x, unflatten = flatten(x)
    pad_width = [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]]
    x = jnp.pad(x, pad_width, mode, **kwargs)
    return unflatten(x)
