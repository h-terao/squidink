from __future__ import annotations

import jax.numpy as jnp
import chex

from . import flatten, to_tuple


def pad(
    x: chex.Array,
    pad_size: int | tuple[int, int],
    mode: str = "constant",
    cval: chex.Scalar = 0,
    reflect_type: str = "even",
) -> chex.Array:
    """Padding operation."""
    pad_size = to_tuple(pad_size)
    x, unflatten = flatten(x)
    pad_width = [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]]
    if mode == "constant":
        x = jnp.pad(x, pad_width, mode, constant_values=cval)
    elif mode in ["reflect", "symmetric"]:
        x = jnp.pad(x, pad_width, mode, reflect_type=reflect_type)
    else:
        x = jnp.pad(x, pad_width, mode)
    return unflatten(x)
