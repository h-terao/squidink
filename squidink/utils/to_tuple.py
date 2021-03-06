from __future__ import annotations
import jax.numpy as jnp


def to_tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if jnp.ndim(x) == 0:
        x = (x, x)
    return x
