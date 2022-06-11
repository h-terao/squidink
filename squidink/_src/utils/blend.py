from __future__ import annotations
import jax.numpy as jnp
import chex


def blend(x1: chex.Array, x2: chex.Array, factor: float) -> chex.Array:
    """Return factor * x1 + (1.-factor) * x2."""
    return jnp.clip(factor * (x1 - x2) + x2, 0, 1)
