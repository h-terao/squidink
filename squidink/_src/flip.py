from __future__ import annotations
import functools

import jax
import jax.random as jr
import chex


__all__ = ["flip", "hflip", "vflip", "random_flip", "random_hflip", "random_vflip"]


def flip(x: chex.Array, horizontal: bool = False, vertical: bool = False) -> chex.Array:
    if vertical:
        x = x[..., ::-1, :, :]

    if horizontal:
        x = x[..., ::-1, :]

    return x


def hflip(x: chex.Array) -> chex.Array:
    """Alias of flip(x, horizontal=True, vertical=False)."""
    return flip(x, horizontal=True)


def vflip(x: chex.Array) -> chex.Array:
    """Alias of flip(x, horizontal=False, vertical=True)."""
    return flip(x, vertical=True)


def random_flip(
    rng: chex.PRNGKey,
    x: chex.Array,
    p: float = 0.5,
    horizontal: bool = False,
    vertical: bool = False,
) -> chex.Array:
    return jax.lax.cond(
        jr.uniform(rng) < p,
        functools.partial(flip, horizontal=horizontal, vertical=vertical),
        lambda x: x,
        x,
    )


def random_hflip(rng: chex.PRNGKey, x: chex.Array, p: float = 0.5) -> chex.Array:
    return random_flip(rng, x, p, horizontal=True)


def random_vflip(rng: chex.PRNGKey, x: chex.Array, p: float = 0.5) -> chex.Array:
    return random_flip(rng, x, p, vertical=True)
