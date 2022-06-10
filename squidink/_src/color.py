"""Modify from: https://github.com/4rtemi5/imax/blob/master/imax/color_transforms.py"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import einops
import chex

from .utils import flatten, blend


__all__ = [
    "rgb2gray",
    "solarize",
    "solarize_add",
    "color",
    "contrast",
    "brightness",
    "posterize",
    "autocontrast",
    "equalize",
    "invert",
]


def rgb2gray(x: chex.Array) -> chex.Array:
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return jnp.stack([v, v, v], axis=-1)


def solarize(x: chex.Array, threshold: float = 0.5) -> chex.Array:
    return jnp.where(x < threshold, x, 1 - x)


def solarize_add(x: chex.Array, threshold: float = 0.5, addition: float = 0) -> chex.Array:
    return jnp.where(x < threshold, x + addition, x).clip(0, 1)


def color(x: chex.Array, factor: float) -> chex.Array:
    return blend(x, rgb2gray(x), factor)


def contrast(x: chex.Array, factor: float) -> chex.Array:
    degenerate = rgb2gray(x)
    degenerate = jnp.mean(degenerate, axis=(-1, -2, -3), keepdims=True)
    return blend(x, degenerate, factor)


def brightness(x: chex.Array, factor: float) -> chex.Array:
    degenerate = jnp.zeros_like(x)
    return blend(x, degenerate, factor)


def posterize(x: chex.Array, bits: int) -> chex.Array:
    shift = 8 - bits
    degenerate = (255 * x).astype(jnp.uint8)
    degenerate = jnp.left_shift(jnp.right_shift(degenerate, shift), shift)
    degenerate = degenerate.astype(x.dtype) / 255.0
    return x + jax.lax.stop_gradient(degenerate - x)  # STE.


def autocontrast(x: chex.Array) -> chex.Array:
    @jax.jit
    def scale_channel(carry, xi):
        low = jnp.min(xi)
        high = jnp.max(xi)

        def _scale_values(v: chex.Array) -> chex.Array:
            scale = 1.0 / (high - low)
            offset = -low * scale
            v = v * scale + offset
            return jnp.clip(v, 0, 1)

        xi = jax.lax.cond(high > low, _scale_values, lambda v: v, xi)
        return carry, xi

    x, unflatten = flatten(x)
    C = x.shape[-1]
    x = einops.rearrange(x, "B H W C -> (B C) H W")
    _, x = jax.lax.scan(scale_channel, jnp.zeros(()), x)
    x = einops.rearrange(x, "(B C) H W -> B H W C", C=C)
    x = unflatten(x)
    return x


def equalize(x: chex.Array) -> chex.Array:
    def build_lut(histo, step):
        lut = (jnp.cumsum(histo) + (step // 2)) // step
        lut = jnp.concatenate([jnp.array([0]), lut[:-1]], axis=0)
        return jnp.clip(lut, 0, 255)

    @jax.jit
    def scale_channel(carry, xi):
        new_xi = (xi * 255).astype(jnp.int32)
        histo = jnp.histogram(new_xi, bins=255, range=(0, 255))[0]
        last_nonzero = jnp.argmax(histo[::-1] > 0)
        step = (jnp.sum(histo) - jnp.take(histo[::-1], last_nonzero)) // 255

        new_xi = jax.lax.cond(
            step == 0,
            lambda x: x.astype("uint8"),
            lambda x: jnp.take(build_lut(histo, step), x).astype("uint8"),
            new_xi,
        )

        new_xi = (new_xi / 255).astype(xi.dtype)
        return carry, new_xi

    x, unflatten = flatten(x)
    C = x.shape[-1]

    degenerate = einops.rearrange(x, "B H W C -> (B C) H W")
    _, degenerate = jax.lax.scan(scale_channel, jnp.zeros(()), degenerate)
    degenerate = einops.rearrange(degenerate, "(B C) H W -> B H W C", C=C)

    out = x + jax.lax.stop_gradient(degenerate - x)  # STE.
    return unflatten(out)


def invert(x: chex.Array) -> chex.Array:
    return 1.0 - x
