from __future__ import annotations

import jax.numpy as jnp
import chex

from .utils import warp_perspective

__all__ = [
    "rotate",
    "rot90",
    "translate",
    "translate_x",
    "translate_y",
    "shear",
    "shear_x",
    "shear_y",
]


def rotate(
    x: chex.Array,
    angle: float,
    center: tuple[float, float] | None,
    order=0,
    mode="constant",
    cval=0,
) -> chex.Array:
    if center is None:
        *_, H, W, _ = x.shape
        center = ((H - 1) / 2, (W - 1) / 2)

    center_y, center_x = center
    angle = angle * jnp.pi / 180

    shift_x = center_x - center_x * jnp.cos(angle) + center_y * jnp.sin(angle)
    shift_y = center_y - center_x * jnp.sin(angle) - center_y * jnp.cos(angle)

    M = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), shift_x],
            [jnp.sin(angle), jnp.cos(angle), shift_y],
            [0, 0, 1],
        ],
        dtype=x.dtype,
    )
    return warp_perspective(x, M, order, mode, cval)


def rot90(x: chex.Array, n: int = 1) -> chex.Array:
    x = jnp.rot90(x, n, axes=(-3, -2))  # [..., H, W, C]
    return x


def translate(
    x: chex.Array,
    shift: tuple[int, int] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    cval: float = 0,
) -> chex.Array:
    shift_y, shift_x = shift
    M = jnp.array(
        [
            [1, 0, -shift_x],  # y-axis
            [0, 1, -shift_y],  # x-axis
            [0, 0, 1],
        ],
        dtype=x.dtype,
    )
    return warp_perspective(x, M, order, mode, cval)


def translate_x(
    x: chex.Array, shift: int, order: int = 0, mode: str = "constant", cval: float = 0
) -> chex.Array:
    return translate(x, (0, shift), order, mode, cval)


def translate_y(
    x: chex.Array, shift: int, order: int = 0, mode: str = "constant", cval: float = 0
) -> chex.Array:
    return translate(x, (shift, 0), order, mode, cval)


def shear(
    x: chex.Array,
    angles: tuple[float, float] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    cval: float = 0,
) -> chex.Array:
    angle_y, angle_x = angles
    angle_x = angle_x * jnp.pi / 180
    angle_y = angle_y * jnp.pi / 180

    M = jnp.array(
        [
            [1, jnp.tan(angle_x), 0],
            [jnp.tan(angle_y), 1, 0],
            [0, 0, 1],
        ],
        dtype=x.dtype,
    )
    return warp_perspective(x, M, order, mode, cval)


def shear_x(
    x: chex.Array, angle: float, order: int = 0, mode: str = "constant", cval: float = 0
) -> chex.Array:
    return shear(x, (0, angle), order, mode, cval)


def shear_y(
    x: chex.Array, angle: float, order: int = 0, mode: str = "constant", cval: float = 0
) -> chex.Array:
    return shear(x, (angle, 0), order, mode, cval)
