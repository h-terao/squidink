from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import chex

from .utils import flatten, pad, to_tuple
from .flip import hflip, vflip


def random_crop(
    rng: chex.PRNGKey,
    x: chex.Array,
    crop_size: int | tuple[int, int],
    pad_size: int | tuple[int, int] = 0,
    mode: str = "constant",
    cval: chex.Scalar = 0,
    reflect_type: str = "even",
) -> chex.Array:
    """Crop a patch from the random position.

    Args:
        rng: A RNG key.
        x: Input data with dimensions (..., height, width, channel).
        crop_size: Desired size of output data.
        pad_size: If specified, padding input data.
        mode: Padding mode.
        cval: Constant value to pad. Only used when mode is constant.
        reflect_type (str): Reflection type. "even" or "odd".
            Only used when mode is reflect or summetric.

    Returns:
        Randomly cropped patch.

    """
    crop_size = to_tuple(crop_size)
    x = pad(x, pad_size, mode, cval, reflect_type)
    x, unflatten = flatten(x)
    N, H, W, C = x.shape

    rng_x, rng_y = jr.split(rng)
    y_offset = jr.randint(rng_y, (), 0, H - crop_size[0] + 1)
    x_offset = jr.randint(rng_x, (), 0, W - crop_size[1] + 1)

    slice_sizes = (N, crop_size[0], crop_size[1], C)
    x = jax.lax.dynamic_slice(x, (0, y_offset, x_offset, 0), slice_sizes)

    return unflatten(x)


def center_crop(
    x: chex.Array,
    crop_size: int | tuple[int, int],
    pad_size: int | tuple[int, int] = 0,
    mode: str = "constant",
    cval: chex.Scalar = 0,
    reflect_type: str = "even",
) -> chex.Array:
    """Crop a central patch.

    Args:
        x: Input data with dimensions (..., height, width, channel).
        crop_size: Desired size of output data.
        pad_size: If specified, padding input data.
        mode: Padding mode.
        cval: Constant value to pad. Only used when mode is constant.
        reflect_type (str): Reflection type. "even" or "odd".
            Only used when mode is reflect or summetric.

    Returns:
        Central patch.

    """
    crop_size = to_tuple(crop_size)

    x = pad(x, pad_size, mode, cval, reflect_type)
    x, unflatten = flatten(x)
    N, H, W, C = x.shape

    y_offset = (H - crop_size[0] + 1) // 2
    x_offset = (W - crop_size[1] + 1) // 2

    slice_sizes = (N, crop_size[0], crop_size[1], C)
    x = jax.lax.dynamic_slice(x, (0, y_offset, x_offset, 0), slice_sizes)

    return unflatten(x)


def five_crop(
    x: chex.Array,
    crop_size: int | tuple[int, int],
    pad_size: int | tuple[int, int] = 0,
    new_axis: int = 0,
    mode: str = "constant",
    cval: chex.Scalar = 0,
    reflect_type: str = "even",
) -> chex.Array:
    """Crop five patches from the fiven image.

    Args:
        new_axis (int): Axis index to stack the cropped patches.
    """
    crop_size = to_tuple(crop_size)

    x = pad(x, pad_size, mode, cval, reflect_type)
    x, unflatten = flatten(x)
    N, H, W, C = x.shape

    slice_sizes = (N, crop_size, crop_size, C)

    # Crop five patches.
    upper_left = unflatten(jax.lax.dynamic_slice(x, (0, 0, 0, 0), slice_sizes))
    lower_left = unflatten(jax.lax.dynamic_slice(x, (0, H - crop_size[0], 0, 0), slice_sizes))
    center = center_crop(x, crop_size)
    upper_right = unflatten(jax.lax.dynamic_slice(x, (0, 0, W - crop_size[1], 0), slice_sizes))
    lower_right = unflatten(
        jax.lax.dynamic_slice(x, (0, H - crop_size[0], W - crop_size[1], 0), slice_sizes)
    )

    return jnp.stack([upper_left, lower_left, center, upper_right, lower_right], axis=new_axis)


def ten_crop(
    x: chex.Array,
    crop_size: int | tuple[int, int],
    pad_size: int | tuple[int, int] = 0,
    new_axis: int = 0,
    vertical: bool = False,
    mode: str = "constant",
    cval: chex.Scalar = 0,
    reflect_type: str = "even",
) -> chex.Array:
    """Crop five patches from the fiven image.

    Args:
        new_axis (int): Axis index to stack the cropped patches.
        vertical (bool): If True, vertically flip the given array, and apply five_crop to it.
            Otherwise, flip the given array horizontally.
    """
    flip_x = vflip(x) if vertical else hflip(x)
    return jnp.concatenate(
        [
            five_crop(x, crop_size, pad_size, new_axis, mode, cval, reflect_type),
            five_crop(flip_x, crop_size, pad_size, new_axis, mode, cval, reflect_type),
        ],
        new_axis,
    )
