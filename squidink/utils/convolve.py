from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import einops
import chex

from . import flatten


def convolve(
    x: chex.Array,
    kernel: chex.Array | Callable[[chex.Array], chex.Array],
    kernel_shape: int | tuple[int, int] | None = None,
) -> chex.Array:
    """Apply kernel or filter to image.

    Args:
        img: Image array that be applied to convolve operation.
        kernel: A kernel array or the callable object.
                If callable object, it must transform [N,H,W] array to [N] array,
                where (H, W) is the shape of kernel and N is the number of patches to convolve.
        kernel_shape: Height and width of kernel.
                      This value is required when you specify the callable object as kernel.
                      If kernel is array, this argument is ignored
                      and `kernel.shape` is used instead of it.

    Return:
        The convolved array.

    Example:
        Apply 9x9 median filter to images.
        >>> convolve(img, (9, 9), lambda x: jnp.median(x, axis=(1, 2)))
    """
    x, unflatten = flatten(x)
    C = x.shape[-1]

    if isinstance(kernel, jnp.ndarray):
        kernel_shape = kernel.shape
    if isinstance(kernel_shape, int):
        kernel_shape = [kernel_shape, kernel_shape]
    assert kernel_shape is not None

    col = jax.lax.conv_general_dilated_patches(
        einops.rearrange(x, "N H W C -> N C H W"),
        kernel_shape,
        window_strides=(1, 1),
        padding="SAME",
    )

    H, W = col.shape[-2:]
    col = einops.rearrange(
        col, "N (C kH kW) H W -> (N H W C) kH kW", kH=kernel_shape[0], kW=kernel_shape[1]
    )

    if isinstance(kernel, jnp.ndarray):
        kernel = kernel[None]
        degenerate = jnp.sum(kernel * col, axis=(-1, -2))
    else:
        degenerate = kernel(col)

    degenerate = einops.rearrange(degenerate, "(N H W C) -> N H W C", H=H, W=W, C=C)
    return unflatten(degenerate)
