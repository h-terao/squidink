"""Optical flow utilities."""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy import ndimage as ndi
import chex
import einops

from .utils import flatten


def warp_by_flow(
    x: chex.Array, flow: chex.Array, order: int = 0, mode: str = "constant", cval: float = 0
) -> chex.Array:
    """Warp images by flow.

    Args:
        x: Image array that has a shape of [..., H, W, C].
        flow: Flow array that has a shape of [H, W, 2].
        order: Interpolation order.
        mode: Padding mode.
        cval: Color value.

    Returns:
        Estimated previous frame of x warped from flow.
    """
    x, unflatten = flatten(x)
    _, H, W, C = x.shape

    x = einops.rearrange(x, "B H W C -> (B C) H W")
    N = len(x)

    y_t, x_t = jnp.mgrid[:H, :W]
    y_coords = y_t + H * flow[..., 1]
    x_coords = x_t + W * flow[..., 0]
    coords_to_map = jnp.stack(
        [
            einops.repeat(jnp.arange(N), "N -> N H W", H=H, W=W),
            einops.repeat(y_coords, "H W -> N H W", N=N),
            einops.repeat(x_coords, "H W -> N H W", N=N),
        ],
        axis=0,
    )

    x = ndi.map_coordinates(x, coords_to_map, order, mode, cval)
    x = einops.rearrange(x, "(B C) H W -> B H W C", C=C)
    return unflatten(x)
