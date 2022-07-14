from __future__ import annotations

import jax.numpy as jnp
from jax.scipy import ndimage as ndi
import einops
import chex

from .flatten import flatten


def warp_perspective(
    x: chex.Array, M: chex.Array, order: int = 0, mode: str = "constant", cval: float = 0
) -> chex.Array:
    x, unflatten = flatten(x)
    _, H, W, C = x.shape

    x = einops.rearrange(x, "B H W C -> (B C) H W")
    N = len(x)

    x_t, y_t = jnp.meshgrid(jnp.arange(0, W), jnp.arange(0, H))
    pixel_coords = jnp.stack([x_t, y_t, jnp.ones_like(x_t)]).astype(jnp.float32)
    x_coords, y_coords, _ = jnp.einsum("ij,jkl->ikl", M, pixel_coords)

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
