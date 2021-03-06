from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import einops
import chex

from .utils import flatten, to_tuple


def _cutout_mask(rng: chex.PRNGKey, x: chex.Array, mask_size: int | tuple[int, int]) -> chex.Array:
    """Create cutout mask.

    Args:
        rng: JAX RNG key.
        x: Input array.
        mask_size: Mask shape.

    Return:
        Binary mask for cutout.
    """
    mask_size = to_tuple(mask_size)
    mask_size_half = (mask_size[0] // 2, mask_size[1] // 2)

    x, unflatten = flatten(x)
    N, H, W, C = x.shape

    mask = jnp.ones((H + mask_size[0], W + mask_size[1]))

    y_rng, x_rng = jr.split(rng)
    start_indices = [jr.randint(y_rng, (), 0, H + 1), jr.randint(x_rng, (), 0, W + 1)]
    mask = jax.lax.dynamic_update_slice(
        mask,
        update=jnp.zeros(mask_size),
        start_indices=start_indices,
    )

    mask = jax.lax.dynamic_slice(mask, mask_size_half, (H, W))
    mask = einops.repeat(mask, "H W -> N H W C", N=N, C=C)
    return unflatten(mask)


def cutout(
    rng: chex.PRNGKey, x: chex.Array, mask_size: int | tuple[int, int], cval: float = 0.5
) -> chex.Array:
    """Apply cutout to input array.

    Args:
        rng: JAX RNG key.
        x: Input array.
        mask_size: Mask shape.

    Return:
        Transformed array.
    """
    mask = _cutout_mask(rng, x, mask_size)
    return jnp.where(mask, x, jnp.full_like(x, cval))
