from __future__ import annotations
import jax
import jax.numpy as jnp
import chex


def quantize(x: chex.Array) -> chex.Array:
    """Quantize float array as an uint8 array.

    Args:
        x: Input array

    Returns:
        Quantized array.
    """
    quantized = jnp.floor(255 * x + 0.5) / 255.0
    quantized = jnp.clip(quantized, 0, 1)
    return x + jax.lax.stop_gradient(quantized - x)
