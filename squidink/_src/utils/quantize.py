import jax
import jax.numpy as jnp
import chex


def quantize(x: chex.Array) -> chex.Array:
    """Quantize float array."""
    quantized = jnp.floor(255 * x + 0.5) / 255.0
    return x + jax.lax.stop_gradient(quantized - x)
