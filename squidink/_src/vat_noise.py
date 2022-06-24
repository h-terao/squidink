from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import chex


def vat_noise(
    rng: chex.PRNGKey,
    x: chex.Array,
    grad_fn: Callable,
    axis: int | list[int] = -1,
    eps: float = 1.0,
    num_iters: int = 1,
) -> chex.Array:
    """Noise for virtual adversarial training.

    Args:
        rng: A PRNG key.
        x: Input array.
        grad_fn: Gradient function that returns grads w.r.t noise.
        axis: A sample axis. For images, specify [-1, -2, -3].
        eps: Epsilon.
        num_iters: Number of steps to update the adversarial noise.

    Returns:
        Adversarial noises.

    Examples:
        >>> x = jnp.zeros((128, 10))
        >>> y = jnp.ones((128, 10))
        >>> grad_fn = jax.grad(lambda z: jnp.abs(x+z, y))
        >>> z = vat_noise(jr.PRNGKey(0), x, grad_fn)
    """

    def normalize(z: chex.Array) -> chex.Array:
        z /= jnp.linalg.norm(z, ord=2, axis=axis, keepdims=True) + 1e-6
        return z

    z = jr.normal(rng, x.shape, x.dtype)
    z, _ = jax.lax.scan(
        lambda z, _: (grad_fn(eps * normalize(z)), _),
        z,
        jnp.arange(num_iters),
    )
    return normalize(z)
