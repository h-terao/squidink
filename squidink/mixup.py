from __future__ import annotations
import jax.random as jr
import chex


def mixup(
    rng: chex.PRNGKey,
    x: chex.Array,
    y: chex.Array,
    beta: float = 0.5,
) -> tuple[chex.Array, chex.Array]:
    """Apply mixup to arrays."""
    batch_size = x.shape[0]
    perm_rng, mix_rng = jr.split(rng)

    index = jr.permutation(perm_rng, batch_size)
    new_x, new_y = x[index], y[index]

    v = jr.beta(mix_rng, beta, beta, dtype=x.dtype)
    new_x = v * x + (1 - v) * new_x
    new_y = v * y + (1 - v) * new_y

    return new_x, new_y
