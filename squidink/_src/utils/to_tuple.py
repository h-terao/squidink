import jax.numpy as jnp


def to_tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if jnp.array(x).ndim == 0:
        x = (x, x)
    return x
