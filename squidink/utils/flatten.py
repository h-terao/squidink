from __future__ import annotations
from typing import Callable
import chex


def flatten(x: chex.Array) -> tuple[chex.Array, Callable]:
    """Reshape array from [..., H, W, C] to [N, H, W, C]."""
    *batch_shape, H, W, C = x.shape
    x = x.reshape(-1, H, W, C)

    def unflatten(x: chex.Array) -> chex.Array:
        _, H, W, C = x.shape
        return x.reshape(*batch_shape, H, W, C)

    return x, unflatten
