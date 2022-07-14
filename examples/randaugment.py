from __future__ import annotations
from typing import Callable
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import chex

from . import functional as F


def default_augment_space(num_bins: int) -> dict[str, tuple[chex.Array, bool]]:
    return {
        "ShearX": (jnp.linspace(0, 0.3, num_bins), True),
        "ShearY": (jnp.linspace(0, 0.3, num_bins), True),
        "TranslateX": (jnp.linspace(0, 150.0 / 331.0, num_bins), True),
        "TranslateY": (jnp.linspace(0, 150.0 / 331.0, num_bins), True),
        "Rotate": (jnp.linspace(0, 30, num_bins), True),
        "Brightness": (jnp.linspace(0, 0.9, num_bins), True),
        "Color": (jnp.linspace(0, 0.9, num_bins), True),
        "Contrast": (jnp.linspace(0, 0.9, num_bins), True),
        "Sharpness": (jnp.linspace(0, 0.9, num_bins), True),
        "Posterize": (8 - jnp.round(jnp.arange(num_bins) / (num_bins - 1) / 4), False),
        "Solarize": (jnp.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (jnp.zeros(num_bins), False),
        "Equalize": (jnp.zeros(num_bins), False),
        "Invert": (jnp.zeros(num_bins), False),
        "Identity": (jnp.zeros(num_bins), False),
    }


def build_randaugment(
    num_layers: int,
    num_bins: int | None = None,
    augment_space: dict[str, tuple[chex.Array, bool]] | None = None,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.5,
) -> Callable[[chex.PRNGKey, chex.Array], chex.Array]:
    """Create the RandAugment function.

    Args:
        num_layers (int): Number of operations to transform images.
        num_bins (int): Number of bins.
        augment_space (dict): Augmentation space.
        order (int): The order of the spline interpolation.
        mode (str): The mode parameter determines how the input array is
            extended beyond its boundaries. "reflect", "grid-mirror", "constant",
            "grid-contant", "nearest", "mirror", "grid-wrap", and "wrap" are supported.
        cval (float): Value to fill past edges of input if mode is "constant".

    Return:
        RangAugment function.

    Example:
        >>> x = jnp.zeros((16, 128, 128, 3))  # 16 images.
        >>> randaugment = build_randaugment(...)
        >>> x = randaugment(rng, x)  # Apply same augmentation to images.
        >>> rngs = jr.split(len(x))
        >>> x = jax.vmap(randaugment)(rngs, x)  # Apply different augmentation to images.
    """

    if augment_space is None:
        assert num_bins is not None, "Specify num_bins when augment_space=None."
        augment_space = default_augment_space(num_bins)

    # Check num_bins.
    for key, val in augment_space.items():
        magnitudes, signed = val
        _num_bins = len(magnitudes)
        if num_bins is None:
            num_bins = _num_bins
        assert num_bins == _num_bins, f"{key} has different number of magnitude bins."

    @jax.jit
    def shear_x(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.shear_x(x, v, order=order, mode=mode, cval=cval)

    @jax.jit
    def shear_y(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.shear_y(x, v, order=order, mode=mode, cval=cval)

    @jax.jit
    def translate_x(x: chex.Array, idx: int, magnitudes: chex.Array):
        width = x.shape[1]
        v = width * magnitudes[idx]
        return F.translate_x(x, v, order=order, mode=mode, cval=cval)

    @jax.jit
    def translate_y(x: chex.Array, idx: int, magnitudes: chex.Array):
        height = x.shape[0]
        v = height * magnitudes[idx]
        return F.translate_y(x, v, order=order, mode=mode, cval=cval)

    @jax.jit
    def rotate(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.rotate(x, v, order=order, mode=mode, cval=cval)

    @jax.jit
    def brightness(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.brightness(x, v)

    @jax.jit
    def color(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.color(x, v)

    @jax.jit
    def contrast(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.contrast(x, v)

    @jax.jit
    def sharpness(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = 1.0 + magnitudes[idx]
        return F.sharpness(x, v)

    @jax.jit
    def posterize(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = jnp.int32(magnitudes[idx])
        return F.posterize(x, v)

    @jax.jit
    def solarize(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.solarize(x, v)

    @jax.jit
    def solarize_add(x: chex.Array, idx: int, magnitudes: chex.Array):
        v = magnitudes[idx]
        return F.solarize_add(x, addition=v)

    @jax.jit
    def autocontrast(x: chex.Array, idx: int, magnitudes: chex.Array):
        return F.autocontrast(x)

    @jax.jit
    def equalize(x: chex.Array, idx: int, magnitudes: chex.Array):
        return F.equalize(x)

    @jax.jit
    def invert(x: chex.Array, idx: int, magnitudes: chex.Array):
        return F.invert(x)

    @jax.jit
    def identity(x: chex.Array, idx: int, magnitudes: chex.Array):
        return x

    operations = {
        "ShearX": shear_x,
        "ShearY": shear_y,
        "TranslateX": translate_x,
        "TranslateY": translate_y,
        "Rotate": rotate,
        "Brightness": brightness,
        "Color": color,
        "Contrast": contrast,
        "Sharpness": sharpness,
        "Posterize": posterize,
        "Solarize": solarize,
        "SolarizeAdd": solarize_add,
        "AutoContrast": autocontrast,
        "Equalize": equalize,
        "Invert": invert,
        "Identity": identity,
    }

    branches = []
    for key, (magnitudes, signed) in augment_space.items():
        op = operations[key]
        if signed:
            magnitudes = jnp.concatenate([magnitudes, -magnitudes])
        else:
            magnitudes = jnp.concatenate([magnitudes, magnitudes])
        branches.append(functools.partial(op, magnitudes=magnitudes))

    def body(carry, item):
        op_idx, mag_idx = item
        carry = jax.lax.switch(op_idx, branches, carry, mag_idx)
        return carry, jnp.array(0)

    @jax.jit
    def fun(rng: chex.PRNGKey, x: chex.Array) -> chex.Array:
        op_rng, mag_rng = jr.split(rng)
        op_idxs = jr.randint(op_rng, [num_layers], 0, len(branches))
        mag_idxs = jr.randint(mag_rng, [num_layers], 0, 2 * num_bins)
        new_x, _ = jax.lax.scan(body, x, xs=[op_idxs, mag_idxs])
        return new_x

    return fun


if __name__ == "__main__":
    pass
