from __future__ import annotations
import os
import sys

import jax.random as jr
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
import squidink.functional as T


rng = jr.PRNGKey(1234)
img = jnp.array(Image.open("../figures/logo.png")) / 255.0
figs = {}
figs["original"] = img

# Crop
figs["random_crop"] = T.random_crop(rng, img, 128)
figs["center_crop"] = T.center_crop(img, 128)

# Flip
figs["hflip"] = T.hflip(img)
figs["vflip"] = T.vflip(img)

# Color
figs["solarize"] = T.solarize(img)
figs["solarize_add"] = T.solarize_add(img, addition=0.25)
figs["color"] = T.color(img, 2.0)
figs["contrast"] = T.contrast(img, 2.0)
figs["brightness"] = T.brightness(img, 2.0)
figs["posterize"] = T.posterize(img, 4)
figs["autocontrast"] = T.autocontrast(img)
figs["equalize"] = T.equalize(img)
figs["invert"] = T.invert(img)
figs["sharpness"] = T.sharpness(img, 2.0)

# Blur
figs["mean_blur"] = T.mean_blur(img, 1.0, 13)
figs["median_blur"] = T.median_blur(img, 1.0, 13)

# Geometry
figs["rotate"] = T.rotate(img, 30)
figs["rot90"] = T.rot90(img, 1)
figs["translate_x"] = T.translate_x(img, 30)
figs["translate_y"] = T.translate_y(img, 30)
figs["shear_x"] = T.shear_x(img, 30)
figs["shear_y"] = T.shear_y(img, 30)

# Salt pepper
figs["salt"] = T.salt(rng, img, 0.05)
figs["pepper"] = T.pepper(rng, img, 0.05)

# Cutout
figs["cutout"] = T.cutout(rng, img, 112)


# Draw.
row = 7
col = 4
plt.figure(figsize=(10, 10))
assert row * col >= len(figs)

for i, key in enumerate(figs):
    plt.subplot(row, col, i + 1)
    plt.imshow(figs[key])
    plt.title(key)
    plt.axis("off")

plt.tight_layout()
plt.savefig("transformed.png")
