"""羊毛毡 / 黏土质感（与 B 图一致）：毛糙边缘、鼓起的明暗、细纤维噪声。"""
import numpy as np
from PIL import Image, ImageDraw
from common import blur, to_arr, fbm, rng, shift, over, W, H

_n = rng(21).normal(0, 1, (H, W)).astype(np.float32)
FINE = np.clip(blur(np.clip(_n * 0.18 + 0.5, 0, 1), 0.8) * 1.6 - 0.3, 0, 1)
FINE = 0.6 * FINE + 0.4 * fbm(H, W, 23, octaves=2, base=90)
SS = 2


def felt(rgba, seed=0, puff=10, fuzz=0.25, light=0.35, sh=6):
    rgb, a = rgba[..., :3], rgba[..., 3]
    n = np.roll(FINE, (seed * 37) % H, axis=0)
    a2 = np.clip(blur(a, 1.4) + (n - 0.5) * fuzz * 2.2 * (1 - np.abs(a * 2 - 1)), 0, 1)
    ab = blur(a, puff)
    emb = ab - shift(ab, sh, sh)
    shade = 0.82 + 0.25 * ab + light * emb
    rgb = rgb * shade[..., None] * (0.92 + 0.16 * n[..., None])
    return np.concatenate([rgb, a2[..., None]], -1)


def layer():
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def done(im):
    return to_arr(im.resize((W, H), Image.LANCZOS))


def comp(dst, rgba):
    return over(dst, rgba[..., :3], rgba[..., 3])


def C(c):
    return tuple(int(v * 255) if max(c) <= 1 else int(v) for v in c) + (255,)
