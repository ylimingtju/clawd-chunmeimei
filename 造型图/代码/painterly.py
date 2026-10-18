"""油画厚涂：把清晰底图转成多层笔触，并按笔触厚度做凹凸光照。"""
import math
import numpy as np
from PIL import Image, ImageDraw
from common import blur, to_arr, to_img, rng, shift, fbm


def flow_field(base, smooth=12):
    """按底图结构算笔触方向：沿着边缘走（垂直于梯度）。"""
    lum = base.mean(-1)
    ls = blur(lum, smooth / 3)
    gy, gx = np.gradient(ls)
    # 结构张量平滑，避免方向抖动
    jxx, jyy, jxy = blur(gx * gx, smooth), blur(gy * gy, smooth), blur(gx * gy, smooth)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    mag = np.sqrt(gx * gx + gy * gy)
    return ang, mag


def paint(base, passes, seed=0, default_angle=None, height_gain=1.0, scale_map=None, jitter=0.08):
    """passes: [(数量, 长度范围, 宽度范围, 只画边缘处?)]。返回 (画面, 厚度图)。"""
    H, W = base.shape[:2]
    g = rng(seed)
    ang, mag = flow_field(base)
    if default_angle is not None:
        # 结构弱的地方用给定方向（比如天空的横向笔触）
        w = np.clip(mag / (np.percentile(mag, 75) + 1e-6), 0, 1)
        da = default_angle if np.ndim(default_angle) else np.full_like(ang, default_angle)
        ang = np.arctan2(np.sin(ang) * w + np.sin(da) * (1 - w), np.cos(ang) * w + np.cos(da) * (1 - w))
    canvas = Image.fromarray((np.clip(blur(base, 6), 0, 1) * 255).astype(np.uint8))
    height = Image.new("L", (W, H), 0)
    dc = ImageDraw.Draw(canvas)
    dh = ImageDraw.Draw(height)
    edge_p = mag / mag.sum()
    for (n, (l0, l1), (w0, w1), edges_only) in passes:
        if edges_only:
            idx = g.choice(H * W, size=n, p=edge_p.ravel())
            ys, xs = idx // W, idx % W
        else:
            ys, xs = g.integers(0, H, n), g.integers(0, W, n)
        for yy, xx in zip(ys, xs):
            c = base[yy, xx]
            c = np.clip(c * g.uniform(1 - jitter, 1 + jitter) + g.normal(0, 0.015, 3), 0, 1)
            sc = 1.0 if scale_map is None else float(scale_map[yy, xx])
            a = ang[yy, xx] + g.normal(0, 0.12)
            L = g.uniform(l0, l1) / 2 * sc
            dx, dy = math.cos(a) * L, math.sin(a) * L
            wdt = max(1, int(g.uniform(w0, w1) * sc))
            p0, p1 = (xx - dx, yy - dy), (xx + dx, yy + dy)
            dc.line([p0, p1], fill=tuple(int(v * 255) for v in c), width=wdt)
            dh.line([p0, p1], fill=int(g.uniform(90, 255)), width=wdt)
    img = to_arr(canvas)
    h = to_arr(height)
    return img, h


def impasto(img, h, strength=0.9, light=(-1, -1)):
    """按厚度图做凹凸：受光面提亮、背光面压暗，另加画布纹理。"""
    hb = blur(h, 1.2)
    lx, ly = light
    emb = hb - shift(hb, 2 * ly, 2 * lx)
    H, W = h.shape
    weave_y, weave_x = np.mgrid[0:H, 0:W].astype(np.float32)
    canvas_tex = 0.5 + 0.25 * np.sin(weave_x * 1.7) * np.sin(weave_y * 1.7)
    out = img * (1 + strength * emb[..., None] * 0.9) + np.clip(emb, 0, 1)[..., None] * 0.08 * strength
    out = out * (0.97 + 0.04 * canvas_tex[..., None])
    return np.clip(out, 0, 1)
