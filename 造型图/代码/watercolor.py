"""透明水彩：颜料按减色法叠在纸上；湿边积色、颗粒沉淀、晕染外渗、铅笔底稿。"""
import numpy as np
from PIL import Image, ImageDraw
from common import blur, to_arr, fbm, rng, W, H

_cache = {}


def paper(h=H, w=W, seed=500):
    """冷压水彩纸：暖白 + 细纹理。返回 (纸色, 纹理)。"""
    tex = 0.6 * fbm(h, w, seed, octaves=2, base=220) + 0.4 * fbm(h, w, seed + 1, octaves=3, base=60)
    col = np.ones((h, w, 3), np.float32) * np.array([0.985, 0.97, 0.94])
    col = col * (0.96 + 0.04 * tex[..., None])
    return col, tex


def _noise(seed, base):
    k = (seed, base)
    if k not in _cache:
        _cache[k] = fbm(H, W, seed, octaves=4, base=base)
    return _cache[k]


def wash(canvas, mask, color, density=0.75, seed=0, bleed=4.0, edge=0.55, gran=0.25, tex=None, var=0.25):
    """一层水彩：mask(0..1) 为笔刷覆盖区域；color 为颜料色（越深越浓）。
    只在遮罩外接框（加上晕染余量）里计算，结果与整幅计算一致。"""
    ys, xs = np.nonzero(mask > 1e-3)
    if len(ys) == 0:
        return canvas
    pad = int(3 * bleed + 24)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    if (y1 - y0) * (x1 - x0) < 0.8 * mask.size:
        out = canvas.copy()
        out[y0:y1, x0:x1] = _wash(canvas[y0:y1, x0:x1], mask[y0:y1, x0:x1], color, density, seed, bleed, edge, gran,
                                  None if tex is None else tex[y0:y1, x0:x1], var, (y0, y1, x0, x1))
        return out
    return _wash(canvas, mask, color, density, seed, bleed, edge, gran, tex, var, None)


def _wash(canvas, mask, color, density, seed, bleed, edge, gran, tex, var, box):
    n = _noise(seed % 50 + 600, 9)
    n2 = _noise(seed % 50 + 700, 4)
    if box is not None:
        y0, y1, x0, x1 = box
        n, n2 = n[y0:y1, x0:x1], n2[y0:y1, x0:x1]
    m = blur(mask, bleed)
    m = np.clip((m + (n - 0.5) * 0.35 - 0.25) * 1.8, 0, 1)   # 不规则外渗的边
    rim = np.clip(m - blur(m, 5), 0, 1) * 2.2                  # 湿边积色
    d = m * density * (1 - var * 0.5 + var * n2) + rim * edge * density
    if tex is not None:
        d = d * (1 + gran * (tex - 0.5) * 2)                    # 颜料颗粒落在纸纹凹处
    d = np.clip(d, 0, 1)[..., None]
    c = np.array(color, np.float32)
    return canvas * (1 - d * (1 - c))


def wet_bloom(canvas, cx, cy, r, color, seed=0, density=0.6, tex=None):
    """湿画法的“花椰菜”水渍：中心淡、边缘有不规则深色环。"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    X0, X1, Y0, Y1 = int(max(cx - r * 1.6, 0)), int(min(cx + r * 1.6, W)), int(max(cy - r * 1.6, 0)), int(min(cy + r * 1.6, H))
    sub = canvas[Y0:Y1, X0:X1]
    xs, ys = x[Y0:Y1, X0:X1] - cx, y[Y0:Y1, X0:X1] - cy
    ang = np.arctan2(ys, xs)
    g = rng(seed)
    ph = g.uniform(0, 6, 3)
    rr = r * (1 + 0.12 * np.sin(ang * 5 + ph[0]) + 0.07 * np.sin(ang * 11 + ph[1]) + 0.05 * np.sin(ang * 17 + ph[2]))
    d = np.sqrt(xs ** 2 + ys ** 2) / rr
    body = np.clip(1 - d, 0, 1) ** 0.5 * 0.55
    ring = np.exp(-((d - 0.93) / 0.07) ** 2)
    dens = np.clip((body + ring * 0.9) * density, 0, 1)
    if tex is not None:
        dens = dens * (1 + 0.3 * (tex[Y0:Y1, X0:X1] - 0.5) * 2)
    c = np.array(color, np.float32)
    canvas[Y0:Y1, X0:X1] = sub * (1 - np.clip(dens, 0, 1)[..., None] * (1 - c))
    return canvas


def pencil(canvas, polylines, seed=0, alpha=0.28, width=2):
    """铅笔底稿线：轻微抖动、断续。"""
    g = rng(seed)
    im = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(im)
    for pl in polylines:
        pts = [((px + g.normal(0, 0.8)) * 2, (py + g.normal(0, 0.8)) * 2) for px, py in pl]
        d.line(pts, fill=int(255 * g.uniform(0.6, 1)), width=width * 2, joint="curve")
    a = to_arr(im.resize((W, H), Image.LANCZOS)) * alpha
    return canvas * (1 - a[..., None] * np.array([0.65, 0.66, 0.7]))


def gouache_dots(canvas, mask, count, seed=0, r=(1.5, 3.5), color=(1, 1, 1)):
    """不透明白色水粉点（亮片高光）。"""
    g = rng(seed)
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return canvas
    idx = g.choice(len(xs), size=min(count, len(xs)), replace=False)
    im = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(im)
    for i in idx:
        rr = g.uniform(*r)
        d.ellipse([(xs[i] - rr) * 2, (ys[i] - rr) * 2, (xs[i] + rr) * 2, (ys[i] + rr) * 2], fill=255)
    a = to_arr(im.resize((W, H), Image.LANCZOS))[..., None] * 0.92
    return canvas * (1 - a) + np.array(color) * a
