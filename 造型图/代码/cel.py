"""06 赛璐璐风格的可复用部分（与 S06_有趣.py 的画法一致，供动画调用）。"""
import math
import numpy as np
from PIL import Image, ImageDraw
from common import *

SS = 2
INK = np.array([0.06, 0.05, 0.08])
ORANGE = np.array(CLAWD_ORANGE) / 255
ORANGE_SH = ORANGE * np.array([0.78, 0.66, 0.68])


def erode(m, r):
    return np.clip((blur(m, r) - 0.5) * 6 + 0.5, 0, 1) * (m > 0.5)


def cel_clawd(S, parts, light_split=11.0, eyes_closed=None, back=False, outline=0.32, arms=None, legs=None, bbox=(-4, -3, 21, 13)):
    extra = {name: shapes for name, shapes, _, _ in parts}
    closed = [(3.2, 3.8, 5.0, 4.3), (11.0, 3.8, 12.8, 4.3)]
    extra["closed"] = closed if eyes_closed == "both" else closed[:1] if eyes_closed == "left" else [(0, 0, 0, 0)]
    cm, (ox, oy), (pw, ph) = clawd_masks(S, int(S * 2), arms=arms, legs=legs, bbox=bbox, extra=extra)
    gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
    gx = (gx_ - ox) / S
    body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
    union = body.copy()
    col = np.zeros((ph, pw, 3), np.float32)
    col[:] = ORANGE
    col = np.where((gx > light_split)[..., None] & (body > 0.5)[..., None], ORANGE_SH, col)
    edges = np.clip(body - erode(body, 1.2), 0, 1)
    for name, shapes, c, sh in parts:
        m = cm[name]
        cc = np.where((gx > light_split)[..., None], np.array(sh), np.array(c))
        col = over(col, cc, m)
        union = np.maximum(union, m)
        edges = np.maximum(edges, np.clip(m - erode(m, 1.0), 0, 1))
    if not back:
        e = {None: cm["eyes"], "left": cm["eyes"] * (gx > 8), "both": 0 * cm["eyes"]}[eyes_closed]
        col = over(col, INK, e)
        col = over(col, INK, cm["closed"])
    out = np.clip(blur(union, S * outline * 0.45) * 5, 0, 1)
    rgba = np.dstack([np.ones((ph, pw, 3)) * INK, out])
    rgba[..., :3] = over(rgba[..., :3], col, union)
    rgba[..., :3] = over(rgba[..., :3], INK, edges * 0.9)
    return rgba


def pearls_shapes(n_rows=4):
    shapes = []
    for k in range(n_rows):
        y0 = 5.4 + k * 0.55
        sag = 0.5 + k * 0.35
        for i in range(15):
            t = i / 14
            cx, cy = 1.2 + 13.6 * t, y0 + sag * (1 - (2 * t - 1) ** 2)
            shapes.append([(cx + 0.28 * math.cos(a), cy + 0.28 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 10)])
    return shapes


SKIRT = [(-0.3, 6.6), (16.3, 6.6), (17.6, 9.0), (19.0, 11.6), (17.0, 12.0), (14.0, 11.7), (11.0, 12.0), (8.0, 11.7),
         (5.0, 12.0), (2.0, 11.7), (-1.0, 12.0), (-3.0, 11.6), (-1.6, 9.0)]
SKIRT_FOLDS = [[(sx, 7.0), (sx + 0.15, 7.2), (sx * 1.15 - 1.2 + 0.15, 11.8), (sx * 1.15 - 1.2, 11.8)] for sx in (3.0, 6.0, 10.0, 13.0)]
YELLOW, YELLOW_SH = (1.0, 0.80, 0.10), (0.92, 0.55, 0.06)
PEARL, PEARL_SH = (0.98, 0.96, 0.92), (0.82, 0.8, 0.82)
check_proportions(extra={"skirt": [SKIRT]})


def outfit_parts():
    return [("skirt", [SKIRT], YELLOW, YELLOW_SH), ("folds", SKIRT_FOLDS, YELLOW_SH, (0.8, 0.42, 0.05)),
            ("pearls", pearls_shapes(), PEARL, PEARL_SH)]


def star4(img, sx, sy, r, color=(1.0, 0.98, 0.9)):
    pts = []
    for i in range(8):
        a = i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.22
        pts.append((sx + rr * math.cos(a), sy + rr * math.sin(a)))
    im = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(im).polygon([(px * SS, py * SS) for px, py in pts], fill=255)
    return over(img, np.array(color), to_arr(im.resize((W, H), Image.LANCZOS)))


_cache = {}


def corridor(with_crew=True):
    """06-a 的镜面长廊背景；with_crew=True 时画上远处 5 个镜像（不含最前面那只，由动画里的主角代替）。"""
    key = ("corridor", with_crew)
    if key in _cache:
        return _cache[key]
    x, y = grid()
    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.array([0.05, 0.05, 0.1])
    img += np.clip(1 - y / H, 0, 1)[..., None] * np.array([0.04, 0.05, 0.12])
    FLOOR_Y, VX, VY = 820, 1700, 560
    img = np.where((y > FLOOR_Y)[..., None], np.array([0.09, 0.08, 0.14]), img)
    seams = [80 + (VX - 80) * (k / 8) ** 1.6 for k in range(9)]
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for k, sx in enumerate(seams):
        top = 40 + (VY - 260 - 40) * (k / 8) ** 1.6
        bot = H + 10 - (H - VY - 60) * (k / 8) ** 1.6
        d.line([(sx * SS, top * SS), (sx * SS, bot * SS)], fill=255, width=int(max(2, 6 - k * 0.5) * SS))
    img = over(img, np.array([0.55, 0.58, 0.7]), to_arr(im.resize((W, H), Image.LANCZOS)))
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for k in range(len(seams) - 1):
        a_, b_ = seams[k], seams[k + 1]
        w_ = b_ - a_
        d.polygon([((a_ + w_ * 0.15) * SS, 0), ((a_ + w_ * 0.32) * SS, 0), ((a_ + w_ * 0.05) * SS, FLOOR_Y * SS), ((a_ - w_ * 0.12) * SS, FLOOR_Y * SS)], fill=255)
    img = img + to_arr(im.resize((W, H), Image.LANCZOS))[..., None] * np.array([0.05, 0.06, 0.1])
    for k, sx in enumerate(seams[:-1]):
        img = star4(img, sx, 70 + (VY - 300) * (k / 8) ** 1.6, 26 - k * 2)
    if with_crew:
        crew = [(730, 860, 18, 0, dict(arms=[(-1, 3, 0, 5), rigid_arm("R", 75)])), (1040, 820, 14, 10, {}),
                (1280, 790, 11, 0, dict(back=True)), (1460, 770, 9, 0, dict(eyes_closed="left")), (1640, 752, 8, -6, {})]
        for (cx, foot, S, ang, kw) in crew[::-1]:
            spr = cel_clawd(S, outfit_parts(), **kw)
            cy = foot - 7 * S
            refl = spr[::-1].copy()
            refl[..., 3] *= 0.22
            img = paste_rgba(img, refl, cx, foot + (foot - cy) - 0.2 * S, -ang)
            img = paste_rgba(img, spr, cx, cy, ang)
    _cache[key] = img
    return img
