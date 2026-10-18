"""v4（画框版）公共工具：skia 矢量遮罩、高精度模糊与噪声、Clawd 几何。
所有函数都用 float32、0..1；不经过 8 位中间结果，避免色带。"""
import math
import os
import sys

import numpy as np
import skia
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import (BODY, ARMS, LEGS, EYES, CLAWD_ORANGE, rigid_arm, check_proportions)  # noqa: E402,F401

FONT_LABEL = os.path.join(ROOT, "字体", "Noto_Serif_SC", "NotoSerifSC-Light.ttf")
W, H = 1920, 1080


# ---------- 数值工具 ----------
def gauss(a, sigma):
    if sigma <= 0:
        return a
    if a.ndim == 3:
        return ndimage.gaussian_filter(a, (sigma, sigma, 0), mode="nearest")
    return ndimage.gaussian_filter(a, sigma, mode="nearest")


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def value_noise(h, w, cell, seed):
    """单层平滑值噪声，cell 为特征尺寸（像素），返回约 0..1。"""
    g = np.random.default_rng(seed)
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    small = g.random((gh, gw)).astype(np.float32)
    zy = np.linspace(0, h / cell, h, dtype=np.float32) + 1
    zx = np.linspace(0, w / cell, w, dtype=np.float32) + 1
    yy, xx = np.meshgrid(zy, zx, indexing="ij")
    return ndimage.map_coordinates(small, [yy, xx], order=3, mode="nearest").astype(np.float32)


def fbm(h, w, cell, seed, octaves=4, gain=0.5):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(h, w, max(cell / (2 ** o), 1.5), seed + 97 * o)
        tot += amp
        amp *= gain
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def streak_noise(h, w, length, width, angle_deg, seed):
    """方向性条纹噪声（刷痕、纤维）：各向异性模糊白噪声。"""
    g = np.random.default_rng(seed)
    n = g.random((h, w)).astype(np.float32)
    a = math.radians(angle_deg)
    n = ndimage.rotate(n, -angle_deg, reshape=False, mode="wrap", order=1)
    n = ndimage.gaussian_filter(n, (width, length), mode="wrap")
    n = ndimage.rotate(n, angle_deg, reshape=False, mode="wrap", order=1)
    n = (n - n.mean()) / (n.std() + 1e-6)
    return n.astype(np.float32)


def over(dst, rgb, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return dst * (1 - a) + np.asarray(rgb, np.float32) * a


def shift(a, dy, dx):
    """非循环平移（亚像素）。"""
    if a.ndim == 3:
        return ndimage.shift(a, (dy, dx, 0), order=1, mode="constant", cval=0)
    return ndimage.shift(a, (dy, dx), order=1, mode="constant", cval=0)


def ishift(a, dy, dx):
    """整数像素非循环平移，空出来的部分补 0。"""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def downsample(a, k):
    if k == 1:
        return a
    h, w = a.shape[0] // k, a.shape[1] // k
    a = a[: h * k, : w * k]
    sh = (h, k, w, k) + a.shape[2:]
    return a.reshape(sh).mean(axis=(1, 3))


# ---------- skia 矢量遮罩 ----------
def mask(w, h, draw):
    """draw(canvas, paint)：在白色抗锯齿画笔下绘制，返回 0..1 的 alpha。"""
    surf = skia.Surface(w, h)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    p = skia.Paint(AntiAlias=True, Color=skia.ColorWHITE)
    draw(c, p)
    return surf.makeImageSnapshot().toarray()[..., 3].astype(np.float32) / 255


def path_from(pts, close=True, smooth_k=0.0):
    """点列 → skia.Path。smooth_k>0 时用 Catmull-Rom 转三次贝塞尔得到圆滑曲线。"""
    p = skia.Path()
    pts = [tuple(map(float, q)) for q in pts]
    if smooth_k <= 0 or len(pts) < 3:
        p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
    else:
        n = len(pts)
        p.moveTo(*pts[0])
        rng_ = range(n) if close else range(n - 1)
        for i in rng_:
            p0 = pts[(i - 1) % n] if (close or i > 0) else pts[i]
            p1, p2 = pts[i], pts[(i + 1) % n]
            p3 = pts[(i + 2) % n] if (close or i + 2 < n) else p2
            c1 = (p1[0] + (p2[0] - p0[0]) * smooth_k / 3, p1[1] + (p2[1] - p0[1]) * smooth_k / 3)
            c2 = (p2[0] - (p3[0] - p1[0]) * smooth_k / 3, p2[1] - (p3[1] - p1[1]) * smooth_k / 3)
            p.cubicTo(*c1, *c2, *p2)
    if close:
        p.close()
    return p


def poly_mask(w, h, pts, smooth_k=0.0):
    pa = path_from(pts, True, smooth_k)
    return mask(w, h, lambda c, p: c.drawPath(pa, p))


def stroke_mask(w, h, pts, width, smooth_k=1.0, cap=True):
    pa = path_from(pts, False, smooth_k)

    def d(c, p):
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(width)
        p.setStrokeCap(skia.Paint.kRound_Cap if cap else skia.Paint.kButt_Cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
        c.drawPath(pa, p)
    return mask(w, h, d)


def taper_stroke(pts, w0, w1, wmid=None):
    """变宽笔画（铁线描/剪纸细条）：沿折线两侧偏移，返回闭合多边形点列。"""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    t = np.linspace(0, 1, n)
    if wmid is None:
        wid = w0 + (w1 - w0) * t
    else:
        wid = np.where(t < 0.5, w0 + (wmid - w0) * t * 2, wmid + (w1 - wmid) * (t - 0.5) * 2)
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    left = pts + nrm * wid[:, None] / 2
    right = pts - nrm * wid[:, None] / 2
    return np.concatenate([left, right[::-1]], 0)


# ---------- Clawd 几何（格坐标 → 像素） ----------
def clawd_rects(arms=None, legs=None):
    arms = arms if arms is not None else ARMS
    legs = legs if legs is not None else LEGS
    return BODY, arms, legs


def rect_pts(r):
    x0, y0, x1, y1 = r
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def to_px(pts, ox, oy, S, ang=0.0, pivot=(8, 12)):
    """格坐标点 → 画布像素。(ox, oy) 是 pivot（默认脚底中心）在画布上的位置，ang 为整体旋转（度）。"""
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for x, y in pts:
        dx, dy = (x - pivot[0]) * S, (y - pivot[1]) * S
        out.append((ox + dx * ca - dy * sa, oy + dx * sa + dy * ca))
    return out


def clawd_union_mask(w, h, ox, oy, S, arms=None, legs=None, ang=0.0, ss_round=0.0):
    """Clawd 剪影（身体+手臂+腿 一个整体），ss_round 为圆角半径（像素）。"""
    body, arms, legs = clawd_rects(arms, legs)
    shapes = [body] + [a for a in arms] + [l for l in legs]

    def d(c, p):
        for r in shapes:
            if isinstance(r[0], tuple):          # rigid_arm 返回的是多边形
                c.drawPath(path_from(to_px(r, ox, oy, S, ang)), p)
            else:
                pts = to_px(rect_pts(r), ox, oy, S, ang)
                c.drawPath(path_from(pts), p)
    return mask(w, h, d)


def label_layer(text, size, w, h, cx, cy, tracking=0.18):
    """展签文字遮罩：思源宋体细体，字距 tracking×字号，水平居中于 cx，cy 为字身中线。"""
    tf = skia.Typeface.MakeFromFile(FONT_LABEL)
    font = skia.Font(tf, size)
    font.setSubpixel(True)
    font.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    adv = [font.measureText(ch) for ch in text]
    total = sum(adv) + tracking * size * (len(text) - 1)
    x = cx - total / 2
    base = cy + size * 0.36

    def d(c, p):
        xx = x
        for ch, a in zip(text, adv):
            c.drawString(ch, xx, base, font, p)
            xx += a + tracking * size
    return mask(w, h, d)
