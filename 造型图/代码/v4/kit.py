"""各画心脚本共用的绘图小工具：画心 2 倍分辨率坐标、遮罩、曲线、勾边、Clawd 遮罩。"""
import math

import numpy as np
from scipy import ndimage

from common4 import (poly_mask, stroke_mask, mask, path_from, taper_stroke, clawd_union_mask, to_px, rect_pts,
                     rigid_arm, check_proportions, LEGS, gauss, smooth, fbm, value_noise, ishift, over, downsample)
import frame

K = 2
PW, PH = frame.PW * K, frame.PH * K
Y, X = np.mgrid[0:PH, 0:PW].astype(np.float32)
EYES_G = ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5))


def P(pts):
    return [(x * K, y * K) for x, y in pts]


def M(pts, smooth_k=0.0):
    """1 倍画心坐标点列 → 2 倍分辨率遮罩。"""
    return poly_mask(PW, PH, P(pts), smooth_k)


def SM(pts, width, smooth_k=1.0, cap=True):
    return stroke_mask(PW, PH, P(pts), width * K, smooth_k, cap)


def circle(cx, cy, r, n=64, ry=None, a0=0.0):
    ry = r if ry is None else ry
    return [(cx + r * math.cos(a0 + 2 * math.pi * i / n), cy + ry * math.sin(a0 + 2 * math.pi * i / n)) for i in range(n)]


def rot_pts(pts, cx, cy, ang_deg):
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]


def catmull(ctrl, n=200):
    c = np.asarray(ctrl, np.float64)
    c = np.vstack([c[0] * 2 - c[1], c, c[-1] * 2 - c[-2]])
    out = []
    segs = len(c) - 3
    for i in range(segs):
        p0, p1, p2, p3 = c[i:i + 4]
        for t in np.linspace(0, 1, max(n // segs, 4), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(c[-2])
    return np.array(out)


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def outline(m, r=2):
    return np.clip(m - ndimage.grey_erosion(m, footprint=disk(r)), 0, 1)


def dilate(m, r):
    return ndimage.grey_dilation(m, footprint=disk(r))


def clawd(fx, fy, S, arms=None, ang=0.0):
    """返回（剪影遮罩, 眼睛遮罩），fx/fy 为脚底中心（1 倍画心坐标）。"""
    cm = clawd_union_mask(PW, PH, fx * K, fy * K, S * K, arms=arms, ang=ang)
    em = np.clip(sum(M(to_px(rect_pts(e), fx, fy, S, ang)) for e in EYES_G), 0, 1)
    return cm, em


def garment(pts_grid, fx, fy, S, ang=0.0, smooth_k=0.0, arms=None, name="garment"):
    """格坐标的服装轮廓 → 遮罩；先做比例检查（下摆不低于 12 格）。"""
    check_proportions(arms=arms, extra={name: [pts_grid]})
    return M(to_px(pts_grid, fx, fy, S, ang), smooth_k)


def g2p(pts_grid, fx, fy, S, ang=0.0):
    return to_px(pts_grid, fx, fy, S, ang)


def mic_px(px, py, ux, uy, s):
    """手持演唱麦克风（球形网罩 + 金属箍 + 上粗下细的手柄）。(px, py) 为握点（画心像素），(ux, uy) 指向网罩，
    s 为一格的像素数。返回各部件遮罩 dict：handle / band / head / grille（网罩上的交叉网纹）/ shine（高光）。"""
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    vx, vy = -uy, ux

    def at(t, w):
        return (px + (ux * t + vx * w) * s, py + (uy * t + vy * w) * s)

    handle = M([at(-2.1, -0.27), at(-2.25, -0.18), at(-2.25, 0.18), at(-2.1, 0.27), at(1.5, 0.37), at(1.5, -0.37)])
    band = M([at(1.45, -0.42), at(1.95, -0.47), at(1.95, 0.47), at(1.45, 0.42)])
    hc = at(2.72, 0)
    head = M([at(2.72 + 0.86 * math.cos(a), 0.84 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64, endpoint=False)])
    # 网罩：两组斜向细线
    gx, gy = (X / K - hc[0]) / s, (Y / K - hc[1]) / s
    gu, gv = gx * ux + gy * uy, gx * vx + gy * vy
    grille = np.maximum(smooth(0.06, 0.0, np.abs(((gu + gv) / 0.26) % 1 - 0.5) * 0.26),
                        smooth(0.06, 0.0, np.abs(((gu - gv) / 0.26) % 1 - 0.5) * 0.26)) * head
    shine = M([at(2.72 + 0.55 * math.cos(a) + 0.18, -0.30 + 0.2 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]) * head
    return {"handle": handle, "band": band, "head": head, "grille": grille, "shine": shine}


def mic_held(arm, fx, fy, S, lean=12.0, ang=0.0):
    """握在 1×2 手臂块里的麦克风：手柄竖直穿过手臂外半段，网罩在脸侧、略向身体倾斜 lean 度。
    arm 为 rigid_arm 的四个角（格）；画完麦克风后，调用方应再把手臂块盖上，才像握住。"""
    cx = sum(p[0] for p in arm) / 4
    cy = sum(p[1] for p in arm) / 4
    side = 1 if cx > 8 else -1
    (px, py), = g2p([(cx + 0.15 * side, cy)], fx, fy, S, ang)
    a = math.radians(lean * side + ang)
    return mic_px(px, py, -math.sin(a), -math.cos(a), S)
