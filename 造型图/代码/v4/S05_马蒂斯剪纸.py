"""05 我管不住我自己 · 马蒂斯剪纸 · 样张（05-a 晨光阳台）。
画法：每一块形状都是刷过水粉的彩纸剪出来的——有刷痕、轻微色差、手剪的微小起伏；
纸片一层层叠在一起，每层在下面投下极浅的阴影（光从左上来）。Clawd 是一整张橙色纸剪出来的。
用法：python S05_马蒂斯剪纸.py 输出目录"""
import math
import os
import sys

import numpy as np

from common4 import (ishift, gauss, fbm, value_noise, streak_noise, over, poly_mask, stroke_mask, downsample,
                     taper_stroke, clawd_union_mask, to_px, rect_pts, rigid_arm, check_proportions, LEGS)
import frame

K = 2                                    # 画心按 2 倍分辨率绘制再缩小
PW, PH = frame.PW * K, frame.PH * K
G = np.random.default_rng(5)

# 马蒂斯色板
SKY = (0.60, 0.79, 0.92)
ULTRA = (0.11, 0.19, 0.62)
EMERALD = (0.08, 0.52, 0.38)
LEMON = (0.98, 0.84, 0.18)
CORAL = (0.95, 0.58, 0.55)
VERMIL = (0.89, 0.27, 0.17)
BLACK = (0.07, 0.07, 0.09)
WHITE = (0.97, 0.95, 0.90)
PINK = (0.96, 0.66, 0.74)
ORANGE = (202 / 255, 124 / 255, 94 / 255)

STREAKS = [streak_noise(PH, PW, 70, 7, a, 40 + i) for i, a in enumerate((8, 35, 80, 125))]
FINE = [streak_noise(PH, PW, 18, 0.8, a, 60 + i) for i, a in enumerate((8, 35, 80, 125))]
MOTTLE = fbm(PH, PW, 260, 9, octaves=3) - 0.5
FIBER = fbm(PH, PW, 6, 77, octaves=2) - 0.5


def jitter(pts, amp=0.7, seed=0):
    """手剪的微小起伏（1 倍坐标里约 amp 像素）。"""
    g = np.random.default_rng(seed)
    pts = np.asarray(pts, np.float64)
    off = g.normal(0, amp, pts.shape)
    off = (off + np.roll(off, 1, 0) + np.roll(off, -1, 0)) / 3
    return pts + off


def M(pts, smooth_k=0.0, jit=0.6, seed=None):
    """1 倍画心坐标的点列 → 2 倍分辨率遮罩。"""
    seed = int(G.integers(1 << 30)) if seed is None else seed
    p = jitter(pts, jit, seed) if jit else np.asarray(pts, np.float64)
    return poly_mask(PW, PH, [(x * K, y * K) for x, y in p], smooth_k)


def circle(cx, cy, r, n=48, ry=None):
    ry = r if ry is None else ry
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


class Paper:
    """一层层往上贴纸。"""

    def __init__(self, base):
        self.img = np.ones((PH, PW, 3), np.float32) * np.array(base, np.float32)
        self.img *= (1 + 0.02 * STREAKS[0] + 0.03 * MOTTLE + 0.02 * FIBER)[..., None]

    def stick(self, m, color, streak=None, lift=1.0, tex=1.0):
        if m.max() <= 0:
            return
        # 纸片投在下层上的阴影：近处一圈较实、外面一圈很柔
        sh = (gauss(ishift(m, int(3 * K * lift), int(2 * K * lift)), 2.2 * K) * 0.16 +
              gauss(ishift(m, int(7 * K * lift), int(5 * K * lift)), 7 * K) * 0.10) * lift
        self.img *= (1 - np.clip(sh - m, 0, 0.35))[..., None]
        k = int(G.integers(4)) if streak is None else streak
        t = 1 + tex * (0.026 * STREAKS[k] + 0.008 * FINE[k] + 0.045 * MOTTLE * (1 if k % 2 else -1) + 0.018 * FIBER)
        col = np.array(color, np.float32)[None, None] * t[..., None]
        self.img = over(self.img, col, m)


def algae(bx, by, length, width, ang_deg, lobes=5, bend=0.25, phase=0.0):
    """马蒂斯的海藻叶：弯曲的主脉，两侧交错的尖叶瓣。"""
    a0 = math.radians(ang_deg)
    n = 90
    spine, left, right = [], [], []
    x, y, a = bx, by, a0
    for i in range(n + 1):
        t = i / n
        spine.append((x, y))
        a += bend / n * math.sin(math.pi * t + phase) * 2.2
        x += math.cos(a) * length / n
        y += math.sin(a) * length / n
    spine = np.array(spine)
    d = np.gradient(spine, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, n + 1)
    env = width * np.sqrt(np.clip(1 - t, 0, 1)) * (0.3 + 0.7 * np.sin(np.pi * np.clip(t * 1.15, 0, 1)))
    wob = np.cumsum(np.random.default_rng(int(phase * 100) + lobes).normal(0, 0.05, n + 1))
    lw = env * (0.22 + 0.78 * np.abs(np.sin(np.pi * lobes * t * (1 + 0.15 * t) + phase + wob)) ** 0.55)
    rw = env * (0.22 + 0.78 * np.abs(np.sin(np.pi * lobes * t * (1 + 0.1 * t) + phase + math.pi / 2 - wob)) ** 0.55)
    left = spine + nrm * lw[:, None]
    right = spine - nrm * rw[:, None]
    return np.concatenate([left, right[::-1]], 0)


def shirt(x0, y0, w, h):
    return [(x0, y0), (x0 + w * 0.32, y0 + h * 0.04), (x0 + w * 0.5, y0 + h * 0.12), (x0 + w * 0.68, y0 + h * 0.04),
            (x0 + w, y0), (x0 + w * 1.12, y0 + h * 0.28), (x0 + w * 0.9, y0 + h * 0.34), (x0 + w * 0.86, y0 + h),
            (x0 + w * 0.14, y0 + h), (x0 + w * 0.1, y0 + h * 0.34), (x0 - w * 0.12, y0 + h * 0.28)]


def baluster(cx, y0, y1, w):
    """阿拉伯花饰式的栏杆柱（对称花瓶轮廓）。"""
    prof = []
    for i in range(41):
        t = i / 40
        r = w * (0.22 + 0.30 * math.sin(math.pi * t) ** 2 + 0.16 * math.sin(3 * math.pi * t) ** 2)
        prof.append((r, y0 + (y1 - y0) * t))
    return [(cx + r, y) for r, y in prof] + [(cx - r, y) for r, y in prof[::-1]]


def note(x, y, s, flag=True):
    head = circle(x, y, 9 * s, 24, 6.5 * s)
    head = [(x + (px - x) * math.cos(-0.4) - (py - y) * math.sin(-0.4), y + (px - x) * math.sin(-0.4) + (py - y) * math.cos(-0.4)) for px, py in head]
    stem = [(x + 7 * s, y - 2 * s), (x + 10 * s, y - 2 * s), (x + 10 * s, y - 40 * s), (x + 7 * s, y - 40 * s)]
    fl = [(x + 10 * s, y - 40 * s), (x + 24 * s, y - 28 * s), (x + 22 * s, y - 20 * s), (x + 10 * s, y - 30 * s)]
    return [head, stem] + ([fl] if flag else [])


def render():
    P = Paper(SKY)
    # ---------- 远景：天空、太阳、海 ----------
    sun = []
    for i in range(72):
        a = 2 * math.pi * i / 72
        r = 48 + (16 if i % 6 in (0, 1) else 0) + G.normal(0, 1.2)
        sun.append((985 + r * math.cos(a), 128 + r * math.sin(a)))
    P.stick(M(sun, 0.6, 0.3), LEMON, lift=0.6)
    for cx, cy, s in [(585, 300, 0.9), (1068, 236, 0.5)]:
        cl = [(cx - 70 * s, cy), (cx - 48 * s, cy - 20 * s), (cx - 18 * s, cy - 30 * s), (cx + 14 * s, cy - 24 * s),
              (cx + 44 * s, cy - 34 * s), (cx + 74 * s, cy - 14 * s), (cx + 86 * s, cy + 6 * s), (cx - 60 * s, cy + 8 * s)]
        P.stick(M(cl, 0.9, 1.0), WHITE, lift=0.6)
    sea = [(280, 336), (1130, 330), (1130, 450), (280, 450)]
    P.stick(M(sea, 0, 0.8), ULTRA, streak=0, lift=0.7)
    for i in range(9):
        wx, wy = 330 + i * 92 + G.normal(0, 10), 362 + (i % 3) * 22
        wv = taper_stroke([(wx + k * 4, wy + 4 * math.sin(k / 2.5)) for k in range(12)], 2.5, 1.0, 4.0)
        P.stick(M(wv, 0.8, 0.2), WHITE, lift=0.3, tex=0.5)

    # ---------- 晾衣绳与衣物 ----------
    line = [(300, 70)] + [(300 + t * 830, 70 + 48 * math.sin(math.pi * t)) for t in np.linspace(0.02, 1, 30)]
    P.stick(stroke_mask(PW, PH, [(x * K, y * K) for x, y in line], 2.2 * K), BLACK, lift=0.5, tex=0.2)

    def line_y(x):
        return 70 + 48 * math.sin(math.pi * (x - 300) / 830)
    P.stick(M(shirt(430, line_y(470) + 2, 92, 118), 0.4, 0.8), PINK, streak=2)
    P.stick(M([(600, line_y(615)), (630, line_y(615)), (632, line_y(615) + 70), (660, line_y(615) + 92),
               (650, line_y(615) + 108), (612, line_y(615) + 88)], 0.5, 0.6), LEMON, streak=1)
    tx = 720
    towel = [(tx, line_y(tx) + 1), (tx + 96, line_y(tx + 96) + 1), (tx + 100, line_y(tx + 96) + 132), (tx + 4, line_y(tx) + 128)]
    P.stick(M(towel, 0, 0.7), WHITE, streak=2)
    for k in range(3):
        yy = line_y(tx) + 30 + k * 30
        P.stick(M([(tx + 2, yy), (tx + 99, yy + 4), (tx + 99, yy + 13), (tx + 2, yy + 9)], 0, 0.5), ULTRA, lift=0.4)
    for x in (452, 500, 608, 628, 735, 805):
        P.stick(M([(x - 3, line_y(x) - 8), (x + 3, line_y(x) - 8), (x + 3, line_y(x) + 12), (x - 3, line_y(x) + 12)], 0, 0.2), VERMIL, lift=0.5, tex=0.4)

    # ---------- 左侧：房子的墙、落地窗与百叶 ----------
    P.stick(M([(-10, -10), (300, -10), (300, 640), (-10, 640)], 0, 0.6), CORAL, streak=2, lift=1.2)
    P.stick(M([(64, 66), (248, 66), (248, 545), (64, 545)], 0, 0.6), ULTRA, streak=2, lift=0.6)
    curtain = [(72, 70), (130, 70), (122, 200), (138, 330), (116, 470), (128, 540), (72, 540)]
    P.stick(M(curtain, 0.7, 1.0), LEMON, lift=0.6)
    for side, x0, x1 in (("L", 10, 66), ("R", 248, 330)):
        P.stick(M([(x0, 60), (x1, 66 if side == "L" else 66), (x1, 548), (x0, 556)], 0, 0.5), EMERALD, lift=1.0)
        for k in range(17):
            yy = 80 + k * 27
            P.stick(M([(x0 + 6, yy), (x1 - 6, yy + 3), (x1 - 6, yy + 11), (x0 + 6, yy + 8)], 0, 0.3), (0.05, 0.36, 0.27), lift=0.25, tex=0.5)

    # ---------- 栏杆（矮，在 Clawd 身后）与地面 ----------
    P.stick(M([(330, 428), (1130, 428), (1130, 442), (330, 442)], 0, 0.6), ULTRA, lift=0.8)
    for k in range(15):
        P.stick(M(baluster(358 + k * 54, 442, 545, 22), 0.5, 0.4), ULTRA, lift=0.8)
    P.stick(M([(-10, 545), (1130, 545), (1130, 640), (-10, 640)], 0, 0.6), EMERALD, streak=0, lift=1.2)
    # 黑白几何地毯
    rug = [(410, 588), (950, 588), (980, 640), (380, 640)]
    P.stick(M(rug, 0, 0.6), WHITE, lift=0.5)
    for k in range(16):
        x = 410 + k * 34
        P.stick(M([(x, 596), (x + 17, 618), (x + 34, 596), (x + 34, 606), (x + 17, 628), (x, 606)], 0, 0.3), BLACK, lift=0.3, tex=0.4)

    # ---------- 盆栽：马蒂斯海藻叶 ----------
    leaves = [(205, 520, 210, 34, -100, 5, 0.5, 0.0, EMERALD), (190, 520, 170, 28, -128, 4, -0.4, 1.0, (0.04, 0.33, 0.25)),
              (222, 520, 185, 30, -72, 4, 0.5, 2.0, (0.20, 0.62, 0.30)), (200, 520, 120, 22, -150, 3, -0.3, 0.5, EMERALD)]
    for bx, by, ln, wd, ang, lob, bend, ph, col in leaves:
        P.stick(M(algae(bx, by, ln, wd, ang, lob, bend, ph), 0.6, 0.4), col, lift=0.9)
    P.stick(M([(150, 515), (262, 515), (248, 600), (164, 600)], 0.2, 0.6), VERMIL, lift=1.1)

    # ---------- 小桌与唱机 ----------
    TX = 760
    P.stick(M([(TX - 9, 494), (TX + 9, 494), (TX + 7, 584), (TX - 7, 584)], 0, 0.4), ULTRA, lift=1.0)
    P.stick(M([(TX - 54, 590), (TX - 30, 580), (TX + 30, 580), (TX + 54, 590), (TX + 50, 596), (TX - 50, 596)], 0.5, 0.4), ULTRA, lift=1.0)
    P.stick(M(circle(TX, 492, 104, 64, 17), 0, 0.5), WHITE, streak=0, lift=1.1)
    box = [(TX - 74, 446), (TX + 74, 446), (TX + 78, 482), (TX - 78, 482)]
    P.stick(M(box, 0.15, 0.4), LEMON, lift=1.0)
    P.stick(M([(TX - 60, 456), (TX - 46, 456), (TX - 46, 466), (TX - 60, 466)], 0, 0.2), VERMIL, lift=0.4)
    P.stick(M(circle(TX + 2, 440, 66, 64, 15), 0, 0.3), BLACK, lift=0.8, tex=0.5)
    for rr in (52, 38):
        ring = stroke_mask(PW, PH, [((TX + 2 + rr * math.cos(a)) * K, (440 + rr * 0.23 * math.sin(a)) * K) for a in np.linspace(0.3, 2.6, 30)], 1.0 * K)
        P.img = over(P.img, (0.22, 0.22, 0.26), ring * 0.8)
    P.stick(M(circle(TX + 2, 440, 18, 32, 5), 0, 0.2), VERMIL, lift=0.3)
    arm_pts = taper_stroke([(TX + 66, 428), (TX + 52, 436), (TX + 30, 440), (TX + 16, 438)], 5, 3)
    P.stick(M(arm_pts, 0.3, 0.1), WHITE, lift=0.6, tex=0.4)
    # 飘出来的音符
    for (x, y, s, fl) in [(852, 396, 0.9, True), (912, 352, 1.05, False), (978, 322, 0.95, True), (1042, 292, 0.8, True)]:
        for part in note(x, y, s, fl):
            P.stick(M(part, 0.2, 0.2), BLACK, lift=0.7, tex=0.4)

    # ---------- Clawd：一整张橙色纸 ----------
    S, FX, FY = 17, 540, 612
    arms = [rigid_arm("L", 38), rigid_arm("R", -32)]
    top = [(-0.4, 5.3), (16.4, 5.3), (16.9, 10.5), (-0.9, 10.5)]
    check_proportions(arms=arms, legs=LEGS, extra={"top": [top]})
    cm = clawd_union_mask(PW, PH, FX * K, FY * K, S * K, arms=arms)
    P.stick(cm, ORANGE, streak=1, lift=1.3)
    tm = poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(jitter(top, 0.04, 3), FX, FY, S)])
    P.stick(tm, WHITE, streak=0, lift=0.7)
    for k in range(4):
        yy = 6.1 + k * 1.15
        st = [(-0.6, yy), (16.6, yy), (16.7, yy + 0.48), (-0.7, yy + 0.48)]
        sm = poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(jitter(st, 0.03, 10 + k), FX, FY, S)]) * tm
        P.stick(sm, ULTRA, lift=0.3, tex=0.6)
    for ex in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        em = poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(rect_pts(ex), FX, FY, S)])
        P.stick(em, BLACK, lift=0.3, tex=0.3)

    # ---------- 落在它头上的小鸟 ----------
    bx, by = 505, 408
    body = [(bx - 30, by - 14), (bx - 12, by - 26), (bx + 12, by - 25), (bx + 26, by - 16), (bx + 36, by - 22),
            (bx + 30, by - 10), (bx + 14, by - 1), (bx - 18, by - 2)]
    P.stick(M(body, 0.7, 0.3), WHITE, lift=0.8)
    wing = [(bx - 8, by - 20), (bx + 18, by - 44), (bx + 22, by - 36), (bx + 8, by - 14)]
    P.stick(M(wing, 0.6, 0.3), SKY, lift=0.5)
    P.stick(M([(bx - 30, by - 15), (bx - 42, by - 12), (bx - 30, by - 9)], 0, 0.1), LEMON, lift=0.3, tex=0.3)
    P.stick(M(circle(bx - 20, by - 17, 2.6, 12), 0, 0), BLACK, lift=0.1, tex=0)
    for lx in (bx - 6, bx + 6):
        P.stick(M([(lx, by - 2), (lx + 2, by - 2), (lx + 2, by + 1), (lx, by + 1)], 0, 0), BLACK, lift=0.1, tex=0)

    # 整张画轻微的纸面颗粒与暖色晨光
    img = P.img * (1 + 0.025 * FIBER)[..., None]
    img = img * np.array([1.0, 0.99, 0.96])
    return np.clip(downsample(img, K), 0, 1)


def star(cx, cy, r, n=5, rot=0.0, irr=0.18, seed=0):
    """马蒂斯《爵士》里那种手剪的不规则星星。"""
    g = np.random.default_rng(seed)
    pts = []
    for i in range(n * 2):
        a = rot + math.pi * i / n
        rr = (r if i % 2 == 0 else r * 0.42) * (1 + g.uniform(-irr, irr))
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def render_b():
    """05-b：致敬马蒂斯《爵士》——深蓝底上撒着手剪的黄星；Clawd 穿银色亮片短裙，亮片是剪碎的银纸。"""
    P = Paper(ULTRA)
    g = np.random.default_rng(55)
    # 两侧的红色剪纸幕布、上方一条黑白的波浪檐
    for side in (-1, 1):
        x0 = 0 if side < 0 else 1120
        cur = [(x0, -10), (x0 - side * 170, -10), (x0 - side * 120, 180), (x0 - side * 190, 360), (x0 - side * 110, 640), (x0, 640)]
        P.stick(M(cur, 0.6, 1.0), VERMIL, streak=2, lift=1.2)
    val = [(-10, -10), (1130, -10)] + [(1130 - i * 40, 46 + 16 * (i % 2)) for i in range(30)] + [(-10, 46)]
    P.stick(M(val, 0.3, 0.6), BLACK, lift=1.0)
    # 黄星
    for i in range(22):
        x, y = g.uniform(190, 930), g.uniform(80, 420)
        if abs(x - 560) < 170 and y > 200:
            continue
        P.stick(M(star(x, y, g.uniform(14, 30), 5, g.uniform(0, 1), 0.2, i), 0.0, 0.5), LEMON, lift=0.8)
    # 白色的大圆：追光（剪纸）
    P.stick(M(circle(560, 330, 190, 120), 0, 1.2), WHITE, streak=0, lift=0.9)
    # 舞台：一条黑色剪纸台面 + 前沿的白色波浪
    P.stick(M([(-10, 540), (1130, 540), (1130, 640), (-10, 640)], 0, 0.6), BLACK, lift=1.2)
    wave = [(-10, 540)] + [(i * 28, 540 + 9 * math.sin(i * 1.3)) for i in range(42)] + [(1130, 556), (-10, 556)]
    P.stick(M(wave, 0.5, 0.4), WHITE, lift=0.5)
    # Clawd：银色亮片短裙
    S, FX, FY = 18, 560, 548
    arms = [rigid_arm("L", 30), rigid_arm("R", 30)]
    dress = [(-0.3, 5.4), (16.3, 5.4), (17.0, 9.6), (-1.0, 9.6)]
    check_proportions(arms=arms, legs=LEGS, extra={"dress": [dress]})
    cm = clawd_union_mask(PW, PH, FX * K, FY * K, S * K, arms=arms)
    P.stick(cm, ORANGE, streak=1, lift=1.3)
    dm = poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(dress, FX, FY, S)])
    P.stick(dm, (0.78, 0.80, 0.84), streak=0, lift=0.8)
    silv = [(0.92, 0.93, 0.96), (0.66, 0.68, 0.74), (0.98, 0.98, 1.0), (0.80, 0.84, 0.92)]
    for k in range(70):                                    # 剪碎的银纸亮片，一片片贴上去
        gx, gy = g.uniform(-0.4, 16.4), g.uniform(5.6, 9.4)
        (px, py), = to_px([(gx, gy)], FX, FY, S)
        r = g.uniform(4.0, 7.5)
        pm = M(circle(px, py, r, 10, r * g.uniform(0.7, 1.0)), 0, 0.3) * dm
        P.stick(pm, silv[k % 4], lift=0.25, tex=0.4)
    for ex in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        em = poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(rect_pts(ex), FX, FY, S)])
        P.stick(em, BLACK, lift=0.3, tex=0.3)
    # 手持麦克风：黑纸剪的手柄、银纸网罩（网纹用剪刀刻出的细缝），右臂这块纸最后再贴一次，压住手柄
    from kit import mic_held
    mc = mic_held(arms[1], FX, FY, S, lean=12)
    P.stick(mc["handle"], BLACK, lift=0.6)
    P.stick(mc["band"], (0.55, 0.56, 0.60), lift=0.6)
    P.stick(mc["head"], (0.80, 0.82, 0.86), lift=0.6)
    P.img = P.img * (1 - 0.35 * mc["grille"])[..., None]
    P.stick(mc["shine"], (0.95, 0.96, 0.98), lift=0.1, tex=0.2)
    P.stick(poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(arms[1], FX, FY, S)]), ORANGE, streak=1, lift=0.8)
    img = P.img * (1 + 0.025 * FIBER)[..., None]
    return np.clip(downsample(img, K), 0, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = render()
    full = frame.compose(painting, (0.92, 0.77, 0.63), "我管不住我自己")
    frame.save(full, os.path.join(out_dir, "05_我管不住我自己_马蒂斯剪纸.png"))
    frame.save(painting, os.path.join(out_dir, "05_画心.png"))
    pb = render_b()
    frame.save(frame.compose(pb, (0.92, 0.77, 0.63), "我管不住我自己"), os.path.join(out_dir, "05b_我管不住我自己_爵士舞台.png"))
    frame.save(pb, os.path.join(out_dir, "05b_画心.png"))
