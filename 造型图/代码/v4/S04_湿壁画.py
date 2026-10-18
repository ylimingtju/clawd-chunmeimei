"""04 假如我们还爱着 · 文艺复兴湿壁画 · 样张（04-a 回廊里的连续叙事）。
画法：一点透视的拱廊（皮耶罗·德拉·弗朗切斯卡式），矿物颜料褪成粉灰调；灰泥的细裂纹、按天分块（giornate）的接缝、
局部缺损露出白灰泥；光环用金箔加压印点纹。同一只 Clawd 在画里出现三次：侧身、四分之三侧身、正面。
三次的外形和明暗先用三维算出，再按湿壁画平涂成 2D。
用法：python S04_湿壁画.py 输出目录"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import clawd3d as c3
from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, gauss, fbm, value_noise, smooth, over, downsample,
                 ishift, outline, clawd, garment, g2p, rigid_arm, check_proportions)
import frame

G = np.random.default_rng(4)
INTONACO = np.array([0.90, 0.86, 0.78])
GR = value_noise(PH, PW, 1.5, 41) - 0.5
FN = fbm(PH, PW, 5, 42, 2) - 0.5
BLOT = fbm(PH, PW, 120, 43, 4)
VP = (560, 318)


class Fresco:
    def __init__(self):
        self.img = INTONACO * (1 + 0.04 * GR + 0.04 * FN)[..., None] * np.ones((PH, PW, 1), np.float32)

    def paint(self, m, color, shade=None, fade=1.0):
        col = np.asarray(color, np.float32)
        col = col[None, None] if col.ndim == 1 else col
        if shade is not None:
            col = col * shade[..., None]
        # 湿壁画的颜料和石灰一起干：颜色发粉、发干，有颗粒
        col = col * (1 + 0.05 * GR + 0.03 * FN)[..., None]
        col = col * 0.86 + INTONACO * 0.14
        cov = m * (1 - 0.25 * fade * smooth(0.62, 0.9, BLOT))
        self.img = over(self.img, col, np.clip(cov, 0, 1))

    def sinopia(self, m, w=1, color=(0.42, 0.24, 0.17), op=0.45):
        self.img = over(self.img, color, outline(m, w) * op)

    def age(self):
        n = fbm(PH, PW, 140, 44, 3)
        crack = smooth(0.994, 0.999, 1 - np.abs(2 * n - 1)) * smooth(0.5, 0.75, fbm(PH, PW, 260, 48, 2))
        self.img *= (1 - 0.14 * crack)[..., None]
        # giornate：几条不规则的分块接缝
        for seed, pts in [(1, [(0, 210), (300, 196), (520, 214), (760, 190), (1120, 206)]), (2, [(680, 0), (690, 200), (660, 420), (700, 630)])]:
            seam = SM(catmull(pts, 80), 0.7, 0.8)
            self.img *= (1 - 0.10 * seam)[..., None]
            self.img = over(self.img, (0.95, 0.92, 0.86), ishift(seam, K, K) * 0.25)
        # 缺损：露出底层的白灰泥，带一点粗糙
        loss = smooth(0.77, 0.79, fbm(PH, PW, 40, 45, 5)) * smooth(0.6, 0.75, fbm(PH, PW, 300, 46, 2))
        edge_w = smooth(170 * K, 30 * K, np.minimum.reduce([X, PW - X, Y, PH - Y]))
        loss = loss * edge_w
        self.img = self.img * (1 - loss[..., None]) + np.array([0.93, 0.91, 0.86]) * (1 + 0.06 * GR)[..., None] * loss[..., None]
        self.img *= (1 - 0.25 * gauss(np.clip(ishift(loss, 2 * K, 2 * K) - loss, 0, 1), 0.8 * K))[..., None]
        # 整体褪色：暗部被石灰抬亮
        self.img = self.img * 0.92 + 0.06


def persp(x, y_floor, depth):
    """地面上的点：depth 0 在画面前沿，1 在灭点。"""
    return (x + (VP[0] - x) * depth, y_floor + (VP[1] - y_floor) * depth)


def sprite_clawd(az, px_cell, cloth=(0.10, 0.10, 0.11)):
    """三维算出外形与明暗：黑色无袖衣装是包住身体下部的一块。"""
    extra = [((-0.15, 2.0, -4.15), (16.15, 6.5, 4.15), "cloth")]
    boxes = c3.clawd_boxes(depth=8, extra=extra)
    colors = {"body": c3.ORANGE, "eye": np.array([0.05, 0.04, 0.05]), "cloth": np.array(cloth)}
    ar = math.radians(az)
    Lx, Ly, Lz = -0.55, 0.75, -0.45                     # 光跟着相机走：每个身影都从画面左前方受光
    light = (Lx * math.cos(ar) + Lz * math.sin(ar), Ly, -Lx * math.sin(ar) + Lz * math.cos(ar))
    spr = c3.sprite(boxes, px_cell, az_deg=az, elev_deg=6, light_dir=light, colors=colors, ss=2, shadow_samples=4)
    return spr


def place(Fr, spr, cx, foot):
    h, w = spr.shape[:2]
    ys, xs = np.nonzero(spr[..., 3] > 0.5)
    by = ys.max()
    x0, y0 = int(cx * K - w / 2), int(foot * K - by)
    rgb = np.zeros((PH, PW, 3), np.float32)
    a = np.zeros((PH, PW), np.float32)
    rgb[y0:y0 + h, x0:x0 + w] = spr[..., :3]
    a[y0:y0 + h, x0:x0 + w] = spr[..., 3]
    # 平涂化：把三维明暗压成 3 级柔和的色阶，像湿壁画的明暗分层
    lum = rgb.mean(-1)
    step = gauss(np.round(lum * 6) / 6, 0.7 * K)
    col = rgb / (lum[..., None] + 1e-3) * step[..., None] * 1.45
    col = col * 0.85 + gauss(col, 1.5 * K) * 0.15
    # 地上的柔影
    sh = gauss(M(circle(cx + 10, foot + 2, w / K * 0.45, 48, 8)), 6 * K)
    Fr.img *= (1 - 0.22 * sh)[..., None]
    Fr.paint(a, np.clip(col, 0, 1))
    Fr.sinopia(np.clip(a, 0, 1), 1, op=0.55)


def render():
    Fr = Fresco()
    # ---------- 回廊（一点透视）：左侧柱廊外是花园，右侧是盲拱墙 ----------
    def at(xf, yf, d):
        return (xf + (VP[0] - xf) * d, yf + (VP[1] - yf) * d)
    DB = 0.62
    Fr.paint(M([(-5, -5), (1125, -5), (1125, 640), (-5, 640)]), (0.56, 0.66, 0.76),
             shade=(1.06 - 0.22 * smooth(0, 420, Y / K)))
    lawn = M([(-5, 420), (420, 420), (420, 640), (-5, 640)])
    Fr.paint(lawn, (0.46, 0.56, 0.38), shade=(0.9 + 0.2 * smooth(420, 640, Y / K)))
    for (x, h) in [(40, 230), (110, 270), (180, 200), (250, 250), (320, 190)]:
        cyp = [(x, 420 - h), (x + 16, 420 - h * 0.7), (x + 20, 420 - h * 0.3), (x + 16, 424), (x - 16, 424), (x - 20, 420 - h * 0.3), (x - 16, 420 - h * 0.7)]
        cm = M(cyp, 0.7)
        Fr.paint(cm, (0.20, 0.33, 0.23), shade=(1.12 - 0.4 * smooth(x - 20, x + 20, X / K)))
        Fr.sinopia(cm, 1, op=0.3)
    # 拱顶
    c0, c1 = at(-40, -40, DB), at(1160, -40, DB)
    ceil = M([(-40, -40), (1160, -40), c1, c0])
    Fr.paint(ceil, (0.78, 0.68, 0.66), shade=(0.85 + 0.2 * smooth(-40, c0[1], Y / K)))
    for d in (0.12, 0.27, 0.42, 0.55):
        a_, b_ = at(-40, -40, d), at(1160, -40, d)
        Fr.paint(SM([a_, ((a_[0] + b_[0]) / 2, a_[1] + 24 * (1 - d)), b_], 3.2 * (1 - d), 1.0, False) * ceil, (0.64, 0.52, 0.50))
    # 右墙与盲拱
    r0, r1 = at(1160, -40, DB), at(1160, 640, DB)
    wall = M([(1160, -40), r0, r1, (1160, 640)])
    Fr.paint(wall, (0.84, 0.74, 0.70), shade=(0.9 + 0.12 * smooth(r0[0], 1120, X / K)))
    for d0, d1 in [(0.06, 0.2), (0.24, 0.36), (0.40, 0.49), (0.52, 0.58)]:
        p00, p01 = at(1160, 120, d0), at(1160, 640, d0)
        p10, p11 = at(1160, 120, d1), at(1160, 640, d1)
        niche = M([p00, p10, (p11[0], p11[1] - 2), (p01[0], p01[1] - 2)])
        Fr.paint(niche, (0.74, 0.64, 0.62))
        Fr.sinopia(niche, 1, op=0.35)
    # 地面：汇向灭点的地砖
    f0, f1 = at(-40, 640, DB), at(1160, 640, DB)
    floor = M([(-40, 640), (1160, 640), f1, f0])
    floor = np.clip(floor - lawn * smooth(f0[1] + 1, f0[1] - 1, Y / K) * 0, 0, 1)
    Fr.paint(floor, (0.82, 0.72, 0.64))
    for i in range(-10, 22):
        x = -40 + i * 64
        Fr.paint(SM([(x, 640), at(x, 640, DB)], 0.9, 0, False) * floor, (0.58, 0.46, 0.42))
    for d in [1 - 0.74 ** k for k in range(1, 9)]:
        if d > DB:
            break
        a_, b_ = at(-40, 640, d), at(1160, 640, d)
        Fr.paint(SM([a_, b_], 0.9, 0, False) * floor, (0.58, 0.46, 0.42))
    # 左侧柱廊（透视里的柱子）
    for d in (0.12, 0.27, 0.42, 0.55):
        top, bot = at(-40, 150, d), at(-40, 640, d)
        w = 30 * (1 - d)
        cm = M([(top[0] - w / 2, top[1]), (top[0] + w / 2, top[1]), (bot[0] + w / 2, bot[1]), (bot[0] - w / 2, bot[1])])
        Fr.paint(cm, (0.86, 0.78, 0.74), shade=(1.05 - 0.25 * smooth(top[0] - w / 2, top[0] + w / 2, X / K)))
        Fr.sinopia(cm, 1, op=0.35)
        a2 = at(-40, -40, d)
        Fr.paint(M([a2, (a2[0] + w, a2[1]), (top[0] + w / 2, top[1]), (top[0] - w / 2, top[1])]), (0.80, 0.70, 0.66))
    # 后墙
    back = M([c0, c1, f1, f0])
    Fr.paint(back, (0.88, 0.80, 0.74))
    Fr.sinopia(back, 1, op=0.4)
    # 后墙的金箔光环（M 段 Clawd 会走进去）
    hcx, hcy, hr = 560, 300, 56
    hm = M(circle(hcx, hcy, hr, 120))
    gold = np.array([0.86, 0.67, 0.30])
    gl = 0.9 + 0.25 * smooth(0.3, 0.9, fbm(PH, PW, 14, 47, 3)) - 0.15 * smooth(hcx - hr, hcx + hr, X / K)
    Fr.img = over(Fr.img, gold[None, None] * gl[..., None], hm)
    for ring in (hr - 6, hr - 14):
        for i in range(int(ring * 0.9)):
            a = 2 * math.pi * i / int(ring * 0.9)
            Fr.img *= (1 - 0.35 * M(circle(hcx + ring * math.cos(a), hcy + ring * math.sin(a), 1.0, 8)))[..., None]
    Fr.sinopia(hm, 1, (0.45, 0.30, 0.12), 0.7)

    # ---------- 三个身影：侧身 → 四分之三侧身 → 正面 ----------
    for (cx, az) in [(240, -90), (560, -45), (880, 0)]:
        place(Fr, sprite_clawd(az, 9 * K), cx, 572)

    # ---------- 前景拱廊：柱子、拱、檐壁 ----------
    arch = np.zeros((PH, PW), np.float32)
    cols = [64, 404, 716, 1056]
    for x in cols:
        arch = np.maximum(arch, M([(x - 22, 160), (x + 22, 160), (x + 22, 640), (x - 22, 640)]))
    arch = np.maximum(arch, M([(-5, -5), (1125, -5), (1125, 92), (-5, 92)]))
    for a_, b_ in zip(cols[:-1], cols[1:]):
        cx, r = (a_ + b_) / 2, (b_ - a_) / 2 - 22
        spandrel = [(a_, 92), (b_, 92), (b_, 160)] + [(cx + r * math.cos(t), 160 - r * 0.42 * math.sin(t)) for t in np.linspace(0, math.pi, 40)] + [(a_, 160)]
        arch = np.maximum(arch, M(spandrel))
    shade = 0.96 + 0.08 * smooth(0, 1120, X / K) - 0.06 * smooth(92, 640, Y / K)
    Fr.paint(arch, (0.88, 0.80, 0.76), shade=shade)
    Fr.sinopia(arch, 1, op=0.45)
    for x in cols:                               # 柱身的竖向明暗、柱头柱础
        u = (X / K - x) / 22
        cyl = 1.08 - 0.3 * smooth(-1, 1, u)
        cm = M([(x - 22, 160), (x + 22, 160), (x + 22, 640), (x - 22, 640)])
        Fr.paint(cm, (0.86, 0.78, 0.74), shade=cyl)
        Fr.paint(M([(x - 32, 150), (x + 32, 150), (x + 26, 172), (x - 26, 172)]), (0.80, 0.72, 0.68))
        Fr.paint(M([(x - 30, 600), (x + 30, 600), (x + 34, 620), (x - 34, 620)]), (0.80, 0.72, 0.68))
    # 檐壁：一条带圆饰的饰带；拱肩上的青色圆章
    Fr.paint(M([(-5, 40), (1125, 40), (1125, 62), (-5, 62)]), (0.62, 0.40, 0.36))
    for i in range(30):
        Fr.paint(M(circle(20 + i * 38, 51, 6, 20)), (0.88, 0.82, 0.74))
    for a_, b_ in zip(cols[:-1], cols[1:]):
        for x in (a_ + 44, b_ - 44):
            rm = M(circle(x, 118, 18, 40))
            Fr.paint(rm, (0.36, 0.48, 0.60))
            Fr.sinopia(rm, 1, op=0.5)
    # 前景地面一条
    Fr.paint(M([(-5, 600), (1125, 600), (1125, 640), (-5, 640)]), (0.76, 0.66, 0.60))
    # 拱顶飘落的玫瑰花瓣
    for i in range(24):
        x, y = G.uniform(100, 1020), G.uniform(110, 560)
        pet = rot_pts(circle(x, y, 5, 20, 3), x, y, G.uniform(0, 180))
        Fr.paint(M(pet), (0.74, 0.34, 0.36))
    Fr.age()
    return Fr.img


def render_b():
    """04-b：致敬波提切利——金箔底，Clawd 穿金色长裙、戴羽毛头饰，站在海上的大扇贝里。"""
    Fr = Fresco()
    # 金箔底：分块贴的金箔（方格接缝）、压印点纹、磨损处露出红色底泥
    gold = np.array([0.84, 0.66, 0.30])
    sheet = 0.95 + 0.06 * (fbm(PH, PW, 120, 50, 3) - 0.5) - 0.10 * smooth(0, 1120, X / K)
    gm = M([(-5, -5), (1125, -5), (1125, 470), (-5, 470)])
    Fr.img = over(Fr.img, gold[None, None] * sheet[..., None], gm)
    for x in range(0, 1120, 70):
        Fr.img *= (1 - 0.08 * SM([(x, 0), (x, 470)], 0.6, 0, False))[..., None]
    for y in range(0, 470, 70):
        Fr.img *= (1 - 0.08 * SM([(0, y), (1120, y)], 0.6, 0, False))[..., None]
    edge_w = smooth(160 * K, 20 * K, np.minimum.reduce([X, PW - X, Y]))
    bole = smooth(0.78, 0.82, fbm(PH, PW, 30, 51, 4)) * gm * edge_w
    Fr.img = over(Fr.img, (0.58, 0.26, 0.18), bole * 0.8)
    # 头光一样的压印圆环（在 Clawd 身后）
    pass
    # 海：波提切利式的 V 形浪纹
    sea = M([(-5, 470), (1125, 470), (1125, 640), (-5, 640)])
    Fr.paint(sea, (0.44, 0.62, 0.60), shade=(0.9 + 0.2 * smooth(470, 640, Y / K)))
    G2 = np.random.default_rng(14)
    for i in range(90):
        x, y = G2.uniform(0, 1120), G2.uniform(480, 630)
        w = 6 + (y - 470) * 0.06
        Fr.paint(SM([(x - w, y - w * 0.5), (x, y), (x + w, y - w * 0.5)], 1.0, 0, False), (0.92, 0.94, 0.88))
    # 扇贝：在 Clawd 身后展开的大扇形，放射状的棱一明一暗，边缘是波浪形的扇贝口
    hx, hy, RX, RY = 560, 600, 330, 300
    n_rib = 17
    rim = []
    for t in np.linspace(0, math.pi, 400):
        bump = 1 + 0.045 * abs(math.sin(n_rib * t))
        rim.append((hx + RX * bump * math.cos(t), hy - RY * bump * math.sin(t)))
    sm = M([(hx, hy)] + rim, 0.0)
    ang = np.arctan2(hy - Y / K, X / K - hx)
    rib = 0.5 + 0.5 * np.cos(ang * n_rib * 2)
    rr = np.hypot((X / K - hx) / RX, (Y / K - hy) / RY)
    shade = (0.86 + 0.18 * rib) * (0.92 + 0.12 * smooth(0.2, 0.95, rr))
    Fr.paint(sm, (0.94, 0.80, 0.74), shade=shade)
    for k in range(n_rib):
        t = (k + 0.5) / n_rib * math.pi
        Fr.paint(SM([(hx, hy), (hx + RX * 1.04 * math.cos(t), hy - RY * 1.04 * math.sin(t))], 0.9, 0, False) * sm, (0.70, 0.50, 0.46))
    for f in (0.35, 0.62, 0.86):                           # 生长纹
        arc = [(hx + RX * f * math.cos(t), hy - RY * f * math.sin(t)) for t in np.linspace(0.05, math.pi - 0.05, 80)]
        Fr.paint(SM(arc, 0.7, 0.8, False) * sm, (0.78, 0.60, 0.56))
    Fr.sinopia(sm, 1, op=0.6)
    hinge = M([(hx - 46, hy), (hx + 46, hy), (hx + 30, hy + 26), (hx - 30, hy + 26)])
    Fr.paint(hinge, (0.88, 0.74, 0.68))
    Fr.sinopia(hinge, 1, op=0.5)
    # Clawd：金色长裙 + 羽毛头饰
    FX, FY, S = 560, 590, 15
    arms = [rigid_arm("L", 0), rigid_arm("R", 0)]
    cm, em = clawd(FX, FY, S, arms)
    Fr.paint(cm, (0.80, 0.50, 0.38), shade=(1.06 - 0.22 * smooth(FX - 8 * S, FX + 8 * S, X / K)))
    gown = garment([(-0.2, 5.4), (16.2, 5.4), (17.0, 8.6), (18.0, 12.0), (-2.0, 12.0), (-1.0, 8.6)], FX, FY, S, arms=arms, name="gown")
    Fr.paint(gown, (0.86, 0.66, 0.30), shade=(1.08 - 0.30 * smooth(FX - 9 * S, FX + 9 * S, X / K)))
    for gx in (2.0, 5.0, 8.0, 11.0, 14.0):
        Fr.paint(SM(g2p([(gx, 6.0), (gx + (gx - 8) * 0.2, 11.9)], FX, FY, S), 1.3, 0.8, False), (0.66, 0.46, 0.20))
    Fr.sinopia(gown, 1, op=0.5)
    Fr.sinopia(cm, 1, op=0.5)
    Fr.paint(em, (0.16, 0.10, 0.08))
    # 羽毛头饰：从顶上扇形伸出的几根羽毛（在身体之上，不影响下摆规则）
    for k, ang in enumerate((-38, -18, 0, 18, 38)):
        a_ = math.radians(ang - 90)
        bx, by = g2p([(8 + ang / 12, 0.2)], FX, FY, S)[0]
        L = 120 - abs(ang) * 0.8
        spine = [(bx + L * t * math.cos(a_) + 6 * math.sin(t * 3) * (1 if ang >= 0 else -1), by + L * t * math.sin(a_)) for t in np.linspace(0, 1, 20)]
        for side in (-1, 1):
            barbs = []
            for t in np.linspace(0.05, 1, 16):
                px, py = spine[int(t * 19)]
                w = 19 * math.sin(math.pi * min(t * 1.1, 1))
                barbs.append((px + side * w * math.cos(a_ + math.pi / 2), py + side * w * math.sin(a_ + math.pi / 2)))
            pts = [spine[0]] + barbs + [spine[-1]]
            Fr.paint(M(pts, 0.6), (0.97, 0.94, 0.86) if k % 2 == 0 else (0.92, 0.84, 0.62))
        Fr.paint(SM(spine, 1.0, 0.8, False), (0.70, 0.56, 0.30))
    Fr.paint(M(g2p([(5.6, -0.6), (10.4, -0.6), (10.0, 0.4), (6.0, 0.4)], FX, FY, S)), (0.78, 0.60, 0.26))
    # 飘落的玫瑰
    for i in range(16):
        x, y = G2.uniform(80, 1040), G2.uniform(60, 440)
        rot = G2.uniform(0, 6.28)
        petals = [(x + 9 * (0.75 + 0.25 * math.cos(5 * t)) * math.cos(t + rot), y + 9 * (0.75 + 0.25 * math.cos(5 * t)) * math.sin(t + rot)) for t in np.linspace(0, 2 * math.pi, 60)]
        rm = M(petals, 0.4)
        Fr.paint(rm, (0.86, 0.52, 0.56), shade=(1.05 - 0.2 * smooth(x - 9, x + 9, X / K)))
        Fr.sinopia(rm, 1, op=0.45)
        sp = [(x + 4.5 * (1 - t / 2.4) * math.cos(t * 2.6 + rot), y + 4.5 * (1 - t / 2.4) * math.sin(t * 2.6 + rot)) for t in np.linspace(0, 2.2, 24)]
        Fr.paint(SM(sp, 0.8, 0.8, False), (0.60, 0.28, 0.32))
        lf = rot_pts([(x + 7, y + 4), (x + 18, y + 2), (x + 10, y + 9)], x, y, math.degrees(rot))
        Fr.paint(M(lf, 0.5), (0.40, 0.52, 0.34))
    Fr.age()
    return np.clip(downsample(Fr.img, K), 0, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = np.clip(downsample(render(), K), 0, 1)
    frame.save(frame.compose(painting, (0.80, 0.76, 0.69), "假如我们还爱着"), os.path.join(out_dir, "04_假如我们还爱着_湿壁画.png"))
    frame.save(painting, os.path.join(out_dir, "04_画心.png"))
    pb = render_b()
    frame.save(frame.compose(pb, (0.80, 0.76, 0.69), "假如我们还爱着"), os.path.join(out_dir, "04b_假如我们还爱着_金箔扇贝.png"))
    frame.save(pb, os.path.join(out_dir, "04b_画心.png"))
