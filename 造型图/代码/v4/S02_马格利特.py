"""02 纯妹妹 · 马格利特式超现实油画 · 样张两张：
  02-a 竹林红裙（《空白签名》：远处的竹竿反而挡在 Clawd 身前）
  02-b 聊斋（《光之帝国》：白天的天空下是夜里的楼阁；书生已变成戴帽的青蛙）
画法：平滑冷静的写实油画——柔和的明暗过渡、干净的边、细微的画布纹和笔触。
用法：python S02_马格利特.py 输出目录"""
import math
import os
import sys

import numpy as np

from common4 import streak_noise
from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, value_noise, smooth, over, downsample, ishift, dilate, outline)
import frame

G = np.random.default_rng(2)
CANVAS = (np.sin(X * 2 * np.pi / (3.4 * K)) * np.sin(Y * 2 * np.pi / (3.4 * K))).astype(np.float32)
BRUSH = [gauss(streak_noise(PH, PW, 40, 5, a, 70 + i), 1.5) for i, a in enumerate((25, 155, 40))]
SOFT = fbm(PH, PW, 160, 5, 3) - 0.5
ORANGE = np.array([202, 124, 94]) / 255


def oil(img, k=0, amt=1.0):
    t = 1 + amt * (0.008 * CANVAS + 0.006 * BRUSH[k] + 0.018 * SOFT)
    return img * t[..., None]


def vgrad(top, bot, y0=0, y1=None):
    y1 = PH / K if y1 is None else y1
    t = np.clip((Y / K - y0) / (y1 - y0), 0, 1)[..., None]
    return np.asarray(top, np.float32) * (1 - t) + np.asarray(bot, np.float32) * t


def lance(cx, cy, length, width, ang):
    """竹叶：两头尖的披针形。"""
    pts = []
    for i in range(25):
        t = i / 24
        pts.append((t * length, width * math.sin(math.pi * t) ** 0.8 * (1 - 0.3 * t)))
    pts += [(t, -w) for t, w in pts[::-1]]
    a = math.radians(ang)
    return [(cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)) for x, y in pts]


def trunk(img, x0, w, col, fog, fogk, lean=0.0, y_top=-10, y_bot=640, nodes=110, seed=0):
    """一根竹竿：圆柱明暗（左亮右暗）+ 竹节。返回（新图, 遮罩）。"""
    g = np.random.default_rng(seed)
    yy = Y / K
    xc = x0 + lean * (yy - 320)
    u = (X / K - xc) / (w / 2)
    m = smooth(1.0, 0.92, np.abs(u)) * smooth(y_top - 1, y_top + 1, yy) * smooth(y_bot + 1, y_bot - 1, yy)
    shade = 1.08 - 0.42 * smooth(-0.9, 1.0, u) + 0.12 * np.exp(-((u + 0.45) / 0.18) ** 2)
    c = np.asarray(col, np.float32)[None, None] * shade[..., None]
    ny = g.uniform(0, nodes)
    node = np.zeros_like(yy)
    while ny < y_bot:
        node = np.maximum(node, np.exp(-((yy - ny) / 1.4) ** 2))
        c = c + (np.exp(-((yy - ny + 4) / 2.4) ** 2) * 0.10)[..., None]
        ny += nodes * g.uniform(0.85, 1.15)
    c = c * (1 - 0.45 * node)[..., None]
    c = c * (1 - fogk) + np.asarray(fog, np.float32) * fogk
    return over(img, c, m), m


def bamboo_scene():
    img = vgrad((0.70, 0.75, 0.77), (0.86, 0.85, 0.81), 0, 520)
    ground = vgrad((0.74, 0.74, 0.70), (0.47, 0.50, 0.45), 520, 630)
    img = np.where((Y / K > 520)[..., None], ground, img)
    hz = np.exp(-((Y / K - 520) / 26) ** 2)
    img = img * (1 - 0.35 * hz[..., None]) + np.array([0.88, 0.87, 0.83]) * 0.35 * hz[..., None]
    fog = np.array([0.84, 0.84, 0.81])
    for i in range(26):                                    # 远景竹：细、淡、融进雾
        img, _ = trunk(img, G.uniform(-20, 1140), G.uniform(8, 14), (0.38, 0.44, 0.40), fog, 0.62, G.uniform(-0.01, 0.01), y_bot=528, nodes=70, seed=i)
    for i in range(12):                                    # 中景
        img, _ = trunk(img, G.uniform(-20, 1140), G.uniform(18, 26), (0.22, 0.27, 0.23), fog, 0.32, G.uniform(-0.015, 0.015), y_bot=560, nodes=95, seed=40 + i)

    FX, FY, S = 548, 572, 15
    # 近处的一根粗竹本应挡在前面，却被画在 Clawd 身后
    img, _ = trunk(img, 318, 44, (0.10, 0.13, 0.11), fog, 0.0, 0.004, y_bot=640, nodes=130, seed=90)
    img, _ = trunk(img, 1010, 52, (0.09, 0.12, 0.10), fog, 0.0, -0.006, y_bot=640, nodes=140, seed=91)
    img, _ = trunk(img, 120, 40, (0.10, 0.13, 0.11), fog, 0.0, 0.006, y_bot=640, nodes=125, seed=92)
    # Clawd 的柔和投影
    sh = M(circle(FX + 40, FY + 2, 150, 64, 14))
    img *= (1 - 0.28 * gauss(sh, 10 * K))[..., None]

    # ---------- Clawd：高领红色薄纱长裙 ----------
    arms = [rigid_arm("L", 28), rigid_arm("R", -18)]
    check_proportions(arms=arms)
    cm, em = clawd(FX, FY, S, arms)
    body_shade = 1.06 - 0.22 * smooth(FX - 8 * S, FX + 8 * S, X / K)
    img = over(img, ORANGE[None, None] * body_shade[..., None], cm)
    img = over(img, (0.10, 0.07, 0.07), em)
    dress = [(-0.15, -0.12), (16.15, -0.12), (16.2, 7.6), (17.2, 11.9), (13.6, 11.9), (13.1, 8.6), (12.6, 11.9),
             (-1.1, 11.9), (-0.2, 7.6)]
    dm = gauss(garment(dress, FX, FY, S, smooth_k=0.15, arms=arms, name="dress"), 0.6 * K)
    fold = 0.5 + 0.5 * np.sin((X / K - FX) / S * 2.4) * smooth(FY - 6 * S, FY, Y / K)
    red = np.array([0.74, 0.07, 0.10])[None, None] * (0.86 + 0.22 * fold - 0.18 * smooth(FX - 8 * S, FX + 9 * S, X / K))[..., None]
    edge = np.clip(dm - gauss(dm, 5 * K) * 1.0, 0, 1) * 2.2
    dens = np.clip(0.34 + 0.16 * fold + 0.40 * edge, 0, 0.85)[..., None] * dm[..., None]
    img = img * (1 - dens) + (img * red * 1.55 + red * 0.06) * dens
    sheen = np.exp(-((X / K - (FX - 3 * S)) / (1.2 * S)) ** 2) * dm * 0.10
    img = img + sheen[..., None] * np.array([1.0, 0.8, 0.8])
    collar = garment([(-0.15, -0.12), (16.15, -0.12), (16.15, 0.9), (-0.15, 0.9)], FX, FY, S, name="collar")
    img = over(img, (0.62, 0.05, 0.08), collar * 0.85)
    # 胸前的白色刺绣花枝
    br = catmull(g2p([(3.4, 9.6), (5.2, 8.1), (7.0, 7.3), (8.9, 6.1), (11.6, 5.6)], FX, FY, S), 80)
    img = over(img, (0.96, 0.93, 0.90), SM(br, 1.3, 0.8) * 0.95)
    for (gx, gy, r) in [(5.6, 7.4, 0.42), (8.0, 6.3, 0.5), (10.5, 5.4, 0.38), (6.9, 8.4, 0.3), (11.7, 6.4, 0.3)]:
        (px, py), = g2p([(gx, gy)], FX, FY, S)
        for k in range(5):
            a = k * 2 * math.pi / 5 + 0.3
            img = over(img, (0.97, 0.95, 0.92), M(circle(px + math.cos(a) * r * S * 0.55, py + math.sin(a) * r * S * 0.55, r * S * 0.5, 20)) * 0.95)
        img = over(img, (0.92, 0.78, 0.40), M(circle(px, py, r * S * 0.25, 12)))

    # 飘落的竹叶，一片落在钳子尖上
    for (x, y, a) in [(330, 160, 40), (720, 120, -20), (860, 300, 70), (240, 420, 15), (680, 250, 120)]:
        img = over(img, (0.17, 0.25, 0.18), M(lance(x, y, 46, 6, a), 0.4))
    (ax, ay), = g2p([(-1.6, 2.9)], FX, FY, S)
    img = over(img, (0.18, 0.28, 0.19), M(lance(ax - 34, ay - 6, 44, 6, -12), 0.4))
    # 近景竹叶丛（画面顶部压下来）
    for i in range(18):
        x = 60 + i * 64 + G.uniform(-20, 20)
        img = over(img, (0.08, 0.12, 0.09), M(lance(x, G.uniform(-14, 24), G.uniform(70, 110), G.uniform(7, 10), G.uniform(60, 120)), 0.4))
    return oil(img, 0)


def cloud_blob(img, cx, cy, s, seed):
    """马格利特的云：柔软的白团，底部带一点灰蓝。"""
    g = np.random.default_rng(seed)
    m = np.zeros((PH, PW), np.float32)
    for _ in range(9):
        ox, oy, r = g.uniform(-90, 90) * s, g.uniform(-18, 14) * s, g.uniform(26, 48) * s
        m = np.maximum(m, M(circle(cx + ox, cy + oy, r, 48)))
    m = m * smooth(cy + 26 * s, cy + 10 * s, Y / K)
    m = gauss(m, 1.6 * K)
    inner = gauss(m, 14 * K)
    col = np.array([0.98, 0.98, 0.97])[None, None] * (1 - 0.22 * smooth(cy - 30 * s, cy + 24 * s, Y / K))[..., None]
    col = col * (0.92 + 0.08 * inner[..., None]) + np.array([0.0, 0.02, 0.06]) * smooth(cy, cy + 26 * s, Y / K)[..., None]
    return over(img, col, m)


def frog(img, cx, cy, s):
    """戴书生帽（幞头）的青蛙。"""
    body = [(cx - 40 * s, cy), (cx - 46 * s, cy - 22 * s), (cx - 30 * s, cy - 44 * s), (cx, cy - 50 * s), (cx + 30 * s, cy - 44 * s),
            (cx + 46 * s, cy - 22 * s), (cx + 40 * s, cy)]
    bm = M(body, 0.8)
    sh = 1.1 - 0.4 * smooth(cx - 46 * s, cx + 46 * s, X / K)
    img = over(img, np.array([0.30, 0.52, 0.24])[None, None] * sh[..., None], bm)
    belly = M(circle(cx, cy - 12 * s, 26 * s, 40, 13 * s), 0)
    img = over(img, (0.80, 0.80, 0.52), belly * 0.9)
    for sx in (-1, 1):                                     # 鼓起的腮、眼睛
        img = over(img, (0.36, 0.60, 0.30), M(circle(cx + sx * 34 * s, cy - 26 * s, 12 * s, 32)))
        img = over(img, (0.26, 0.46, 0.20), M(circle(cx + sx * 20 * s, cy - 50 * s, 11 * s, 32)))
        img = over(img, (0.95, 0.80, 0.30), M(circle(cx + sx * 20 * s, cy - 51 * s, 7 * s, 28)))
        img = over(img, (0.05, 0.05, 0.04), M(circle(cx + sx * 20 * s, cy - 51 * s, 2.6 * s, 16, 5.5 * s)))
        img = over(img, (0.24, 0.44, 0.19), M([(cx + sx * 40 * s, cy), (cx + sx * 58 * s, cy - 2 * s), (cx + sx * 60 * s, cy + 3 * s), (cx + sx * 34 * s, cy + 4 * s)], 0.5))
    hat = [(cx - 22 * s, cy - 58 * s), (cx - 20 * s, cy - 78 * s), (cx, cy - 86 * s), (cx + 20 * s, cy - 78 * s), (cx + 22 * s, cy - 58 * s)]
    img = over(img, (0.06, 0.06, 0.07), M(hat, 0.6))
    for sx in (-1, 1):                                     # 幞头两侧的长翅
        wing = [(cx + sx * 18 * s, cy - 70 * s), (cx + sx * 64 * s, cy - 76 * s), (cx + sx * 66 * s, cy - 70 * s), (cx + sx * 18 * s, cy - 64 * s)]
        img = over(img, (0.07, 0.07, 0.08), M(wing, 0.3))
    return img


def liaozhai_scene():
    # 白天的天空
    img = vgrad((0.36, 0.56, 0.84), (0.74, 0.85, 0.94), 0, 330)
    for (cx, cy, s, sd) in [(220, 96, 1.0, 1), (640, 60, 0.8, 2), (960, 140, 1.15, 3)]:
        img = cloud_blob(img, cx, cy, s, sd)
    # 夜：树丛剪影、楼阁、池水
    night = vgrad((0.05, 0.07, 0.09), (0.02, 0.03, 0.04), 280, 630)
    trees = np.zeros((PH, PW), np.float32)
    crowns = [(40, 330, 140, 0.95), (200, 300, 110, 0.95), (300, 360, 100, 0.9), (870, 360, 110, 0.9), (990, 300, 130, 0.95), (1110, 340, 120, 0.95), (150, 250, 46, 3.6)]
    for (cx, cy, r, sy) in crowns:
        trees = np.maximum(trees, M(circle(cx, cy, r, 80, r * sy)))
        for k in range(60):
            a = 2 * math.pi * k / 60 + G.uniform(-0.04, 0.04)
            rr = G.uniform(0.07, 0.13) * r
            trees = np.maximum(trees, M(circle(cx + r * math.cos(a), cy + r * sy * math.sin(a), rr, 18)))
    trees = np.maximum(trees, M([(0, 360), (1120, 360), (1120, 640), (0, 640)]))
    trees = gauss(trees, 1.2 * K)
    leaf = fbm(PH, PW, 7, 21, 3)
    lit = np.clip(trees - ishift(trees, 6 * K, 0), 0, 1)      # 树冠上沿被天光照到的一圈
    tcol = night * (0.8 + 0.45 * leaf[..., None]) + np.array([0.05, 0.07, 0.08]) * gauss(lit, 2 * K)[..., None]
    img = over(img, tcol, trees)
    # 楼阁：歇山顶、檐角上翘
    # 墙面与门扇
    img = over(img, (0.09, 0.065, 0.055), M([(360, 330), (760, 330), (760, 548), (360, 548)]))
    for x0 in (366, 726):
        for k in range(6):
            img = over(img, (0.13, 0.09, 0.07), SM([(x0 + 4, 350 + k * 32), (x0 + 26, 350 + k * 32)], 1.2, 0, False))
    top = [(288, 292), (330, 306), (380, 312), (430, 300), (470, 276), (498, 252), (512, 238), (518, 228), (602, 228), (608, 238),
           (622, 252), (650, 276), (690, 300), (740, 312), (790, 306), (832, 292)]
    eave = [(824, 302), (780, 318), (700, 326), (560, 330), (420, 326), (340, 318), (296, 302)]
    img = over(img, (0.055, 0.055, 0.065), M(top + eave, 0.25))
    for k in range(22):                                   # 瓦垄
        t = (k + 0.5) / 22
        x_e = 330 + 460 * t
        x_r = 518 + 84 * t
        img = over(img, (0.10, 0.10, 0.12), SM([(x_r, 234), ((x_r + x_e) / 2, 286), (x_e, 322)], 1.0, 0.8, False) * 0.8)
    img = over(img, (0.03, 0.03, 0.04), M([(512, 224), (608, 224), (612, 232), (508, 232)]))
    for sx in (-1, 1):                                    # 正脊两端的鸱吻
        x = 560 + sx * 52
        img = over(img, (0.04, 0.04, 0.05), M([(x, 232), (x + sx * 4, 210), (x + sx * 12, 206), (x + sx * 10, 222), (x + sx * 8, 232)], 0.4))
    for x in (382, 738):
        img = over(img, (0.09, 0.07, 0.06), M([(x - 9, 336), (x + 9, 336), (x + 9, 548), (x - 9, 548)]))
    # 窗洞：屋里透出暖红的光
    WX0, WY0, WX1, WY1 = 400, 340, 720, 528
    inner = vgrad((0.62, 0.20, 0.10), (0.92, 0.50, 0.22), WY0, WY1)
    glow = np.exp(-(((X / K - 560) / 170) ** 2 + ((Y / K - 470) / 90) ** 2))
    inner = inner * (0.8 + 0.4 * glow[..., None])
    wm = M([(WX0, WY0), (WX1, WY0), (WX1, WY1), (WX0, WY1)])
    img = over(img, inner, wm)
    # 屋内：书案、蜡烛、书、青蛙、Clawd
    img = over(img, (0.20, 0.08, 0.05), M([(592, 470), (708, 470), (708, 482), (592, 482)]))
    for x in (600, 698):
        img = over(img, (0.18, 0.07, 0.05), M([(x - 4, 482), (x + 4, 482), (x + 4, 528), (x - 4, 528)]))
    img = over(img, (0.96, 0.90, 0.78), M([(612, 466), (640, 462), (644, 470), (612, 470)]))
    img = over(img, (0.94, 0.88, 0.76), M([(612, 466), (586, 462), (582, 470), (612, 470)]))
    img = over(img, (0.92, 0.86, 0.72), M([(690, 430), (697, 430), (697, 470), (690, 470)]))
    flame = np.exp(-(((X / K - 693.5) / 4.5) ** 2 + ((Y / K - 422) / 9) ** 2))
    img = img + (flame * 1.2)[..., None] * np.array([1.0, 0.85, 0.5]) + (np.exp(-(((X / K - 693) / 60) ** 2 + ((Y / K - 430) / 60) ** 2)) * 0.25)[..., None] * np.array([1, 0.6, 0.3])
    img = frog(img, 650, 468, 0.55)
    FX, FY, S = 502, 528, 9.5
    arms = [rigid_arm("L", 0), rigid_arm("R", 32)]
    cm, em = clawd(FX, FY, S, arms)
    # 烛光从右边来：右侧亮、左侧沉进暖暗里；身体与书案、墙之间加一圈柔和的接触阴影
    rim = 0.62 + 0.55 * smooth(FX - 8 * S, FX + 9 * S, X / K)
    img *= (1 - 0.35 * gauss(cm, 6 * K) * (1 - cm))[..., None]
    img = over(img, ORANGE[None, None] * rim[..., None] * np.array([1.08, 0.86, 0.72]), cm)
    img = img + (gauss(np.clip(cm - ishift(cm, 0, -3 * K), 0, 1), 1.0 * K) * 0.35)[..., None] * np.array([1.0, 0.7, 0.4])
    img = over(img, (0.08, 0.05, 0.04), em)
    robe = [(-0.3, 5.4), (16.3, 5.4), (17.0, 11.9), (-1.0, 11.9)]
    rm = garment(robe, FX, FY, S, arms=arms, name="robe")
    img = over(img, np.array([0.98, 0.90, 0.80])[None, None] * (0.70 + 0.4 * smooth(FX - 8 * S, FX + 9 * S, X / K))[..., None], rm)
    for gx in (3.0, 7.0, 11.0, 14.5):                     # 白袍的竖褶阴影
        img = img * (1 - 0.10 * gauss(SM(g2p([(gx, 6.0), (gx + 0.3, 11.8)], FX, FY, S), 1.2, 0, False), 2 * K) * rm)[..., None]
    for (gx, gy) in [(3.6, 8.2), (6.4, 7.2), (9.0, 6.4), (12.6, 8.8), (14.2, 10.6), (2.2, 10.4), (10.8, 10.2)]:
        (px, py), = g2p([(gx, gy)], FX, FY, S)
        for k in range(5):                                 # 白袍上暗色的五瓣梅
            a = k * 2 * math.pi / 5
            img = over(img, (0.36, 0.24, 0.28), M(circle(px + 2.4 * math.cos(a), py + 2.4 * math.sin(a), 1.9, 14)) * rm)
        img = over(img, (0.80, 0.62, 0.30), M(circle(px, py, 0.9, 10)) * rm)
    br = g2p([(0.6, 11.4), (3.6, 8.2), (6.4, 7.2), (9.0, 6.4), (11.0, 6.0)], FX, FY, S)
    img = over(img, (0.32, 0.24, 0.24), SM(catmull(br, 40), 0.9, 0.8) * rm)
    # 窗格：冰裂纹边框
    frame_m = np.clip(wm - M([(WX0 + 16, WY0 + 16), (WX1 - 16, WY0 + 16), (WX1 - 16, WY1), (WX0 + 16, WY1)]), 0, 1)
    lat = np.zeros_like(frame_m)
    for i in range(40):
        a = G.uniform(0, math.pi)
        x, y = G.uniform(WX0, WX1), G.uniform(WY0, WY1)
        lat = np.maximum(lat, SM([(x - 30 * math.cos(a), y - 30 * math.sin(a)), (x + 30 * math.cos(a), y + 30 * math.sin(a))], 2.2, 0, False))
    img = over(img, (0.11, 0.06, 0.04), np.clip(frame_m * 0.85 + lat * frame_m, 0, 1))
    img = over(img, (0.12, 0.06, 0.04), outline(wm, 3))
    # 檐下的红灯笼（对应《光之帝国》的路灯）
    for lx in (322, 798):
        img = over(img, (0.05, 0.04, 0.04), SM([(lx, 306), (lx, 340)], 1.0, 0, False))
        lm = M(circle(lx, 362, 15, 48, 20))
        lc = np.array([0.95, 0.30, 0.14])[None, None] * (1.15 - 0.55 * smooth(0, 1, np.abs(X / K - lx) / 15))[..., None]
        img = over(img, lc, lm)
        for dx in (-8, 0, 8):
            img = over(img, (0.55, 0.10, 0.07), SM([(lx + dx * 0.6, 343), (lx + dx, 362), (lx + dx * 0.6, 381)], 0.7, 0.8, False) * lm)
        img = over(img, (0.12, 0.06, 0.04), M([(lx - 9, 340), (lx + 9, 340), (lx + 9, 345), (lx - 9, 345)]))
        img = over(img, (0.12, 0.06, 0.04), M([(lx - 9, 379), (lx + 9, 379), (lx + 9, 384), (lx - 9, 384)]))
        img = over(img, (0.70, 0.12, 0.08), SM([(lx, 384), (lx, 402)], 1.6, 0, False))
        g_ = np.exp(-(((X / K - lx) / 52) ** 2 + ((Y / K - 362) / 52) ** 2))
        img = img + (g_ * 0.30)[..., None] * np.array([1.0, 0.38, 0.18])
    # 池水与倒影
    pond = M([(0, 560), (1120, 560), (1120, 640), (0, 640)])
    water = vgrad((0.03, 0.04, 0.06), (0.02, 0.025, 0.035), 560, 630)
    img = over(img, water, pond)
    for i in range(26):
        y = 566 + i * 2.6
        w = 150 * (1 - i / 30) + G.uniform(-14, 14)
        x = 560 + G.uniform(-10, 10)
        img = over(img, (0.70, 0.32, 0.14), SM([(x - w, y), (x + w, y)], 0.9, 0, False) * (0.55 - i * 0.017))
    return oil(img, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    wall = (0.24, 0.34, 0.30)
    a = np.clip(downsample(bamboo_scene(), K), 0, 1)
    frame.save(frame.compose(a, wall, "纯妹妹"), os.path.join(out_dir, "02a_纯妹妹_竹林红裙.png"))
    frame.save(a, os.path.join(out_dir, "02a_画心.png"))
    b = np.clip(downsample(liaozhai_scene(), K), 0, 1)
    frame.save(frame.compose(b, wall, "纯妹妹"), os.path.join(out_dir, "02b_纯妹妹_聊斋.png"))
    frame.save(b, os.path.join(out_dir, "02b_画心.png"))
