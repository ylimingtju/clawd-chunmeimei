"""02 纯妹妹 · 马格利特式超现实油画 · 117 BPM（0:17.51–0:27.77）。拍号 b 从 01 结束的那一刻算起。
E  b0–3   画里的镜头往后拉：01 的拱门舞台原来是竹林里画架上的一块画布。Clawd 从画布里走出来，
          走出画布边缘的部分变成油画质感、换上红裙（手里的麦克风留在画里）；它一出画布，画布上的画就变成
          身后那片竹林的延续（《人类的处境》）。墙色和展签在它完全走出画布的那一拍切换。
a  b3–11  雾里的墨竹林，横移跟拍、分层视差。近处的粗竹本该挡在前面，却从 Clawd 身后滑过（《空白签名》）。
          竹叶按拍点飘落，Clawd 用左钳接住一片，原地转一圈，红裙下摆飘起来。
M  b11–14 Clawd 像拨门帘一样拨开两根竹竿，竹缝里是白天天空下的夜楼阁（《光之帝国》）；
          它走进竹缝，越过去的部分换上聊斋素白长衣，竹竿一路向两边分开，整幅画变成楼阁。
b  b14–20 楼阁里书生在案前读书，烛光摇曳。Clawd 走近，用钳子点了一下书，书生在重拍上「噗」地变成戴帽的青蛙。
          青蛙每 2 拍跳一下，Clawd 学它跳；最后青蛙鼓起腮。画框里慢慢推近。"""
import math

import numpy as np
from scipy.ndimage import map_coordinates

from kit import (K, PW, PH, X, Y, M, SM, circle, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, smooth, over, downsample, ishift, outline)
import anim_core as A
import frame
import S02_马格利特 as S2
from S02_马格利特 import oil, vgrad, lance, trunk, cloud_blob, ORANGE
import A01_还有什么更好的 as A1
from S01_瑞士海报 import Press

FPS = 30
T0 = A1.T0 + 13 * A1.BEAT
BEAT = 60 / 117
WALL1, LABEL1 = A1.WALL1, A1.LABEL1
WALL2, LABEL2 = (0.24, 0.34, 0.30), "纯妹妹"
FOG = np.array([0.84, 0.84, 0.81])
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 20 * BEAT) * FPS)

# 画架上的画布（E 段结束时的位置和大小）
SC, OX, OY = 0.45, 120, 104
ZC, ZOOM01 = (560, 400), 1.08                       # 01 结尾的推镜

_CACHE = {}
G = np.random.default_rng(22)
# 竹子：世界坐标（随镜头横移，视差系数不同）
FAR = [(G.uniform(-200, 1900), G.uniform(8, 14), G.uniform(-0.01, 0.01), i) for i in range(40)]
MID = [(G.uniform(-200, 1900), G.uniform(18, 26), G.uniform(-0.015, 0.015), 40 + i) for i in range(18)]
NEAR = [(318, 44, 0.004, 90), (1010, 52, -0.006, 91), (120, 40, 0.006, 92), (1500, 48, 0.003, 93), (1900, 46, -0.004, 94), (-260, 44, 0.004, 95)]
GATE = [(1308, 40), (1380, 42)]                     # M 段被拨开的两根（世界坐标，近景层）
PAN = 95.0                                          # a 段每拍横移的像素


def bt(t):
    return (t - T0) / BEAT


def cam_x(b):
    """镜头（=Clawd 的世界位置 - 548）的横移量：a 段匀速跟拍，接住竹叶转圈时慢一点，M 段前停住。"""
    if b < 3:
        return 0.0
    run = A.seg(b, 3.0, 11.0)
    return PAN * 8 * (0.9 * run + 0.1 * A.ease_io(run))


# ---------- 竹林 ----------
def ground():
    if "ground" not in _CACHE:
        img = vgrad((0.70, 0.75, 0.77), (0.86, 0.85, 0.81), 0, 520)
        gr = vgrad((0.74, 0.74, 0.70), (0.47, 0.50, 0.45), 520, 630)
        img = np.where((Y / K > 520)[..., None], gr, img)
        hz = np.exp(-((Y / K - 520) / 26) ** 2)
        img = img * (1 - 0.35 * hz[..., None]) + np.array([0.88, 0.87, 0.83]) * 0.35 * hz[..., None]
        _CACHE["ground"] = img.astype(np.float32)
    return _CACHE["ground"]


def grove(cam, gate_bend=0.0, gate_spread=0.0):
    """竹林背景（不含 Clawd）。gate_bend/gate_spread：被拨开的两根竹竿往两边弯/移。返回 (img, 竹缝遮罩)。"""
    img = ground().copy()
    for (x, w, lean, sd) in FAR:
        xs = x - cam * 0.35
        if -40 < xs < 1160:
            img, _ = trunk(img, xs, w, (0.38, 0.44, 0.40), FOG, 0.62, lean, y_bot=528, nodes=70, seed=sd)
    for (x, w, lean, sd) in MID:
        xs = x - cam * 0.65
        if -40 < xs < 1160:
            img, _ = trunk(img, xs, w, (0.22, 0.27, 0.23), FOG, 0.32, lean, y_bot=560, nodes=95, seed=sd)
    for (x, w, lean, sd) in NEAR:                     # 近景粗竹：移得比 Clawd 还快，却在它身后
        xs = x - cam * 1.25
        if -60 < xs < 1180:
            img, _ = trunk(img, xs, w, (0.10, 0.13, 0.11), FOG, 0.0, lean, y_bot=640, nodes=130, seed=sd)
    gap = None
    if cam > 300:
        xl, xr = GATE[0][0] - cam * 1.0, GATE[1][0] - cam * 1.0
        yy = Y / K
        # 竹竿根扎在地里：中段（Clawd 的高度）被拨得最开，往上也张开，根部几乎不动
        d = (yy - 400) / np.where(yy < 400, 700.0, 260.0)
        prof = np.clip(1 - d ** 2, 0, 1)
        cl = xl - (gate_bend * prof + gate_spread)
        cr = xr + (gate_bend * prof + gate_spread)
        for c, w, sd in ((cl, GATE[0][1], 96), (cr, GATE[1][1], 97)):
            u = (X / K - c) / (w / 2)
            m = smooth(1.0, 0.92, np.abs(u))
            shade = 1.08 - 0.42 * smooth(-0.9, 1.0, u) + 0.12 * np.exp(-((u + 0.45) / 0.18) ** 2)
            col = np.array([0.10, 0.13, 0.11])[None, None] * shade[..., None]
            nodes = np.zeros_like(yy)
            for ny in np.arange(40 + sd % 7 * 9, 640, 128):
                nodes = np.maximum(nodes, np.exp(-((yy - ny) / 1.4) ** 2))
            col = col * (1 - 0.45 * nodes)[..., None]
            img = over(img, col, m)
        gap = smooth(cl + GATE[0][1] / 2 - 0.5, cl + GATE[0][1] / 2 + 0.5, X / K) * smooth(cr - GATE[1][1] / 2 + 0.5, cr - GATE[1][1] / 2 - 0.5, X / K)
    return img, gap


def leaves(img, b, cam):
    """按拍点飘落的竹叶（屏幕空间，带一点视差）。"""
    g = np.random.default_rng(7)
    for k in range(0, 20):
        x0, a0, sp = g.uniform(160, 1080), g.uniform(0, 180), g.uniform(0.8, 1.2)
        u = (b - (1 + k * 0.5)) / 2.2
        if not 0 <= u <= 1:
            continue
        x = x0 + 40 * math.sin(u * 7 + k) - cam * 0.2 * (u)
        y = -20 + 660 * u * sp
        img = over(img, (0.17, 0.25, 0.18), M(lance(x, y, 44, 6, a0 + 140 * u), 0.4))
    return img


def top_leaves(img, cam):
    g = np.random.default_rng(3)
    for i in range(30):
        x = (60 + i * 64 + g.uniform(-20, 20) - cam * 1.4) % 1920 - 300
        if -120 < x < 1240:
            img = over(img, (0.08, 0.12, 0.09), M(lance(x, g.uniform(-14, 24), g.uniform(70, 110), g.uniform(7, 10), g.uniform(60, 120)), 0.4))
    return img


# ---------- Clawd：红裙（油画） ----------
def feet_from_center(cx, cy, S, ang):
    a = math.radians(ang)
    return cx - 6 * S * math.sin(a), cy + 6 * S * math.cos(a)


def red_clawd(img, FX, FY, S, ang=0.0, arms=None, flare=0.0, mask=None, leaf=False):
    arms = arms or [rigid_arm("L", 28), rigid_arm("R", -18)]
    check_proportions(arms=arms)
    mm = 1.0 if mask is None else mask
    cm, em = clawd(FX, FY, S, arms, ang)
    sh = M(circle(FX + 40, max(FY, 560) + 2, 150 * S / 15, 64, 14))      # 柔和投影（落在地上）
    img = img * (1 - 0.28 * gauss(sh, 10 * K) * (mm if mask is not None else 1))[..., None]
    body_shade = 1.06 - 0.22 * smooth(FX - 8 * S, FX + 8 * S, X / K)
    img = over(img, ORANGE[None, None] * body_shade[..., None], cm * mm)
    img = over(img, (0.10, 0.07, 0.07), em * mm)
    fl = flare * 1.6
    dress = [(-0.15, -0.12), (16.15, -0.12), (16.2, 7.6), (17.2 + fl, 11.9), (13.6, 11.9), (13.1, 8.6), (12.6, 11.9),
             (-1.1 - fl, 11.9), (-0.2, 7.6)]
    dm = gauss(garment(dress, FX, FY, S, ang, smooth_k=0.15, arms=arms, name="dress"), 0.6 * K) * mm
    fold = 0.5 + 0.5 * np.sin((X / K - FX) / S * 2.4) * smooth(FY - 6 * S, FY, Y / K)
    red = np.array([0.74, 0.07, 0.10])[None, None] * (0.86 + 0.22 * fold - 0.18 * smooth(FX - 8 * S, FX + 9 * S, X / K))[..., None]
    edge = np.clip(dm - gauss(dm, 5 * K) * 1.0, 0, 1) * 2.2
    dens = np.clip(0.34 + 0.16 * fold + 0.40 * edge, 0, 0.85)[..., None] * dm[..., None]
    img = img * (1 - dens) + (img * red * 1.55 + red * 0.06) * dens
    img = img + (np.exp(-((X / K - (FX - 3 * S)) / (1.2 * S)) ** 2) * dm * 0.10)[..., None] * np.array([1.0, 0.8, 0.8])
    collar = garment([(-0.15, -0.12), (16.15, -0.12), (16.15, 0.9), (-0.15, 0.9)], FX, FY, S, ang, name="collar")
    img = over(img, (0.62, 0.05, 0.08), collar * 0.85 * mm)
    br = catmull(g2p([(3.4, 9.6), (5.2, 8.1), (7.0, 7.3), (8.9, 6.1), (11.6, 5.6)], FX, FY, S, ang), 80)
    img = over(img, (0.96, 0.93, 0.90), SM(br, 1.3 * S / 15, 0.8) * 0.95 * mm)
    for (gx, gy, r) in [(5.6, 7.4, 0.42), (8.0, 6.3, 0.5), (10.5, 5.4, 0.38), (6.9, 8.4, 0.3), (11.7, 6.4, 0.3)]:
        (px, py), = g2p([(gx, gy)], FX, FY, S, ang)
        for k in range(5):
            a = k * 2 * math.pi / 5 + 0.3
            img = over(img, (0.97, 0.95, 0.92), M(circle(px + math.cos(a) * r * S * 0.55, py + math.sin(a) * r * S * 0.55, r * S * 0.5, 20)) * 0.95 * mm)
        img = over(img, (0.92, 0.78, 0.40), M(circle(px, py, r * S * 0.25, 12)) * mm)
    if leaf:                                            # 接住的那片竹叶：粘在左钳尖上
        (ax, ay), = g2p([(-1.6, 2.9)], FX, FY, S, ang)
        img = over(img, (0.18, 0.28, 0.19), M(lance(ax - 34 * S / 15, ay - 6, 44 * S / 15, 6 * S / 15, -12 + ang), 0.4) * mm)
    return img


# ---------- 聊斋楼阁 ----------
def liaozhai_static():
    """楼阁整幅（不含 Clawd、书生/青蛙、烛焰），以及压在最上面的窗格。"""
    if "lz" in _CACHE:
        return _CACHE["lz"]
    img = vgrad((0.36, 0.56, 0.84), (0.74, 0.85, 0.94), 0, 330)
    for (cx, cy, s, sd) in [(220, 96, 1.0, 1), (640, 60, 0.8, 2), (960, 140, 1.15, 3)]:
        img = cloud_blob(img, cx, cy, s, sd)
    night = vgrad((0.05, 0.07, 0.09), (0.02, 0.03, 0.04), 280, 630)
    trees = np.zeros((PH, PW), np.float32)
    g = np.random.default_rng(5)
    for (cx, cy, r, sy) in [(40, 330, 140, 0.95), (200, 300, 110, 0.95), (300, 360, 100, 0.9), (870, 360, 110, 0.9), (990, 300, 130, 0.95), (1110, 340, 120, 0.95), (150, 250, 46, 3.6)]:
        trees = np.maximum(trees, M(circle(cx, cy, r, 80, r * sy)))
        for k in range(60):
            a = 2 * math.pi * k / 60 + g.uniform(-0.04, 0.04)
            rr = g.uniform(0.07, 0.13) * r
            trees = np.maximum(trees, M(circle(cx + r * math.cos(a), cy + r * sy * math.sin(a), rr, 18)))
    trees = np.maximum(trees, M([(0, 360), (1120, 360), (1120, 640), (0, 640)]))
    trees = gauss(trees, 1.2 * K)
    leaf = fbm(PH, PW, 7, 21, 3)
    lit = np.clip(trees - ishift(trees, 6 * K, 0), 0, 1)
    img = over(img, night * (0.8 + 0.45 * leaf[..., None]) + np.array([0.05, 0.07, 0.08]) * gauss(lit, 2 * K)[..., None], trees)
    img = over(img, (0.09, 0.065, 0.055), M([(360, 330), (760, 330), (760, 548), (360, 548)]))
    for x0 in (366, 726):
        for k in range(6):
            img = over(img, (0.13, 0.09, 0.07), SM([(x0 + 4, 350 + k * 32), (x0 + 26, 350 + k * 32)], 1.2, 0, False))
    top = [(288, 292), (330, 306), (380, 312), (430, 300), (470, 276), (498, 252), (512, 238), (518, 228), (602, 228), (608, 238),
           (622, 252), (650, 276), (690, 300), (740, 312), (790, 306), (832, 292)]
    eave = [(824, 302), (780, 318), (700, 326), (560, 330), (420, 326), (340, 318), (296, 302)]
    img = over(img, (0.055, 0.055, 0.065), M(top + eave, 0.25))
    for k in range(22):
        t = (k + 0.5) / 22
        img = over(img, (0.10, 0.10, 0.12), SM([(518 + 84 * t, 234), ((518 + 84 * t + 330 + 460 * t) / 2, 286), (330 + 460 * t, 322)], 1.0, 0.8, False) * 0.8)
    img = over(img, (0.03, 0.03, 0.04), M([(512, 224), (608, 224), (612, 232), (508, 232)]))
    for sx in (-1, 1):
        x = 560 + sx * 52
        img = over(img, (0.04, 0.04, 0.05), M([(x, 232), (x + sx * 4, 210), (x + sx * 12, 206), (x + sx * 10, 222), (x + sx * 8, 232)], 0.4))
    for x in (382, 738):
        img = over(img, (0.09, 0.07, 0.06), M([(x - 9, 336), (x + 9, 336), (x + 9, 548), (x - 9, 548)]))
    WX0, WY0, WX1, WY1 = 400, 340, 720, 528
    inner = vgrad((0.62, 0.20, 0.10), (0.92, 0.50, 0.22), WY0, WY1)
    glow = np.exp(-(((X / K - 560) / 170) ** 2 + ((Y / K - 470) / 90) ** 2))
    wm = M([(WX0, WY0), (WX1, WY0), (WX1, WY1), (WX0, WY1)])
    img = over(img, inner * (0.8 + 0.4 * glow[..., None]), wm)
    img = over(img, (0.20, 0.08, 0.05), M([(592, 470), (708, 470), (708, 482), (592, 482)]))
    for x in (600, 698):
        img = over(img, (0.18, 0.07, 0.05), M([(x - 4, 482), (x + 4, 482), (x + 4, 528), (x - 4, 528)]))
    img = over(img, (0.92, 0.86, 0.72), M([(690, 430), (697, 430), (697, 470), (690, 470)]))     # 蜡烛
    # 窗格（最后压在上面）
    frame_m = np.clip(wm - M([(WX0 + 16, WY0 + 16), (WX1 - 16, WY0 + 16), (WX1 - 16, WY1), (WX0 + 16, WY1)]), 0, 1)
    lat = np.zeros_like(frame_m)
    for i in range(40):
        a = g.uniform(0, math.pi)
        x, y = g.uniform(WX0, WX1), g.uniform(WY0, WY1)
        lat = np.maximum(lat, SM([(x - 30 * math.cos(a), y - 30 * math.sin(a)), (x + 30 * math.cos(a), y + 30 * math.sin(a))], 2.2, 0, False))
    lat_a = np.clip(np.clip(frame_m * 0.85 + lat * frame_m, 0, 1) + outline(wm, 3), 0, 1)
    # 灯笼、池水
    post = np.zeros_like(img)
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
        img = img + (np.exp(-(((X / K - lx) / 52) ** 2 + ((Y / K - 362) / 52) ** 2)) * 0.30)[..., None] * np.array([1.0, 0.38, 0.18])
    img = over(img, vgrad((0.03, 0.04, 0.06), (0.02, 0.025, 0.035), 560, 630), M([(0, 560), (1120, 560), (1120, 640), (0, 640)]))
    for i in range(26):
        y = 566 + i * 2.6
        w = 150 * (1 - i / 30) + g.uniform(-14, 14)
        x = 560 + g.uniform(-10, 10)
        img = over(img, (0.70, 0.32, 0.14), SM([(x - w, y), (x + w, y)], 0.9, 0, False) * (0.55 - i * 0.017))
    _CACHE["lz"] = (img.astype(np.float32), lat_a.astype(np.float32), wm)
    return _CACHE["lz"]


def candle(img, t):
    fl = 1.0 + 0.10 * math.sin(t * 23) + 0.06 * math.sin(t * 37 + 1) + 0.05 * math.sin(t * 61 + 2)
    dx = 1.2 * math.sin(t * 17)
    flame = np.exp(-(((X / K - 693.5 - dx) / 4.5) ** 2 + ((Y / K - 422) / (9 * fl)) ** 2))
    img = img + (flame * 1.2)[..., None] * np.array([1.0, 0.85, 0.5])
    img = img + (np.exp(-(((X / K - 693) / 60) ** 2 + ((Y / K - 430) / 60) ** 2)) * 0.25 * fl)[..., None] * np.array([1, 0.6, 0.3])
    return img


def scholar(img, cx, base):
    """书生：坐在案后，只露上半身；素色长衫、幞头，双手捧着一卷书。烛光从右边来。"""
    lit = lambda c, x0, x1: np.asarray(c)[None, None] * (0.72 + 0.5 * smooth(x0, x1, X / K))[..., None]
    torso = [(cx - 24, base), (cx - 20, base - 30), (cx - 12, base - 38), (cx + 12, base - 38), (cx + 20, base - 30), (cx + 24, base)]
    img = over(img, lit((0.34, 0.36, 0.42), cx - 24, cx + 26), M(torso, 0.5))
    img = over(img, (0.70, 0.62, 0.50), M([(cx - 6, base - 38), (cx, base - 30), (cx + 6, base - 38)]))      # 交领
    img = over(img, lit((0.86, 0.68, 0.54), cx - 10, cx + 10), M(circle(cx, base - 48, 10, 40, 11)))
    for sx in (-1, 1):
        img = over(img, (0.10, 0.07, 0.06), M(circle(cx + sx * 3.6, base - 48, 1.0, 10)))
    img = over(img, (0.06, 0.06, 0.07), M([(cx - 10, base - 54), (cx - 9, base - 64), (cx, base - 68), (cx + 9, base - 64), (cx + 10, base - 54)], 0.6))
    for sx in (-1, 1):
        img = over(img, (0.07, 0.07, 0.08), M([(cx + sx * 8, base - 62), (cx + sx * 34, base - 65), (cx + sx * 35, base - 62), (cx + sx * 8, base - 59)], 0.3))
    book = [(cx - 16, base - 16), (cx, base - 12), (cx + 16, base - 16), (cx + 16, base - 4), (cx, base), (cx - 16, base - 4)]
    img = over(img, lit((0.92, 0.86, 0.72), cx - 16, cx + 16), M(book))
    img = over(img, (0.40, 0.30, 0.24), SM([(cx, base - 12), (cx, base)], 0.6, 0, False))
    for sx in (-1, 1):                                     # 袖口捧着书
        img = over(img, lit((0.30, 0.32, 0.38), cx - 24, cx + 26), M(circle(cx + sx * 18, base - 8, 6, 20, 5)))
    return img


def frog(img, cx, cy, s, puff=0.0):
    """S2.frog，加上「鼓腮」：puff 0..1，两腮鼓成两个大泡。"""
    img = S2.frog(img, cx, cy, s)
    if puff > 0:
        r = (12 + 9 * puff) * s
        for sx in (-1, 1):
            m = M(circle(cx + sx * (34 + 4 * puff) * s, cy - 24 * s, r, 32))
            img = over(img, np.array([0.55, 0.72, 0.40])[None, None] * (0.9 + 0.2 * smooth(cx - 60 * s, cx + 60 * s, X / K))[..., None], m)
            img = over(img, (0.85, 0.92, 0.75), M(circle(cx + sx * (34 + 4 * puff) * s - 3 * s, cy - 28 * s, r * 0.3, 16)) * 0.6)
    return img


def robe_clawd(img, FX, FY, S, ang=0.0, arms=None, mask=None):
    arms = arms or [rigid_arm("L", 0), rigid_arm("R", 32)]
    mm = 1.0 if mask is None else mask
    cm, em = clawd(FX, FY, S, arms, ang)
    rim = 0.62 + 0.55 * smooth(FX - 8 * S, FX + 9 * S, X / K)
    img *= (1 - 0.35 * gauss(cm, 6 * K) * (1 - cm) * mm)[..., None]
    img = over(img, ORANGE[None, None] * rim[..., None] * np.array([1.08, 0.86, 0.72]), cm * mm)
    img = img + (gauss(np.clip(cm - ishift(cm, 0, -3 * K), 0, 1), 1.0 * K) * 0.35 * mm)[..., None] * np.array([1.0, 0.7, 0.4])
    img = over(img, (0.08, 0.05, 0.04), em * mm)
    rm = garment([(-0.3, 5.4), (16.3, 5.4), (17.0, 11.9), (-1.0, 11.9)], FX, FY, S, ang, arms=arms, name="robe") * mm
    img = over(img, np.array([0.98, 0.90, 0.80])[None, None] * (0.70 + 0.4 * smooth(FX - 8 * S, FX + 9 * S, X / K))[..., None], rm)
    for gx in (3.0, 7.0, 11.0, 14.5):
        img = img * (1 - 0.10 * gauss(SM(g2p([(gx, 6.0), (gx + 0.3, 11.8)], FX, FY, S, ang), 1.2, 0, False), 2 * K) * rm)[..., None]
    for (gx, gy) in [(3.6, 8.2), (6.4, 7.2), (9.0, 6.4), (12.6, 8.8), (14.2, 10.6), (2.2, 10.4), (10.8, 10.2)]:
        (px, py), = g2p([(gx, gy)], FX, FY, S, ang)
        r = S / 9.5
        for k in range(5):
            a = k * 2 * math.pi / 5
            img = over(img, (0.36, 0.24, 0.28), M(circle(px + 2.4 * r * math.cos(a), py + 2.4 * r * math.sin(a), 1.9 * r, 14)) * rm)
        img = over(img, (0.80, 0.62, 0.30), M(circle(px, py, 0.9 * r, 10)) * rm)
    br = g2p([(0.6, 11.4), (3.6, 8.2), (6.4, 7.2), (9.0, 6.4), (11.0, 6.0)], FX, FY, S, ang)
    img = over(img, (0.32, 0.24, 0.24), SM(catmull(br, 40), 0.9 * S / 9.5, 0.8) * rm)
    return img


# ---------- E 段：画架上的画布 ----------
def canvas_rect(b):
    u = A.ease_io(A.seg(b, 0.0, 1.3))
    sc = A.lerp(1.0, SC, u)
    ox, oy = A.lerp(0, OX, u), A.lerp(0, OY, u)
    return sc, ox, oy


def to_canvas(p, sc, ox, oy):
    """01 画心坐标（推镜前）→ 当前画心坐标。"""
    zx = ZC[0] + (p[0] - ZC[0]) * ZOOM01
    zy = ZC[1] + (p[1] - ZC[1]) * ZOOM01
    return ox + sc * zx, oy + sc * zy


def stage_on_canvas(sc, ox, oy):
    """把 01 的拱门舞台（K 倍分辨率）按画布的位置和大小贴进当前画心。"""
    src = A1.stage_static()
    qx = ZC[0] + ((X / K - ox) / sc - ZC[0]) / ZOOM01
    qy = ZC[1] + ((Y / K - oy) / sc - ZC[1]) / ZOOM01
    out = np.stack([map_coordinates(src[..., c], [qy * K, qx * K], order=1, mode="nearest") for c in range(3)], -1)
    return out


def easel(img, sc, ox, oy):
    """画架：三条木腿、托住画布的横档、顶上的夹子；画布右边和下边露出一点厚度。"""
    x0, y0, x1, y1 = ox, oy, ox + 1120 * sc, oy + 630 * sc
    wood = lambda x0_, x1_: np.array([0.36, 0.24, 0.15])[None, None] * (1.15 - 0.45 * smooth(x0_, x1_, X / K))[..., None]
    cxm = (x0 + x1) / 2
    for (xa, xb, w) in [(cxm - 20, x0 + 40, 14), (cxm + 20, x1 - 40, 14), (cxm, cxm + 6, 12)]:
        top = (xa, y0 - 30)
        bot = (xb + (xb - cxm) * 0.25, 600)
        m = SM([top, bot], w * sc / SC if sc < 0.99 else w, 0, False)
        img = over(img, wood(min(top[0], bot[0]) - 10, max(top[0], bot[0]) + 10), m)
    img = over(img, wood(x0, x1), M([(x0 - 14, y1), (x1 + 14, y1), (x1 + 14, y1 + 14), (x0 - 14, y1 + 14)]))
    img = over(img, wood(cxm - 20, cxm + 20), M([(cxm - 16, y0 - 12), (cxm + 16, y0 - 12), (cxm + 16, y0 + 8), (cxm - 16, y0 + 8)]))
    side = M([(x1, y0 + 3), (x1 + 6, y0), (x1 + 6, y1 - 3), (x1, y1)])
    img = over(img, (0.78, 0.74, 0.66), side)
    img *= (1 - 0.25 * gauss(M([(x0 + 8, y1), (x1 + 8, y1), (x1 + 8, y1 + 6), (x0 + 8, y1 + 6)]), 4 * K))[..., None]
    return img


# ---------- 一帧 ----------
def e_pose(b):
    """E 段 Clawd：舞台上的位置（画布坐标）→ 走出画布到竹林里 (548, 572)。返回 (fx, fy, S, inside_ratio)。"""
    sc, ox, oy = canvas_rect(b)
    fx0, fy0 = to_canvas((A1.STAGE_C, A1.STAGE_BASE), sc, ox, oy)
    S0 = 17 * ZOOM01 * sc
    u = A.seg(b, 1.35, 2.9)
    e = A.ease_io(u)
    fx = A.lerp(fx0, 548, e)
    fy = A.lerp(fy0, 572, e)
    S = A.lerp(S0, 15, e)
    fy -= 4 * abs(math.sin(math.pi * u * 3)) * (S / 15)          # 走路的起伏（3 步）
    return fx, fy, S


def painting(t, b):
    if b < 3.0:
        sc, ox, oy = canvas_rect(b)
        img, _ = grove(0.0)
        img = easel(img, sc, ox, oy)
        cmask = M([(ox, oy), (ox + 1120 * sc, oy), (ox + 1120 * sc, oy + 630 * sc), (ox, oy + 630 * sc)])
        fx, fy, S = e_pose(b)
        out = e_outside(b, fx, fy, S, sc, ox, oy)
        if not out:
            St = Press()
            St.img = stage_on_canvas(sc, ox, oy)
            pose = (fx, fy, S, 0.0, 0.0, 34.0 * (1 - A.seg(b, 1.35, 2.2)), 0.0)
            A1.draw_clawd(St, 13.0, pose, dressed=True)
            A1.cord_and_mic(St, 13.0, pose)
            cv = St.img
        else:                                          # 走出来之后：画布上画的就是后面的竹林
            cv = img * (1 + 0.02 * S2.CANVAS[..., None])
        img = img * (1 - cmask[..., None]) + cv * cmask[..., None]
        img = img * (1 - 0.4 * outline(cmask, 1))[..., None]
        arms = [rigid_arm("L", 28 * A.seg(b, 1.6, 2.6)), rigid_arm("R", -18 * A.seg(b, 1.6, 2.6))]
        img = red_clawd(img, fx, fy, S, 0.0, arms, mask=(1 - cmask))
        img = top_leaves(img, 0.0) if b > 1.0 else img
        return oil(img, 0), out
    if b < 11.0:
        cam = cam_x(b)
        img, _ = grove(cam)
        img = leaves(img, b, cam)
        # 接住竹叶、转一圈
        spin = A.ease_io(A.seg(b, 7.4, 8.4))
        ang = 360 * spin
        hopy = 34 * math.sin(math.pi * A.seg(b, 7.4, 8.4))
        S = 15
        step = A.seg(b, 3.0, 11.0) * 8
        bob = 0 if 7.4 <= b < 8.4 else 4 * abs(math.sin(math.pi * step))
        cx, cy = 548, 572 - 6 * S - hopy - bob
        fx, fy = feet_from_center(cx, cy, S, ang)
        if b >= 6.4 and b < 7.0:                     # 那片竹叶落向左钳
            u = A.seg(b, 6.4, 7.0)
            (ax, ay), = g2p([(-1.6, 2.9)], fx, fy, S, ang)
            img = over(img, (0.18, 0.28, 0.19), M(lance(A.lerp(ax - 120, ax - 34, u), A.lerp(ay - 260, ay - 6, u), 44, 6, A.lerp(150, -12, u)), 0.4))
        arms = [rigid_arm("L", 28 + 20 * math.sin(math.pi * A.seg(b, 6.8, 7.2))), rigid_arm("R", -18)]
        img = red_clawd(img, fx, fy, S, ang, arms, flare=math.sin(math.pi * spin), leaf=b >= 7.0)
        img = top_leaves(img, cam)
        return oil(img, 0), True
    if b < 14.0:
        cam = cam_x(11.0)
        bend = 70 * A.back_out(A.seg(b, 11.3, 12.0)) + 260 * A.ease_in(A.seg(b, 12.3, 13.6))
        spread = 700 * A.ease_in(A.seg(b, 12.6, 13.9), 2.2)
        img, gap = grove(cam, bend, spread)
        lz, lat_a, wm = liaozhai_static()
        lzi = candle(lz.copy(), t)
        lzi = scholar(lzi, 650, 470)
        lzi = over(lzi, (0.11, 0.06, 0.04), lat_a)
        img = img * (1 - gap[..., None]) + lzi * gap[..., None]
        u = A.ease_io(A.seg(b, 11.9, 13.8))
        fx = A.lerp(548, 470, u) + 52 * math.sin(math.pi * A.seg(b, 11.0, 12.0)) * (1 - u)
        fy = A.lerp(572, 528, u) - 3 * abs(math.sin(math.pi * A.seg(b, 11.9, 13.8) * 4))
        S = A.lerp(15, 9.5, u)
        raise_ = math.sin(math.pi * A.seg(b, 11.1, 12.4))
        arms_r = [rigid_arm("L", 28 + 30 * raise_), rigid_arm("R", -18 + 50 * raise_)]
        img = red_clawd(img, fx, fy, S, 0, arms_r, mask=(1 - gap), leaf=True)
        img = robe_clawd(img, fx, fy, S, 0, [rigid_arm("L", 0), rigid_arm("R", 32 * raise_)], mask=gap)
        return oil(img, 1 if u > 0.5 else 0), True
    # b 段：聊斋
    lz, lat_a, wm = liaozhai_static()
    img = candle(lz.copy(), t)
    walk = A.seg(b, 14.0, 15.4)
    fx = A.lerp(470, 556, A.ease_io(walk))
    fy = 528 - 3 * abs(math.sin(math.pi * walk * 3))
    poke = math.sin(math.pi * A.seg(b, 15.4, 16.1))
    hop_c = 18 * math.sin(math.pi * A.seg(b, 18.0, 18.5)) + 14 * math.sin(math.pi * A.seg(b, 19.3, 19.75))
    fy -= hop_c
    poof = A.seg(b, 16.0, 16.7)
    if b < 16.12:
        img = scholar(img, 650, 470)
    else:
        hop_f = 22 * math.sin(math.pi * A.seg(b, 17.0, 17.5)) + 22 * math.sin(math.pi * A.seg(b, 19.0, 19.5))
        img = frog(img, 650, 468 - hop_f, 0.55, puff=A.ease_out(A.seg(b, 19.55, 19.95)))
    img = robe_clawd(img, fx, fy, 9.5, 0, [rigid_arm("L", 0), rigid_arm("R", 32 + 24 * poke)])
    if 0 < poof < 1:                                   # 「噗」：一团白烟先鼓起再散开
        s = 0.35 + 0.35 * A.ease_out(poof, 3)
        sm = np.zeros((PH, PW), np.float32)
        g = np.random.default_rng(11)
        for _ in range(8):
            ox_, oy_, r = g.uniform(-30, 30) * s, g.uniform(-26, 14) * s, g.uniform(16, 28) * s
            sm = np.maximum(sm, M(circle(650 + ox_, 448 + oy_, r, 40)))
        sm = gauss(sm, 1.5 * K) * (1 - A.ease_in(A.seg(poof, 0.35, 1.0), 1.5))
        img = over(img, np.array([0.98, 0.95, 0.90])[None, None] * (0.85 + 0.15 * gauss(sm, 6 * K))[..., None], sm)
    img = over(img, (0.11, 0.06, 0.04), lat_a)
    return oil(img, 1), True


def e_outside(b, fx, fy, S, sc, ox, oy):
    """Clawd 是否已完全走出画布。"""
    if b >= 3.0:
        return True
    pts = g2p([(-1, 0), (17, 0), (17, 12), (-1, 12)], fx, fy, S)
    x1, y1 = ox + 1120 * sc, oy + 630 * sc
    return all(not (ox <= x <= x1 and oy <= y <= y1) for x, y in pts) and b > 1.4


def frame_at(i):
    t = i / FPS
    b = bt(t)
    p, switched = painting(t, b)
    p = np.clip(downsample(p, K), 0, 1)
    if b >= 14.0:
        p = A.zoom_painting(p, 1 + 0.25 * A.ease_io(A.seg(b, 14.0, 20.0)), 600, 455)
    if switched:
        return A.compose(p, WALL2, LABEL2)
    return A.compose(p, WALL1, LABEL1)


def warm():
    A1.warm()
    ground()
    liaozhai_static()
    for w, l in ((WALL1, LABEL1), (WALL2, LABEL2)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a02_{i:04d}.png")
