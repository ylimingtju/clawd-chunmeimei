"""07 我表示理解 · 王家卫港片 · 114 BPM（1:08.77–1:18.77）。拍号 b 从 06 结束算起。
画面「抽帧」：画心内容每 3 帧（10 格/秒）换一次，每格叠一层上一格的残影；胶片颗粒每帧都变。
E  b0–3   光管从左往右一根根熄灭，Clawd 走向右边的黑暗。最后一根光管闪了闪，变成地下通道里的荧光灯：
          它的光从右往左铺开，被照到的部分变成胶片画面，Clawd 换上过大的黑皮夹克、戴上窄墨镜。
a  b3–11  荧光灯下的地下通道。Clawd 站着不动，路人糊成拖影从身边走过，墙上的钟按拍点跳秒。
          它掏出一罐凤梨罐头，举到眼前看了看保质期，拉开拉环吃了一口；然后转身，背对我们走向出口。
M  b11–14 出口的方框原来是冰室墙上的一台老电视，正在放演唱会；Clawd 走进屏幕，电视推满画面，
          进去的部分换上白缎长裙和白手套。
b  b14–19 武汉现场，舞台灯拖出慢门光轨，Clawd 一只钳子向前伸出。轻微晃动。"""
import math

import numpy as np
from scipy import ndimage

from kit import (K, PW, PH, X, Y, M, SM, circle, clawd, garment, g2p, rigid_arm, check_proportions, gauss, fbm, smooth, over,
                 downsample, ishift, outline, dilate)
from common4 import rect_pts
import anim_core as A
import frame
import S07_王家卫 as S7
from S07_王家卫 import quad, lerp, hblur, VP, BX0, BY0, BX1, BY1, ORANGE
import A06_有趣 as A6

FPS = 30
T0 = A6.T0 + 24 * A6.BEAT
BEAT = 60 / 114
WALL6, LABEL6 = A6.WALL6, A6.LABEL6
WALL7, LABEL7 = (0.13, 0.25, 0.20), "我表示理解"
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 19 * BEAT) * FPS)
HERO = (664, 604, 16)
_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


# ---------- 通道静态层 ----------
def tunnel_static():
    if "tun" in _CACHE:
        return _CACHE["tun"]
    img = np.zeros((PH, PW, 3), np.float32)
    TL, TR, BR, BL = (0, 0), (1120, 0), (1120, 630), (0, 630)
    bTL, bTR, bBR, bBL = (BX0, BY0), (BX1, BY0), (BX1, BY1), (BX0, BY1)
    walls = [(quad(TL, bTL, bBL, BL), 0.95), (quad(TR, bTR, bBR, BR), 0.80)]
    for m, k in walls:
        depth = smooth(0, 1, 1 - np.minimum(np.abs(X / K - VP[0]) / 650, 1))
        img = over(img, np.array([0.55, 0.64, 0.52])[None, None] * (k * (1 - 0.55 * depth))[..., None], m)
    ceil = quad(TL, TR, bTR, bTL)
    img = over(img, np.array([0.30, 0.36, 0.30])[None, None] * (1 - 0.5 * smooth(0, BY0, Y / K))[..., None], ceil)
    floor = quad(BL, BR, bBR, bBL)
    img = over(img, np.array([0.16, 0.20, 0.17])[None, None] * (1 - 0.4 * smooth(630, BY1, Y / K))[..., None], floor)
    tiles = np.zeros((PH, PW), np.float32)
    for k in range(1, 26):
        y = k * 25
        for (p, q, mm) in [((0, y), (BX0, BY0 + (BY1 - BY0) * y / 630), walls[0][0]), ((1120, y), (BX1, BY0 + (BY1 - BY0) * y / 630), walls[1][0])]:
            tiles = np.maximum(tiles, SM([p, q], 0.7, 0, False) * mm)
    for d in [1 - 0.86 ** k for k in range(1, 30)]:
        for (p, q, mm) in [(lerp(TL, bTL, d), lerp(BL, bBL, d), walls[0][0]), (lerp(TR, bTR, d), lerp(BR, bBR, d), walls[1][0])]:
            tiles = np.maximum(tiles, SM([p, q], 0.7 * (1 - d) + 0.3, 0, False) * mm)
    img = img * (1 - 0.35 * tiles[..., None])
    lights = np.zeros((PH, PW), np.float32)
    for d in [1 - 0.80 ** k for k in range(1, 14)]:
        for side in (-0.5, 0.5):
            cx = lerp((560 + side * 520, 0), (650 + side * 40, BY0), d)
            w = 70 * (1 - d) + 6
            h = 7 * (1 - d) + 1.5
            lights = np.maximum(lights, M([(cx[0] - w, cx[1] - h), (cx[0] + w, cx[1] - h), (cx[0] + w, cx[1] + h), (cx[0] - w, cx[1] + h)]) * ceil)
    img = img + lights[..., None] * np.array([1.2, 1.25, 1.0])
    refl = ishift(np.flipud(lights), int(2 * VP[1] * K - (PH - 1)), 0) * floor
    refl = ndimage.gaussian_filter(refl, (10 * K, 2.0 * K)) * (0.6 + 0.4 * fbm(PH, PW, 12, 71, 2))
    img = img + (refl * 1.6)[..., None] * np.array([0.8, 1.0, 0.75])
    ex = quad(bTL, bTR, bBR, bBL)
    _CACHE["tun"] = (img.astype(np.float32), floor, ex)
    return _CACHE["tun"]


def concert_static():
    """07-b 的现场（不含 Clawd；光轨另画成会动的）。"""
    if "con" in _CACHE:
        return _CACHE["con"]
    img = np.zeros((PH, PW, 3), np.float32) + np.array([0.03, 0.04, 0.06])
    haze = fbm(PH, PW, 140, 72, 3)
    img = img + (haze * 0.10)[..., None] * np.array([0.5, 0.7, 0.8])
    g = np.random.default_rng(77)
    for i in range(9):
        x0 = 80 + i * 120
        col = [np.array([1.0, 0.35, 0.65]), np.array([0.35, 0.75, 1.0]), np.array([1.0, 0.75, 0.35])][i % 3]
        x1 = x0 + g.uniform(-160, 160)
        cone = M([(x0 - 6, 20), (x0 + 6, 20), (x1 + 90, 640), (x1 - 90, 640)])
        img = img + (gauss(cone, 6 * K) * (0.10 + 0.12 * haze))[..., None] * col
        img = img + (np.exp(-(((X / K - x0) / 9) ** 2 + ((Y / K - 22) / 6) ** 2)) * 1.4)[..., None] * col
    img = over(img, (0.05, 0.06, 0.07), M([(0, 520), (1120, 520), (1120, 640), (0, 640)]))
    img = img + (np.exp(-(((X / K - 560) / 260) ** 2 + ((Y / K - 528) / 18) ** 2)) * 0.35)[..., None] * np.array([0.9, 0.95, 1.0])
    crowd = np.zeros((PH, PW), np.float32)
    for i in range(30):
        x, y, r = g.uniform(-20, 1140), g.uniform(578, 640), g.uniform(22, 40)
        crowd = np.maximum(crowd, M(circle(x, y, r, 32, r * 1.1)))
    crowd = ndimage.gaussian_filter(crowd, (1.5 * K, 6 * K))
    phones = []
    for i in range(14):
        phones.append((g.uniform(40, 1080), g.uniform(560, 610)))
    arcs = [(g.uniform(150, 970), g.uniform(60, 260), g.uniform(80, 220), g.uniform(0, 2 * math.pi), g.uniform(1.0, 2.2)) for i in range(7)]
    _CACHE["con"] = (img.astype(np.float32), crowd.astype(np.float32), phones, arcs)
    return _CACHE["con"]


def concert(b, cl=None):
    img, crowd, phones, arcs = concert_static()
    img = img.copy()
    for i, (cx, cy, r, a0, span) in enumerate(arcs):        # 慢门光轨：摇头灯转着画出的弧，随时间往前走
        a0 = a0 + 0.8 * b
        arc = [(cx + r * math.cos(a0 + t), cy + r * 0.45 * math.sin(a0 + t)) for t in np.linspace(0, span, 40)]
        col = [np.array([1.0, 0.4, 0.7]), np.array([0.4, 0.8, 1.0]), np.array([1.0, 0.85, 0.5])][i % 3]
        m = SM(arc, 2.2, 0.8, False)
        img = img + (m * 0.9 + gauss(m, 4 * K) * 0.5)[..., None] * col
    cm = None
    if cl is not None:
        img, cm = satin_clawd(img, *cl)
    img = img * (1 - 0.92 * crowd[..., None]) + np.array([0.02, 0.025, 0.03]) * 0.92 * crowd[..., None]
    rim = np.clip(crowd - ishift(crowd, 4 * K, 0), 0, 1)
    img = img + (gauss(rim, 1.2 * K) * 0.55)[..., None] * np.array([0.75, 0.80, 0.95])
    for k, (x, y) in enumerate(phones):
        ph = M([(x - 7, y - 11), (x + 7, y - 11), (x + 7, y + 11), (x - 7, y + 11)])
        ph = ndimage.gaussian_filter(ph, (0.6 * K, 4 * K)) * (0.7 + 0.3 * math.sin(b * 3 + k))
        img = img + (ph * 1.2)[..., None] * np.array([0.85, 0.95, 1.0])
    return img, cm


def satin_clawd(img, FX, FY, S, reach=1.0, mask=None):
    mm = 1.0 if mask is None else mask
    arms = [rigid_arm("L", 0), rigid_arm("R", 34 * reach)]
    check_proportions(arms=arms)
    cm, em = clawd(FX, FY, S, arms)
    img = over(img, ORANGE[None, None] * (1.05 - 0.25 * smooth(FX - 8 * S, FX + 8 * S, X / K))[..., None], cm * mm)
    gown = garment([(-0.2, 5.4), (16.2, 5.4), (16.9, 8.6), (18.0, 12.0), (-2.0, 12.0), (-0.9, 8.6)], FX, FY, S, arms=arms, name="gown") * mm
    u = (X / K - FX) / (9 * S)
    satin = 0.78 + 0.22 * np.cos(u * 7.0) ** 2 + 0.25 * np.exp(-((u + 0.25) / 0.08) ** 2)
    img = over(img, np.array([0.93, 0.92, 0.88])[None, None] * satin[..., None], gown)
    gl = np.clip(sum(M(g2p(a, FX, FY, S)) for a in arms), 0, 1) * mm
    img = over(img, (0.96, 0.95, 0.92), gl)
    img = over(img, (0.06, 0.05, 0.05), em * mm)
    img = img + (gauss(np.clip(cm - ishift(cm, 3 * K, 0), 0, 1), 1.5 * K) * 0.4 * mm)[..., None] * np.array([0.9, 0.85, 1.0])
    return img, cm * mm


def jacket_clawd(img, FX, FY, S, arms, can=None, back=False, mask=None):
    """07 的 Clawd：黑皮夹克 + 白 T + 窄墨镜；can=(状态, 抬高 0..1)；back=True 时是背影。"""
    mm = 1.0 if mask is None else mask
    check_proportions(arms=arms)
    cm_, em = clawd(FX, FY, S, arms)
    body = ORANGE[None, None] * 1.05 * np.ones((PH, PW, 1))
    body = body * (0.75 + 0.45 * (1 - smooth(FY - 12 * S, FY - 6 * S, Y / K)))[..., None] * np.array([0.92, 1.05, 0.85])
    img = over(img, body, cm_ * mm)
    jm = garment([(-0.5, 4.7), (16.5, 4.7), (16.9, 10.9), (-0.9, 10.9)], FX, FY, S, smooth_k=0.1, arms=arms, name="jacket")
    sleeves = np.clip(sum(M(g2p(a, FX, FY, S)) for a in arms), 0, 1)
    leather = np.clip(jm + sleeves, 0, 1) * mm
    jc = np.array([0.045, 0.05, 0.05])[None, None] * np.ones((PH, PW, 1))
    hl = np.zeros((PH, PW), np.float32)
    for (gx0, gy0, gx1, gy1, w) in [(1.2, 5.4, 2.0, 10.4, 0.45), (14.4, 5.4, 15.0, 10.4, 0.35), (4.6, 6.0, 5.2, 10.6, 0.25), (11.2, 6.0, 11.6, 10.6, 0.22)]:
        hl = np.maximum(hl, SM(g2p([(gx0, gy0), (gx1, gy1)], FX, FY, S), w * S, 0, True))
    hl = np.maximum(hl, gauss(np.clip(sleeves - ishift(sleeves, 3 * K, 0), 0, 1), K))
    jc = jc + gauss(hl, 2.5 * K)[..., None] * np.array([0.34, 0.40, 0.34])
    img = over(img, jc, leather)
    img = over(img, (0.0, 0.0, 0.0), outline(leather, 1) * 0.7)
    if back:                                             # 背影：竖起的领子从后面看是一道黑边，背中缝
        img = over(img, (0.07, 0.08, 0.08), garment([(-0.5, 3.9), (16.5, 3.9), (16.5, 4.9), (-0.5, 4.9)], FX, FY, S, name="collar") * mm)
        img = over(img, (0.0, 0.0, 0.0), SM(g2p([(8.0, 4.9), (8.0, 10.9)], FX, FY, S), 0.08 * S, 0, False) * mm * 0.8)
    else:
        tee = garment([(6.4, 4.7), (9.6, 4.7), (9.2, 10.9), (6.8, 10.9)], FX, FY, S, name="tee") * mm
        img = over(img, (0.86, 0.88, 0.80), tee)
        img = over(img, (0.0, 0.0, 0.0), outline(tee, 1) * 0.5)
        for side in (-1, 1):
            x0 = 6.4 if side < 0 else 9.6
            lm = garment([(x0, 4.7), (x0 + side * 4.4, 3.9), (x0 + side * 5.6, 5.0), (x0 + side * 2.2, 8.4), (x0, 8.0)], FX, FY, S, name="lapel") * mm
            img = over(img, np.array([0.08, 0.09, 0.09])[None, None] * np.ones((PH, PW, 1)) + gauss(outline(lm, 2), K)[..., None] * 0.25, lm)
            img = over(img, (0.0, 0.0, 0.0), outline(lm, 1) * 0.9)
        for k in range(10):
            (zx, zy), = g2p([(6.5, 5.2 + k * 0.56)], FX, FY, S)
            img = over(img, (0.75, 0.78, 0.70), M([(zx - 1.4 * S / 16, zy), (zx + 1.4 * S / 16, zy), (zx + 1.4 * S / 16, zy + 2.2 * S / 16), (zx - 1.4 * S / 16, zy + 2.2 * S / 16)]) * mm)
        gl = np.zeros((PH, PW), np.float32)
        for (x0, x1) in ((2.5, 5.8), (10.2, 13.5)):
            gl = np.maximum(gl, garment([(x0, 3.0), (x1, 3.0), (x1 - 0.2, 4.6), (x0 + 0.2, 4.6)], FX, FY, S, smooth_k=0.2, name="lens"))
        gl = np.maximum(gl, SM(g2p([(5.8, 3.4), (8.0, 3.15), (10.2, 3.4)], FX, FY, S), 0.12 * S, 0.8, False))
        gl = np.maximum(gl, SM(g2p([(0.0, 3.3), (2.5, 3.3)], FX, FY, S), 0.12 * S, 0, False))
        gl = np.maximum(gl, SM(g2p([(13.5, 3.3), (16.0, 3.3)], FX, FY, S), 0.12 * S, 0, False))
        img = over(img, (0.02, 0.02, 0.025), gl * mm)
        for x0 in (2.9, 10.6):
            img = img + (SM(g2p([(x0, 3.35), (x0 + 1.6, 3.3)], FX, FY, S), 0.16 * S, 0, True) * 0.7 * mm)[..., None] * np.array([0.8, 1.0, 0.8])
    img = img + (gauss(np.clip(cm_ - ishift(cm_, 3 * K, 0), 0, 1), 1.2 * K) * 0.35 * mm)[..., None] * np.array([0.6, 0.8, 0.55])
    if can is not None:
        img = can_draw(img, arms[0], FX, FY, S, can, jc, mm)
    return img, cm_ * mm


def can_draw(img, arm, FX, FY, S, can, jc, mm):
    state, lift = can
    ax = sum(p[0] for p in arm) / 4 - 1.0
    ay = sum(p[1] for p in arm) / 4 + 0.9
    (cx_, cy_), = g2p([(ax, ay)], FX, FY, S)
    rw, rh, ey = 1.25 * S, 1.7 * S, 0.32 * S
    canm = M([(cx_ - rw, cy_ - rh)] + [(cx_ + rw * math.cos(t), cy_ + rh + ey * math.sin(t)) for t in np.linspace(math.pi, 0, 30)] + [(cx_ + rw, cy_ - rh)]) * mm
    cyl = 0.62 + 0.55 * np.exp(-((X / K - (cx_ - 0.35 * rw)) / (0.45 * rw)) ** 2) - 0.25 * smooth(cx_, cx_ + rw, X / K)
    img = over(img, np.array([0.72, 0.74, 0.70])[None, None] * cyl[..., None], canm)
    lab = canm * M([(cx_ - rw - 2, cy_ - rh + 0.45 * S), (cx_ + rw + 2, cy_ - rh + 0.45 * S)] +
                   [(cx_ + rw * math.cos(t) * 1.02, cy_ + rh - 0.45 * S + ey * math.sin(t)) for t in np.linspace(0, math.pi, 30)])
    img = over(img, np.array([0.92, 0.74, 0.18])[None, None] * cyl[..., None], lab)
    for yy in (cy_ - rh + 0.75 * S, cy_ + rh - 0.75 * S):
        img = over(img, np.array([0.70, 0.16, 0.10])[None, None] * cyl[..., None], SM([(cx_ + rw * math.cos(t), yy + ey * math.sin(t)) for t in np.linspace(math.pi, 0, 24)], 0.13 * S, 0.5, False) * canm)
    pine = M(circle(cx_ - 0.1 * S, cy_ + 0.2 * S, 0.55 * S, 32, 0.72 * S)) * lab
    img = over(img, np.array([0.88, 0.56, 0.10])[None, None] * cyl[..., None], pine)
    for dx in (-0.32, 0.0, 0.32):
        img = over(img, (0.18, 0.48, 0.16), M([(cx_ - 0.1 * S + (dx - 0.12) * S, cy_ - 0.45 * S), (cx_ - 0.1 * S + dx * 1.7 * S, cy_ - 1.05 * S), (cx_ - 0.1 * S + (dx + 0.12) * S, cy_ - 0.45 * S)]) * lab)
    lid = M(circle(cx_, cy_ - rh, rw, 48, ey)) * mm
    img = over(img, np.array([0.80, 0.82, 0.78])[None, None] * (0.85 + 0.25 * (1 - smooth(cx_ - rw, cx_ + rw, X / K)))[..., None], lid)
    if state == "open":                                  # 拉开了：罐口一个深色的开口，拉环翻起来
        img = over(img, (0.18, 0.12, 0.05), M(circle(cx_ - 0.2 * rw, cy_ - rh, 0.45 * rw, 32, ey * 0.55)) * mm)
        img = over(img, (0.60, 0.62, 0.60), M([(cx_ + 0.15 * rw, cy_ - rh), (cx_ + 0.45 * rw, cy_ - rh - 0.6 * S), (cx_ + 0.6 * rw, cy_ - rh - 0.5 * S), (cx_ + 0.3 * rw, cy_ - rh + 0.05 * S)]) * mm)
    else:
        img = over(img, (0.55, 0.57, 0.55), np.clip(M(circle(cx_ + 0.15 * rw, cy_ - rh, 0.32 * rw, 24, 0.13 * S)) - M(circle(cx_ + 0.15 * rw, cy_ - rh, 0.18 * rw, 24, 0.07 * S)), 0, 1) * mm)
    img = over(img, (0.0, 0.0, 0.0), outline(np.clip(canm + lid, 0, 1), 1) * 0.6)
    sl = M(g2p(arm, FX, FY, S)) * mm
    img = over(img, jc, sl)
    img = over(img, (0.0, 0.0, 0.0), outline(sl, 1) * 0.7)
    return img


def passersby(img, t):
    """抽帧拖影的路人：一直从左右两边穿过画面。"""
    specs = [(300, 600, 300, 1, (0.26, 0.30, 0.26), 7, 170), (930, 610, 330, -1, (0.45, 0.16, 0.12), 7, 190),
             (480, 470, 150, 1, (0.30, 0.34, 0.30), 6, 60), (820, 452, 120, -1, (0.22, 0.24, 0.36), 6, 50),
             (1060, 640, 380, -1, (0.12, 0.13, 0.12), 5, 260)]
    for (x0, foot, h, dirn, col, n, v) in specs:
        x = (x0 + dirn * v * t + 200) % 1520 - 200
        fig = np.clip(M(circle(x, foot - h * 0.9, h * 0.07, 32, h * 0.085)) + M([(x - h * 0.14, foot - h * 0.8), (x + h * 0.14, foot - h * 0.8),
                      (x + h * 0.12, foot - h * 0.38), (x + h * 0.08, foot), (x - h * 0.08, foot), (x - h * 0.12, foot - h * 0.38)], 0.5), 0, 1)
        trail = np.zeros((PH, PW), np.float32)
        for k in range(n):
            ghost = hblur(ishift(fig, 0, int(-dirn * k * h * 0.12 * K)), h * 0.035 * K)
            trail = np.maximum(trail, np.clip(ghost * 1.3, 0, 1) * (1 - k / n) ** 1.1)
        rim = np.clip(trail - ishift(trail, 4 * K, 0), 0, 1)
        img = img * (1 - 0.78 * trail[..., None]) + np.array(col) * 0.78 * trail[..., None]
        img = img + (rim * 0.25)[..., None] * np.array([0.8, 0.9, 0.7])
    return img


def clock(img, b):
    ccx, ccy = 214, 170
    cm = M(circle(ccx, ccy, 30, 64, 34))
    img = over(img, (0.80, 0.82, 0.70), cm)
    img = over(img, (0.10, 0.10, 0.08), outline(cm, 3))
    for i in range(12):
        a = 2 * math.pi * i / 12
        img = over(img, (0.15, 0.15, 0.12), SM([(ccx + 24 * math.cos(a), ccy + 28 * math.sin(a)), (ccx + 28 * math.cos(a), ccy + 32 * math.sin(a))], 1.4, 0, False))
    sec = math.floor(b) + A.back_out(A.seg(b % 1.0, 0.0, 0.15), 3.0)             # 按拍点跳一格
    for (a, L, w) in [(-2.0, 16, 2.6), (-0.6 + sec * 0.003, 24, 1.8), (1.3 + sec * 2 * math.pi / 60, 26, 0.9)]:
        img = over(img, (0.08, 0.08, 0.06), SM([(ccx, ccy), (ccx + L * math.cos(a), ccy + L * 1.12 * math.sin(a))], w, 0, False))
    return img


def reflection(img, cm, FY, floor):
    ref = ishift(np.flipud(cm), int(-(PH - 2 * FY * K)), 0) * floor
    ref = ndimage.gaussian_filter(ref, (6 * K, 1 * K))
    return img * (1 - 0.35 * ref[..., None]) + np.array([0.20, 0.14, 0.10]) * 0.35 * ref[..., None]


# ---------- 一帧 ----------
def a_hero(b):
    """a 段：站着 → 掏罐头 → 举到眼前看 → 拉开 → 吃一口 → 转身走向出口。返回绘制参数。"""
    FX, FY, S = HERO
    if b < 5.0:
        return dict(FX=FX, FY=FY, S=S, arms=[rigid_arm("L", -24), rigid_arm("R", 10)], can=None, back=False)
    up = A.ease_io(A.seg(b, 5.0, 5.6)) - A.ease_io(A.seg(b, 7.6, 8.0)) * 0.3
    arms = [rigid_arm("L", -24 - 40 * up), rigid_arm("R", 10 + 30 * math.sin(math.pi * A.seg(b, 7.0, 7.5)))]
    state = "open" if b >= 7.3 else "shut"
    bite = math.sin(math.pi * A.seg(b, 8.0, 8.8))
    if b < 9.0:
        return dict(FX=FX, FY=FY - 6 * bite, S=S, arms=arms, can=(state, up), back=False)
    # 转身走向出口：背影，越走越远（向灭点缩小）
    w = A.ease_in(A.seg(b, 9.2, 11.0), 1.3)
    ex_foot = (650, BY1)
    fx = A.lerp(FX, ex_foot[0], w)
    fy = A.lerp(FY, ex_foot[1] + 6, w) - 3 * abs(math.sin(math.pi * 6 * w))
    S2 = A.lerp(S, 3.2, w)
    return dict(FX=fx, FY=fy, S=S2, arms=[rigid_arm("L", 0), rigid_arm("R", 0)], can=None, back=True)


def tunnel_frame(t, b, exit_layer=None, hero=True):
    base, floor, ex = tunnel_static()
    img = base.copy()
    if exit_layer is None:
        out = np.zeros((PH, PW, 3), np.float32) + np.array([0.05, 0.05, 0.08])
        for (x, y, r, c) in [(610, 270, 18, (1.2, 0.20, 0.35)), (680, 258, 14, (0.25, 0.55, 1.3)), (640, 320, 22, (1.3, 0.55, 0.15)), (700, 330, 10, (0.9, 0.2, 1.1))]:
            out = out + (np.exp(-(((X / K - x) / r) ** 2 + ((Y / K - y) / (r * 0.6)) ** 2)) * (0.85 + 0.15 * math.sin(t * 5 + x)))[..., None] * np.array(c)
        img = over(img, out, ex)
    else:
        img = over(img, exit_layer, ex)
    img = clock(img, b)
    img = passersby(img, t)
    cm = np.zeros((PH, PW), np.float32)
    if hero:
        h = a_hero(b)
        img, cm = jacket_clawd(img, h["FX"], h["FY"], h["S"], h["arms"], h["can"], h["back"])
        img = reflection(img, cm, h["FY"], floor)
    return img, cm


def step_time(t):
    """抽帧：10 格/秒。"""
    return math.floor(t * 10) / 10


def painting(t, b):
    if b < 3.0:
        # E：06 的光管舞台，光管一根根熄灭；最后一根闪了闪，荧光灯的光从右往左铺开
        st = A6.stage_static().copy()
        off = [A.ease_out(A.seg(b, 0.25 + 0.38 * k, 0.45 + 0.38 * k)) for k in range(5)]
        xs = [120, 250, 380, 740, 870]
        for k, x in enumerate(xs):
            if off[k] > 0:
                tube = M([(x - 48, 20), (x + 48, 20), (x + 48, 518), (x - 48, 518)])
                st = st * (1 - 0.85 * off[k] * tube[..., None] * smooth(0.3, 0.6, st.mean(-1))[..., None])
        flick = 1.0 if b < 2.0 else (0.3 if int(b * 12) % 2 else 1.0)
        tube6 = M([(1000 - 48, 20), (1000 + 48, 20), (1000 + 48, 518), (1000 - 48, 518)])
        st = st * (1 - (1 - flick) * 0.8 * tube6[..., None] * smooth(0.3, 0.6, st.mean(-1))[..., None])
        st = st * (1 - 0.55 * A.seg(b, 0.3, 2.2))                 # 舞台整体暗下去
        walk = A.ease_io(A.seg(b, 0.0, 2.6))
        fx = A.lerp(560, HERO[0], walk)
        fy = A.lerp(560, HERO[1], walk) - 4 * abs(math.sin(math.pi * 4 * walk))
        S = A.lerp(16, HERO[2], walk)
        edge = A.lerp(1060, -20, A.ease_in(A.seg(b, 2.3, 3.0), 1.4))
        right = smooth(edge - 8, edge + 8, X / K)
        C = A6.Comic()
        C.img = st
        A6.grey_clawd(C, fx, fy, S, [rigid_arm("L", 32), rigid_arm("R", 32)], mask=(1 - right))
        tun, cm = tunnel_frame(t, 3.0, hero=False)
        tun, cm = jacket_clawd(tun, fx, fy, S, [rigid_arm("L", -24), rigid_arm("R", 10)], mask=right)
        lit = np.exp(-((X / K - edge) / 20) ** 2)[..., None] * np.array([0.6, 0.7, 0.5]) * 0.6 * (edge < 1050)
        tun_g = S7.grade(tun, cm)
        img = C.img * (1 - right[..., None]) + tun_g * right[..., None] + lit
        return img, edge < fx - 9 * S, True
    if b < 11.0:
        img, cm = tunnel_frame(t, b)
        return S7.grade(img, cm), True, True
    # M：出口原来是冰室墙上的老电视，正在放演唱会；Clawd 走进屏幕，电视推满画面
    zoom = A.ease_in(A.seg(b, 11.6, 13.6), 1.7)
    con, _ = concert(b)
    tv = A.ease_io(A.seg(b, 11.0, 11.6))
    # 出口里换成演唱会画面（缩小贴进出口），出口四周长出电视机的外框
    sx, sy = (BX1 - BX0) / 1120, (BY1 - BY0) / 630
    qx = (X / K - BX0) / sx
    qy = (Y / K - (BY0 + BY1) / 2) / sx + 315            # 等比缩放：屏幕按宽度对齐，上下裁掉
    from scipy.ndimage import map_coordinates
    screen = np.stack([map_coordinates(con[..., c], [qy * K, qx * K], order=1, mode="nearest") for c in range(3)], -1)
    screen = screen * (0.85 + 0.15 * np.cos((Y / K - BY0) / sy * 0.9) ** 2)[..., None]      # 显像管的扫描线
    img, cm = tunnel_frame(t, 11.0 + min(b - 11.0, 0.0), exit_layer=screen * tv + np.array([0.05, 0.05, 0.08]) * (1 - tv), hero=False)
    bez = np.clip(M([(BX0 - 26 * tv, BY0 - 22 * tv), (BX1 + 26 * tv, BY0 - 22 * tv), (BX1 + 26 * tv, BY1 + 40 * tv), (BX0 - 26 * tv, BY1 + 40 * tv)]) -
                  M([(BX0, BY0), (BX1, BY0), (BX1, BY1), (BX0, BY1)]), 0, 1)
    img = over(img, np.array([0.30, 0.22, 0.14])[None, None] * (0.8 + 0.3 * smooth(BY0, BY1, Y / K))[..., None], bez * tv)
    for kx in (BX1 + 8, BX1 + 16):
        img = over(img, (0.70, 0.66, 0.55), M(circle(kx, BY1 + 18, 3.5, 16)) * tv)
    # Clawd 背影走到屏幕前，越过电视框的部分换装
    h = a_hero(11.0)
    fx, fy, S = h["FX"], h["FY"], h["S"]
    img, cm = jacket_clawd(img, fx, fy, S, h["arms"], None, True, mask=1 - M([(BX0, BY0), (BX1, BY0), (BX1, BY1), (BX0, BY1)]) * (zoom > 0))
    if zoom <= 0:
        return S7.grade(img, cm), True, True
    # 推进电视：外面的通道放大，屏幕里的演唱会按当前屏幕大小直接取样（不放大小图）；最后再统一调色、加颗粒
    z = A.lerp(1.0, 1120 / (BX1 - BX0), zoom)
    cx, cy = (BX0 + BX1) / 2, (BY0 + BY1) / 2
    dst = A.lerp2((cx, cy), (560, 315), zoom)
    outer = A3_warp(img, z, (cx, cy), dst)
    rx0, ry0 = dst[0] + (BX0 - cx) * z, dst[1] + (BY0 - cy) * z
    rx1, ry1 = dst[0] + (BX1 - cx) * z, dst[1] + (BY1 - cy) * z
    con2, cm2 = concert(b, (560, 530, 16, 0.0)) if zoom >= 0.5 else (con, None)
    sxz, syz = (rx1 - rx0) / 1120, (ry1 - ry0) / 630
    qx = (X / K - rx0) / sxz
    qy = (Y / K - (ry0 + ry1) / 2) / sxz + 315
    scr = np.stack([map_coordinates(con2[..., c], [qy * K, qx * K], order=1, mode="nearest") for c in range(3)], -1)
    scan = (0.85 + 0.15 * np.cos((Y / K - ry0) / syz * 0.9) ** 2) * (1 - zoom) + zoom
    rm = M([(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)])
    g = outer * (1 - rm[..., None]) + scr * scan[..., None] * rm[..., None]
    hero_m = cm2 * rm if cm2 is not None else np.zeros((PH, PW), np.float32)
    if zoom >= 0.5:                                      # 进了屏幕的 Clawd：白缎长裙（按屏幕缩放画在演唱会画面里）
        pass
    return S7.grade(g, hero_m), True, True
    return g, True, True


def A3_warp(src, z, c_src, c_dst):
    import A03_向日葵朝着夜 as A3
    return A3.warp(src, z, c_src, c_dst)


def frame_at(i):
    t = i / FPS
    ts = step_time(t)
    b = bt(ts)
    bb = bt(t)
    if b >= 14.0:
        reach = A.ease_out(A.seg(b, 14.0, 14.8))
        con, cm = concert(b, (560, 530, 16, 0.4 + 0.6 * reach))
        p = S7.grade(con, cm)
        sw = True
    else:
        p, out, sw = painting(ts, b)
        if b < 3.0:
            sw = out
    # 抽帧残影：叠一层上一格
    if 3.0 <= b < 14.0 and b >= 3.2:
        prev_b = bt(ts - 0.1)
        if prev_b < 14.0 and prev_b >= 3.0:
            pp, _, _ = painting(ts - 0.1, prev_b)
            p = p * 0.78 + pp * 0.22
    p = np.clip(downsample(p, K), 0, 1)
    # 每帧都变的颗粒
    g = np.random.default_rng(i).normal(0, 1, (p.shape[0] // 2, p.shape[1] // 2)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[:p.shape[0], :p.shape[1]]
    if bb >= 3.0:
        p = np.clip(p + (0.022 * g)[..., None], 0, 1)
    if bb >= 14.0 or (3.0 <= bb < 11.0):                     # 手持的轻微晃动
        dx = 3 * math.sin(t * 1.7) + 1.5 * math.sin(t * 4.1)
        dy = 2 * math.sin(t * 2.3 + 1)
        p = A.zoom_painting(p, 1.03, 560 + dx, 315 + dy)
    return A.compose(p, WALL7 if sw else WALL6, LABEL7 if sw else LABEL6)


def warm():
    A6.warm()
    tunnel_static()
    concert_static()
    for w, l in ((WALL6, LABEL6), (WALL7, LABEL7)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a07_{i:04d}.png")
