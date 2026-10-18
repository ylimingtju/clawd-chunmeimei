"""05 我管不住我自己 · 马蒂斯剪纸 · 80 BPM（0:49.48–0:59.23）。画心动作「一拍二」。拍号 b 从 04 结束算起。
E  b0–3   Clawd 举起钳子当剪刀，在金箔背景上「咔嚓咔嚓」按拍剪出一道圆弧；剪下的那块壁画像纸一样卷着掉下去，
          后面露出晨光阳台。Clawd 跳进切口，跨过切口的部分变成剪纸、换上蓝白条纹的居家上衣。
a  b3–8   晨光阳台（矮栏杆在 Clawd 身后）。Clawd 推开剪纸百叶窗，海藻形的阳光照进来；按拍点把剪纸衣服挂上晾衣绳；
          把唱片放到唱机上，唱片转起来、飘出剪纸音符；盆栽的叶子随拍子摇；一只剪纸小鸟落到它头上。画框里缓慢右移。
M  b8–10  一朵大剪纸花开出来，Clawd 在拍点上跳进花心；进去的部分换上银色亮片短裙，花心张开成《爵士》的舞台。
b  b10–13 蓝底黄星，亮片是剪碎的银纸，随拍点闪烁。慢慢推近。"""
import math

import numpy as np

from common4 import (ishift, gauss, over, poly_mask, stroke_mask, downsample, taper_stroke, clawd_union_mask, to_px, rect_pts,
                     rigid_arm, check_proportions, LEGS, smooth)
import anim_core as A
import frame
import S05_马蒂斯剪纸 as S5
from S05_马蒂斯剪纸 import (K, PW, PH, Paper, jitter, circle, algae, shirt, baluster, note, star, SKY, ULTRA, EMERALD, LEMON,
                          CORAL, VERMIL, BLACK, WHITE, PINK, ORANGE, FIBER)
import A04_假如我们还爱着 as A4

FPS = 30
T0 = A4.T0 + 12 * A4.BEAT
BEAT = 60 / 80
WALL4, LABEL4 = A4.WALL4, A4.LABEL4
WALL5, LABEL5 = (0.92, 0.77, 0.63), "我管不住我自己"
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 13 * BEAT) * FPS)
HOLE = (850, 260, 190)
FLOWER = (840, 470, 130)
TX = 760
Yk, Xk = np.mgrid[0:PH, 0:PW].astype(np.float32)
_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


def M(pts, smooth_k=0.0, jit=0.6, seed=1):
    """固定种子的手剪轮廓（动画里每帧剪口一样，不会抖）。"""
    p = jitter(pts, jit, seed) if jit else np.asarray(pts, np.float64)
    return poly_mask(PW, PH, [(x * K, y * K) for x, y in p], smooth_k)


def line_y(x):
    return 70 + 48 * math.sin(math.pi * (x - 300) / 830)


# ---------- 阳台：静态层 ----------
def balcony_static():
    if "balcony" in _CACHE:
        return _CACHE["balcony"]
    S5.G = np.random.default_rng(5)
    G = S5.G
    P = Paper(SKY)
    sun = []
    for i in range(72):
        a = 2 * math.pi * i / 72
        r = 48 + (16 if i % 6 in (0, 1) else 0) + G.normal(0, 1.2)
        sun.append((985 + r * math.cos(a), 128 + r * math.sin(a)))
    P.stick(M(sun, 0.6, 0.3, 11), LEMON, lift=0.6, streak=1)
    for j, (cx, cy, s) in enumerate([(585, 300, 0.9), (1068, 236, 0.5)]):
        cl = [(cx - 70 * s, cy), (cx - 48 * s, cy - 20 * s), (cx - 18 * s, cy - 30 * s), (cx + 14 * s, cy - 24 * s),
              (cx + 44 * s, cy - 34 * s), (cx + 74 * s, cy - 14 * s), (cx + 86 * s, cy + 6 * s), (cx - 60 * s, cy + 8 * s)]
        P.stick(M(cl, 0.9, 1.0, 20 + j), WHITE, lift=0.6, streak=0)
    P.stick(M([(280, 336), (1130, 330), (1130, 450), (280, 450)], 0, 0.8, 30), ULTRA, streak=0, lift=0.7)
    for i in range(9):
        wx, wy = 330 + i * 92 + G.normal(0, 10), 362 + (i % 3) * 22
        wv = taper_stroke([(wx + k * 4, wy + 4 * math.sin(k / 2.5)) for k in range(12)], 2.5, 1.0, 4.0)
        P.stick(M(wv, 0.8, 0.2, 40 + i), WHITE, lift=0.3, tex=0.5, streak=0)
    ln = [(300, 70)] + [(300 + t * 830, 70 + 48 * math.sin(math.pi * t)) for t in np.linspace(0.02, 1, 30)]
    P.stick(stroke_mask(PW, PH, [(x * K, y * K) for x, y in ln], 2.2 * K), BLACK, lift=0.5, tex=0.2, streak=0)
    P.stick(M([(-10, -10), (300, -10), (300, 640), (-10, 640)], 0, 0.6, 50), CORAL, streak=2, lift=1.2)
    P.stick(M([(64, 66), (248, 66), (248, 545), (64, 545)], 0, 0.6, 51), ULTRA, streak=2, lift=0.6)
    P.stick(M([(72, 70), (130, 70), (122, 200), (138, 330), (116, 470), (128, 540), (72, 540)], 0.7, 1.0, 52), LEMON, lift=0.6, streak=1)
    P.stick(M([(330, 428), (1130, 428), (1130, 442), (330, 442)], 0, 0.6, 53), ULTRA, lift=0.8, streak=0)
    for k in range(15):
        P.stick(M(baluster(358 + k * 54, 442, 545, 22), 0.5, 0.4, 60 + k), ULTRA, lift=0.8, streak=k % 4)
    P.stick(M([(-10, 545), (1130, 545), (1130, 640), (-10, 640)], 0, 0.6, 80), EMERALD, streak=0, lift=1.2)
    P.stick(M([(410, 588), (950, 588), (980, 640), (380, 640)], 0, 0.6, 81), WHITE, lift=0.5, streak=0)
    for k in range(16):
        x = 410 + k * 34
        P.stick(M([(x, 596), (x + 17, 618), (x + 34, 596), (x + 34, 606), (x + 17, 628), (x, 606)], 0, 0.3, 90 + k), BLACK, lift=0.3, tex=0.4, streak=1)
    # 小桌与唱机（唱片另画）
    P.stick(M([(TX - 9, 494), (TX + 9, 494), (TX + 7, 584), (TX - 7, 584)], 0, 0.4, 110), ULTRA, lift=1.0, streak=0)
    P.stick(M([(TX - 54, 590), (TX - 30, 580), (TX + 30, 580), (TX + 54, 590), (TX + 50, 596), (TX - 50, 596)], 0.5, 0.4, 111), ULTRA, lift=1.0, streak=0)
    P.stick(M(circle(TX, 492, 104, 64, 17), 0, 0.5, 112), WHITE, streak=0, lift=1.1)
    P.stick(M([(TX - 74, 446), (TX + 74, 446), (TX + 78, 482), (TX - 78, 482)], 0.15, 0.4, 113), LEMON, lift=1.0, streak=1)
    P.stick(M([(TX - 60, 456), (TX - 46, 456), (TX - 46, 466), (TX - 60, 466)], 0, 0.2, 114), VERMIL, lift=0.4, streak=2)
    P.stick(M([(150, 515), (262, 515), (248, 600), (164, 600)], 0.2, 0.6, 115), VERMIL, lift=1.1, streak=2)
    _CACHE["balcony"] = P.img.copy()
    return _CACHE["balcony"]


def sticker(img):
    P = Paper(SKY)
    P.img = img
    return P


def shutters(P, b):
    """剪纸百叶窗：b3.5 被推开（像门一样翻过去，经过侧对时变窄）。"""
    o = A.back_out(A.seg(b, 3.35, 3.75), 1.4)
    for k, (hinge, w_closed, w_open) in enumerate([(64, 92, -54), (248, -92, 82)]):
        w = A.lerp(w_closed, w_open, o)
        x0, x1 = sorted((hinge, hinge + w))
        if x1 - x0 < 2:
            continue
        P.stick(M([(x0, 60), (x1, 66), (x1, 548), (x0, 556)], 0, 0.5, 120 + k), EMERALD, lift=1.0, streak=0)
        for j in range(17):
            yy = 80 + j * 27
            P.stick(M([(x0 + 6, yy), (x1 - 6, yy + 3), (x1 - 6, yy + 11), (x0 + 6, yy + 8)], 0, 0.3, 130 + j), (0.05, 0.36, 0.27), lift=0.25, tex=0.5, streak=1)


def sunlight(P, b):
    """百叶窗推开后，海藻形的阳光落在墙和地上。"""
    g = A.ease_out(A.seg(b, 3.6, 4.1))
    if g <= 0:
        return
    shapes = [(330, 600, 120, 26, -20, 4, 0.4, 0.3), (120, 600, 90, 22, -60, 3, -0.3, 1.2), (260, 300, 110, 22, -80, 4, 0.3, 2.1)]
    for k, (bx, by, ln, wd, ang, lob, bend, ph) in enumerate(shapes):
        m = M(algae(bx, by, ln * g, wd * g, ang, lob, bend, ph), 0.6, 0.4, 140 + k)
        P.img = over(P.img, (1.0, 0.92, 0.45), m * 0.55)


def plant(P, b):
    sw = 0.25 * math.sin(math.pi * b)
    leaves = [(205, 520, 210, 34, -100, 5, 0.5, 0.0, EMERALD), (190, 520, 170, 28, -128, 4, -0.4, 1.0, (0.04, 0.33, 0.25)),
              (222, 520, 185, 30, -72, 4, 0.5, 2.0, (0.20, 0.62, 0.30)), (200, 520, 120, 22, -150, 3, -0.3, 0.5, EMERALD)]
    for k, (bx, by, ln, wd, ang, lob, bend, ph, col) in enumerate(leaves):
        P.stick(M(algae(bx, by, ln, wd, ang + 6 * math.sin(math.pi * b + k), lob, bend + sw, ph), 0.6, 0.4, 150 + k), col, lift=0.9, streak=k % 4)
    P.stick(M([(150, 515), (262, 515), (248, 600), (164, 600)], 0.2, 0.6, 115), VERMIL, lift=1.1, streak=2)


def clothes(P, b, claw):
    """按拍点挂上晾衣绳：b4 粉衬衫、b4.5 黄袜子、b5 条纹毛巾。挂之前从 Clawd 的钳子那里升上去。"""
    items = []
    tx = 720
    items.append((4.0, (470, line_y(470)), lambda dx, dy: (M(shirt(430 + dx, line_y(470) + 2 + dy, 92, 118), 0.4, 0.8, 160), PINK)))
    items.append((4.5, (615, line_y(615)), lambda dx, dy: (M([(600 + dx, line_y(615) + dy), (630 + dx, line_y(615) + dy), (632 + dx, line_y(615) + 70 + dy),
                                                           (660 + dx, line_y(615) + 92 + dy), (650 + dx, line_y(615) + 108 + dy), (612 + dx, line_y(615) + 88 + dy)], 0.5, 0.6, 161), LEMON)))
    items.append((5.0, (768, line_y(768)), lambda dx, dy: (M([(tx + dx, line_y(tx) + 1 + dy), (tx + 96 + dx, line_y(tx + 96) + 1 + dy), (tx + 100 + dx, line_y(tx + 96) + 132 + dy),
                                                           (tx + 4 + dx, line_y(tx) + 128 + dy)], 0, 0.7, 162), WHITE)))
    for k, (bb, anchor, make) in enumerate(items):
        u = A.seg(b, bb - 0.4, bb)
        if u <= 0:
            continue
        e = A.back_out(u, 1.2)
        dx = (claw[0] - anchor[0]) * (1 - e)
        dy = (claw[1] - anchor[1] - 60) * (1 - e)
        m, col = make(dx, dy)
        P.stick(m, col, streak=2, lift=0.8)
        if k == 2:
            for j in range(3):
                yy = line_y(tx) + 30 + j * 30 + dy
                P.stick(M([(tx + 2 + dx, yy), (tx + 99 + dx, yy + 4), (tx + 99 + dx, yy + 13), (tx + 2 + dx, yy + 9)], 0, 0.5, 170 + j), ULTRA, lift=0.4, streak=0)
        if u >= 1:
            for x in ((452, 500), (608, 628), (735, 805))[k]:
                P.stick(M([(x - 3, line_y(x) - 8), (x + 3, line_y(x) - 8), (x + 3, line_y(x) + 12), (x - 3, line_y(x) + 12)], 0, 0.2, 180 + x), VERMIL, lift=0.5, tex=0.4, streak=2)


def record(P, b, claw):
    """唱片：从 Clawd 的钳子落到唱机上，然后转起来、飘出音符。"""
    u = A.ease_io(A.seg(b, 5.8, 6.3))
    if b < 5.5:
        return
    cx, cy = A.lerp(claw[0], TX + 2, u), A.lerp(claw[1] - 20, 440, u)
    P.stick(M(circle(cx, cy, 66, 64, 15), 0, 0.3, 190), BLACK, lift=0.8, tex=0.5, streak=0)
    spin = max(0.0, b - 6.3) * 2.2
    for rr in (52, 38):
        ring = stroke_mask(PW, PH, [((cx + rr * math.cos(a + spin)) * K, (cy + rr * 0.23 * math.sin(a + spin)) * K) for a in np.linspace(0.3, 2.6, 30)], 1.0 * K)
        P.img = over(P.img, (0.22, 0.22, 0.26), ring * 0.8)
    P.stick(M(circle(cx, cy, 18, 32, 5), 0, 0.2, 191), VERMIL, lift=0.3, streak=2)
    if u >= 1:
        P.stick(M(taper_stroke([(TX + 66, 428), (TX + 52, 436), (TX + 30, 440), (TX + 16, 438)], 5, 3), 0.3, 0.1, 192), WHITE, lift=0.6, tex=0.4, streak=0)
        for k, (x, y, s, fl) in enumerate([(852, 396, 0.9, True), (912, 352, 1.05, False), (978, 322, 0.95, True), (1042, 292, 0.8, True)]):
            nu = A.seg(b, 6.5 + 0.4 * k, 7.5 + 0.4 * k)
            if nu <= 0:
                continue
            ox, oy = (TX + 40 - x) * (1 - A.ease_out(nu)), (430 - y) * (1 - A.ease_out(nu)) - 20 * max(0.0, b - 7.5 - 0.4 * k)
            for j, part in enumerate(note(x + ox, y + oy + 6 * math.sin(b * 3 + k), s * min(1.0, nu * 2), fl)):
                P.stick(M(part, 0.2, 0.2, 200 + 10 * k + j), BLACK, lift=0.7, tex=0.4, streak=0)


def bird(P, b, head):
    u = A.ease_out(A.seg(b, 6.9, 7.5))
    if u <= 0:
        return
    bx, by = A.lerp(1150, head[0], u), A.lerp(120, head[1], u) - 30 * math.sin(math.pi * u)
    flap = math.sin(b * 14) * (1 - u) * 14
    body = [(bx - 30, by - 14), (bx - 12, by - 26), (bx + 12, by - 25), (bx + 26, by - 16), (bx + 36, by - 22),
            (bx + 30, by - 10), (bx + 14, by - 1), (bx - 18, by - 2)]
    P.stick(M(body, 0.7, 0.3, 220), WHITE, lift=0.8, streak=0)
    P.stick(M([(bx - 8, by - 20), (bx + 18, by - 44 - flap), (bx + 22, by - 36 - flap), (bx + 8, by - 14)], 0.6, 0.3, 221), SKY, lift=0.5, streak=0)
    P.stick(M([(bx - 30, by - 15), (bx - 42, by - 12), (bx - 30, by - 9)], 0, 0.1, 222), LEMON, lift=0.3, tex=0.3, streak=1)
    P.stick(M(circle(bx - 20, by - 17, 2.6, 12), 0, 0, 223), BLACK, lift=0.1, tex=0, streak=0)


def home_clawd(P, FX, FY, S, arms, ang=0.0, mask=None):
    mm = 1.0 if mask is None else mask
    top = [(-0.4, 5.3), (16.4, 5.3), (16.9, 10.5), (-0.9, 10.5)]
    check_proportions(arms=arms, legs=LEGS, extra={"top": [top]})
    cm = clawd_union_mask(PW, PH, FX * K, FY * K, S * K, arms=arms, ang=ang) * mm
    P.stick(cm, ORANGE, streak=1, lift=1.3)
    tm = poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(jitter(top, 0.04, 3), FX, FY, S, ang)]) * mm
    P.stick(tm, WHITE, streak=0, lift=0.7)
    for k in range(4):
        yy = 6.1 + k * 1.15
        st = [(-0.6, yy), (16.6, yy), (16.7, yy + 0.48), (-0.7, yy + 0.48)]
        P.stick(poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(jitter(st, 0.03, 10 + k), FX, FY, S, ang)]) * tm, ULTRA, lift=0.3, tex=0.6, streak=0)
    for ex in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        P.stick(poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(rect_pts(ex), FX, FY, S, ang)]) * mm, BLACK, lift=0.3, tex=0.3, streak=0)


# ---------- 《爵士》舞台 ----------
def jazz_static():
    if "jazz" in _CACHE:
        return _CACHE["jazz"]
    P = Paper(ULTRA)
    g = np.random.default_rng(55)
    for side in (-1, 1):
        x0 = 0 if side < 0 else 1120
        cur = [(x0, -10), (x0 - side * 170, -10), (x0 - side * 120, 180), (x0 - side * 190, 360), (x0 - side * 110, 640), (x0, 640)]
        P.stick(M(cur, 0.6, 1.0, 300 + side), VERMIL, streak=2, lift=1.2)
    val = [(-10, -10), (1130, -10)] + [(1130 - i * 40, 46 + 16 * (i % 2)) for i in range(30)] + [(-10, 46)]
    P.stick(M(val, 0.3, 0.6, 303), BLACK, lift=1.0, streak=0)
    for i in range(22):
        x, y = g.uniform(190, 930), g.uniform(80, 420)
        if abs(x - 560) < 170 and y > 200:
            continue
        P.stick(M(star(x, y, g.uniform(14, 30), 5, g.uniform(0, 1), 0.2, i), 0.0, 0.5, 310 + i), LEMON, lift=0.8, streak=i % 4)
    P.stick(M(circle(560, 330, 190, 120), 0, 1.2, 340), WHITE, streak=0, lift=0.9)
    P.stick(M([(-10, 540), (1130, 540), (1130, 640), (-10, 640)], 0, 0.6, 341), BLACK, lift=1.2, streak=0)
    wave = [(-10, 540)] + [(i * 28, 540 + 9 * math.sin(i * 1.3)) for i in range(42)] + [(1130, 556), (-10, 556)]
    P.stick(M(wave, 0.5, 0.4, 342), WHITE, lift=0.5, streak=0)
    _CACHE["jazz"] = P.img.copy()
    return _CACHE["jazz"]


def sequin_clawd(P, FX, FY, S, b, ang=0.0, mask=None, mic=True):
    from kit import mic_px
    mm = 1.0 if mask is None else mask
    arms = [rigid_arm("L", 30), rigid_arm("R", 30)]
    dress = [(-0.3, 5.4), (16.3, 5.4), (17.0, 9.6), (-1.0, 9.6)]
    check_proportions(arms=arms, legs=LEGS, extra={"dress": [dress]})
    cm = clawd_union_mask(PW, PH, FX * K, FY * K, S * K, arms=arms, ang=ang) * mm
    P.stick(cm, ORANGE, streak=1, lift=1.3)
    dm = poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(dress, FX, FY, S, ang)]) * mm
    P.stick(dm, (0.78, 0.80, 0.84), streak=0, lift=0.8)
    silv = [(0.92, 0.93, 0.96), (0.66, 0.68, 0.74), (0.98, 0.98, 1.0), (0.80, 0.84, 0.92)]
    g = np.random.default_rng(56)
    beat = int(math.floor(b))
    for k in range(70):
        gx, gy = g.uniform(-0.4, 16.4), g.uniform(5.6, 9.4)
        (px, py), = to_px([(gx, gy)], FX, FY, S, ang)
        r = g.uniform(4.0, 7.5) * S / 18
        pm = M(circle(px, py, r, 10, r * g.uniform(0.7, 1.0)), 0, 0.3, 400 + k) * dm
        c = silv[(k + beat) % 4]                       # 随拍点轮换明暗：亮片在闪
        P.stick(pm, c, lift=0.25, tex=0.4, streak=0)
    for ex in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        P.stick(poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(rect_pts(ex), FX, FY, S, ang)]) * mm, BLACK, lift=0.3, tex=0.3, streak=0)
    if mic:
        from kit import mic_held
        mc = mic_held(arms[1], FX, FY, S, lean=12, ang=ang)
        P.stick(mc["handle"] * mm, BLACK, lift=0.6, streak=0)
        P.stick(mc["band"] * mm, (0.55, 0.56, 0.60), lift=0.6, streak=0)
        P.stick(mc["head"] * mm, (0.80, 0.82, 0.86), lift=0.6, streak=0)
        P.img = P.img * (1 - 0.35 * mc["grille"] * mm)[..., None]
        P.stick(poly_mask(PW, PH, [(x * K, y * K) for x, y in to_px(arms[1], FX, FY, S, ang)]) * mm, ORANGE, streak=1, lift=0.8)


def zoom_k(src, z, c_src, c_dst):
    from scipy.ndimage import map_coordinates
    qx = c_src[0] + (Xk / K - c_dst[0]) / z
    qy = c_src[1] + (Yk / K - c_dst[1]) / z
    return np.stack([map_coordinates(src[..., c], [qy * K, qx * K], order=1, mode="nearest") for c in range(3)], -1)


# ---------- 一帧 ----------
def a_pose(b):
    """a 段 Clawd：(fx, fy, arms, 钳子位置)。"""
    S = 17
    if b < 3.4:
        x = A.lerp(640, 400, A.ease_io(A.seg(b, 3.0, 3.35)))
        arms = [rigid_arm("L", 38), rigid_arm("R", -32)]
    elif b < 3.9:
        x = 400
        up = math.sin(math.pi * A.seg(b, 3.3, 3.8))
        arms = [rigid_arm("L", 38 - 70 * up), rigid_arm("R", -32 + 40 * up)]
    elif b < 5.6:
        x = A.lerp(400, 640, A.ease_io(A.seg(b, 3.9, 5.5)))
        lift = abs(math.sin(math.pi * 2 * (b - 3.9)))
        arms = [rigid_arm("L", 38 - 40 * lift), rigid_arm("R", -32 + 60 * lift)]
    else:
        x = 640
        put = math.sin(math.pi * A.seg(b, 5.6, 6.3))
        arms = [rigid_arm("L", 38), rigid_arm("R", -32 + 62 * put)]
    bob = 4 * abs(math.sin(math.pi * 2 * b)) if (3.0 <= b < 3.35 or 3.9 <= b < 5.5) else 0
    fy = 612 - bob
    claw = (x + 8.6 * S, fy - 9 * S)
    return x, fy, S, arms, claw


def a_scene(b, with_clawd=True):
    P = sticker(balcony_static().copy())
    sunlight(P, b)
    shutters(P, b)
    plant(P, b)
    x, fy, S, arms, claw = a_pose(b)
    clothes(P, b, claw)
    record(P, b, claw)
    if with_clawd:
        home_clawd(P, x, fy, S, arms)
        bird(P, b, (x - 35, fy - 12 * S))
    return P.img


def flower(P, b, open_):
    """大剪纸花：珊瑚色的大花瓣一片片开出来，柠檬黄的花心。"""
    cx, cy, R = FLOWER
    for k in range(8):
        a = k * math.pi / 4 + 0.2
        r = R * open_
        pet = [(cx + r * 0.25 * math.cos(a + d), cy + r * 0.25 * math.sin(a + d)) for d in (-0.5, 0.5)]
        tip = (cx + r * math.cos(a), cy + r * math.sin(a))
        pts = [pet[0], (cx + r * 0.75 * math.cos(a - 0.35), cy + r * 0.75 * math.sin(a - 0.35)), tip, (cx + r * 0.75 * math.cos(a + 0.35), cy + r * 0.75 * math.sin(a + 0.35)), pet[1]]
        P.stick(M(pts, 0.8, 0.6, 500 + k), PINK if k % 2 else CORAL, lift=0.9, streak=k % 4)
    P.stick(M(circle(cx, cy, R * 0.36 * open_, 48), 0, 0.6, 520), LEMON, lift=0.7, streak=1)


def painting(t, b):
    if b < 3.0:
        return e_frame(t, b)
    if b < 8.0:
        img = a_scene(b)
        z = 1.07
        cx = A.lerp(510, 610, A.seg(b, 3.0, 8.0))
        return ("zoom", img, z, cx), True
    # M：大剪纸花开出来，Clawd 跳进花心，花心张开成舞台
    P = sticker(a_scene(8.0, with_clawd=False))
    op = A.back_out(A.seg(b, 8.0, 8.4), 1.3)
    flower(P, b, op)
    hx, hy, hr = FLOWER
    ju = A.seg(b, 8.4, 9.2)
    fx, fy = A.hop((640, 612), (hx, hy + 60), ju, 160)
    S = A.lerp(17, 9, A.ease_io(ju))
    reveal = hr * 0.29 * op + A.ease_in(A.seg(b, 9.0, 10.0), 1.7) * 1300     # 花心就是通往舞台的窗口，外面留一圈柠檬黄
    s_st = A.lerp(0.25, 1.0, A.ease_io(A.seg(b, 9.0, 10.0)))
    c_now = A.lerp2((hx, hy), (560, 400), A.ease_io(A.seg(b, 9.0, 10.0)))
    stg = sticker(zoom_k(jazz_static(), s_st, (560, 400), c_now))
    fin = A.ease_io(A.seg(b, 9.2, 10.0))
    fx, fy, S = A.lerp(fx, 560, fin), A.lerp(fy, 548, fin), A.lerp(S, 18, fin)
    disk = M(circle(c_now[0], c_now[1], reveal, 120), 0, 0, 1)
    sequin_clawd(stg, fx, fy, S, b, mask=disk, mic=b > 9.6)
    home_clawd(P, fx, fy, S, [rigid_arm("L", 30), rigid_arm("R", 30)], mask=(1 - disk))
    img = P.img * (1 - disk[..., None]) + stg.img * disk[..., None]
    return img, True


def e_frame(t, b):
    """04 结尾的金箔扇贝：沿圆弧剪开，剪下的一块卷着掉下去，后面是阳台。"""
    shell = A4.shell_static()
    hx, hy, hr = HOLE
    cut = sum(A.ease_out(A.seg(b, k, k + 0.25)) for k in (0.5, 1.0, 1.5)) / 3      # 三下「咔嚓」
    fall = A.ease_in(A.seg(b, 1.75, 2.6), 1.6)
    grow = A.ease_in(A.seg(b, 2.2, 3.0), 1.8) * 1300
    # 背后的阳台（百叶窗还关着）
    behind = a_scene(3.0, with_clawd=False)
    # 金箔这一层（不含 Clawd）
    gold = A4.age(A4.roses(shell.copy(), 12.0 + b * BEAT / A4.BEAT))
    hole = M(circle(hx, hy, hr + grow, 160), 0, 0.0, 1)
    open_m = hole if fall > 0 else np.zeros_like(hole)
    img = gold * (1 - open_m[..., None]) + behind * open_m[..., None]
    # 剪下来的那块：绕剪口左下方往下翻落，背面（纸背）越翻越多
    if 0 < fall < 1:
        ang = 70 * fall
        dy = 700 * fall ** 1.5
        from scipy.ndimage import map_coordinates
        a = math.radians(ang)
        px, py = hx - hr * 0.7, hy + hr * 0.7
        qx = px + ((Xk / K - px) * math.cos(a) + (Yk / K - py - dy) * math.sin(a))
        qy = py + (-(Xk / K - px) * math.sin(a) + (Yk / K - py - dy) * math.cos(a)) / max(0.25, math.cos(a * 0.8))
        src_in = (np.hypot(qx - hx, qy - hy) < hr).astype(np.float32)
        piece = np.stack([map_coordinates(gold[..., c], [qy * K, qx * K], order=1, mode="nearest") for c in range(3)], -1)
        curl = (0.75 + 0.25 * np.clip((qy - (hy - hr)) / (2 * hr), 0, 1))
        piece = piece * curl[..., None] * (1 - 0.5 * fall) + np.array([0.90, 0.86, 0.78]) * 0.5 * fall
        src_in = gauss(src_in, 0.5 * K)
        img *= (1 - 0.3 * gauss(src_in, 8 * K))[..., None]
        img = img * (1 - src_in[..., None]) + piece * src_in[..., None]
    # 剪口：已剪开的那段弧上一道细黑缝
    if fall <= 0 and cut > 0:
        th = np.linspace(math.pi * 0.75, math.pi * 0.75 - 2 * math.pi * cut, max(4, int(160 * cut)))
        arc = [(hx + hr * math.cos(a), hy + hr * math.sin(a)) for a in th]
        seam = stroke_mask(PW, PH, [(x * K, y * K) for x, y in arc], 1.6 * K)
        img = img * (1 - 0.7 * seam[..., None])
    # Clawd：先在原地「咔嚓」，然后跳进切口
    up = abs(math.sin(math.pi * 2 * min(b, 1.75)))
    ju = A.seg(b, 2.0, 2.95)
    fx, fy = A.hop((560, 590), (640, 612), ju, 170)
    S = A.lerp(15, 17, A.ease_io(ju))
    inside = M(circle(hx, hy, hr + grow, 160), 0, 0.0, 1) if fall > 0 else np.zeros((PH, PW), np.float32)
    img = A4.gold_clawd(img, fx, fy, S, 12.0, mask=(1 - inside))
    P = sticker(img)
    home_clawd(P, fx, fy, S, [rigid_arm("L", 38), rigid_arm("R", -32)], mask=inside)
    img = P.img
    if b < 2.0:                                         # 剪刀一样开合的右钳：在钳尖画一道小闪光
        (cx_, cy_), = to_px([(17.2, 3.0)], 560, 590, 15)
        img = over(img, (1.0, 0.98, 0.9), M(circle(cx_ + 6, cy_ - 4, 3 + 3 * up, 12), 0, 0, 2) * 0.8)
    out = ju >= 1 or (grow > 300)
    return img, out


def frame_at(i):
    t = i / FPS
    b = bt(A.twos(i))
    if b >= 10.0:
        P = sticker(jazz_static().copy())
        sequin_clawd(P, 560, 548, 18, b, ang=2.0 * math.sin(math.pi * (b - 10.0)))
        p, sw = P.img * (1 + 0.025 * FIBER)[..., None], True
    else:
        p, sw = painting(t, b)
    if isinstance(p, tuple):
        _, img, z, cx = p
        p = np.clip(downsample(img * (1 + 0.025 * FIBER)[..., None], K), 0, 1)
        p = A.zoom_painting(p, z, cx, 360)
    else:
        p = np.clip(downsample(p * (1 + 0.025 * FIBER)[..., None] if b < 10 else p, K), 0, 1)
    bb = bt(t)
    if bb >= 10.0:
        p = A.zoom_painting(p, 1 + 0.08 * A.ease_io(A.seg(bb, 10.0, 13.0)), 560, 400)
    return A.compose(p, WALL5 if sw else WALL4, LABEL5 if sw else LABEL4)


def warm():
    A4.warm()
    balcony_static()
    jazz_static()
    for w, l in ((WALL4, LABEL4), (WALL5, LABEL5)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a05_{i:04d}.png")
