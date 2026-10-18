"""04 假如我们还爱着 · 文艺复兴湿壁画 · 63 BPM（0:38.05–0:49.48）。画心动作「一拍二」。拍号 b 从 03 结束算起。
E  b0–2   Clawd 走到舞台左边，掀起青海波幕布的一角：原来这张木版画是贴在灰泥墙上的一张纸。纸从左下角被揭开、翻起，
          纸背后是湿壁画的回廊；Clawd 钻到纸后面，越过纸边的部分变成湿壁画质感、换上黑色无袖衣装。
a  b2–7   拱廊里 Clawd 从左往右走过柱子，每过一根柱子，原地留下一个它的身影：侧身、四分之三侧身、正面
          （同一人物在一幅画里出现多次的连续叙事）。玫瑰花瓣从拱顶飘落，最后几个身影一起回头。
M  b7–9   身影汇合成一个，Clawd 走进后墙上的金箔光环；镜头推进光环，进去的部分换上金色长裙和羽毛头饰。
b  b9–12  致敬波提切利：金箔底、大扇贝，羽毛随拍点颤动，玫瑰飘落。画框里慢慢拉远。"""
import math

import numpy as np

from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, smooth, over, downsample, ishift, outline)
import anim_core as A
import frame
import S04_湿壁画 as S4
from S04_湿壁画 import Fresco, INTONACO, GR, VP
import A03_向日葵朝着夜 as A3

FPS = 30
T0 = A3.T0 + 12 * A3.BEAT
BEAT = 60 / 63
WALL3, LABEL3 = A3.WALL3, A3.LABEL3
WALL4, LABEL4 = (0.80, 0.76, 0.69), "假如我们还爱着"
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 12 * BEAT) * FPS)
HALO = (560, 300, 56)
FOOT = 572
GHOSTS = [(240, -90), (560, -45), (880, 0)]
_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


# ---------- 静态层 ----------
def age_maps():
    if "age" in _CACHE:
        return _CACHE["age"]
    n = fbm(PH, PW, 140, 44, 3)
    crack = smooth(0.994, 0.999, 1 - np.abs(2 * n - 1)) * smooth(0.5, 0.75, fbm(PH, PW, 260, 48, 2))
    seams = [SM(catmull(pts, 80), 0.7, 0.8) for pts in ([(0, 210), (300, 196), (520, 214), (760, 190), (1120, 206)], [(680, 0), (690, 200), (660, 420), (700, 630)])]
    loss = smooth(0.77, 0.79, fbm(PH, PW, 40, 45, 5)) * smooth(0.6, 0.75, fbm(PH, PW, 300, 46, 2))
    loss = loss * smooth(170 * K, 30 * K, np.minimum.reduce([X, PW - X, Y, PH - Y]))
    lsh = 0.25 * gauss(np.clip(ishift(loss, 2 * K, 2 * K) - loss, 0, 1), 0.8 * K)
    mul = (1 - 0.14 * crack)
    for s in seams:
        mul = mul * (1 - 0.10 * s)
    sover = np.clip(sum(ishift(s, K, K) for s in seams) * 0.25, 0, 1)
    _CACHE["age"] = (mul.astype(np.float32), sover.astype(np.float32), loss.astype(np.float32), (1 - lsh).astype(np.float32))
    return _CACHE["age"]


def age(img):
    mul, sover, loss, lsh = age_maps()
    img = img * mul[..., None]
    img = over(img, (0.95, 0.92, 0.86), sover)
    img = img * (1 - loss[..., None]) + np.array([0.93, 0.91, 0.86]) * (1 + 0.06 * GR)[..., None] * loss[..., None]
    img = img * lsh[..., None]
    return img * 0.92 + 0.06


def loggia_static():
    """回廊：后层（人物之后画的都在前层）与前层拱廊（颜色 + 覆盖）。"""
    if "loggia" in _CACHE:
        return _CACHE["loggia"]
    # 直接复用样张的 render()：把 place 换成「记下此刻的画面」，人物之后画的就是前层拱廊
    orig_place = S4.place
    orig_age = Fresco.age
    snaps = {}

    def fake_place(Fr_, spr, cx, foot):
        if "back" not in snaps:
            snaps["back"] = Fr_.img.copy()
    S4.place = fake_place
    Fresco.age = lambda self: None
    orig_rot = S4.rot_pts
    S4.rot_pts = lambda pts, *a, **k: [(0, 0), (0.01, 0), (0, 0.01)]    # 样张里静止的花瓣不要，动画里另画飘落的
    G_old = S4.G
    S4.G = np.random.default_rng(4)
    try:
        full = S4.render()
    finally:
        S4.place = orig_place
        Fresco.age = orig_age
        S4.rot_pts = orig_rot
        S4.G = G_old
    back = snaps["back"]
    # 前层 = 拱廊与前景地面：两张图不同的地方
    diff = np.abs(full - back).max(-1)
    front_a = gauss(np.clip(diff * 40, 0, 1), 0.3 * K)
    _CACHE["loggia"] = (back.astype(np.float32), full.astype(np.float32), front_a.astype(np.float32))
    return _CACHE["loggia"]


def sprite(az):
    az = int(round(az / 5.0) * 5)
    key = ("spr", az)
    if key not in _CACHE:
        _CACHE[key] = S4.sprite_clawd(az, 9 * K)
    return _CACHE[key]


def place(img, az, cx, foot, alpha=1.0):
    """把三维算好的身影平涂进壁画（与样张 place() 相同的做法，结果直接返回 img）。"""
    Fr = Fresco()
    Fr.img = img
    spr = sprite(az)
    h, w = spr.shape[:2]
    ys, xs = np.nonzero(spr[..., 3] > 0.5)
    by = ys.max()
    x0, y0 = int(cx * K - w / 2), int(foot * K - by)
    rgb = np.zeros((PH, PW, 3), np.float32)
    a = np.zeros((PH, PW), np.float32)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    ww, hh = min(w - sx0, PW - dx0), min(h - sy0, PH - dy0)
    if ww <= 0 or hh <= 0:
        return img
    rgb[dy0:dy0 + hh, dx0:dx0 + ww] = spr[sy0:sy0 + hh, sx0:sx0 + ww, :3]
    a[dy0:dy0 + hh, dx0:dx0 + ww] = spr[sy0:sy0 + hh, sx0:sx0 + ww, 3]
    lum = rgb.mean(-1)
    step = gauss(np.round(lum * 6) / 6, 0.7 * K)
    col = rgb / (lum[..., None] + 1e-3) * step[..., None] * 1.45
    col = col * 0.85 + gauss(col, 1.5 * K) * 0.15
    sh = gauss(M(circle(cx + 10, foot + 2, w / K * 0.45, 48, 8)), 6 * K)
    Fr.img = Fr.img * (1 - 0.22 * sh * alpha)[..., None]
    Fr.paint(a * alpha, np.clip(col, 0, 1))
    Fr.sinopia(np.clip(a * alpha, 0, 1), 1, op=0.55)
    return Fr.img


def petals(img, b, n=24, y_top=110):
    Fr = Fresco()
    Fr.img = img
    g = np.random.default_rng(44)
    for i in range(n):
        x0, ph, sp = g.uniform(100, 1020), g.uniform(0, 4), g.uniform(0.7, 1.2)
        y = y_top + ((b - 2 + ph) * 60 * sp) % 480
        x = x0 + 18 * math.sin(b * 1.7 + i)
        rot = g.uniform(0, 180) + 40 * b
        Fr.paint(M(rot_pts(circle(x, y, 5, 20, 3), x, y, rot)), (0.74, 0.34, 0.36))
    return Fr.img


def flat_fresco_clawd(img, fx, fy, S, mask=None):
    """E 段：二维的湿壁画 Clawd（黑色无袖衣装包住身体下部），和三维身影同样的平涂法。"""
    Fr = Fresco()
    Fr.img = img
    mm = 1.0 if mask is None else mask
    arms = [rigid_arm("L", 0), rigid_arm("R", 0)]
    cm, em = clawd(fx, fy, S, arms)
    Fr.paint(cm * mm, (0.80, 0.50, 0.38), shade=(1.06 - 0.22 * smooth(fx - 8 * S, fx + 8 * S, X / K)))
    cloth = garment([(-0.15, 5.5), (16.15, 5.5), (16.15, 10.0), (-0.15, 10.0)], fx, fy, S, name="cloth") * mm
    Fr.paint(cloth, (0.12, 0.12, 0.13))
    Fr.paint(em * mm, (0.10, 0.07, 0.06))
    Fr.sinopia(np.clip(cm * mm, 0, 1), 1, op=0.5)
    return Fr.img


# ---------- 04-b：金箔扇贝 ----------
def shell_static():
    if "shell" in _CACHE:
        return _CACHE["shell"]
    orig_age = Fresco.age
    orig_clawd = S4.clawd
    Fresco.age = lambda self: None
    snaps = {}

    def stop_clawd(*a, **k):
        snaps["img"] = True
        raise StopIteration
    import types
    # 复制 render_b 画 Clawd 之前的部分：在第一次调用 clawd() 时截住
    Fr_holder = {}
    orig_init = Fresco.__init__

    def init(self):
        orig_init(self)
        Fr_holder["fr"] = self
    Fresco.__init__ = init
    S4.clawd = stop_clawd
    try:
        S4.render_b()
    except StopIteration:
        pass
    finally:
        Fresco.age = orig_age
        S4.clawd = orig_clawd
        Fresco.__init__ = orig_init
    _CACHE["shell"] = Fr_holder["fr"].img.astype(np.float32)
    return _CACHE["shell"]


def gold_clawd(img, FX, FY, S, b, mask=None):
    Fr = Fresco()
    Fr.img = img
    mm = 1.0 if mask is None else mask
    arms = [rigid_arm("L", 0), rigid_arm("R", 0)]
    cm, em = clawd(FX, FY, S, arms)
    Fr.paint(cm * mm, (0.80, 0.50, 0.38), shade=(1.06 - 0.22 * smooth(FX - 8 * S, FX + 8 * S, X / K)))
    gown = garment([(-0.2, 5.4), (16.2, 5.4), (17.0, 8.6), (18.0, 12.0), (-2.0, 12.0), (-1.0, 8.6)], FX, FY, S, arms=arms, name="gown") * mm
    Fr.paint(gown, (0.86, 0.66, 0.30), shade=(1.08 - 0.30 * smooth(FX - 9 * S, FX + 9 * S, X / K)))
    for gx in (2.0, 5.0, 8.0, 11.0, 14.0):
        Fr.paint(SM(g2p([(gx, 6.0), (gx + (gx - 8) * 0.2, 11.9)], FX, FY, S), 1.3 * S / 15, 0.8, False) * mm, (0.66, 0.46, 0.20))
    Fr.sinopia(gown, 1, op=0.5)
    Fr.sinopia(cm * mm, 1, op=0.5)
    Fr.paint(em * mm, (0.16, 0.10, 0.08))
    # 羽毛头饰：每拍颤一下
    q = S / 15
    for k, ang0 in enumerate((-38, -18, 0, 18, 38)):
        tremble = 5 * math.exp(-((b % 1.0) / 0.18)) * math.sin((b % 1.0) * 40 + k) * (1 if b >= 9 else 0)
        ang = ang0 + tremble
        a_ = math.radians(ang - 90)
        bx, by = g2p([(8 + ang0 / 12, 0.2)], FX, FY, S)[0]
        L = (120 - abs(ang0) * 0.8) * q
        spine = [(bx + L * t * math.cos(a_) + 6 * q * math.sin(t * 3) * (1 if ang0 >= 0 else -1), by + L * t * math.sin(a_)) for t in np.linspace(0, 1, 20)]
        for side in (-1, 1):
            barbs = []
            for t in np.linspace(0.05, 1, 16):
                px, py = spine[int(t * 19)]
                w = 19 * q * math.sin(math.pi * min(t * 1.1, 1))
                barbs.append((px + side * w * math.cos(a_ + math.pi / 2), py + side * w * math.sin(a_ + math.pi / 2)))
            Fr.paint(M([spine[0]] + barbs + [spine[-1]], 0.6) * mm, (0.97, 0.94, 0.86) if k % 2 == 0 else (0.92, 0.84, 0.62))
        Fr.paint(SM(spine, 1.0 * q, 0.8, False) * mm, (0.70, 0.56, 0.30))
    Fr.paint(M(g2p([(5.6, -0.6), (10.4, -0.6), (10.0, 0.4), (6.0, 0.4)], FX, FY, S)) * mm, (0.78, 0.60, 0.26))
    return Fr.img


def roses(img, b):
    Fr = Fresco()
    Fr.img = img
    g = np.random.default_rng(14)
    for i in range(16):
        x0, ph = g.uniform(80, 1040), g.uniform(0, 5)
        rot = g.uniform(0, 6.28) + 0.5 * b
        x = x0 + 14 * math.sin(b * 1.3 + i)
        y = 40 + ((b + ph) * 34) % 430
        pts = [(x + 9 * (0.75 + 0.25 * math.cos(5 * t)) * math.cos(t + rot), y + 9 * (0.75 + 0.25 * math.cos(5 * t)) * math.sin(t + rot)) for t in np.linspace(0, 2 * math.pi, 60)]
        rm = M(pts, 0.4)
        Fr.paint(rm, (0.86, 0.52, 0.56), shade=(1.05 - 0.2 * smooth(x - 9, x + 9, X / K)))
        Fr.sinopia(rm, 1, op=0.45)
        sp = [(x + 4.5 * (1 - t / 2.4) * math.cos(t * 2.6 + rot), y + 4.5 * (1 - t / 2.4) * math.sin(t * 2.6 + rot)) for t in np.linspace(0, 2.2, 24)]
        Fr.paint(SM(sp, 0.8, 0.8, False), (0.60, 0.28, 0.32))
        Fr.paint(M(rot_pts([(x + 7, y + 4), (x + 18, y + 2), (x + 10, y + 9)], x, y, math.degrees(rot)), 0.5), (0.40, 0.52, 0.34))
    return Fr.img


# ---------- E：揭开木版画 ----------
def paper_img():
    """03 结尾那张木版画（推近 1.08 倍，不含 Clawd）。"""
    if "paper" not in _CACHE:
        _CACHE["paper"] = A3.warp(A3.stage_static(), 1.08, (560, 400), (560, 400))
    return _CACHE["paper"]


def peel(b):
    """纸从左下角被揭起：折线垂直于对角线方向，沿对角线往右上推进。返回 (未揭部分遮罩, 翻起的纸背遮罩, 纸背明暗)。"""
    d = A.ease_in(A.seg(b, 0.85, 2.0), 1.6) * 1400
    if d <= 0:
        return np.ones((PH, PW), np.float32), np.zeros((PH, PW), np.float32), None
    u = np.array([1120.0, -630.0])
    u /= np.linalg.norm(u)
    c = np.array([0.0, 630.0])
    s = (X / K - c[0]) * u[0] + (Y / K - c[1]) * u[1]
    attached = smooth(d - 0.6, d + 0.6, s)
    # 纸背：把已揭起的部分沿折线翻过来
    rx = X / K - 2 * (s - d) * u[0]
    ry = Y / K - 2 * (s - d) * u[1]
    inside = (rx >= 0) & (rx <= 1120) & (ry >= 0) & (ry <= 630)
    flap = (inside & (s > d)).astype(np.float32)
    flap = gauss(flap, 0.4 * K)
    shade = 0.80 + 0.2 * np.clip((s - d) / 160, 0, 1)
    return attached, flap, shade


def e_frame(t, b):
    back, full, front_a = loggia_static()
    # 湿壁画回廊（纸后面）
    wall = back.copy()
    att, flap, fshade = peel(b)
    # Clawd：舞台中央 → 走到左边掀纸 → 钻到纸后面，变小、落到回廊左侧
    w = A.ease_io(A.seg(b, 0.0, 0.85))
    sx = A.lerp(560, 300, w)
    sy = 400 + (540 - 400) * 1.08
    S0 = 17 * 1.08
    slip = A.ease_io(A.seg(b, 1.05, 1.85))
    fx = A.lerp(sx, 120, slip)
    fy = A.lerp(sy, FOOT, slip) - 4 * abs(math.sin(math.pi * w * 3)) - 30 * math.sin(math.pi * slip)
    S = A.lerp(S0, 9, slip)
    turn = A.seg(b, 1.85, 2.0)
    # 纸后面的 Clawd（湿壁画）
    if turn > 0:
        wall = place(wall, -90 * A.ease_io(turn), fx, FOOT)
    else:
        wall = flat_fresco_clawd(wall, fx, fy, S, mask=(1 - att))
    wall = wall * (1 - front_a[..., None]) + full * front_a[..., None]
    wall = age(wall)
    # 还贴着的纸（木版画）+ 纸上那部分 Clawd
    Pr = A3.printer_on(paper_img())
    if turn <= 0:
        lift = math.sin(math.pi * A.seg(b, 0.7, 1.3))
        sway = 0
        fxp, fyp = A3.A2.feet_from_center(fx, fy - 6 * S, S, sway)
        A3.dressed_clawd(Pr, fxp, fyp, S, sway, mask=att, mic=b < 0.9)
    paper = Pr.finish()
    img = wall * (1 - att[..., None]) + paper * att[..., None]
    # 翻起的纸背：和纸色，越靠折线越暗
    if fshade is not None:
        back_col = np.array(A3.WASHI)[None, None] * fshade[..., None] * (1 + 0.03 * A3.S3.FIB[..., None])
        img *= (1 - 0.35 * gauss(flap, 6 * K) * (1 - flap))[..., None]
        img = img * (1 - flap[..., None]) + back_col * flap[..., None]
    out = slip >= 1 or (b > 1.2 and float((att * M(g2p([(-1, 0), (17, 0), (17, 12), (-1, 12)], fx, fy, S))).max()) < 0.01)
    return img, out


# ---------- 一帧 ----------
def walker(b):
    """a 段：走过三根柱子，每过一根留下一个身影。返回 (走着的那个的 x, 朝向, 已留下的身影列表, 走路起伏)。"""
    legs = [(2.0, 2.9, 120, 240, -90), (3.1, 4.0, 240, 560, -45), (4.2, 5.1, 560, 880, 0)]
    left = []
    x, az, bob = 120, -90, 0.0
    for (b0, b1, x0, x1, az1) in legs:
        if b < b0:
            break
        u = A.seg(b, b0, b1)
        x = A.lerp(x0, x1, A.ease_io(u))
        bob = 3 * abs(math.sin(math.pi * u * 3))
        prev_az = -90 if az1 == -90 else (-90 if az1 == -45 else -45)
        az = A.lerp(prev_az, az1, A.ease_io(A.seg(b, b1 - 0.3, b1)))
        if b >= b1:
            left.append((x1, az1))
    return x, az, left, bob


def painting(t, b):
    if b < 2.0:
        return e_frame(t, b)
    back, full, front_a = loggia_static()
    if b < 7.0:
        img = back.copy()
        x, az, left, bob = walker(b)
        look = A.ease_io(A.seg(b, 6.0, 6.6))                      # 最后几个身影一起回头
        for (gx, gaz) in left[:-1] if (left and abs(left[-1][0] - x) < 1) else left:
            img = place(img, A.lerp(gaz, 0, look), gx, FOOT)
        img = place(img, A.lerp(az, 0, look) if x >= 879 else az, x, FOOT - bob)
        img = img * (1 - front_a[..., None]) + full * front_a[..., None]
        img = petals(img, b)
        return age(img), True
    # M：身影汇合 → 走进金箔光环，镜头推进光环
    merge = A.ease_io(A.seg(b, 7.0, 7.6))
    img = back.copy()
    if merge < 1:
        for (gx, gaz) in GHOSTS[:2]:
            img = place(img, 0, A.lerp(gx, 560, merge), FOOT, alpha=1.0)
    walk = A.ease_in(A.seg(b, 7.5, 8.6), 1.4)
    cx = A.lerp(A.lerp(880, 560, merge), 560, walk)
    foot = A.lerp(FOOT, 380, walk)
    img = place(img, 0, cx, foot)
    img = img * (1 - front_a[..., None]) + full * front_a[..., None]
    img = petals(img, b)
    img = age(img)
    e = A.ease_in(A.seg(b, 7.9, 8.9), 1.8)
    z = A.lerp(1.0, 3.2, e)
    hx, hy, hr = HALO
    hc = (hx, hy)
    if z > 1.0005:
        img = A3.warp(img, z, hc, hc)
    rr = hr * z
    reveal = rr + A.ease_in(A.seg(b, 8.5, 9.0), 1.5) * 1400
    st = age(gold_clawd(roses(shell_static().copy(), b), 560, 590, 15, b))
    s_st = A.lerp(min(1.0, rr / 300), 1.0, A.ease_io(A.seg(b, 8.5, 9.0)))
    stw = A3.warp(st, s_st, (560, 380), hc if b < 8.5 else A.lerp2(hc, (560, 380), A.ease_io(A.seg(b, 8.5, 9.0))))
    disk = M(circle(hc[0], hc[1], reveal, 240))
    img = img * (1 - disk[..., None]) + stw * disk[..., None]
    return img, True


def b_frame(t, b):
    img = shell_static().copy()
    img = gold_clawd(img, 560, 590, 15, b)
    img = roses(img, b)
    return age(img)


def frame_at(i):
    t = i / FPS
    b = bt(A.twos(i))
    if b >= 9.0:
        p, sw = b_frame(t, b), True
    else:
        p, sw = painting(t, b)
    p = np.clip(downsample(p, K), 0, 1)
    bb = bt(t)
    if bb >= 9.0:
        p = A.zoom_painting(p, 1.10 - 0.10 * A.ease_io(A.seg(bb, 9.0, 12.0)), 560, 400)   # 慢慢拉远
    return A.compose(p, WALL4 if sw else WALL3, LABEL4 if sw else LABEL3)


def warm():
    A3.warm()
    loggia_static()
    shell_static()
    age_maps()
    paper_img()
    for az in range(-90, 1, 5):
        sprite(az)
    for w, l in ((WALL3, LABEL3), (WALL4, LABEL4)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a04_{i:04d}.png")
