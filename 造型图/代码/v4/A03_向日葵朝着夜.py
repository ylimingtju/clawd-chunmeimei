"""03 向日葵朝着夜 · 浮世绘木版 · 70 BPM（0:27.77–0:38.05）。画心动作「一拍二」。拍号 b 从 02 结束算起。
E  b0–2  窗外先印上木版画的夜（从窗框往外漫开）。青蛙往窗外一跳，Clawd 跟着跳出窗户；跳出窗框的部分变成木版画法、
         卸掉造型恢复素身。它一出窗，墙色和展签切换；那扇亮着的窗缩成一点，变成一只萤火虫。
a  b2–7  月下的向日葵田。Clawd 落进花田坐下；向日葵按拍点一丛接一丛转头望向月亮，萤火虫按拍点亮起；
         一棵向日葵弯腰把 Clawd 托到花盘上，茎一拍一拍长高，把它送向月亮。镜头从低处慢慢拉开。
M  b7–9  Clawd 从花盘上跳向月亮，镜头跟着推进月亮；月亮里面就是舞台，进去的部分换上驼色不对称上衣、古铜褶裙、
         紫色长手套；月亮变成圆形追光。
b  b9–12 青海波幕布的舞台，Clawd 站在圆形光斑里，紫手套握着麦克风。慢慢推近。"""
import math

import numpy as np
from scipy.ndimage import map_coordinates

from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, smooth, over, downsample, ishift, outline, mic_held)
from common4 import rect_pts
import anim_core as A
import frame
import S03_浮世绘 as S3
from S03_浮世绘 import Print, petals, leaf, sunflower, WASHI, PRUSSIAN, INDIGO, SUMI, SPECK, GRAIN
import A02_纯妹妹 as A2

FPS = 30
T0 = A2.T0 + 20 * A2.BEAT
BEAT = 60 / 70
WALL2, LABEL2 = A2.WALL2, A2.LABEL2
WALL3, LABEL3 = (0.13, 0.17, 0.30), "向日葵朝着夜"
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 12 * BEAT) * FPS)
MOON = (820, 140, 64)
SPOT = (560, 300, 214)
CLAWD_COL = (0.86, 0.52, 0.37)
_CACHE = {}

# 02 结尾：楼阁的窗（推近 1.25 倍之后在画面上的位置）
Z02, ZC02 = 1.25, (600, 455)
WIN = (ZC02[0] + (400 - ZC02[0]) * Z02, ZC02[1] + (340 - ZC02[1]) * Z02, ZC02[0] + (720 - ZC02[0]) * Z02, ZC02[1] + (528 - ZC02[1]) * Z02)


def bt(t):
    return (t - T0) / BEAT


def z02(p):
    return ZC02[0] + (p[0] - ZC02[0]) * Z02, ZC02[1] + (p[1] - ZC02[1]) * Z02


# ---------- 只在遮罩范围内计算的木版印刷（比整幅计算快很多） ----------
def _bbox(m, pad):
    rows = np.flatnonzero(m.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m[rows[0]:rows[-1] + 1].any(0))
    y0, y1 = max(0, rows[0] - pad), min(m.shape[0], rows[-1] + 1 + pad)
    x0, x1 = max(0, cols[0] - pad), min(m.shape[1], cols[-1] + 1 + pad)
    return slice(y0, y1), slice(x0, x1)


class PrintR(Print):
    def block(self, m, color, reg=(0.6, 0.4), speck=0.06, grain=0.0):
        sl = _bbox(m, int(4 * K))
        if sl is None:
            return
        mm = m[sl]
        self.keys[sl] *= (1 - gauss(mm, 0.4 * K))
        mm = ishift(mm, int(round(reg[1] * K)), int(round(reg[0] * K)))
        col = np.asarray(color, np.float32)
        col = col[sl] if col.ndim == 3 else col[None, None]
        tex = (1 - speck * SPECK[sl] + grain * 0.022 * GRAIN[sl])[..., None]
        self.img[sl] = over(self.img[sl], self.paper[sl] * col * tex * 1.06, gauss(mm, 0.3 * K))

    def key(self, m, w=1.2):
        r = max(1, int(round(w * K)))
        sl = _bbox(m, r + 2)
        if sl is None:
            return
        self.keys[sl] = np.maximum(self.keys[sl], outline(m[sl], r))

    def keyline(self, pts, w, smooth_k=1.0):
        m = SM(pts, w, smooth_k)
        sl = _bbox(m, 1)
        if sl is not None:
            self.keys[sl] = np.maximum(self.keys[sl], m[sl])


def printer_on(img):
    """在一张已经印好的画上继续加印（新的墨线版）。"""
    Pr = PrintR()
    Pr.img = img.copy()
    return Pr


# ---------- 向日葵田 ----------
def field_flowers():
    if "flowers" in _CACHE:
        return _CACHE["flowers"]
    gF = np.random.default_rng(303)
    clumps = [(250, 0.25), (360, 0.75), (470, 0.4), (560, 0.9), (780, 0.55), (870, 0.2), (960, 0.8), (1060, 0.45)]
    fl = []
    while len(fl) < 46:
        cxl, cd = clumps[gF.integers(len(clumps))]
        d = float(np.clip(cd + gF.normal(0, 0.22), 0, 1))
        x = cxl + gF.normal(0, 40 + 30 * d)
        r = (13 + 34 * d) * gF.uniform(0.8, 1.2)
        top = 392 + 150 * d + gF.uniform(-30, 12) * (0.4 + d)
        if not (190 < x < 1110) or abs(x - 660) < r + 14:
            continue
        if any(math.hypot(x - a[0], top - a[1]) < 0.8 * (r + a[2]) for a in fl):
            continue
        fl.append((x, top, r, d))
    out = []
    for k, (x, top, r, d) in enumerate(sorted(fl, key=lambda f: f[3])):
        tilt = gF.uniform(-0.35, 0.35)
        sq_back = -gF.uniform(0.4, 0.75)
        sq_front = 0.35 + 0.5 * gF.uniform(0.6, 1.0)
        band = int(np.clip((x - 190) / 920 * 4, 0, 3))
        out.append(dict(x=x, top=top, r=r, d=d, tilt=tilt, sq0=sq_back, sq1=sq_front, band=band, seed=500 + k))
    _CACHE["flowers"] = out
    return out


def sky_static():
    """天空、月亮、远山、霞带、远处花田（每个进程算一次）。"""
    if "sky" in _CACHE:
        return _CACHE["sky"]
    Pr = PrintR()
    t = np.clip(Y / K / 330, 0, 1)
    bok = (1 - smooth(0.0, 0.62, t))[..., None]
    sky = PRUSSIAN * bok + np.array([0.58, 0.66, 0.74]) * (1 - bok)
    sky = sky * (1 - 0.10 * smooth(0.7, 1.0, t))[..., None]
    Pr.block(M([(-5, -5), (1125, -5), (1125, 340), (-5, 340)]), sky, reg=(0, 0), speck=0.05, grain=1.0)
    mx, my, mr = MOON
    halo = np.exp(-((np.hypot(X / K - mx, Y / K - my) - mr) / 30).clip(0) ** 2) * 0.25
    Pr.img = over(Pr.img, Pr.paper * np.array([0.96, 0.94, 0.86]), halo * smooth(mr, mr + 2, np.hypot(X / K - mx, Y / K - my)))
    Pr.img = over(Pr.img, Pr.paper * np.array([1.0, 0.98, 0.88]), M(circle(mx, my, mr, 120)))
    hills = [(-5, 340), (120, 318), (260, 330), (420, 300), (560, 322), (700, 306), (880, 330), (1010, 312), (1125, 326), (1125, 380), (-5, 380)]
    Pr.block(M(hills, 0.6), INDIGO * np.array([0.9, 0.9, 1.0]))
    for (y0, x0, x1, h) in [(292, -10, 470, 22), (346, 700, 1130, 18)]:
        bm = M([(x0, y0), (x1, y0), (x1 + 16, y0 + h / 2), (x1, y0 + h), (x0, y0 + h)], 0.9)
        fade = smooth(x1 + 10, x1 - 120, X / K) * 0.6 + 0.4
        Pr.block(bm, np.array([0.93, 0.84, 0.80]), reg=(0, 0), speck=0.02)
        Pr.img = over(Pr.img, Pr.paper * np.array([0.96, 0.92, 0.88]), bm * fade * smooth(y0 + h, y0, Y / K))
    fld = M([(-5, 370), (1125, 370), (1125, 640), (-5, 640)])
    Pr.block(fld, np.array([0.26, 0.34, 0.27])[None, None] * (1.05 - 0.45 * smooth(380, 630, Y / K))[..., None] * np.ones((PH, PW, 1)), speck=0.08)
    g = np.random.default_rng(31)
    for i in range(70):
        x, y = g.uniform(0, 1120), g.uniform(378, 420)
        r = 3 + (y - 378) * 0.06
        Pr.block(M(circle(x, y, r, 16, r * 0.6)), (0.80, 0.60, 0.20), reg=(0, 0), speck=0.02)
    _CACHE["sky"] = (Pr.img.copy(), Pr.keys.copy())
    return _CACHE["sky"]


def foreground_static():
    """左侧前景那株高大的向日葵：单独印在一张透明层上（颜色 + 覆盖 + 墨线）。"""
    if "fg" in _CACHE:
        return _CACHE["fg"]
    Pr = PrintR()
    base = Pr.img.copy()
    cover = np.zeros((PH, PW), np.float32)
    g = np.random.default_rng(41)
    fx = 120
    st = catmull([(fx + 30, 640), (fx + 10, 400), (fx - 6, 200), (fx + 20, 60)], 60)
    parts = []
    sm = SM(st, 16, 1.0)
    parts.append((sm, (0.24, 0.38, 0.22)))
    for (ly, side, L) in [(500, 1, 150), (380, -1, 130), (260, 1, 120)]:
        parts.append((M(leaf(fx + 8, ly, L, 46, -20 if side > 0 else 200), 0.5), (0.22, 0.36, 0.21)))
    parts.append((M(petals(fx + 40, 60, 120, 20, 0.5, 0.7), 0.4), (0.90, 0.66, 0.18)))
    parts.append((M(circle(fx + 40, 60, 58, 60, 40, 0.5)), (0.34, 0.20, 0.09)))
    for m, c in parts:
        Pr.block(m, c)
        Pr.key(m, 1.0)
        cover = np.maximum(cover, m)
    for (ly, side, L) in [(500, 1, 150), (380, -1, 130), (260, 1, 120)]:
        Pr.keyline([(fx + 8, ly), (fx + 8 + side * L * 0.8, ly - side * 20)], 0.8, 0.5)
    for i in range(60):
        a, rr = g.uniform(0, 2 * math.pi), g.uniform(0, 50)
        Pr.block(M(circle(fx + 40 + rr * math.cos(a), 60 + rr * 0.69 * math.sin(a), 2.0, 8)), (0.58, 0.40, 0.16), reg=(0, 0), speck=0)
    _CACHE["fg"] = (Pr.img, gauss(cover, 0.3 * K), Pr.keys)
    return _CACHE["fg"]


def turn_of(f, b):
    """一丛一丛按拍点转头：第 band 丛在第 3+band 拍转（0.4 拍转完）。"""
    return A.ease_out(A.seg(b, 3.0 + f["band"], 3.4 + f["band"]), 2.5)


def hero_flower(b):
    """托 Clawd 的那株：先弯腰到 Clawd 身边，托起它，再每拍长高一截。返回 (花盘中心 x, y, 茎的控制点)。"""
    bend = A.ease_io(A.seg(b, 3.6, 4.1))
    lift = A.ease_io(A.seg(b, 4.1, 4.6))
    grow = sum(A.back_out(A.seg(b, k, k + 0.35)) for k in (5.0, 6.0, 7.0))
    hy_low = 590
    hx = A.lerp(700, 640, bend)
    hy = A.lerp(470, hy_low, bend)
    hy = A.lerp(hy, 520, lift)
    hx = A.lerp(hx, 652, lift)
    hy = hy - grow * 74
    hx = hx + grow * 3
    ctrl = [(660, 640), (A.lerp(690, 662, lift) + 6 * (1 - grow / 3), (640 + hy) / 2 + 20 * bend * (1 - lift)), (hx, hy)]
    return hx, hy, ctrl


def clawd_in_field(b):
    """a 段 Clawd 的位置：落进花田坐下 → 被花盘托起。返回 (fx, fy, S)。"""
    hx, hy, _ = hero_flower(b)
    if b < 4.1:
        return 600, 612, 8.0
    lift = A.ease_io(A.seg(b, 4.1, 4.6))
    return A.lerp(600, hx - 4, lift), A.lerp(612, hy + 4, lift), 8.0


def field(b, with_clawd=True, cl_override=None):
    """完整的花田画面（K 倍分辨率，墨线已压印）。"""
    img0, keys0 = sky_static()
    Pr = PrintR()
    Pr.img, Pr.keys = img0.copy(), keys0.copy()
    for f in field_flowers():
        u = turn_of(f, b)
        sq = A.lerp(f["sq0"], f["sq1"], u)
        if abs(sq) < 0.06:
            sq = 0.06 if u > 0.5 else -0.06
        fa = -0.3 + f["tilt"] if sq < 0 else -0.5 + 0.3 * u + f["tilt"]
        sunflower(Pr, f["x"], 640, f["top"], f["r"], fa, sq, 2.5 + 3.5 * f["d"], 1 + int(f["d"] > 0.45) + int(f["d"] > 0.8), f["seed"])
    hx, hy, ctrl = hero_flower(b)
    stem = catmull(ctrl, 40)
    sm = SM(stem, 8, 1.0)
    Pr.block(sm, (0.30, 0.44, 0.27))
    Pr.key(sm, 0.8)
    for k, (sy, side) in enumerate([(0.62, 1), (0.42, -1), (0.25, 1)]):
        p = stem[int(sy * (len(stem) - 1))]
        lm = M(leaf(p[0], p[1], 78 * 1.3, 78 * 0.4, -25 * side + (0 if side > 0 else 180)), 0.5)
        Pr.block(lm, (0.27, 0.42, 0.25))
        Pr.key(lm, 0.8)
    face = -0.1
    pm = M(petals(hx, hy, 78, 16, face, 0.3), 0.4)
    Pr.block(pm, (0.92, 0.70, 0.20))
    Pr.key(pm, 0.8)
    cm_ = M(circle(hx, hy, 78 * 0.48, 40, 78 * 0.48 * 0.3, face), 0)
    Pr.block(cm_, (0.38, 0.22, 0.10))
    Pr.key(cm_, 0.8)
    if with_clawd:
        fx, fy, S = cl_override or clawd_in_field(b)
        arms = [rigid_arm("L", 0), rigid_arm("R", 30)] if b >= 4.6 else [rigid_arm("L", 20), rigid_arm("R", 20)]
        check_proportions(arms=arms)
        cm, em = clawd(fx, fy, S, arms)
        Pr.block(cm, CLAWD_COL)
        Pr.key(cm, 1.0)
        Pr.block(em, SUMI, reg=(0, 0), speck=0)
        if b >= 4.1:
            front = M(petals(hx, hy + 6, 78, 16, face, 0.3), 0.4) * smooth(hy + 2, hy + 8, Y / K)
            Pr.block(front, (0.92, 0.70, 0.20))
            Pr.key(front * smooth(hy + 4, hy + 10, Y / K), 0.8)
    # 萤火虫：按拍点一只只亮起来
    g = np.random.default_rng(8)
    for i in range(22):
        x, y = g.uniform(80, 1100), g.uniform(240, 600)
        on = A.ease_out(A.seg(b, 2.0 + (i % 5) + g.uniform(0, 0.2), 2.3 + (i % 5)))
        if on <= 0:
            continue
        Pr.block(M(circle(x, y, 6 * on, 20)), (0.88, 0.92, 0.62), reg=(0, 0), speck=0.02)
        Pr.block(M(circle(x, y, 2.2 * on, 12)), (1.0, 1.0, 0.80), reg=(0, 0), speck=0)
    fg_img, fg_cov, fg_keys = foreground_static()
    Pr.keys = Pr.keys * (1 - fg_cov) + fg_keys * fg_cov
    Pr.img = Pr.img * (1 - fg_cov[..., None]) + fg_img * fg_cov[..., None]
    return Pr.finish()


# ---------- 舞台 ----------
def stage_static():
    if "stage" in _CACHE:
        return _CACHE["stage"]
    Pr = PrintR()
    Pr.block(M([(-5, -5), (1125, -5), (1125, 520), (-5, 520)]), PRUSSIAN * 1.05, reg=(0, 0), speck=0.05, grain=0.6)
    R_ = 46
    xx, yy = X / K, Y / K
    pat = np.full((PH, PW), -1, np.int32)
    for row in range(26):
        cy_ = -20 + row * R_ * 0.5 + R_
        off = (row % 2) * R_
        dx = ((xx - off + R_) % (2 * R_)) - R_
        d = np.hypot(dx, yy - cy_) / R_
        inside = (d < 1.0) & (yy < cy_ + 0.5)
        ring = np.clip(((1.0 - d) / 0.22).astype(np.int32), 0, 3)
        pat = np.where(inside, ring, pat)
    light = (pat % 2 == 0) & (pat >= 0)
    Pr.block(gauss(light.astype(np.float32), 0.5 * K) * (yy < 520), np.array([0.80, 0.84, 0.86]), reg=(0, 0), speck=0.04)
    Pr.block(M(circle(560, 300, 230, 160)), np.array([0.96, 0.90, 0.68]), reg=(0, 0), speck=0.03)
    Pr.block(M(circle(560, 300, 214, 160)), np.array([1.0, 0.97, 0.88]), reg=(0, 0), speck=0.02)
    fl = M([(-5, 520), (1125, 520), (1125, 640), (-5, 640)])
    Pr.block(fl, np.array([0.74, 0.55, 0.32]), speck=0.06, grain=1.0)
    Pr.key(fl, 0.9)
    for i in range(1, 6):
        Pr.keyline([(-5, 520 + i * 22), (1125, 520 + i * 22)], 0.6, 0)
    Pr.block(M(circle(560, 552, 260, 120, 26)), np.array([0.92, 0.80, 0.56]), reg=(0, 0), speck=0.04)
    for side in (-1, 1):
        x0 = 0 if side < 0 else 1120
        cur = [(x0, -5), (x0 + side * -150, -5), (x0 + side * -120, 200), (x0 + side * -170, 400), (x0 + side * -110, 640), (x0, 640)]
        cm = M(cur, 0.6)
        Pr.block(cm, INDIGO * 0.9, speck=0.06)
        Pr.key(cm, 1.0)
        for k in range(4):
            xk = x0 + side * -(28 + k * 28)
            Pr.keyline([(xk, 0), (xk + side * 10, 300), (xk - side * 8, 630)], 0.7, 1.0)
    _CACHE["stage"] = Pr.finish()
    return _CACHE["stage"]


def dressed_clawd(Pr, FX, FY, S, ang=0.0, mask=None, mic=True):
    mm = 1.0 if mask is None else mask
    arms = [rigid_arm("L", 0), rigid_arm("R", 30)]
    check_proportions(arms=arms)
    cm, em = clawd(FX, FY, S, arms, ang)
    Pr.block(cm * mm, CLAWD_COL)
    Pr.key(cm * mm, 1.0)
    top = garment([(-0.3, 5.4), (16.3, 5.4), (16.3, 7.8), (-0.3, 8.6)], FX, FY, S, ang, arms=arms, name="top")
    strap = garment([(13.0, 5.4), (16.3, 1.8), (16.3, 3.6), (14.8, 5.4)], FX, FY, S, ang, name="strap")
    ts = np.clip(top + strap, 0, 1) * mm
    Pr.block(ts, (0.74, 0.58, 0.40))
    Pr.key(ts, 0.9)
    skirt = garment([(-0.3, 8.6), (16.3, 7.8), (17.4, 11.9), (-1.4, 11.9)], FX, FY, S, ang, name="skirt") * mm
    Pr.block(skirt, (0.58, 0.36, 0.18))
    Pr.key(skirt, 0.9)
    for k in range(9):
        gx = 0.4 + k * 1.9
        Pr.keys = np.maximum(Pr.keys, SM(g2p([(gx, 8.6 - 0.05 * gx), (gx + (gx - 8) * 0.13, 11.8)], FX, FY, S, ang), 0.7, 0) * skirt)
    gl = np.clip(sum(M(g2p(a, FX, FY, S, ang)) for a in arms), 0, 1) * mm
    Pr.block(gl, (0.48, 0.30, 0.56))
    Pr.key(gl, 0.9)
    Pr.block(em * mm, SUMI, reg=(0, 0), speck=0)
    if mic:
        mc = mic_held(arms[1], FX, FY, S, lean=12, ang=ang)
        for part in ("handle", "head", "shine", "band", "grille"):
            mc[part] = mc[part] * mm
        Pr.block(mc["handle"], (0.15, 0.14, 0.14), reg=(0, 0))
        Pr.key(mc["handle"], 0.7)
        Pr.block(mc["head"], (0.66, 0.66, 0.64), reg=(0, 0))
        Pr.key(mc["head"], 0.8)
        Pr.keys = np.maximum(Pr.keys, mc["grille"] * 0.55)
        Pr.block(mc["shine"], (0.86, 0.86, 0.82), reg=(0, 0), speck=0)
        Pr.block(mc["band"], (0.40, 0.40, 0.40), reg=(0, 0))
        Pr.key(mc["band"], 0.6)
        gR = M(g2p(arms[1], FX, FY, S, ang)) * mm
        Pr.block(gR, (0.48, 0.30, 0.56), reg=(0, 0))
        Pr.key(gR, 0.9)


def plain_clawd(Pr, FX, FY, S, ang=0.0, mask=None, arms=None):
    mm = 1.0 if mask is None else mask
    arms = arms or [rigid_arm("L", 20), rigid_arm("R", 20)]
    cm, em = clawd(FX, FY, S, arms, ang)
    Pr.block(cm * mm, CLAWD_COL)
    Pr.key(cm * mm, 1.0)
    Pr.block(em * mm, SUMI, reg=(0, 0), speck=0)


def warp(src, z, c_src, c_dst):
    """K 倍分辨率的图：把 src 中以 c_src 为中心的内容放大 z 倍放到 c_dst。"""
    qx = c_src[0] + (X / K - c_dst[0]) / z
    qy = c_src[1] + (Y / K - c_dst[1]) / z
    return np.stack([map_coordinates(src[..., c], [qy * K, qx * K], order=1, mode="nearest") for c in range(3)], -1)


# ---------- E：楼阁的窗 ----------
def room_layer(t, b, frog_pos, clawd_pose):
    """02 结尾的窗里（油画），推近 1.25 倍后的样子。frog_pos/clawd_pose 为 None 表示已经离开。"""
    lz, lat_a, wm = A2.liaozhai_static()
    img = A2.candle(lz.copy(), t)
    if frog_pos is not None:
        img = A2.frog(img, frog_pos[0], frog_pos[1], 0.55, puff=max(0.0, 1 - A.seg(b, 0.0, 0.15)))
    if clawd_pose is not None:
        img = A2.robe_clawd(img, *clawd_pose)
    img = over(img, (0.11, 0.06, 0.04), lat_a)
    img = A2.oil(img, 1)
    return warp(img, Z02, ZC02, ZC02)


def win_mask(scale=1.0, target=(480, 260)):
    x0, y0, x1, y1 = WIN
    cx, cy = A.lerp(target[0], (x0 + x1) / 2, scale), A.lerp(target[1], (y0 + y1) / 2, scale)
    hw, hh = (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
    return M([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]), (cx, cy, hw, hh)


def e_frame(t, b):
    # 青蛙：02 坐标 (650, 468) → 往右下跳出窗，落进花田再跳出画面
    fu = A.seg(b, 0.15, 0.95)
    f0 = z02((650, 468))
    frog_scr = A.hop(f0, (900, 600), fu, 120) if fu < 1 else A.hop((900, 600), (1180, 560), A.seg(b, 0.95, 1.5), 60)
    # Clawd：02 坐标 (556, 528) S9.5 → 落进花田 (600, 612) S8
    cu = A.seg(b, 0.75, 1.7)
    c0 = z02((556, 528))
    cs = A.hop(c0, (600, 612), A.ease_io(cu) * 0.5 + cu * 0.5, 130)
    S = A.lerp(9.5 * Z02, 8.0, A.ease_io(cu))
    # 木版画的夜从窗框往外漫开
    x0, y0, x1, y1 = WIN
    dx = np.maximum(np.maximum(x0 - X / K, X / K - x1), 0)
    dy = np.maximum(np.maximum(y0 - Y / K, Y / K - y1), 0)
    dist = np.hypot(dx, dy)
    spread = A.ease_out(A.seg(b, 0.0, 0.45)) * 900
    wood_m = smooth(spread, spread - 30, dist) * (dist > 0)
    shrink = A.ease_in(A.seg(b, 1.65, 2.0), 1.6)
    wmask, (wcx, wcy, hw, hh) = win_mask(1 - shrink)
    wmask = wmask * (1 if shrink < 1 else 0)
    # 窗里（油画）
    # 青蛙在窗里的那部分仍是油画（窗外的部分由下面的木版青蛙接上）
    fr_room = ((frog_scr[0] - ZC02[0]) / Z02 + ZC02[0], (frog_scr[1] - ZC02[1]) / Z02 + ZC02[1]) if fu < 0.6 else None
    lz_room = room_layer(t, b, fr_room, None)
    # 窗外：木版花田（全背着月亮），以及 02 的外景（还没被盖住的部分）
    fieldimg = field(2.0, with_clawd=False)
    ext02 = None
    if spread < 900:
        lz, lat_a, wm = A2.liaozhai_static()
        ext02 = warp(A2.oil(A2.candle(lz.copy(), t), 1), Z02, ZC02, ZC02)
        outside = ext02 * (1 - wood_m[..., None]) + fieldimg * wood_m[..., None]
    else:
        outside = fieldimg
    # 窗里的内容随窗一起缩小
    if shrink > 0:
        room_s = warp(lz_room, 1 - shrink + 1e-3, ((x0 + x1) / 2, (y0 + y1) / 2), (wcx, wcy))
    else:
        room_s = lz_room
    img = outside * (1 - wmask[..., None]) + room_s * wmask[..., None]
    # 窗里那部分的 Clawd（油画、素白长衣）
    if cu < 1 and shrink < 0.05:
        cx_, cy_ = cs
        rp = ((cx_ - ZC02[0]) / Z02 + ZC02[0], (cy_ - ZC02[1]) / Z02 + ZC02[1], S / Z02, 0.0, None, None)
        inner = room_layer(t, b, fr_room, rp)
        img = img * (1 - wmask[..., None]) + inner * wmask[..., None]
    # 窗外的部分：木版画法
    Pr = printer_on(img)
    if fu > 0:
        s_ = 0.55 * Z02                                  # 和窗里的青蛙同样大小
        fx_, fy_ = frog_scr
        out_m = (1 - wmask)
        body = M([(fx_ - 40 * s_, fy_), (fx_ - 46 * s_, fy_ - 22 * s_), (fx_ - 30 * s_, fy_ - 44 * s_), (fx_, fy_ - 50 * s_), (fx_ + 30 * s_, fy_ - 44 * s_),
                  (fx_ + 46 * s_, fy_ - 22 * s_), (fx_ + 40 * s_, fy_)], 0.8) * out_m
        Pr.block(body, (0.40, 0.62, 0.30))
        Pr.key(body, 0.8)
        belly = M(circle(fx_, fy_ - 12 * s_, 26 * s_, 40, 13 * s_)) * out_m
        Pr.block(belly, (0.86, 0.84, 0.56), reg=(0, 0))
        for sx in (-1, 1):
            em_ = M(circle(fx_ + sx * 20 * s_, fy_ - 51 * s_, 8 * s_, 24)) * out_m
            Pr.block(em_, (0.95, 0.82, 0.32), reg=(0, 0))
            Pr.key(em_, 0.6)
            Pr.block(M(circle(fx_ + sx * 20 * s_, fy_ - 51 * s_, 2.6 * s_, 12, 5.5 * s_)) * out_m, SUMI, reg=(0, 0), speck=0)
        hat = M([(fx_ - 22 * s_, fy_ - 58 * s_), (fx_ - 20 * s_, fy_ - 78 * s_), (fx_, fy_ - 86 * s_), (fx_ + 20 * s_, fy_ - 78 * s_), (fx_ + 22 * s_, fy_ - 58 * s_)], 0.6) * out_m
        Pr.block(hat, SUMI, reg=(0, 0))
        for sx in (-1, 1):
            Pr.block(M([(fx_ + sx * 18 * s_, fy_ - 70 * s_), (fx_ + sx * 64 * s_, fy_ - 76 * s_), (fx_ + sx * 66 * s_, fy_ - 70 * s_), (fx_ + sx * 18 * s_, fy_ - 64 * s_)], 0.3) * out_m, SUMI, reg=(0, 0))
    if cu > 0:
        plain_clawd(Pr, cs[0], cs[1], S, 0, mask=(1 - wmask))
    if 0 < shrink and shrink >= 1:
        pass
    img = Pr.finish()
    if shrink >= 0.98 or (0.0 < shrink):                 # 缩成一点之后亮成一只萤火虫
        glow = A.ease_out(A.seg(b, 1.85, 2.0))
        if glow > 0:
            img = over(img, (0.88, 0.92, 0.62), M(circle(480, 260, 6 * glow, 20)))
            img = over(img, (1.0, 1.0, 0.80), M(circle(480, 260, 2.2 * glow, 12)))
    out = cu >= 1 or all(not (x0 <= x <= x1 and y0 <= y <= y1) for x, y in g2p([(-1, 0), (17, 0), (17, 12), (-1, 12)], cs[0], cs[1], S))
    return img, out and b > 0.9


# ---------- 一帧 ----------
def painting(t, b):
    if b < 2.0:
        return e_frame(t, b)
    if b < 7.0:
        img = field(b)
        z = A.lerp(1.18, 1.0, A.ease_io(A.seg(b, 2.0, 6.6)))       # 从低处慢慢拉开
        if z > 1.0005:
            img = warp(img, z, (600, 560), (600, 560))
        return img, True
    # M：跳向月亮，镜头推进月亮
    u = A.seg(b, 7.0, 9.0)
    e = A.ease_in(A.seg(b, 7.2, 8.7), 1.8)
    z = A.lerp(1.0, 2.6, e)
    mx, my, mr = MOON
    mc = A.lerp2((mx, my), (SPOT[0], SPOT[1]), A.ease_io(A.seg(b, 7.2, 8.9)))
    base = field(7.0, with_clawd=False)
    img = warp(base, z, (mx, my), mc) if z > 1.0005 or mc != (mx, my) else base
    # 月亮里的舞台
    rr = mr * z
    reveal_r = rr + A.ease_in(A.seg(b, 8.55, 9.0), 1.5) * 1400
    s_st = min(1.0, rr / SPOT[2]) if b < 8.55 else A.lerp(min(1.0, rr / SPOT[2]), 1.0, A.ease_io(A.seg(b, 8.55, 9.0)))
    stg = warp(stage_static(), s_st, (SPOT[0], SPOT[1]), mc)
    disk = M(circle(mc[0], mc[1], reveal_r, 240))
    img = img * (1 - disk[..., None]) + stg * disk[..., None]
    # Clawd：花盘 → 月亮（场景坐标里飞，按镜头变换）
    hx, hy, _ = hero_flower(7.0)
    ju = A.seg(b, 7.3, 8.8)
    sx, sy = A.hop((hx - 4, hy + 4), (mx, my + 30), ju, 90)
    S_scene = A.lerp(8.0, 17 / 2.6, A.ease_io(ju))
    px, py = mc[0] + (sx - mx) * z, mc[1] + (sy - my) * z
    S = S_scene * z
    fin = A.ease_io(A.seg(b, 8.6, 9.0))
    px, py, S = A.lerp(px, 560, fin), A.lerp(py, 540, fin), A.lerp(S, 17, fin)
    ang = 360 * A.ease_io(A.seg(b, 7.6, 8.6))
    cx_, cy_ = px, py - 6 * S
    fx, fy = A2.feet_from_center(cx_, cy_, S, ang)
    Pr = printer_on(img)
    plain_clawd(Pr, fx, fy, S, ang, mask=(1 - disk))
    dressed_clawd(Pr, fx, fy, S, ang, mask=disk, mic=b > 8.7)
    return Pr.finish(), True


def b_frame(t, b):
    Pr = printer_on(stage_static())
    sway = 2.5 * math.sin(math.pi * (b - 9.0))
    fx, fy = A2.feet_from_center(560, 540 - 6 * 17, 17, sway)
    dressed_clawd(Pr, fx, fy, 17, sway)
    return Pr.finish()


def frame_at(i):
    t = i / FPS
    b = bt(A.twos(i))
    if b >= 9.0:
        p = b_frame(t, b)
        sw = True
    else:
        p, sw = painting(t, b)
    p = np.clip(downsample(p, K), 0, 1)
    bb = bt(t)
    if bb >= 9.0:
        p = A.zoom_painting(p, 1 + 0.08 * A.ease_io(A.seg(bb, 9.0, 12.0)), 560, 400)
    return A.compose(p, WALL3 if sw else WALL2, LABEL3 if sw else LABEL2)


def warm():
    A2.warm()
    sky_static()
    foreground_static()
    stage_static()
    field_flowers()
    for w, l in ((WALL2, LABEL2), (WALL3, LABEL3)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a03_{i:04d}.png")
