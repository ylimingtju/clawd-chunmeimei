"""08 还你茉莉 · 敦煌壁画加摇滚 · 样张（08-a 反弹琵琶）。
画法：土黄灰泥底（水渍、细裂纹、颗粒）→ 矿物颜料平涂并做唐代「凹凸晕染」→ 铁线描勾边 →
颜料磨损、局部剥落露出底层灰泥。飞天是小 Clawd，身后是火焰纹背光（也是摇滚舞台光）。不用任何印章。
用法：python S08_敦煌.py 输出目录"""
import math
import os
import sys

import numpy as np
from scipy import ndimage

from common4 import (ishift, gauss, fbm, value_noise, smooth, over, poly_mask, stroke_mask, downsample, taper_stroke,
                     clawd_union_mask, to_px, rect_pts, rigid_arm, check_proportions, LEGS, mask, path_from)
import frame

K = 2
PW, PH = frame.PW * K, frame.PH * K
G = np.random.default_rng(8)

GROUND = np.array([0.80, 0.66, 0.47])
PLASTER = np.array([0.87, 0.80, 0.67])
LAPIS = (0.20, 0.31, 0.56)
LAPIS_D = (0.13, 0.20, 0.40)
MALA = (0.29, 0.56, 0.47)
MALA_D = (0.17, 0.40, 0.34)
CINNA = (0.74, 0.29, 0.19)
EARTH = (0.58, 0.26, 0.16)
OCHRE = (0.86, 0.64, 0.30)
LEAD = (0.91, 0.87, 0.77)
INK = (0.17, 0.10, 0.07)
CLAWD = (0.78, 0.47, 0.34)

Y, X = np.mgrid[0:PH, 0:PW].astype(np.float32)
FINE = fbm(PH, PW, 3, 1, octaves=2) - 0.5
GRAIN = value_noise(PH, PW, 1.6, 2) - 0.5
WEAR = fbm(PH, PW, 70, 3, octaves=4)
FLAKE_N = fbm(PH, PW, 55, 4, octaves=5)
BRKS = [smooth(0.12, 0.35, value_noise(PH, PW, 9, 50 + i)) for i in range(4)]


def P(pts):
    return [(x * K, y * K) for x, y in pts]


def M(pts, smooth_k=0.0):
    return poly_mask(PW, PH, P(pts), smooth_k)


def circle(cx, cy, r, n=64, ry=None, a0=0.0):
    ry = r if ry is None else ry
    return [(cx + r * math.cos(a0 + 2 * math.pi * i / n), cy + ry * math.sin(a0 + 2 * math.pi * i / n)) for i in range(n)]


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def outline(m, r=2):
    er = ndimage.grey_erosion(m, footprint=disk(r))
    return np.clip(m - er, 0, 1)


class Mural:
    def __init__(self):
        g = GROUND * (1 + 0.10 * (fbm(PH, PW, 400, 10, 3) - 0.5))[..., None]
        stain = smooth(0.55, 0.85, fbm(PH, PW, 260, 12, 4))
        g = g * (1 - 0.08 * stain[..., None]) + np.array([0.45, 0.30, 0.18]) * 0.05 * stain[..., None]
        g *= (1 + 0.05 * FINE + 0.05 * GRAIN)[..., None]
        self.img = g.astype(np.float32)

    def paint(self, m, color, opacity=1.0, shade=0.22, wear=1.0, line=None, line_w=2, grain=1.0):
        if m.max() <= 0:
            return
        col = np.array(color, np.float32)[None, None] * (1 + grain * (0.07 * GRAIN + 0.05 * FINE))[..., None]
        if shade:
            # 凹凸晕染：靠近边缘颜色加深，中间留亮
            inner = gauss(m, 9 * K)
            col = col * (1 - shade * (1 - smooth(0.25, 0.85, inner)))[..., None]
        cov = m * opacity * (1 - wear * 0.35 * smooth(0.72, 0.95, WEAR))
        cov *= (1 - 0.18 * wear * smooth(0.2, 0.0, GRAIN + 0.3))
        self.img = over(self.img, col, np.clip(cov, 0, 1))
        if line is not None:
            self.line(outline(m, line_w), line)

    def line(self, lm, color=INK, opacity=0.9):
        brk = BRKS[int(G.integers(4))]
        self.img = over(self.img, color, np.clip(lm * opacity * (0.55 + 0.45 * brk) * (1 - 0.5 * smooth(0.7, 0.92, WEAR)), 0, 1))

    def stroke(self, pts, w0, w1, color=INK, opacity=0.9, wmid=None):
        self.line(M(taper_stroke(pts, w0, w1, wmid), 0.0), color, opacity)

    def age(self):
        # 裂纹：脊状噪声的细线
        n = fbm(PH, PW, 120, 30, 3)
        ridge = 1 - np.abs(2 * n - 1)
        crack = smooth(0.985, 0.996, ridge) * smooth(0.45, 0.7, fbm(PH, PW, 300, 31, 2))
        self.img *= (1 - 0.22 * crack)[..., None]
        # 剥落：颜料层掉了，露出更浅更粗的灰泥，边缘右下有一道细影
        edge_w = smooth(150 * K, 20 * K, np.minimum.reduce([X, PW - X, Y, PH - Y]))
        flake = smooth(0.70, 0.73, FLAKE_N) * smooth(0.5, 0.7, fbm(PH, PW, 380, 33, 2)) * edge_w
        rough = PLASTER * (1 + 0.10 * GRAIN + 0.08 * FINE)[..., None]
        edge = np.clip(ishift(flake, 2 * K, 2 * K) - flake, 0, 1)
        self.img = self.img * (1 - flake[..., None]) + rough * flake[..., None]
        self.img *= (1 - 0.35 * gauss(edge, 0.8 * K))[..., None]
        # 盐霜与灰尘
        salt = smooth(0.85, 0.97, fbm(PH, PW, 30, 34, 3)) * 0.12
        self.img = over(self.img, (0.93, 0.90, 0.84), salt)
        vign = np.clip(((X / PW - 0.5) ** 2 + (Y / PH - 0.5) ** 2) * 1.4, 0, 1)
        self.img *= (1 - 0.22 * vign)[..., None]


def catmull(ctrl, n=200):
    """Catmull-Rom 曲线采样。"""
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


def ribbon(Mu, ctrl, w0, front, back, twists=3.0, taper=0.25, phase=0.0, line_w=1):
    """会翻转的双色飘带（披帛）：宽度随翻转变化，正反两面两种颜色。"""
    pts = catmull(ctrl, 260)
    n = len(pts)
    t = np.linspace(0, 1, n)
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nr = np.stack([-d[:, 1], d[:, 0]], 1)
    tw = np.cos(np.pi * twists * t + phase)
    wid = w0 * (taper + (1 - taper) * (1 - t) ** 0.6) * (0.18 + 0.82 * np.abs(tw))
    L = pts + nr * wid[:, None] / 2
    R = pts - nr * wid[:, None] / 2
    import skia
    pf, pb = skia.Path(), skia.Path()
    for i in range(n - 1):
        q = [L[i], L[i + 1], R[i + 1], R[i]]
        pa = pf if tw[i] >= 0 else pb
        pa.moveTo(q[0][0] * K, q[0][1] * K)
        for qq in q[1:]:
            pa.lineTo(qq[0] * K, qq[1] * K)
        pa.close()
    mf = mask(PW, PH, lambda c, p: c.drawPath(pf, p))
    mb = mask(PW, PH, lambda c, p: c.drawPath(pb, p))
    Mu.paint(mb, back, shade=0.12, line=None)
    Mu.paint(mf, front, shade=0.12, line=None)
    Mu.line(outline(np.clip(mf + mb, 0, 1), line_w), INK, 0.85)


def flame(cx, cy, a, r0, r1, w, lean=0.32):
    """火焰舌：从内圈长出、朝顺时针方向卷起的尖。"""
    left, right = [], []
    for i in range(31):
        t = i / 30
        r = r0 + (r1 - r0) * t
        c = a + lean * t ** 1.6
        hw = (w / r0) * (1 - t) ** 0.85 * (0.75 + 0.25 * math.sin(math.pi * t))
        left.append((cx + r * math.cos(c - hw), cy + r * math.sin(c - hw)))
        right.append((cx + r * math.cos(c + hw * 0.8), cy + r * math.sin(c + hw * 0.8)))
    return left + right[::-1]


def cloud(cx, cy, s, flip=1):
    """卷云：几个圆团加一条拖尾。"""
    parts = [circle(cx, cy, 20 * s, 40), circle(cx + flip * 24 * s, cy + 5 * s, 15 * s, 40), circle(cx - flip * 20 * s, cy + 7 * s, 13 * s, 40)]
    tail = [(cx - flip * 26 * s, cy + 12 * s), (cx - flip * 72 * s, cy + 22 * s), (cx - flip * 104 * s, cy + 13 * s),
            (cx - flip * 66 * s, cy + 30 * s), (cx - flip * 18 * s, cy + 22 * s)]
    return parts, tail


def spiral(cx, cy, r, turns=1.6, flip=1):
    return [(cx + flip * r * (1 - t / turns * 0.85) * math.cos(t * 2 * math.pi), cy + r * (1 - t / turns * 0.85) * math.sin(t * 2 * math.pi))
            for t in np.linspace(0, turns, 50)]


def jasmine(cx, cy, r, rot):
    pts = []
    for i in range(60):
        a = rot + 2 * math.pi * i / 60
        rr = r * (0.5 + 0.5 * abs(math.cos(2.5 * (a - rot))) ** 0.6)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def pipa(base, tip, width):
    """曲项琵琶：梨形音箱 + 细颈 + 向后折的琴头。"""
    bx, by = base
    tx, ty = tip
    L = math.hypot(tx - bx, ty - by)
    ux, uy = (tx - bx) / L, (ty - by) / L
    vx, vy = -uy, ux
    prof = []
    for i in range(61):
        s_ = i / 60 * 0.86
        if s_ < 0.6:
            q = s_ / 0.6                                     # 梨形：底端圆、最宽处靠下，向上收成颈
            if q < 0.36:
                w = width * 0.5 * math.sqrt(max(0.0, 1 - ((0.36 - q) / 0.36) ** 2))
            else:
                w = width * 0.5 * (1 - (1 - 0.18) * ((q - 0.36) / 0.64) ** 1.4)
        else:
            w = width * (0.090 - 0.025 * (s_ - 0.6))
        prof.append((s_, max(w, width * 0.05)))
    pts = [(bx + ux * L * s_ + vx * w, by + uy * L * s_ + vy * w) for s_, w in prof] + \
          [(bx + ux * L * s_ - vx * w, by + uy * L * s_ - vy * w) for s_, w in prof[::-1]]
    a = math.atan2(uy, ux) + math.radians(70)
    hx, hy = bx + ux * L * 0.86, by + uy * L * 0.86
    hl = L * 0.15
    wn = width * 0.06
    head = [(hx + vx * wn, hy + vy * wn), (hx + math.cos(a) * hl + vx * wn * 0.7, hy + math.sin(a) * hl + vy * wn * 0.7),
            (hx + math.cos(a) * hl - vx * wn * 0.7, hy + math.sin(a) * hl - vy * wn * 0.7), (hx - vx * wn, hy - vy * wn)]
    return pts, head, (ux, uy, vx, vy, L, a, hx, hy, hl)


def eyes(Mu, fx, fy, S, ang):
    for ex in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        Mu.paint(M(to_px(rect_pts(ex), fx, fy, S, ang)), INK, shade=0, wear=0.3, grain=0.3)


def apsara(Mu, fx, fy, S, ang, arms, ribs, carry=None):
    for ctrl, w0, f, b, tw, ph in ribs:
        ribbon(Mu, ctrl, w0, f, b, tw, 0.2, ph)
    check_proportions(arms=arms)
    cm = clawd_union_mask(PW, PH, fx * K, fy * K, S * K, arms=arms, ang=ang)
    Mu.paint(cm, CLAWD, shade=0.32, line=INK, line_w=2)
    band = [(-0.2, 6.8), (16.2, 6.8), (16.2, 8.2), (-0.2, 8.2)]
    Mu.paint(M(to_px(band, fx, fy, S, ang)), MALA, shade=0.1, line=INK, line_w=1)
    eyes(Mu, fx, fy, S, ang)
    if carry:
        carry()


def render():
    Mu = Mural()
    # ---------- 顶部边饰：联珠纹 + 三角垂帐 ----------
    Mu.paint(M([(0, 0), (1120, 0), (1120, 38), (0, 38)]), EARTH, shade=0)
    for i in range(48):
        x = 12 + i * 23.5
        Mu.paint(M(circle(x, 19, 8, 28)), LEAD, shade=0.12, grain=0.6)
        Mu.paint(M(circle(x, 19, 3.2, 18)), LAPIS, shade=0)
    Mu.line(outline(M([(0, 0), (1120, 0), (1120, 38), (0, 38)]), 1), INK, 0.7)
    for i in range(29):
        x = i * 40
        Mu.paint(M([(x, 38), (x + 40, 38), (x + 20, 60)]), MALA if i % 2 else CINNA, shade=0.12, line=INK, line_w=1)
    # 底部卷草边
    Mu.paint(M([(0, 600), (1120, 600), (1120, 630), (0, 630)]), LAPIS_D, shade=0)
    for i in range(15):
        x = i * 80
        vine = [(x + t * 80, 615 + 8 * math.sin(t * 2 * math.pi)) for t in np.linspace(0, 1, 30)]
        Mu.stroke(vine, 2.6, 2.6, OCHRE, 0.85)
        Mu.paint(M(circle(x + 20, 607, 6, 20, 3.5, 0.5)), MALA, shade=0)
        Mu.paint(M(circle(x + 60, 623, 6, 20, 3.5, -0.5)), MALA, shade=0)

    # ---------- 卷云（远） ----------
    for (cx, cy, s, fl, col) in [(140, 330, 1.0, 1, MALA), (990, 360, 0.9, -1, LEAD), (110, 545, 0.8, 1, LEAD), (1010, 548, 1.0, -1, MALA)]:
        parts, tail = cloud(cx, cy, s, fl)
        mm = np.clip(sum(M(p) for p in parts) + M(tail, 0.8), 0, 1)
        Mu.paint(mm, col, shade=0.22, line=INK, line_w=1)
        for p in [(cx, cy, 12 * s), (cx + fl * 24 * s, cy + 5 * s, 8 * s)]:
            Mu.stroke(spiral(p[0], p[1], p[2], 1.4, fl), 2.0, 0.8, INK, 0.7)

    # ---------- 背光：同心纹带 + 旋转火焰 ----------
    CX, CY = 560, 392
    Mu.paint(M(circle(CX, CY, 204, 160)), OCHRE, shade=0.10, line=INK, line_w=2)
    Mu.paint(M(circle(CX, CY, 190, 160)), EARTH, shade=0.22, line=INK, line_w=1)
    for i in range(26):
        a = 2 * math.pi * i / 26
        Mu.paint(M(flame(CX, CY, a, 132, 190, 15), 0.5), CINNA if i % 3 else MALA, shade=0.30, line=INK, line_w=1)
        Mu.paint(M(flame(CX, CY, a + 0.02, 138, 172, 6.5), 0.5), OCHRE if i % 3 else LEAD, shade=0.1)
    Mu.paint(M(circle(CX, CY, 132, 140)), MALA_D, shade=0.15, line=INK, line_w=1)
    for i in range(40):
        a = 2 * math.pi * i / 40
        Mu.paint(M(circle(CX + 126 * math.cos(a), CY + 126 * math.sin(a), 3.2, 14)), LEAD, shade=0, grain=0.4)
    Mu.paint(M(circle(CX, CY, 119, 140)), LAPIS, shade=0.30, line=INK, line_w=1)
    for r in (104, 88):
        Mu.line(outline(M(circle(CX, CY, r, 140)), 1), LEAD, 0.55)
    for i in range(12):                     # 内圈莲瓣放射
        a = 2 * math.pi * i / 12
        pet = [(CX + 30 * math.cos(a - 0.22), CY + 30 * math.sin(a - 0.22)), (CX + 84 * math.cos(a), CY + 84 * math.sin(a)),
               (CX + 30 * math.cos(a + 0.22), CY + 30 * math.sin(a + 0.22))]
        Mu.paint(M(pet, 0.8), LAPIS_D, opacity=0.6, shade=0)

    # ---------- 不鼓自鸣：系着飘带自己飘在空中的乐器 ----------
    # 腰鼓（左）
    drum = [(196, 236), (284, 214), (292, 246), (204, 268)]
    ribbon(Mu, [(200, 252), (150, 238), (110, 262), (70, 240), (36, 252)], 11, MALA, LAPIS, 2.5, 0.3, 0.4)
    Mu.paint(M(drum, 0.35), CINNA, shade=0.32, line=INK, line_w=1)
    for tt in (0.22, 0.5, 0.78):
        a_ = (196 + 88 * tt, 236 - 22 * tt)
        b_ = (204 + 88 * tt, 268 - 22 * tt)
        Mu.line(stroke_mask(PW, PH, [(a_[0] * K, a_[1] * K), (b_[0] * K, b_[1] * K)], 1.3 * K), OCHRE, 0.95)
    Mu.paint(M(circle(196, 252, 6, 20, 16, 0.25)), LEAD, shade=0.1, line=INK, line_w=1)
    Mu.paint(M(circle(288, 230, 6, 20, 16, 0.25)), LEAD, shade=0.1, line=INK, line_w=1)
    # 箜篌（右）：弓形琴身与琴弦
    hx0, hy0 = 930, 196
    arc = [(hx0 + 70 * math.sin(t) * 0.7, hy0 + 110 * (1 - math.cos(t)) * 0.9) for t in np.linspace(0, 2.2, 30)]
    ribbon(Mu, [(hx0, hy0 - 4), (980, 170), (1030, 188), (1080, 160), (1110, 172)], 11, CINNA, OCHRE, 2.6, 0.3, 1.2)
    Mu.paint(M(taper_stroke(arc, 12, 7), 0.0), EARTH, shade=0.25, line=INK, line_w=1)
    for k in range(7):
        t = 0.25 + k * 0.25
        x1, y1 = hx0 + 70 * math.sin(t) * 0.7, hy0 + 110 * (1 - math.cos(t)) * 0.9
        Mu.line(stroke_mask(PW, PH, [(x1 * K, y1 * K), ((hx0 - 6) * K, (hy0 + 160) * K)], 0.7 * K), LEAD, 0.85)
    Mu.paint(M(taper_stroke([(hx0 - 14, hy0 + 150), (hx0 + 30, hy0 + 172)], 7, 7), 0), EARTH, shade=0.2, line=INK, line_w=1)

    # ---------- 飞天（小 Clawd） ----------
    def tray():
        Mu.paint(M(circle(842, 98, 22, 40, 7)), OCHRE, shade=0.2, line=INK, line_w=1)
        for (x, y) in [(830, 91), (843, 88), (856, 92)]:
            Mu.paint(M(jasmine(x, y, 6, 0.3)), LEAD, shade=0.1, line=INK, line_w=1)
    apsara(Mu, 800, 168, 6.5, 12, [rigid_arm("L", -20), rigid_arm("R", 95)],
           [([(790, 140), (870, 128), (950, 140), (1020, 116), (1100, 104)], 12, LAPIS, MALA, 3.2, 0.0),
            ([(760, 150), (700, 124), (640, 120), (590, 92)], 9, CINNA, OCHRE, 2.2, 0.6)], tray)
    apsara(Mu, 330, 150, 6.5, -14, [rigid_arm("L", 85), rigid_arm("R", -25)],
           [([(342, 128), (280, 112), (210, 120), (140, 96), (60, 92)], 12, MALA, CINNA, 3.0, 0.3),
            ([(372, 132), (420, 106), (480, 100), (520, 74)], 9, LAPIS, LEAD, 2.0, 1.0)])

    # ---------- 主角：反弹琵琶（略向后仰，像摇滚吉他手） ----------
    S, FX, FY, ANG = 14, 560, 528, -7
    for (y, rx, n, col) in [(574, 150, 9, CINNA), (562, 128, 8, LEAD), (552, 106, 7, CINNA)]:
        for i in range(n):
            x = FX - rx + (2 * rx) * (i + 0.5) / n
            w = rx / n * 1.3
            petal = [(x - w, y + 16), (x - w * 0.8, y - 4), (x, y - 22), (x + w * 0.8, y - 4), (x + w, y + 16)]
            Mu.paint(M(petal, 0.7), col, shade=0.32, line=INK, line_w=1)
    Mu.paint(M(circle(FX, 532, 110, 64, 9)), OCHRE, shade=0.2, line=INK, line_w=1)
    # 披帛：从肩后绕出，向两侧高高扬起
    ribbon(Mu, [(470, 450), (400, 470), (330, 440), (300, 370), (330, 300), (300, 236), (238, 214)], 20, MALA, LAPIS, 3.4, 0.22, 0.2)
    ribbon(Mu, [(650, 450), (720, 476), (790, 470), (830, 410), (812, 340), (860, 290), (930, 300)], 20, CINNA, OCHRE, 3.2, 0.22, 0.9)
    # 琵琶：横在脑后（反弹琵琶）——梨形音箱从右肩上方整个露出来，琴颈斜向左下，左钳在琴头处按住琴颈
    BXp, BYp = 800, 296
    pts, head, (ux, uy, vx, vy, L, ha, hx, hy, hl) = pipa((BXp, BYp), (318, 372), 168)
    Mu.paint(M(pts, 0.6), (0.62, 0.38, 0.21), shade=0.36, line=INK, line_w=2)
    Mu.paint(M(head), EARTH, shade=0.2, line=INK, line_w=1)
    for k in range(3):
        px_ = hx + math.cos(ha) * hl * (0.3 + 0.25 * k)
        py_ = hy + math.sin(ha) * hl * (0.3 + 0.25 * k)
        Mu.paint(M(taper_stroke([(px_ - vx * 9, py_ - vy * 9), (px_ + vx * 9, py_ + vy * 9)], 3.2, 3.2), 0), OCHRE, shade=0, line=INK, line_w=1)
    for k in range(4):
        p0 = (BXp + ux * L * 0.125 + vx * (k - 1.5) * 10, BYp + uy * L * 0.125 + vy * (k - 1.5) * 10)
        p1 = (BXp + ux * L * 0.86 + vx * (k - 1.5) * 2.4, BYp + uy * L * 0.86 + vy * (k - 1.5) * 2.4)
        Mu.line(stroke_mask(PW, PH, [(p0[0] * K, p0[1] * K), (p1[0] * K, p1[1] * K)], 0.8 * K), LEAD, 0.9)
    # 不画音孔：画成月牙会像一只眯着的眼睛，琵琶看起来像一张脸
    bq = [(BXp + ux * L * s_ + vx * w_, BYp + uy * L * s_ + vy * w_) for s_, w_ in ((0.10, 26), (0.15, 26), (0.15, -26), (0.10, -26))]
    Mu.paint(M(bq), EARTH, shade=0.1, line=INK, line_w=1)          # 覆手：琴弦系在这里
    # Clawd 本体：左钳举过头顶按住琴颈，右钳在身后拨弦
    arms = [rigid_arm("L", 30), rigid_arm("R", 60)]
    skirt = [(0.4, 6.0), (15.6, 6.0), (16.4, 7.6), (17.6, 10.6), (18.2, 11.9), (15.2, 11.4), (12.2, 11.9), (9.2, 11.4), (6.4, 11.9), (3.4, 11.4), (0.4, 11.9), (-2.0, 11.7), (-0.8, 9.4)]
    check_proportions(arms=arms, legs=LEGS, extra={"skirt": [skirt]})
    cm = clawd_union_mask(PW, PH, FX * K, FY * K, S * K, arms=arms, ang=ANG)
    Mu.paint(cm, CLAWD, shade=0.30, line=INK, line_w=2)
    sm = M(to_px(skirt, FX, FY, S, ANG), 0.25)
    Mu.paint(sm, LEAD, opacity=0.94, shade=0.14, line=EARTH, line_w=1, wear=0.25)
    Mu.paint(M(to_px([(-0.5, 6.0), (16.5, 6.0), (16.6, 7.1), (-0.6, 7.1)], FX, FY, S, ANG)), MALA, shade=0.1, line=INK, line_w=1)
    for fxg in (2.4, 5.4, 8.4, 11.4, 14.4):
        fold = to_px([(fxg + 0.4, 7.3), (fxg + 0.1, 8.6), (fxg + 0.5, 9.9), (fxg + 0.9 * (fxg - 8) / 8, 11.3)], FX, FY, S, ANG)
        Mu.stroke(fold, 1.8, 0.7, EARTH, 0.6)
    eyes(Mu, FX, FY, S, ANG)

    # ---------- 抛起的茉莉 ----------
    for i in range(30):
        a = G.uniform(-2.6, -0.55)
        d = G.uniform(150, 330)
        x, y = 560 + d * math.cos(a) * 1.3, 330 + d * math.sin(a) * 0.62
        if not (70 < y < 580) or abs(x - 560) > 520:
            continue
        r = G.uniform(4.5, 8)
        Mu.paint(M(jasmine(x, y, r, G.uniform(0, 6))), LEAD, shade=0.1, line=INK, line_w=1, grain=0.5)
        Mu.paint(M(circle(x, y, r * 0.24, 12)), OCHRE, shade=0, wear=0.2)

    Mu.age()
    return np.clip(downsample(Mu.img, K), 0, 1)


def render_b():
    """08-b：青金石蓝底的摇滚舞台。火焰纹背光变成舞台光；Clawd 穿黑色垂坠长衣、戴红手套；飞天在台下甩飘带。"""
    Mu = Mural()
    Mu.paint(M([(-5, -5), (1125, -5), (1125, 640), (-5, 640)]), LAPIS_D, shade=0, wear=0.6)
    # 顶部联珠纹边饰
    Mu.paint(M([(0, 0), (1120, 0), (1120, 34), (0, 34)]), EARTH, shade=0)
    for i in range(48):
        Mu.paint(M(circle(12 + i * 23.5, 17, 7.5, 28)), LEAD, shade=0.12, grain=0.6)
    CX, CY = 560, 318
    # 背光里放射出的舞台光（白色细光束）
    ang = np.arctan2(Y / K - CY, X / K - CX)
    rr = np.hypot(X / K - CX, Y / K - CY)
    beams = ((np.cos(ang * 12) > 0.985) & (rr > 262)).astype(np.float32) * smooth(700, 280, rr)
    Mu.paint(gauss(beams, 1.6 * K), LEAD, opacity=0.38, shade=0, wear=0.3)
    # 双层火焰纹背光
    Mu.paint(M(circle(CX, CY, 262, 160)), EARTH, shade=0.2, line=INK, line_w=2)
    for i in range(34):
        a = 2 * math.pi * i / 34
        Mu.paint(M(flame(CX, CY, a, 186, 262, 22, 0.36), 0.5), CINNA if i % 3 else OCHRE, shade=0.3, line=INK, line_w=1)
        Mu.paint(M(flame(CX, CY, a + 0.02, 192, 240, 9, 0.36), 0.5), OCHRE if i % 3 else LEAD, shade=0.1)
    Mu.paint(M(circle(CX, CY, 186, 140)), MALA_D, shade=0.15, line=INK, line_w=1)
    for i in range(52):
        a = 2 * math.pi * i / 52
        Mu.paint(M(circle(CX + 178 * math.cos(a), CY + 178 * math.sin(a), 3.4, 14)), LEAD, shade=0, grain=0.4)
    Mu.paint(M(circle(CX, CY, 170, 140)), LAPIS, shade=0.32, line=INK, line_w=1)
    for i in range(16):
        a = 2 * math.pi * i / 16
        Mu.paint(M([(CX + 40 * math.cos(a - 0.12), CY + 40 * math.sin(a - 0.12)), (CX + 160 * math.cos(a), CY + 160 * math.sin(a)),
                    (CX + 40 * math.cos(a + 0.12), CY + 40 * math.sin(a + 0.12))], 0.6), LEAD, opacity=0.35, shade=0)
    # 舞台：阶梯形的台，边上一条卷草纹
    Mu.paint(M([(200, 520), (920, 520), (960, 560), (160, 560)]), EARTH, shade=0.15, line=INK, line_w=1)
    Mu.paint(M([(160, 560), (960, 560), (1000, 600), (120, 600)]), CINNA, shade=0.15, line=INK, line_w=1)
    for i in range(18):
        x = 180 + i * 44
        vine = [(x + t * 44, 580 + 6 * math.sin(t * 2 * math.pi)) for t in np.linspace(0, 1, 20)]
        Mu.stroke(vine, 2.2, 2.2, OCHRE, 0.8)
    # Clawd：黑色垂坠长衣 + 红手套
    S, FX, FY = 16, 560, 522
    arms = [rigid_arm("L", 30), rigid_arm("R", 30)]
    robe = [(-0.3, 5.3), (16.3, 5.3), (17.0, 8.0), (18.2, 11.9), (13.0, 11.6), (8.0, 11.9), (3.0, 11.6), (-2.2, 11.9), (-1.0, 8.0)]
    check_proportions(arms=arms, legs=LEGS, extra={"robe": [robe]})
    cm = clawd_union_mask(PW, PH, FX * K, FY * K, S * K, arms=arms)
    Mu.paint(cm, CLAWD, shade=0.30, line=INK, line_w=2)
    rm = M(to_px(robe, FX, FY, S), 0.2)
    Mu.paint(rm, (0.12, 0.09, 0.08), shade=0.12, line=INK, line_w=1, wear=0.3)
    for gx in (1.5, 4.5, 7.5, 10.5, 13.5):                 # 垂坠的褶：几道弯的浅色线
        fold = to_px([(gx + 0.5, 5.8), (gx, 7.6), (gx + 0.6, 9.4), (gx + 0.8 * (gx - 8) / 8, 11.5)], FX, FY, S)
        Mu.stroke(fold, 2.2, 0.8, (0.40, 0.34, 0.30), 0.7)
    from common4 import rect_pts as _rp
    gl = np.clip(sum(M(to_px(a if isinstance(a[0], tuple) else _rp(a), FX, FY, S)) for a in arms), 0, 1)
    Mu.paint(gl, CINNA, shade=0.2, line=INK, line_w=1)
    eyes(Mu, FX, FY, S, 0)
    # 台下的飞天：甩起长长的飘带
    apsara(Mu, 150, 612, 7.0, -16, [rigid_arm("L", 30), rigid_arm("R", 30)],
           [([(176, 560), (230, 480), (210, 400), (262, 320), (240, 240)], 16, MALA, LAPIS, 3.2, 0.2)])
    apsara(Mu, 970, 612, 7.0, 16, [rigid_arm("L", 30), rigid_arm("R", 30)],
           [([(944, 560), (890, 480), (910, 400), (858, 320), (880, 240)], 16, CINNA, OCHRE, 3.2, 0.8)])
    for i in range(22):
        x, y = G.uniform(60, 1060), G.uniform(60, 500)
        if np.hypot(x - CX, y - CY) < 270:
            continue
        Mu.paint(M(jasmine(x, y, G.uniform(4.5, 7), G.uniform(0, 6))), LEAD, shade=0.1, line=INK, line_w=1, grain=0.5)
    Mu.age()
    return np.clip(downsample(Mu.img, K), 0, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = render()
    full = frame.compose(painting, (0.62, 0.22, 0.17), "还你茉莉")
    frame.save(full, os.path.join(out_dir, "08_还你茉莉_敦煌壁画.png"))
    frame.save(painting, os.path.join(out_dir, "08_画心.png"))
    pb = render_b()
    frame.save(frame.compose(pb, (0.62, 0.22, 0.17), "还你茉莉"), os.path.join(out_dir, "08b_还你茉莉_摇滚舞台.png"))
    frame.save(pb, os.path.join(out_dir, "08b_画心.png"))
