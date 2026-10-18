"""03 向日葵朝着夜 · 浮世绘木版 · 样张（03-a 月下向日葵田）。
画法：和纸上的多版套印——普鲁士蓝天空做 bokashi 渐晕、天空露木纹、大色块有「胡麻摺」颗粒；
墨线版最后压印，色版与墨线有极轻微的错位；月亮是留白的纸色。前景一株高大的向日葵框住画面（广重式构图）。
用法：python S03_浮世绘.py 输出目录"""
import math
import os
import sys

import numpy as np

from common4 import streak_noise
from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, value_noise, smooth, over, downsample, ishift, outline, dilate)
import frame

G = np.random.default_rng(3)
WASHI = np.array([0.93, 0.88, 0.76])
FIB = fbm(PH, PW, 3, 31, 2) - 0.5
LONGFIB = gauss(streak_noise(PH, PW, 14, 0.6, 20, 32), 0.6)
SPECK = smooth(0.55, 0.8, value_noise(PH, PW, 1.4, 33))           # 胡麻摺颗粒
GRAIN = np.sin((Y / K + 14 * (fbm(PH, PW, 260, 34, 2) - 0.5)) / 2.6 * 2 * np.pi) * (0.5 + 0.5 * fbm(PH, PW, 120, 36, 2))   # 木纹
PRUSSIAN = np.array([0.10, 0.18, 0.38])
INDIGO = np.array([0.18, 0.24, 0.40])
SUMI = np.array([0.10, 0.09, 0.09])


class Print:
    def __init__(self):
        self.paper = WASHI * (1 + 0.03 * FIB + 0.008 * LONGFIB)[..., None]
        self.img = self.paper.copy()
        self.keys = np.zeros((PH, PW), np.float32)

    def block(self, m, color, reg=(0.6, 0.4), speck=0.06, grain=0.0):
        """一块色版：颜色印在纸上（不与下层相乘），带颗粒和错位。"""
        self.keys *= (1 - gauss(m, 0.4 * K))                 # 后印的色块会盖住先前的墨线
        m = ishift(m, int(round(reg[1] * K)), int(round(reg[0] * K)))
        col = np.asarray(color, np.float32)
        if col.ndim == 1:
            col = col[None, None]
        tex = (1 - speck * SPECK + grain * 0.022 * GRAIN)[..., None]
        self.img = over(self.img, self.paper * col * tex * 1.06, gauss(m, 0.3 * K))

    def key(self, m, w=1.2):
        """墨线版：记下要压印的轮廓。"""
        self.keys = np.maximum(self.keys, outline(m, max(1, int(round(w * K)))))

    def keyline(self, pts, w, smooth_k=1.0):
        self.keys = np.maximum(self.keys, SM(pts, w, smooth_k))

    def finish(self):
        brk = smooth(0.08, 0.3, value_noise(PH, PW, 6, 35))
        k = gauss(self.keys, 0.35 * K) * (0.75 + 0.25 * brk)
        self.img = over(self.img, SUMI * (1 - 0.04 * SPECK)[..., None], np.clip(k, 0, 1))
        return self.img


def petals(cx, cy, r, n, rot, sq=1.0):
    pts = []
    for i in range(n * 8):
        a = 2 * math.pi * i / (n * 8)
        rr = r * (0.62 + 0.38 * abs(math.cos(n * a / 2)) ** 0.5)
        x, y = rr * math.cos(a), rr * math.sin(a) * sq
        pts.append((cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot)))
    return pts


def leaf(x, y, L, w, ang):
    pts = []
    for i in range(21):
        t = i / 20
        pts.append((t * L, w * math.sin(math.pi * t) ** 0.9 * (1 - 0.35 * t)))
    pts += [(t, -ww) for t, ww in pts[::-1]]
    return rot_pts([(x + px, y + py) for px, py in pts], x, y, ang)


def sunflower(Pr, x, base, top, head_r, face_ang, sq, stem_w=5, leaves=2, seed=0):
    """一株向日葵：茎、叶、花盘。sq<0 表示花盘背对我们（只看到绿色的花托）。"""
    g = np.random.default_rng(seed)
    stem = catmull([(x, base), (x + g.uniform(-6, 6), (base + top) / 2), (x + g.uniform(-4, 4), top)], 40)
    sm = SM(stem, stem_w, 1.0)
    Pr.block(sm, (0.30, 0.44, 0.27))
    Pr.key(sm, 0.8)
    for k in range(leaves):
        ly = base - (base - top) * (0.25 + 0.3 * k)
        side = 1 if k % 2 == 0 else -1
        lm = M(leaf(x, ly, head_r * 1.5, head_r * 0.45, -25 * side + (0 if side > 0 else 180)), 0.5)
        Pr.block(lm, (0.27, 0.42, 0.25))
        Pr.key(lm, 0.8)
        Pr.keyline([(x, ly), (x + side * head_r * 1.2, ly - 6)], 0.6, 0.5)
    if sq >= 0:
        pm = M(petals(x, top, head_r, 16, face_ang, sq), 0.4)
        Pr.block(pm, (0.92, 0.70, 0.20))
        Pr.key(pm, 0.8)
        cm = M(circle(x, top, head_r * 0.48, 40, head_r * 0.48 * sq, face_ang), 0)
        Pr.block(cm, (0.38, 0.22, 0.10))
        Pr.key(cm, 0.8)
        for i in range(10):
            a, rr = g.uniform(0, 2 * math.pi), g.uniform(0, head_r * 0.36)
            Pr.block(M(circle(x + rr * math.cos(a), top + rr * math.sin(a) * sq, 1.4, 8)), (0.62, 0.42, 0.16), reg=(0, 0), speck=0)
    else:
        # 背面：花瓣尖从绿色花托的边缘露出来
        pm = M(petals(x, top, head_r, 16, face_ang + 0.1, -sq), 0.4)
        Pr.block(pm, (0.80, 0.58, 0.16))
        Pr.key(pm, 0.7)
        bm = M(circle(x, top, head_r * 0.74, 40, head_r * 0.74 * (-sq), face_ang), 0)
        Pr.block(bm, (0.34, 0.47, 0.26))
        Pr.key(bm, 0.8)
        for i in range(10):
            a = 2 * math.pi * i / 10 + 0.2
            Pr.keyline([(x + head_r * 0.2 * math.cos(a), top + head_r * 0.2 * (-sq) * math.sin(a)),
                        (x + head_r * 0.66 * math.cos(a), top + head_r * 0.66 * (-sq) * math.sin(a))], 0.45, 0)


def render():
    Pr = Print()
    # ---------- 天空：普鲁士蓝渐晕 + 木纹 ----------
    t = np.clip(Y / K / 330, 0, 1)
    bok = (1 - smooth(0.0, 0.62, t))[..., None]
    sky = PRUSSIAN * bok + np.array([0.58, 0.66, 0.74]) * (1 - bok)
    sky = sky * (1 - 0.10 * smooth(0.7, 1.0, t))[..., None]
    Pr.block(M([(-5, -5), (1125, -5), (1125, 340), (-5, 340)]), sky, reg=(0, 0), speck=0.05, grain=1.0)
    # 月亮：留白的纸色，外圈一层极淡的晕
    halo = np.exp(-((np.hypot(X / K - 820, Y / K - 140) - 64) / 30).clip(0) ** 2) * 0.25
    Pr.img = over(Pr.img, Pr.paper * np.array([0.96, 0.94, 0.86]), halo * smooth(64, 66, np.hypot(X / K - 820, Y / K - 140)))
    Pr.img = over(Pr.img, Pr.paper * np.array([1.0, 0.98, 0.88]), M(circle(820, 140, 64, 120)))
    # 远山与霞（kasumi 横雾带）
    hills = [(-5, 340), (120, 318), (260, 330), (420, 300), (560, 322), (700, 306), (880, 330), (1010, 312), (1125, 326), (1125, 380), (-5, 380)]
    hm = M(hills, 0.6)
    Pr.block(hm, INDIGO * np.array([0.9, 0.9, 1.0]))
    for (y0, x0, x1, h) in [(292, -10, 470, 22), (346, 700, 1130, 18)]:
        band = [(x0, y0), (x1, y0), (x1 + 16, y0 + h / 2), (x1, y0 + h), (x0, y0 + h)]
        bm = M(band, 0.9)
        fade = smooth(x1 + 10, x1 - 120, X / K) * 0.6 + 0.4
        Pr.block(bm, np.array([0.93, 0.84, 0.80])[None, None] * np.ones((PH, PW, 1)), reg=(0, 0), speck=0.02)
        Pr.img = over(Pr.img, Pr.paper * np.array([0.96, 0.92, 0.88]), bm * fade * smooth(y0 + h, y0, Y / K))
    # 远处成片的花田
    fld = M([(-5, 370), (1125, 370), (1125, 640), (-5, 640)])
    fcol = np.array([0.26, 0.34, 0.27])[None, None] * (1.05 - 0.45 * smooth(380, 630, Y / K))[..., None]
    Pr.block(fld, fcol, speck=0.08)
    for i in range(70):
        x, y = G.uniform(0, 1120), G.uniform(378, 420)
        r = 3 + (y - 378) * 0.06
        Pr.block(M(circle(x, y, r, 16, r * 0.6)), (0.80, 0.60, 0.20), reg=(0, 0), speck=0.02)
    # ---------- 中景：成丛的向日葵，远小近大、高低错落，由左往右陆续转向月亮 ----------
    gF = np.random.default_rng(303)
    clumps = [(250, 0.25), (360, 0.75), (470, 0.4), (560, 0.9), (780, 0.55), (870, 0.2), (960, 0.8), (1060, 0.45)]
    flowers = []
    while len(flowers) < 46:
        cxl, cd = clumps[gF.integers(len(clumps))]
        d = float(np.clip(cd + gF.normal(0, 0.22), 0, 1))
        x = cxl + gF.normal(0, 40 + 30 * d)
        r = (13 + 34 * d) * gF.uniform(0.8, 1.2)
        top = 392 + 150 * d + gF.uniform(-30, 12) * (0.4 + d)
        if not (190 < x < 1110) or abs(x - 660) < r + 14:          # 让开托着 Clawd 的那根茎
            continue
        if any(math.hypot(x - fx_, top - ft_) < 0.8 * (r + fr_) for fx_, ft_, fr_, _ in flowers):
            continue
        flowers.append((x, top, r, d))
    for k, (x, top, r, d) in enumerate(sorted(flowers, key=lambda f: f[3])):
        # 转向的进度：左边还背着、右边已朝月；中间夹着几株转得慢的
        ph = float(np.clip((x - 260) / 650 + gF.normal(0, 0.18), 0, 1))
        tilt = gF.uniform(-0.35, 0.35)
        sw, nl = 2.5 + 3.5 * d, 1 + int(d > 0.45) + int(d > 0.8)
        if ph < 0.28:
            sunflower(Pr, x, 640, top, r, -0.3 + tilt, -gF.uniform(0.4, 0.75), sw, nl, 500 + k)
        else:
            sunflower(Pr, x, 640, top, r, -0.5 + 0.3 * ph + tilt, 0.35 + 0.5 * ph * gF.uniform(0.7, 1.1), sw, nl, 500 + k)
    # ---------- 托着 Clawd 的那株：茎一拍一拍长高 ----------
    hx, hy = 660, 300
    sunflower(Pr, hx, 640, hy, 78, -0.1, 0.3, 8, 3, 77)
    S = 8
    arms = [rigid_arm("L", 0), rigid_arm("R", 30)]
    check_proportions(arms=arms)
    cm, em = clawd(hx - 4, hy + 4, S, arms)
    Pr.block(cm, (0.86, 0.52, 0.37))
    Pr.key(cm, 1.0)
    Pr.block(em, SUMI, reg=(0, 0), speck=0)
    front = M(petals(hx, hy + 6, 78, 16, -0.1, 0.3), 0.4) * smooth(hy + 2, hy + 8, Y / K)   # 花盘前沿的花瓣压在脚上
    Pr.block(front, (0.92, 0.70, 0.20))
    Pr.key(front * smooth(hy + 4, hy + 10, Y / K), 0.8)
    # 萤火虫：淡黄绿的点，外面一圈更淡的晕
    for i in range(22):
        x, y = G.uniform(80, 1100), G.uniform(240, 600)
        Pr.block(M(circle(x, y, 6, 20)), (0.88, 0.92, 0.62), reg=(0, 0), speck=0.02)
        Pr.block(M(circle(x, y, 2.2, 12)), (1.0, 1.0, 0.80), reg=(0, 0), speck=0)
    # ---------- 前景：左侧一株高大的向日葵框住画面 ----------
    fx = 120
    st = catmull([(fx + 30, 640), (fx + 10, 400), (fx - 6, 200), (fx + 20, 60)], 60)
    sm = SM(st, 16, 1.0)
    Pr.block(sm, (0.24, 0.38, 0.22))
    Pr.key(sm, 1.0)
    for (ly, side, L) in [(500, 1, 150), (380, -1, 130), (260, 1, 120)]:
        lm = M(leaf(fx + 8, ly, L, 46, -20 if side > 0 else 200), 0.5)
        Pr.block(lm, (0.22, 0.36, 0.21))
        Pr.key(lm, 1.0)
        Pr.keyline([(fx + 8, ly), (fx + 8 + side * L * 0.8, ly - side * 20)], 0.8, 0.5)
    bigp = M(petals(fx + 40, 60, 120, 20, 0.5, 0.7), 0.4)
    Pr.block(bigp, (0.90, 0.66, 0.18))
    Pr.key(bigp, 1.1)
    bc = M(circle(fx + 40, 60, 58, 60, 40, 0.5))
    Pr.block(bc, (0.34, 0.20, 0.09))
    Pr.key(bc, 1.0)
    for i in range(60):
        a, rr = G.uniform(0, 2 * math.pi), G.uniform(0, 50)
        p = rot_pts([(fx + 40 + rr * math.cos(a), 60 + rr * 0.69 * math.sin(a))], fx + 40, 60, 0)
        Pr.block(M(circle(p[0][0], p[0][1], 2.0, 8)), (0.58, 0.40, 0.16), reg=(0, 0), speck=0)
    img = Pr.finish()
    return img


def render_b():
    """03-b：青海波纹样的幕布，月亮落成的圆形追光；Clawd 穿驼色不对称上衣、古铜褶裙，戴紫色长手套。"""
    Pr = Print()
    # 青海波：一层层同心弧，普鲁士蓝与靛蓝交替
    Pr.block(M([(-5, -5), (1125, -5), (1125, 520), (-5, 520)]), PRUSSIAN * 1.05, reg=(0, 0), speck=0.05, grain=0.6)
    R_ = 46
    xx, yy = X / K, Y / K
    pat = np.full((PH, PW), -1, np.int32)                 # 每个像素落在哪一圈弧里（0 外圈 … 3 内圈）
    for row in range(26):
        cy_ = -20 + row * R_ * 0.5 + R_
        off = (row % 2) * R_
        dx = ((xx - off + R_) % (2 * R_)) - R_
        d = np.hypot(dx, yy - cy_) / R_
        inside = (d < 1.0) & (yy < cy_ + 0.5)
        ring = np.clip(((1.0 - d) / 0.22).astype(np.int32), 0, 3)
        pat = np.where(inside, ring, pat)
    light = (pat % 2 == 0) & (pat >= 0)
    m_light = gauss(light.astype(np.float32), 0.5 * K) * (yy < 520)
    Pr.block(m_light, np.array([0.80, 0.84, 0.86]), reg=(0, 0), speck=0.04)
    # 圆形追光：留白的纸色大圆，外圈一层淡黄
    Pr.block(M(circle(560, 300, 230, 160)), np.array([0.96, 0.90, 0.68]), reg=(0, 0), speck=0.03)
    Pr.block(M(circle(560, 300, 214, 160)), np.array([1.0, 0.97, 0.88]), reg=(0, 0), speck=0.02)
    # 台面：木地板
    fl = M([(-5, 520), (1125, 520), (1125, 640), (-5, 640)])
    Pr.block(fl, np.array([0.74, 0.55, 0.32]), speck=0.06, grain=1.0)
    Pr.key(fl, 0.9)
    for i in range(1, 6):
        Pr.keyline([(-5, 520 + i * 22), (1125, 520 + i * 22)], 0.6, 0)
    Pr.block(M(circle(560, 552, 260, 120, 26)), np.array([0.92, 0.80, 0.56]), reg=(0, 0), speck=0.04)
    # 两侧的幕布（深靛蓝，带褶）
    for side in (-1, 1):
        x0 = 0 if side < 0 else 1120
        cur = [(x0, -5), (x0 + side * -150, -5), (x0 + side * -120, 200), (x0 + side * -170, 400), (x0 + side * -110, 640), (x0, 640)]
        cm = M(cur, 0.6)
        Pr.block(cm, INDIGO * 0.9, speck=0.06)
        Pr.key(cm, 1.0)
        for k in range(4):
            xx = x0 + side * -(28 + k * 28)
            Pr.keyline([(xx, 0), (xx + side * 10, 300), (xx - side * 8, 630)], 0.7, 1.0)
    # Clawd
    FX, FY, S = 560, 540, 17
    arms = [rigid_arm("L", 0), rigid_arm("R", 30)]
    check_proportions(arms=arms)
    cm, em = clawd(FX, FY, S, arms)
    Pr.block(cm, (0.86, 0.52, 0.37))
    Pr.key(cm, 1.0)
    top = garment([(-0.3, 5.4), (16.3, 5.4), (16.3, 7.8), (-0.3, 8.6)], FX, FY, S, arms=arms, name="top")
    strap = garment([(13.0, 5.4), (16.3, 1.8), (16.3, 3.6), (14.8, 5.4)], FX, FY, S, name="strap")    # 不对称：一侧斜挎到肩
    Pr.block(np.clip(top + strap, 0, 1), (0.74, 0.58, 0.40))
    Pr.key(np.clip(top + strap, 0, 1), 0.9)
    skirt = garment([(-0.3, 8.6), (16.3, 7.8), (17.4, 11.9), (-1.4, 11.9)], FX, FY, S, name="skirt")
    Pr.block(skirt, (0.58, 0.36, 0.18))
    Pr.key(skirt, 0.9)
    for k in range(9):                                     # 褶裙的竖褶
        gx = 0.4 + k * 1.9
        Pr.keyline(g2p([(gx, 8.6 - 0.05 * gx), (gx + (gx - 8) * 0.13, 11.8)], FX, FY, S), 0.7, 0)
    from common4 import rect_pts as _rp
    gl = np.clip(sum(M(g2p(a if isinstance(a[0], tuple) else _rp(a), FX, FY, S)) for a in arms), 0, 1)
    Pr.block(gl, (0.48, 0.30, 0.56))
    Pr.key(gl, 0.9)
    Pr.block(em, SUMI, reg=(0, 0), speck=0)
    # 紫手套握着的麦克风：黑手柄、银网罩；手套最后再印一次，压在手柄上
    from kit import mic_held
    mc = mic_held(arms[1], FX, FY, S, lean=12)
    Pr.block(mc["handle"], (0.15, 0.14, 0.14), reg=(0, 0))
    Pr.key(mc["handle"], 0.7)
    Pr.block(mc["head"], (0.66, 0.66, 0.64), reg=(0, 0))
    Pr.key(mc["head"], 0.8)
    Pr.keys = np.maximum(Pr.keys, mc["grille"] * 0.55)
    Pr.block(mc["shine"], (0.86, 0.86, 0.82), reg=(0, 0), speck=0)
    Pr.block(mc["band"], (0.40, 0.40, 0.40), reg=(0, 0))
    Pr.key(mc["band"], 0.6)
    gR = M(g2p(arms[1], FX, FY, S))
    Pr.block(gR, (0.48, 0.30, 0.56), reg=(0, 0))
    Pr.key(gR, 0.9)
    return np.clip(downsample(Pr.finish(), K), 0, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = np.clip(downsample(render(), K), 0, 1)
    frame.save(frame.compose(painting, (0.13, 0.17, 0.30), "向日葵朝着夜"), os.path.join(out_dir, "03_向日葵朝着夜_浮世绘.png"))
    frame.save(painting, os.path.join(out_dir, "03_画心.png"))
    pb = render_b()
    frame.save(frame.compose(pb, (0.13, 0.17, 0.30), "向日葵朝着夜"), os.path.join(out_dir, "03b_向日葵朝着夜_月光舞台.png"))
    frame.save(pb, os.path.join(out_dir, "03b_画心.png"))
