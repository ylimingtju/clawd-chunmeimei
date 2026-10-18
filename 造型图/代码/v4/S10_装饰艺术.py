"""10 珠玉 · 装饰艺术 · 样张（10-a 追着珍珠飞向光之巨像）。
画法：深海军蓝底上的金色几何——放射扇纹、浮在云上的金色幻影巨像、程式化的扇贝浪、汇聚的探照灯、细金线的阶梯角边框；
金色用分段的金属渐变（亮带 + 暗带）表现。巨像是半透明的光（45° 侧身看向小 Clawd，下半身化进装饰艺术云纹里）；
Clawd 穿白色钩针镂空流苏长裙，裙上的薄纱向后飘起。
用法：python S10_装饰艺术.py 输出目录"""
import math
import os
import sys

import numpy as np

from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, value_noise, smooth, over, downsample, ishift, outline, dilate)
import frame

G = np.random.default_rng(10)
NAVY = np.array([0.05, 0.07, 0.15])
NAVY2 = np.array([0.10, 0.13, 0.25])
TEAL = np.array([0.08, 0.26, 0.30])
CREAM = np.array([0.95, 0.90, 0.78])
ORANGE = np.array([202, 124, 94]) / 255
FINE = fbm(PH, PW, 3, 101, 2) - 0.5
HZ = 400
HEAD = (790, 150)                       # 幻影巨像头部附近：放射扇纹与探照灯都汇向这里


def gold(axis="y", lo=0.0, hi=1.0, bands=3.0):
    """分段金属金：沿 axis 方向在亮金和暗金之间来回，带一条窄高光。"""
    t = (Y / K if axis == "y" else X / K)
    u = ((t - lo) / max(hi - lo, 1e-6)) * bands
    w = 0.5 + 0.5 * np.cos(2 * np.pi * u)
    c = np.array([0.62, 0.44, 0.20])[None, None] * (1 - w[..., None]) + np.array([0.98, 0.84, 0.52])[None, None] * w[..., None]
    c = c + (np.exp(-((u % 1 - 0.05) / 0.03) ** 2) * 0.25)[..., None]
    return c * (1 + 0.03 * FINE)[..., None]


def deco_cloud(img, x0, y0, rads):
    """一朵装饰艺术云：底边平直，上沿是一串大小不一的半圆云头，每个云头里套两道同心金弧。"""
    lobes, x = [], x0
    for r in rads:
        lobes.append((x + r, r))
        x += r * 1.55
    x1 = x + rads[-1] * 0.45
    pts = [(x0, y0)]
    for cx_, r in lobes:
        pts += [(cx_ + r * math.cos(t), y0 - r * 0.25 - r * 0.9 * math.sin(t)) for t in np.linspace(math.pi, 0, 18)]
    pts += [(x1, y0)]
    m = M(pts, 0.3)
    img = img + (gauss(m, 10 * K) * 0.08)[..., None] * np.array([1.0, 0.9, 0.7])
    img = over(img, NAVY2 * 1.3, m * 0.96)
    for cx_, r in lobes:
        for f in (0.68, 0.38):
            arc = [(cx_ + r * f * math.cos(t), y0 - r * 0.25 - r * 0.9 * f * math.sin(t)) for t in np.linspace(math.pi * 0.95, 0.05, 20)]
            img = over(img, gold("x", 0, 1120, 6), SM(arc, 0.9, 0.5, False) * m * 0.75)
    img = over(img, gold("x", 0, 1120, 6), outline(m, 1))
    return img


def apparition(img, ox, oy, s, yaw=225.0, pitch=10.0):
    """天上的幻影：三维 Clawd 体块按 yaw 转到 45°（正面朝画面左前方），平行投影成平面的金色几何面，
    整体是半透明的光——上半身清楚、往下逐渐散进云里；棱线是亮金细线，外面一圈柔光。(ox, oy) 为脚底中心。"""
    import clawd3d as c3
    check_proportions()
    a, p = math.radians(yaw), math.radians(pitch)
    ca, sa, cp, sp = math.cos(a), math.sin(a), math.cos(p), math.sin(p)

    def proj(x, y, z):
        x = x - 8
        xr, zr = x * ca - z * sa, x * sa + z * ca
        return ox + s * xr, oy - s * (y * cp - zr * sp), y * sp + zr * cp

    faces = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)), ((0, 4, 5, 1), (0, -1, 0)),
             ((2, 3, 7, 6), (0, 1, 0)), ((0, 2, 6, 4), (0, 0, -1)), ((1, 5, 7, 3), (0, 0, 1))]
    L = np.array([-0.55, 0.65, 0.52])
    L /= np.linalg.norm(L)
    polys, eyes = [], []
    for bmin, bmax, name in c3.clawd_boxes():
        cs = [(x, y, z) for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])]
        for idx, n in faces:
            nr = np.array([n[0] * ca - n[2] * sa, n[1], n[0] * sa + n[2] * ca])
            if nr[1] * sp + nr[2] * cp <= 1e-6:
                continue
            q = [proj(*cs[i]) for i in idx]
            sh = 0.55 + 0.45 * max(0.0, float(nr @ L))
            item = (float(np.mean([v[2] for v in q])), [(v[0], v[1]) for v in q], sh)
            (eyes if name == "eye" else polys).append(item)
    polys.sort(key=lambda t: t[0])
    top_y, foot_y = oy - 13.5 * s, oy
    cov = np.zeros(X.shape, np.float32)
    layer = np.zeros(X.shape + (3,), np.float32)
    edges = np.zeros_like(cov)
    for d, q, sh in polys:
        m = M(q)
        layer = over(layer, gold("y", top_y, foot_y, 1.6) * sh, m)
        cov = np.maximum(cov, m)
        edges = edges * (1 - m) + SM(q + [q[0]], 1.3, 0, False) * m
    # 正面的装饰：竖向凹槽、胸前人字纹、腰线（贴在体块正面 z=-4 上投影过来）
    front = lambda gx, gy: proj(gx, 12 - gy, -4.0)[:2]
    fm = M([front(0, 2), front(16, 2), front(16, 10), front(0, 10)])
    deco = np.zeros_like(cov)
    for k in range(1, 16):
        deco = np.maximum(deco, SM([front(k, 6.0), front(k, 9.8)], 0.8, 0, False))
    for k in range(4):
        y = 0.9 + k * 0.9
        deco = np.maximum(deco, SM([front(5.2, y), front(8.0, y + 1.1), front(10.8, y)], 1.3, 0, False))
    deco = np.maximum(deco, SM([front(0, 5.4), front(16, 5.4)], 1.2, 0, False)) * fm
    eye_m = np.zeros_like(cov)
    for d, q, sh in eyes:
        eye_m = np.maximum(eye_m, M(q))
    # 透明度：头和肩最实，往下逐渐化开；叠一层细横纹，像光织成的
    A = 0.12 + 0.34 * smooth(foot_y + 8, foot_y - 9 * s, Y / K)
    A = A * (0.72 + 0.28 * np.cos(Y / K / 3.4 * 2 * np.pi))
    glow = gauss(cov, 16 * K)
    img = img + (glow * 0.22 * smooth(foot_y + 30, top_y + 60, Y / K))[..., None] * np.array([1.0, 0.86, 0.55])
    img = img * (1 - 0.30 * gauss(cov, 1.5 * K) * smooth(foot_y + 20, foot_y - 4 * s, Y / K))[..., None]   # 轮廓里的背景略压暗，让身形从光芒里分出来
    img = img * (1 - (A * cov)[..., None]) + layer * (A * cov)[..., None]
    img = img + (gauss(edges, 1.5 * K) * 0.35 + edges * 0.75 * smooth(foot_y + 6, foot_y - 6 * s, Y / K))[..., None] * np.array([1.0, 0.9, 0.62])
    img = over(img, (1.0, 0.93, 0.70), deco * (0.25 + 0.6 * A))
    # 眼睛是透出去的夜空，四周一圈淡淡的光
    img = over(img, NAVY * 1.1, eye_m * 0.92)
    img = img + (np.clip(gauss(eye_m, 2 * K) - eye_m, 0, 1) * 0.5)[..., None] * np.array([1.0, 0.95, 0.8])
    # 托着它的装饰艺术云纹：圆弧一个挨一个的云头、平底，里面套几道金色弧线；压住腿和下摆，像浮在云上
    for (x0, y0, rads) in [(560, foot_y + 34, (26, 40, 32, 22)), (770, foot_y + 50, (30, 46, 38, 28, 20)), (930, foot_y + 22, (20, 28, 20))]:
        img = deco_cloud(img, x0, y0, rads)
    return img


def render():
    img = NAVY[None, None] * (1 - smooth(0, HZ, Y / K))[..., None] + NAVY2[None, None] * smooth(0, HZ, Y / K)[..., None]
    img = img * (1 + 0.02 * FINE)[..., None]
    # 星：小小的四芒金星
    for i in range(40):
        x, y, r = G.uniform(30, 1090), G.uniform(30, 340), G.uniform(2, 4.5)
        st = M([(x, y - r * 2), (x + r * 0.35, y - r * 0.35), (x + r * 2, y), (x + r * 0.35, y + r * 0.35), (x, y + r * 2),
                (x - r * 0.35, y + r * 0.35), (x - r * 2, y), (x - r * 0.35, y - r * 0.35)])
        img = over(img, (0.95, 0.85, 0.60), st * 0.9)
    # ---------- 放射扇纹（巨像身后） ----------
    cx, cy = HEAD
    ang = np.arctan2(Y / K - cy, X / K - cx)
    rr = np.hypot(X / K - cx, Y / K - cy)
    upper = (Y / K < cy)
    rays = (np.cos(ang * 44) > 0.55) & (Y / K < HZ) & (rr > 240) & (rr < 520)
    img = over(img, gold("y", 0, HZ, 2), rays.astype(np.float32) * 0.55 * smooth(520, 330, rr))
    for r in (240, 380, 520):
        ring = np.clip(M(circle(cx, cy, r + 1.5, 200)) - M(circle(cx, cy, r - 1.5, 200)), 0, 1) * (Y / K < HZ)
        img = over(img, gold("x", 200, 1200, 6), ring)
    # 从海面打上去、汇向幻影的探照灯光束
    for (x0, x1) in [(250, 730), (460, 750), (1110, 810), (960, 790)]:
        beam = M([(x0 - 26, HZ), (x0 + 26, HZ), (x1 + 3, cy), (x1 - 3, cy)])
        img = img + (gauss(beam, 3 * K) * 0.16)[..., None] * np.array([1.0, 0.92, 0.70])
    # ---------- 光之巨像：天上的金色幻影 Clawd，45° 侧身朝向画面左前方，看着飞来的小 Clawd ----------
    img = apparition(img, 800, 336, 21)
    # ---------- 程式化的扇贝浪 ----------
    sea = M([(-5, HZ), (1125, HZ), (1125, 640), (-5, 640)])
    img = over(img, NAVY * 0.8, sea)
    rows = 6
    for r in range(rows):
        t = r / (rows - 1)
        y = HZ + 6 + (640 - HZ) * t ** 1.5
        rad = 18 + 46 * t
        off = (r % 2) * rad
        for i in range(int(1200 / (2 * rad)) + 2):
            x = -rad + off + i * 2 * rad
            sc = M([(x - rad, y + rad * 0.3)] + [(x + rad * math.cos(a), y - rad * 0.9 * math.sin(a) + rad * 0.3) for a in np.linspace(math.pi, 0, 24)] + [(x + rad, y + rad)] + [(x - rad, y + rad)])
            fill = TEAL * (0.45 + 0.4 * t) if (i + r) % 3 else NAVY2 * 1.05
            img = over(img, fill, sc)
            for k, f in enumerate((1.0, 0.55)):
                arc = [(x + rad * f * math.cos(a), y - rad * 0.9 * f * math.sin(a) + rad * 0.3) for a in np.linspace(math.pi, 0, 24)]
                img = over(img, gold("x", 0, 1120, 8), SM(arc, 1.3 if k == 0 else 0.8, 0.5, False) * (0.9 - 0.25 * k))
    # 幻影洒在海面上的一道淡光
    img = img + (np.exp(-((X / K - 760) / 150) ** 2) * smooth(HZ, HZ + 30, Y / K) * smooth(640, HZ, Y / K) * 0.10)[..., None] * np.array([1.0, 0.9, 0.65])
    # ---------- Clawd：白色钩针镂空流苏长裙，拖着三道金色速度线 ----------
    FX, FY, S, ANG = 400, 352, 12, -10
    arms = [rigid_arm("L", 30), rigid_arm("R", 30)]
    check_proportions(arms=arms)
    cm, em = clawd(FX, FY, S, arms, ANG)
    body = ORANGE[None, None] * (1.05 - 0.2 * smooth(FX - 8 * S, FX + 8 * S, X / K))[..., None]
    img = over(img, body, cm)
    img = over(img, (0.06, 0.05, 0.06), em)
    dress = [(-0.3, 5.0), (16.3, 5.0), (16.6, 10.4), (-0.6, 10.4)]
    dm = garment(dress, FX, FY, S, ANG, arms=arms, name="dress")
    # 钩针镂空：菱形网眼露出底下的橙色
    a = math.radians(ANG)
    gu = ((X / K - FX) * math.cos(a) + (Y / K - FY) * math.sin(a)) / (0.9 * S)
    gv = (-(X / K - FX) * math.sin(a) + (Y / K - FY) * math.cos(a)) / (0.9 * S)
    lattice = np.abs((gu + gv) % 1 - 0.5) + np.abs((gu - gv) % 1 - 0.5)
    holes = smooth(0.30, 0.22, lattice)
    img = over(img, CREAM * (1 + 0.03 * FINE)[..., None], dm * (1 - 0.85 * holes))
    # 裙上的薄纱向后飘起（下缘始终不低于 12 格）
    veil = [(-0.4, 5.8), (-4.0, 5.0), (-8.5, 6.2), (-13.0, 4.9), (-18.0, 6.0), (-22.0, 5.2), (-20.5, 6.6), (-16.0, 7.6), (-11.5, 7.4),
            (-7.5, 9.0), (-3.5, 9.8), (-0.4, 9.6)]
    check_proportions(extra={"veil": [veil]})
    vm = M(catmull(g2p(veil + [veil[0]], FX, FY, S, ANG), 160).tolist())
    fade = smooth(FX - 20 * S, FX - 2 * S, X / K)
    vgu = ((X / K - FX) * math.cos(a) + (Y / K - FY) * math.sin(a)) / (0.7 * S)
    vgv = (-(X / K - FX) * math.sin(a) + (Y / K - FY) * math.cos(a)) / (0.7 * S)
    vl = np.abs((vgu + vgv) % 1 - 0.5) + np.abs((vgu - vgv) % 1 - 0.5)
    img = img + (gauss(vm, 6 * K) * 0.10 * (0.4 + 0.6 * fade))[..., None] * np.array([1.0, 0.92, 0.75])
    img = over(img, CREAM * 1.04, vm * (0.42 + 0.40 * fade))
    for off in (6.6, 7.6, 8.5):                          # 纱上顺着飘动方向的几道褶光
        line = [(-0.6, off + 0.2), (-5.0, off - 0.9), (-9.5, off + 0.1), (-14.0, off - 1.3), (-19.0, off - 0.6)]
        img = over(img, (1.0, 0.95, 0.82), SM(catmull(g2p(line, FX, FY, S, ANG), 60), 0.9, 0.8, False) * vm * 0.6)
    img = over(img, gold("x", 0, 1120, 5), outline(vm, 1) * (0.4 + 0.5 * fade))
    fr = []
    for k in range(26):                                      # 流苏：最低到 12 格
        gx = -0.4 + k * 0.67
        fr.append(g2p([(gx, 10.4), (gx - 0.25, 11.95)], FX, FY, S, ANG))
    check_proportions(extra={"fringe": [[(0, 10.4), (0, 11.95)]]})
    for f in fr:
        img = over(img, CREAM, SM(f, 1.1, 0, False))
    img = over(img, gold("x", 0, 1120, 5), SM(g2p([(-0.3, 5.0), (16.3, 5.0)], FX, FY, S, ANG), 1.6, 0, False))
    # 前面那颗被追着的珍珠
    q = (560, 290)
    img = img + (np.exp(-(np.hypot(X / K - q[0], Y / K - q[1]) / 18) ** 2) * 0.5)[..., None] * np.array([1.0, 0.96, 0.88])
    img = over(img, (1.0, 0.98, 0.94), M(circle(*q, 8, 40)))
    # ---------- 阶梯角的细金线边框 ----------
    def stepped(inset, step):
        a_, b_ = inset, inset + step
        return [(b_, a_), (1120 - b_, a_), (1120 - b_, b_), (1120 - a_, b_), (1120 - a_, 630 - b_), (1120 - b_, 630 - b_), (1120 - b_, 630 - a_),
                (b_, 630 - a_), (b_, 630 - b_), (a_, 630 - b_), (a_, b_), (b_, b_)]
    for inset, w in ((14, 1.6), (22, 0.8)):
        pts = stepped(inset, 18)
        img = over(img, gold("x", 0, 1120, 10), SM(pts + [pts[0]], w, 0, False))
    return np.clip(img, 0, 1)


def render_b():
    """10-b：几束探照灯在 Clawd 身后交汇成放射扇；Clawd 穿白色垂坠礼服、戴珍珠耳饰，安静地站在阶梯形舞台上。"""
    img = NAVY[None, None] * (1 - smooth(0, 520, Y / K))[..., None] + NAVY2[None, None] * smooth(0, 520, Y / K)[..., None]
    img = img * (1 + 0.02 * FINE)[..., None]
    cx, cy = 560, 300
    # 交汇的探照灯：从台口两侧斜打上去，在 Clawd 身后交叉
    for (x0, x1) in [(60, 760), (220, 700), (380, 640), (740, 480), (900, 420), (1060, 360)]:
        beam = M([(x0 - 30, 640), (x0 + 30, 640), (x1 + 6, -10), (x1 - 6, -10)])
        img = img + (gauss(beam, 4 * K) * 0.13)[..., None] * np.array([1.0, 0.93, 0.75])
    # 身后的金色放射扇与同心圆
    ang = np.arctan2(Y / K - cy, X / K - cx)
    rr = np.hypot(X / K - cx, Y / K - cy)
    rays = (np.cos(ang * 40) > 0.6) & (rr > 120) & (rr < 330) & (Y / K < 470)
    img = over(img, gold("y", 0, 470, 2), rays.astype(np.float32) * 0.5 * smooth(330, 200, rr))
    for r in (120, 220, 330):
        ring = np.clip(M(circle(cx, cy, r + 1.4, 200)) - M(circle(cx, cy, r - 1.4, 200)), 0, 1) * (Y / K < 470)
        img = over(img, gold("x", 200, 1000, 6), ring)
    # 阶梯形的舞台
    for (yt, yb, w) in [(470, 500, 330), (500, 530, 400), (530, 560, 470)]:
        tm = M([(cx - w, yt), (cx + w, yt), (cx + w, yb), (cx - w, yb)])
        img = over(img, NAVY2 * 1.3, tm)
        img = over(img, gold("x", 0, 1120, 8), SM([(cx - w, yt), (cx + w, yt)], 1.6, 0, False))
        for k in range(-int(w / 40), int(w / 40) + 1):
            img = over(img, gold("x", 0, 1120, 8), SM([(cx + k * 40, yt + 6), (cx + k * 40 + 10, yt + 15), (cx + k * 40 + 20, yt + 6)], 1.0, 0, False) * 0.7)
    img = over(img, NAVY * 0.7, M([(-5, 560), (1125, 560), (1125, 640), (-5, 640)]))
    for k in range(6):
        y = 572 + k * 11
        img = over(img, gold("x", 0, 1120, 6), SM([(cx - 300 + k * 30, y), (cx + 300 - k * 30, y)], 1.0, 0, False) * (0.45 - k * 0.06))
    # Clawd：白色垂坠礼服 + 珍珠耳饰
    FX, FY, S = cx, 470, 16
    arms = [rigid_arm("L", 0), rigid_arm("R", 0)]
    check_proportions(arms=arms)
    cm, em = clawd(FX, FY, S, arms)
    img = over(img, ORANGE[None, None] * (1.05 - 0.2 * smooth(FX - 8 * S, FX + 8 * S, X / K))[..., None], cm)
    gown = garment([(-0.2, 5.4), (16.2, 5.4), (16.8, 8.6), (17.6, 12.0), (-1.6, 12.0), (-0.8, 8.6)], FX, FY, S, arms=arms, name="gown")
    u = (X / K - FX) / (9 * S)
    drape = 0.84 + 0.16 * np.cos(u * 18) ** 2 * smooth(FY - 6 * S, FY, Y / K)
    img = over(img, CREAM[None, None] * drape[..., None], gown)
    for gx in (1.5, 4.5, 8.0, 11.5, 14.5):                 # 垂坠的褶：金色细线
        img = over(img, gold("x", 0, 1120, 6), SM(g2p([(gx, 6.2), (gx + (gx - 8) * 0.14, 11.9)], FX, FY, S), 0.9, 0, False) * 0.6)
    img = over(img, gold("x", 0, 1120, 5), SM(g2p([(-0.2, 5.4), (16.2, 5.4)], FX, FY, S), 1.6, 0, False))
    img = over(img, (0.06, 0.05, 0.06), em)
    for gx, sd in ((-0.15, -1), (16.15, 1)):               # 珍珠耳坠：挂在身体两侧上角——金环、金杆、装饰艺术小扇、一颗水滴珍珠
        (px, py), = g2p([(gx, 1.0)], FX, FY, S)
        px += sd * 4
        img = over(img, gold("x", 0, 1120, 4), np.clip(M(circle(px, py, 5, 24)) - M(circle(px, py, 3, 24)), 0, 1))
        img = over(img, gold("y", py, py + 30, 2), SM([(px, py + 5), (px, py + 16)], 1.4, 0, False))
        fan = [(px, py + 14)] + [(px + 11 * math.cos(t), py + 14 + 11 * math.sin(t)) for t in np.linspace(0.15 * math.pi, 0.85 * math.pi, 16)]
        fm = M(fan)
        img = over(img, gold("y", py + 14, py + 26, 1.5), fm)
        for t in np.linspace(0.3 * math.pi, 0.7 * math.pi, 3):
            img = over(img, (0.40, 0.27, 0.10), SM([(px, py + 15), (px + 10 * math.cos(t), py + 14 + 10 * math.sin(t))], 0.6, 0, False) * fm)
        qx, qy, qr = px, py + 38, 11
        drop = M([(qx + qr * math.cos(t), qy + qr * 1.15 * math.sin(t) if math.sin(t) > 0 else qy + qr * 1.6 * math.sin(t) * (0.55 + 0.45 * abs(math.cos(t))))
                  for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)], 0.3)
        img = img + (np.exp(-(np.hypot(X / K - qx, Y / K - qy) / 26) ** 2) * 0.30)[..., None] * np.array([1.0, 0.96, 0.88])
        pr = np.hypot(X / K - qx + 4, Y / K - qy + 5) / qr
        img = over(img, np.array([0.99, 0.97, 0.93])[None, None] * (1 - 0.32 * smooth(0.2, 1.6, pr))[..., None] + np.array([0.02, 0.0, 0.03]), drop)
        img = over(img, (1.0, 1.0, 1.0), M(circle(qx - 4, qy - 5, 2.6, 16)) * 0.9)
    # 阶梯角的细金线边框
    def stepped(inset, step):
        a_, b_ = inset, inset + step
        return [(b_, a_), (1120 - b_, a_), (1120 - b_, b_), (1120 - a_, b_), (1120 - a_, 630 - b_), (1120 - b_, 630 - b_), (1120 - b_, 630 - a_),
                (b_, 630 - a_), (b_, 630 - b_), (a_, 630 - b_), (a_, b_), (b_, b_)]
    for inset, w in ((14, 1.6), (22, 0.8)):
        pts = stepped(inset, 18)
        img = over(img, gold("x", 0, 1120, 10), SM(pts + [pts[0]], w, 0, False))
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = np.clip(downsample(render(), K), 0, 1)
    frame.save(frame.compose(painting, (0.90, 0.86, 0.78), "珠玉"), os.path.join(out_dir, "10_珠玉_装饰艺术.png"))
    frame.save(painting, os.path.join(out_dir, "10_画心.png"))
    pb = np.clip(downsample(render_b(), K), 0, 1)
    frame.save(frame.compose(pb, (0.90, 0.86, 0.78), "珠玉"), os.path.join(out_dir, "10b_珠玉_光束舞台.png"))
    frame.save(pb, os.path.join(out_dir, "10b_画心.png"))
