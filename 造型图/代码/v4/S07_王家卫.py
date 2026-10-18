"""07 我表示理解 · 王家卫港片 · 样张（07-a 荧光灯下的地下通道）。
画法：2D 平面的“电影画面”——一点透视的瓷砖通道、天花板荧光灯管、潮湿地面的倒影；
路人用抽帧 + 慢门拖成重影，Clawd 是画面里唯一清楚的东西；整体青绿暗部、黄绿中间调、暖色高光，粗颗粒、光晕。
Clawd 穿过大的黑色皮夹克、戴窄墨镜，手里一罐凤梨罐头。
用法：python S07_王家卫.py 输出目录"""
import math
import os
import sys

import numpy as np
from scipy import ndimage

from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, value_noise, smooth, over, downsample, ishift, outline, dilate)
import frame

G = np.random.default_rng(7)
VP = (650, 300)
BX0, BY0, BX1, BY1 = 584, 236, 716, 364             # 通道尽头的出口
ORANGE = np.array([202, 124, 94]) / 255


def quad(a, b, c, d):
    return M([a, b, c, d])


def lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def hblur(img, s):
    return ndimage.gaussian_filter(img, (0, s, 0) if img.ndim == 3 else (0, s))


def render():
    img = np.zeros((PH, PW, 3), np.float32)
    TL, TR, BR, BL = (0, 0), (1120, 0), (1120, 630), (0, 630)
    bTL, bTR, bBR, bBL = (BX0, BY0), (BX1, BY0), (BX1, BY1), (BX0, BY1)
    # ---------- 墙、天花、地面 ----------
    walls = [(quad(TL, bTL, bBL, BL), 0.95), (quad(TR, bTR, bBR, BR), 0.80)]
    for m, k in walls:
        depth = smooth(0, 1, 1 - np.minimum(np.abs(X / K - VP[0]) / 650, 1))
        col = np.array([0.55, 0.64, 0.52])[None, None] * (k * (1 - 0.55 * depth))[..., None]
        img = over(img, col, m)
    ceil = quad(TL, TR, bTR, bTL)
    img = over(img, np.array([0.30, 0.36, 0.30])[None, None] * (1 - 0.5 * smooth(0, BY0, Y / K))[..., None], ceil)
    floor = quad(BL, BR, bBR, bBL)
    img = over(img, np.array([0.16, 0.20, 0.17])[None, None] * (1 - 0.4 * smooth(630, BY1, Y / K))[..., None], floor)
    # 瓷砖缝（墙面）：横缝汇向灭点，竖缝按深度排
    tiles = np.zeros((PH, PW), np.float32)
    for k in range(1, 26):
        y = k * 25
        for (p, q, mm) in [((0, y), (BX0, BY0 + (BY1 - BY0) * y / 630), walls[0][0]), ((1120, y), (BX1, BY0 + (BY1 - BY0) * y / 630), walls[1][0])]:
            tiles = np.maximum(tiles, SM([p, q], 0.7, 0, False) * mm)
    for d in [1 - 0.86 ** k for k in range(1, 30)]:
        for (p, q, mm) in [(lerp(TL, bTL, d), lerp(BL, bBL, d), walls[0][0]), (lerp(TR, bTR, d), lerp(BR, bBR, d), walls[1][0])]:
            tiles = np.maximum(tiles, SM([p, q], 0.7 * (1 - d) + 0.3, 0, False) * mm)
    img = img * (1 - 0.35 * tiles[..., None])
    # ---------- 天花板上的荧光灯管 ----------
    lights = np.zeros((PH, PW), np.float32)
    for d in [1 - 0.80 ** k for k in range(1, 14)]:
        for side in (-0.5, 0.5):
            cx = lerp((560 + side * 520, 0), (650 + side * 40, BY0), d)
            w = 70 * (1 - d) + 6
            h = 7 * (1 - d) + 1.5
            lights = np.maximum(lights, M([(cx[0] - w, cx[1] - h), (cx[0] + w, cx[1] - h), (cx[0] + w, cx[1] + h), (cx[0] - w, cx[1] + h)]) * ceil)
    img = img + lights[..., None] * np.array([1.2, 1.25, 1.0])
    # 地面倒影：灯管在湿地面上拉成竖条
    refl = ishift(np.flipud(lights), int(2 * VP[1] * K - (PH - 1)), 0) * floor     # 以视平线为镜面
    refl = ndimage.gaussian_filter(refl, (10 * K, 2.0 * K)) * (0.6 + 0.4 * fbm(PH, PW, 12, 71, 2))
    img = img + (refl * 1.6)[..., None] * np.array([0.8, 1.0, 0.75])
    # ---------- 出口：外面高架桥下的霓虹 ----------
    ex = quad(bTL, bTR, bBR, bBL)
    out = np.zeros((PH, PW, 3), np.float32) + np.array([0.05, 0.05, 0.08])
    for (x, y, r, c) in [(610, 270, 18, (1.2, 0.20, 0.35)), (680, 258, 14, (0.25, 0.55, 1.3)), (640, 320, 22, (1.3, 0.55, 0.15)), (700, 330, 10, (0.9, 0.2, 1.1))]:
        out = out + (np.exp(-(((X / K - x) / r) ** 2 + ((Y / K - y) / (r * 0.6)) ** 2)) * 1.0)[..., None] * np.array(c)
    for (x0, y0, x1, y1, c) in [(596, 262, 626, 268, (1.4, 0.3, 0.4)), (668, 250, 700, 255, (0.4, 0.7, 1.4)), (626, 316, 664, 322, (1.5, 0.8, 0.3))]:
        out = over(out, c, M([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    img = over(img, out, ex)
    # ---------- 墙上的钟（按拍点跳秒） ----------
    ccx, ccy = 214, 170
    cm = M(circle(ccx, ccy, 30, 64, 34))
    img = over(img, (0.80, 0.82, 0.70), cm)
    img = over(img, (0.10, 0.10, 0.08), outline(cm, 3))
    for i in range(12):
        a = 2 * math.pi * i / 12
        img = over(img, (0.15, 0.15, 0.12), SM([(ccx + 24 * math.cos(a), ccy + 28 * math.sin(a)), (ccx + 28 * math.cos(a), ccy + 32 * math.sin(a))], 1.4, 0, False))
    for (a, L, w) in [(-2.0, 16, 2.6), (-0.6, 24, 1.8), (1.3, 26, 0.9)]:
        img = over(img, (0.08, 0.08, 0.06), SM([(ccx, ccy), (ccx + L * math.cos(a), ccy + L * 1.12 * math.sin(a))], w, 0, False))
    # ---------- 抽帧拖影的路人 ----------
    for (x, foot, h, dirn, col, n) in [(300, 600, 300, 1, (0.26, 0.30, 0.26), 7), (930, 610, 330, -1, (0.45, 0.16, 0.12), 7),
                                       (480, 470, 150, 1, (0.30, 0.34, 0.30), 6), (820, 452, 120, -1, (0.22, 0.24, 0.36), 6),
                                       (1060, 640, 380, -1, (0.12, 0.13, 0.12), 5)]:
        fig = np.clip(M(circle(x, foot - h * 0.9, h * 0.07, 32, h * 0.085)) + M([(x - h * 0.14, foot - h * 0.8), (x + h * 0.14, foot - h * 0.8),
                      (x + h * 0.12, foot - h * 0.38), (x + h * 0.08, foot), (x - h * 0.08, foot), (x - h * 0.12, foot - h * 0.38)], 0.5), 0, 1)
        trail = np.zeros((PH, PW), np.float32)
        for k in range(n):                                  # 抽帧：一格一格的重影，每格自身也被慢门拉长
            ghost = hblur(ishift(fig, 0, int(-dirn * k * h * 0.12 * K)), h * 0.035 * K)
            trail = np.maximum(trail, np.clip(ghost * 1.3, 0, 1) * (1 - k / n) ** 1.1)
        rim = np.clip(trail - ishift(trail, 4 * K, 0), 0, 1)
        img = img * (1 - 0.78 * trail[..., None]) + np.array(col) * 0.78 * trail[..., None]
        img = img + (rim * 0.25)[..., None] * np.array([0.8, 0.9, 0.7])
    # ---------- Clawd：过大的黑皮夹克、窄墨镜、凤梨罐头 ----------
    FX, FY, S = 664, 604, 16
    arms = [rigid_arm("L", -24), rigid_arm("R", 10)]
    check_proportions(arms=arms)
    cm_, em = clawd(FX, FY, S, arms)
    body = ORANGE[None, None] * 1.05 * np.ones((PH, PW, 1))
    body = body * (0.75 + 0.45 * (1 - smooth(FY - 12 * S, FY - 6 * S, Y / K)))[..., None] * np.array([0.92, 1.05, 0.85])
    img = over(img, body, cm_)
    # 皮夹克：衣身 + 两只袖子（手臂整块包上黑皮）+ 竖起的大翻领，胸前敞开露出白 T 恤
    jacket = [(-0.5, 4.7), (16.5, 4.7), (16.9, 10.9), (-0.9, 10.9)]
    jm = garment(jacket, FX, FY, S, smooth_k=0.1, arms=arms, name="jacket")
    from common4 import rect_pts as _rp
    sleeves = np.clip(sum(M(g2p(a if isinstance(a[0], tuple) else _rp(a), FX, FY, S)) for a in arms), 0, 1)
    tee = garment([(6.4, 4.7), (9.6, 4.7), (9.2, 10.9), (6.8, 10.9)], FX, FY, S, name="tee")
    leather = np.clip(jm + sleeves, 0, 1)
    jc = np.array([0.045, 0.05, 0.05])[None, None] * np.ones((PH, PW, 1))
    hl = np.zeros((PH, PW), np.float32)
    for (gx0, gy0, gx1, gy1, w) in [(1.2, 5.4, 2.0, 10.4, 0.45), (14.4, 5.4, 15.0, 10.4, 0.35), (4.6, 6.0, 5.2, 10.6, 0.25), (11.2, 6.0, 11.6, 10.6, 0.22)]:
        hl = np.maximum(hl, SM(g2p([(gx0, gy0), (gx1, gy1)], FX, FY, S), w * S, 0, True))
    hl = np.maximum(hl, gauss(np.clip(sleeves - ishift(sleeves, 3 * K, 0), 0, 1), K))
    jc = jc + gauss(hl, 2.5 * K)[..., None] * np.array([0.34, 0.40, 0.34])
    img = over(img, jc, leather)
    img = over(img, (0.0, 0.0, 0.0), outline(leather, 1) * 0.7)
    img = over(img, (0.86, 0.88, 0.80), tee)
    img = over(img, (0.0, 0.0, 0.0), outline(tee, 1) * 0.5)
    for side in (-1, 1):                                   # 竖起的尖翻领，翻出肩线之外
        x0 = 6.4 if side < 0 else 9.6
        lap = [(x0, 4.7), (x0 + side * 4.4, 3.9), (x0 + side * 5.6, 5.0), (x0 + side * 2.2, 8.4), (x0, 8.0)]
        lm = garment(lap, FX, FY, S, name="lapel")
        img = over(img, np.array([0.08, 0.09, 0.09])[None, None] * np.ones((PH, PW, 1)) + gauss(outline(lm, 2), K)[..., None] * 0.25, lm)
        img = over(img, (0.0, 0.0, 0.0), outline(lm, 1) * 0.9)
    for k in range(10):                                    # 拉链
        (zx, zy), = g2p([(6.5, 5.2 + k * 0.56)], FX, FY, S)
        img = over(img, (0.75, 0.78, 0.70), M([(zx - 1.4, zy), (zx + 1.4, zy), (zx + 1.4, zy + 2.2), (zx - 1.4, zy + 2.2)]))
    # 窄墨镜：两片镜片 + 鼻梁 + 镜腿
    gl = np.zeros((PH, PW), np.float32)
    for (x0, x1) in ((2.5, 5.8), (10.2, 13.5)):
        gl = np.maximum(gl, garment([(x0, 3.0), (x1, 3.0), (x1 - 0.2, 4.6), (x0 + 0.2, 4.6)], FX, FY, S, smooth_k=0.2, name="lens"))
    gl = np.maximum(gl, SM(g2p([(5.8, 3.4), (8.0, 3.15), (10.2, 3.4)], FX, FY, S), 0.12 * S, 0.8, False))
    gl = np.maximum(gl, SM(g2p([(0.0, 3.3), (2.5, 3.3)], FX, FY, S), 0.12 * S, 0, False))
    gl = np.maximum(gl, SM(g2p([(13.5, 3.3), (16.0, 3.3)], FX, FY, S), 0.12 * S, 0, False))
    img = over(img, (0.02, 0.02, 0.025), gl)
    for x0 in (2.9, 10.6):
        img = img + (SM(g2p([(x0, 3.35), (x0 + 1.6, 3.3)], FX, FY, S), 0.16 * S, 0, True) * 0.7)[..., None] * np.array([0.8, 1.0, 0.8])
    img = img + (gauss(np.clip(cm_ - ishift(cm_, 3 * K, 0), 0, 1), 1.2 * K) * 0.35)[..., None] * np.array([0.6, 0.8, 0.55])   # 顶光的边缘亮线
    # 凤梨罐头：握在左臂里——圆柱罐身（椭圆罐盖、罐口卷边、拉环）、黄色标签上一颗凤梨；最后把左袖盖上去，像握住
    (cx_, cy_), = g2p([(-1.5, 4.9)], FX, FY, S)
    rw, rh, ey = 1.25 * S, 1.7 * S, 0.32 * S                  # 半宽、半高、罐顶椭圆的半短轴
    body_pts = [(cx_ - rw, cy_ - rh)] + [(cx_ + rw * math.cos(t), cy_ + rh + ey * math.sin(t)) for t in np.linspace(math.pi, 0, 30)] + [(cx_ + rw, cy_ - rh)]
    canm = M(body_pts)
    cyl = 0.62 + 0.55 * np.exp(-((X / K - (cx_ - 0.35 * rw)) / (0.45 * rw)) ** 2) - 0.25 * smooth(cx_, cx_ + rw, X / K)
    img = over(img, np.array([0.72, 0.74, 0.70])[None, None] * cyl[..., None], canm)
    lab = canm * M([(cx_ - rw - 2, cy_ - rh + 0.45 * S), (cx_ + rw + 2, cy_ - rh + 0.45 * S)] +
                   [(cx_ + rw * math.cos(t) * 1.02, cy_ + rh - 0.45 * S + ey * math.sin(t)) for t in np.linspace(0, math.pi, 30)])
    img = over(img, np.array([0.92, 0.74, 0.18])[None, None] * cyl[..., None], lab)
    for yy in (cy_ - rh + 0.75 * S, cy_ + rh - 0.75 * S):    # 标签上下两道红边
        img = over(img, np.array([0.70, 0.16, 0.10])[None, None] * cyl[..., None],
                   SM([(cx_ + rw * math.cos(t), yy + ey * math.sin(t)) for t in np.linspace(math.pi, 0, 24)], 0.13 * S, 0.5, False) * canm)
    pine = M(circle(cx_ - 0.1 * S, cy_ + 0.2 * S, 0.55 * S, 32, 0.72 * S)) * lab
    img = over(img, np.array([0.88, 0.56, 0.10])[None, None] * cyl[..., None], pine)
    for d in np.arange(-1.0, 1.01, 0.33):
        for sg in (-1, 1):
            img = over(img, (0.50, 0.28, 0.06), SM([(cx_ - 0.1 * S - 0.7 * S, cy_ + 0.2 * S + d * S - sg * 0.7 * S),
                                                    (cx_ - 0.1 * S + 0.7 * S, cy_ + 0.2 * S + d * S + sg * 0.7 * S)], 0.05 * S, 0, False) * pine)
    for dx in (-0.32, 0.0, 0.32):
        img = over(img, (0.18, 0.48, 0.16), M([(cx_ - 0.1 * S + (dx - 0.12) * S, cy_ - 0.45 * S), (cx_ - 0.1 * S + dx * 1.7 * S, cy_ - 1.05 * S),
                                               (cx_ - 0.1 * S + (dx + 0.12) * S, cy_ - 0.45 * S)]) * lab)
    lid = M(circle(cx_, cy_ - rh, rw, 48, ey))
    img = over(img, np.array([0.80, 0.82, 0.78])[None, None] * (0.85 + 0.25 * (1 - smooth(cx_ - rw, cx_ + rw, X / K)))[..., None], lid)
    img = over(img, (0.40, 0.42, 0.40), np.clip(lid - M(circle(cx_, cy_ - rh, rw * 0.84, 48, ey * 0.78)), 0, 1))   # 卷边
    img = over(img, (0.55, 0.57, 0.55), np.clip(M(circle(cx_ + 0.15 * rw, cy_ - rh, 0.32 * rw, 24, 0.13 * S)) -
                                               M(circle(cx_ + 0.15 * rw, cy_ - rh, 0.18 * rw, 24, 0.07 * S)), 0, 1))  # 拉环
    img = over(img, (0.0, 0.0, 0.0), outline(np.clip(canm + lid, 0, 1), 1) * 0.6)
    sl = M(g2p(arms[0], FX, FY, S))
    img = over(img, jc, sl)
    img = over(img, (0.0, 0.0, 0.0), outline(sl, 1) * 0.7)
    # Clawd 在湿地面上的倒影
    ref = ishift(np.flipud(cm_), int(-(PH - 2 * FY * K)), 0) * floor
    ref = ndimage.gaussian_filter(ref, (6 * K, 1 * K))
    img = img * (1 - 0.35 * ref[..., None]) + np.array([0.20, 0.14, 0.10]) * 0.35 * ref[..., None]

    # ---------- 调色、光晕、颗粒 ----------
    hero = gauss(dilate(cm_, 2 * K), 2 * K)[..., None]
    pre = img.copy()
    lum = img.mean(-1, keepdims=True)
    shadow = np.array([0.06, 0.20, 0.17])
    high = np.array([1.05, 0.95, 0.70])
    t = np.clip(lum, 0, 1)
    img = img * (0.75 + 0.25 * t) + shadow * (1 - t) * 0.22 + high * np.clip(t - 0.6, 0, 1) * 0.3
    bright = np.clip(img - 0.85, 0, None)
    halo = gauss(bright, 10 * K) * 1.4 + gauss(bright, 30 * K) * 1.0
    img = img + halo * np.array([1.0, 0.55, 0.35])
    img = img / (1 + 0.35 * img)
    img = img * (1 - 0.45 * hero) + (pre / (1 + 0.3 * pre)) * 0.45 * hero      # 主角保留更多本色
    vig = np.clip(((X / PW - 0.52) ** 2 * 1.2 + (Y / PH - 0.5) ** 2 * 1.6), 0, 1)
    img *= (1 - 0.55 * vig)[..., None]
    grain = (G.normal(0, 1, (PH // 2, PW // 2)).astype(np.float32))
    grain = np.repeat(np.repeat(grain, 2, 0), 2, 1)
    img = img + (0.045 * grain * (0.6 + 0.6 * (1 - img.mean(-1))))[..., None]
    # 轻微的色散
    img[..., 0] = ishift(img[..., 0], 0, 1)
    img[..., 2] = ishift(img[..., 2], 0, -1)
    return np.clip(img * 1.15, 0, 1)


def grade(img, hero):
    """港片调色：青绿暗部、暖高光、光晕、暗角、粗颗粒、轻微色散；hero 区域保留更多本色。"""
    hero = gauss(dilate(hero, 2 * K), 2 * K)[..., None]
    pre = img.copy()
    lum = img.mean(-1, keepdims=True)
    t = np.clip(lum, 0, 1)
    img = img * (0.75 + 0.25 * t) + np.array([0.06, 0.20, 0.17]) * (1 - t) * 0.22 + np.array([1.05, 0.95, 0.70]) * np.clip(t - 0.6, 0, 1) * 0.3
    bright = np.clip(img - 0.85, 0, None)
    img = img + (gauss(bright, 10 * K) * 1.4 + gauss(bright, 30 * K) * 1.0) * np.array([1.0, 0.55, 0.35])
    img = img / (1 + 0.35 * img)
    img = img * (1 - 0.45 * hero) + (pre / (1 + 0.3 * pre)) * 0.45 * hero
    vig = np.clip(((X / PW - 0.52) ** 2 * 1.2 + (Y / PH - 0.5) ** 2 * 1.6), 0, 1)
    img *= (1 - 0.55 * vig)[..., None]
    gr = np.repeat(np.repeat(G.normal(0, 1, (PH // 2, PW // 2)).astype(np.float32), 2, 0), 2, 1)
    img = img + (0.045 * gr * (0.6 + 0.6 * (1 - img.mean(-1))))[..., None]
    img[..., 0] = ishift(img[..., 0], 0, 1)
    img[..., 2] = ishift(img[..., 2], 0, -1)
    return np.clip(img * 1.15, 0, 1)


def render_b():
    """07-b：武汉现场——舞台灯在慢门下拖出光轨，台下观众举着手机；Clawd 穿白缎长裙、戴白手套，一只钳子向前伸。"""
    img = np.zeros((PH, PW, 3), np.float32) + np.array([0.03, 0.04, 0.06])
    haze = fbm(PH, PW, 140, 72, 3)
    img = img + (haze * 0.10)[..., None] * np.array([0.5, 0.7, 0.8])
    # 顶上的灯与穿过烟雾的光柱
    g = np.random.default_rng(77)
    for i in range(9):
        x0 = 80 + i * 120
        col = [np.array([1.0, 0.35, 0.65]), np.array([0.35, 0.75, 1.0]), np.array([1.0, 0.75, 0.35])][i % 3]
        x1 = x0 + g.uniform(-160, 160)
        cone = M([(x0 - 6, 20), (x0 + 6, 20), (x1 + 90, 640), (x1 - 90, 640)])
        img = img + (gauss(cone, 6 * K) * (0.10 + 0.12 * haze))[..., None] * col
        img = img + (np.exp(-(((X / K - x0) / 9) ** 2 + ((Y / K - 22) / 6) ** 2)) * 1.4)[..., None] * col
    # 慢门光轨：转动的摇头灯画出的弧线
    for i in range(7):
        cx, cy, r = g.uniform(150, 970), g.uniform(60, 260), g.uniform(80, 220)
        a0 = g.uniform(0, 2 * math.pi)
        arc = [(cx + r * math.cos(a0 + t), cy + r * 0.45 * math.sin(a0 + t)) for t in np.linspace(0, g.uniform(1.0, 2.2), 40)]
        col = [np.array([1.0, 0.4, 0.7]), np.array([0.4, 0.8, 1.0]), np.array([1.0, 0.85, 0.5])][i % 3]
        m = SM(arc, 2.2, 0.8, False)
        img = img + (m * 0.9 + gauss(m, 4 * K) * 0.5)[..., None] * col
    # 舞台地面
    img = over(img, (0.05, 0.06, 0.07), M([(0, 520), (1120, 520), (1120, 640), (0, 640)]))
    img = img + (np.exp(-(((X / K - 560) / 260) ** 2 + ((Y / K - 528) / 18) ** 2)) * 0.35)[..., None] * np.array([0.9, 0.95, 1.0])
    # Clawd：白缎长裙 + 白手套
    FX, FY, S = 560, 530, 16
    arms = [rigid_arm("L", 0), rigid_arm("R", 34)]
    check_proportions(arms=arms)
    cm, em = clawd(FX, FY, S, arms)
    body = ORANGE[None, None] * (1.05 - 0.25 * smooth(FX - 8 * S, FX + 8 * S, X / K))[..., None]
    img = over(img, body, cm)
    gown = garment([(-0.2, 5.4), (16.2, 5.4), (16.9, 8.6), (18.0, 12.0), (-2.0, 12.0), (-0.9, 8.6)], FX, FY, S, arms=arms, name="gown")
    u = (X / K - FX) / (9 * S)
    satin = 0.78 + 0.22 * np.cos(u * 7.0) ** 2 + 0.25 * np.exp(-((u + 0.25) / 0.08) ** 2)     # 缎面的长条高光
    img = over(img, np.array([0.93, 0.92, 0.88])[None, None] * satin[..., None], gown)
    from common4 import rect_pts as _rp
    gl = np.clip(sum(M(g2p(a if isinstance(a[0], tuple) else _rp(a), FX, FY, S)) for a in arms), 0, 1)
    img = over(img, (0.96, 0.95, 0.92), gl)
    img = over(img, (0.06, 0.05, 0.05), em)
    img = img + (gauss(np.clip(cm - ishift(cm, 3 * K, 0), 0, 1), 1.5 * K) * 0.4)[..., None] * np.array([0.9, 0.85, 1.0])
    # 台下：观众的头与举起的手机（慢门拖影）
    crowd = np.zeros((PH, PW), np.float32)
    for i in range(30):
        x, y, r = g.uniform(-20, 1140), g.uniform(578, 640), g.uniform(22, 40)
        crowd = np.maximum(crowd, M(circle(x, y, r, 32, r * 1.1)))
    crowd = ndimage.gaussian_filter(crowd, (1.5 * K, 6 * K))
    img = img * (1 - 0.92 * crowd[..., None]) + np.array([0.02, 0.025, 0.03]) * 0.92 * crowd[..., None]
    rim = np.clip(crowd - ishift(crowd, 4 * K, 0), 0, 1)          # 舞台光从前面打过来，勾出观众头顶的一圈亮边
    img = img + (gauss(rim, 1.2 * K) * 0.55)[..., None] * np.array([0.75, 0.80, 0.95])
    for i in range(14):
        x, y = g.uniform(40, 1080), g.uniform(560, 610)
        ph = M([(x - 7, y - 11), (x + 7, y - 11), (x + 7, y + 11), (x - 7, y + 11)])
        ph = ndimage.gaussian_filter(ph, (0.6 * K, 4 * K))
        img = img + (ph * 1.2)[..., None] * np.array([0.85, 0.95, 1.0])
    return grade(img, cm)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = np.clip(downsample(render(), K), 0, 1)
    frame.save(frame.compose(painting, (0.13, 0.25, 0.20), "我表示理解"), os.path.join(out_dir, "07_我表示理解_王家卫.png"))
    frame.save(painting, os.path.join(out_dir, "07_画心.png"))
    pb = np.clip(downsample(render_b(), K), 0, 1)
    frame.save(frame.compose(pb, (0.13, 0.25, 0.20), "我表示理解"), os.path.join(out_dir, "07b_我表示理解_武汉现场.png"))
    frame.save(pb, os.path.join(out_dir, "07b_画心.png"))
