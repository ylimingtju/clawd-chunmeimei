"""09 多少的光阴 · 彩色玻璃花窗 · 样张（09-a 日落时的潮间带）。
画法：画心是一个灯箱。天空、海、沙滩按区域切成不规则的玻璃块（泰森多边形），块与块之间是铅条（带一点焊锡的反光）；
每块玻璃颜色略有不同，有拉丝纹、气泡，靠近铅条处透光变暗；细节（眼睛、浪纹、贝壳纹）用深色釉彩画在玻璃上。
Clawd 用几块橙色玻璃拼成，正在用玻璃块堆沙堡，潮水涨上来。透出来的彩光会落在画框周围的墙上。
用法：python S09_彩色玻璃.py 输出目录"""
import math
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from common4 import streak_noise, to_px, rect_pts
from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, value_noise, smooth, over, downsample, ishift, outline, dilate)
from common import BODY, ARMS, LEGS
import frame

G = np.random.default_rng(9)
STREAK = [streak_noise(PH, PW, 50, 2, a, 90 + i) for i, a in enumerate((0, 60, 120))]


def voronoi_labels(region, n, stretch=(1.0, 1.0), seed=0):
    """在 region 遮罩内撒 n 个种子，返回每个像素最近种子的编号（区域外为 -1）。"""
    g = np.random.default_rng(seed)
    ys, xs = np.nonzero(region > 0.5)
    if len(xs) == 0:
        return np.full(region.shape, -1, np.int32)
    idx = g.choice(len(xs), n, replace=False)
    pts = np.stack([xs[idx] / stretch[0], ys[idx] / stretch[1]], 1)
    tree = cKDTree(pts)
    _, lab = tree.query(np.stack([xs / stretch[0], ys / stretch[1]], 1))
    out = np.full(region.shape, -1, np.int32)
    out[ys, xs] = lab
    return out


def render():
    H_, W_ = PH, PW
    L = np.full((H_, W_), -1, np.int32)
    color = {}
    nid = [0]

    def new_label(c):
        nid[0] += 1
        color[nid[0]] = np.asarray(c, np.float32)
        return nid[0]

    yy = Y / K
    # ---------- 背景三块区域，各自切成玻璃 ----------
    HZ = 300
    sky = (yy < HZ).astype(np.float32)
    sea = ((yy >= HZ) & (yy < 300 + 120 + 18 * np.sin(X / K / 70))).astype(np.float32)
    sand = np.clip(1 - sky - sea, 0, 1)
    for reg, n, st, sd, fn in [
        (sky, 70, (1.4, 1.0), 1, lambda cx, cy: np.array([0.30, 0.18, 0.55]) * (1 - cy / HZ) + np.array([1.0, 0.55, 0.22]) * (cy / HZ) ** 1.6),
        (sea, 60, (2.6, 0.8), 2, lambda cx, cy: np.array([0.10, 0.36, 0.62]) * (1 - (cy - HZ) / 140) + np.array([0.16, 0.56, 0.66]) * ((cy - HZ) / 140)),
        (sand, 70, (1.6, 1.0), 3, lambda cx, cy: np.array([0.95, 0.72, 0.36]) * (1 - (cy - 420) / 210 * 0.4)),
    ]:
        lab = voronoi_labels(reg, n, (st[0], st[1]), sd)
        for i in np.unique(lab[lab >= 0]):
            ys, xs = np.nonzero(lab == i)
            cx, cy = xs.mean() / K, ys.mean() / K
            c = fn(cx, cy) * (0.88 + 0.24 * G.random()) + G.normal(0, 0.03, 3)
            L[lab == i] = new_label(np.clip(c, 0.03, 1.2))
    # 夕阳（半个落进海里），一道金色的光路
    sun = M(circle(800, HZ, 58, 96)) * sky
    L[sun > 0.5] = new_label((1.25, 0.92, 0.45))
    for k in range(5):
        y0 = HZ + 10 + k * 20
        w = 50 - k * 7
        m = M([(800 - w, y0), (800 + w, y0 + 3), (800 + w - 6, y0 + 10), (800 - w + 6, y0 + 9)]) * sea
        L[m > 0.5] = new_label((1.15, 0.78, 0.38))
    # 涨上来的潮水边（白色浪花玻璃）
    foam = np.clip(M(catmull([(-10, 440), (200, 452), (420, 446), (640, 462), (900, 450), (1130, 458)], 80).tolist() +
                     catmull([(1130, 476), (900, 468), (640, 480), (420, 464), (200, 470), (-10, 460)], 80).tolist()), 0, 1)
    flab = voronoi_labels(foam, 18, (3.0, 1.0), 4)
    for i in np.unique(flab[flab >= 0]):
        L[flab == i] = new_label(np.array([0.85, 0.94, 0.96]) * (0.9 + 0.15 * G.random()))
    # 被冲到脚边的贝壳、一串将被抹平的脚印
    shell = M([(392, 600), (404, 574), (420, 568), (436, 574), (448, 600), (420, 606)], 0.6)
    L[shell > 0.5] = new_label((1.0, 0.82, 0.78))
    for k, (x, y) in enumerate([(220, 520), (262, 540), (300, 516), (342, 538)]):
        fp = M(circle(x, y, 9, 24, 5))
        L[fp > 0.5] = new_label(np.array([0.80, 0.56, 0.28]) * (1 - 0.1 * k))
    # ---------- 玻璃块堆的沙堡 ----------
    for (x0, y0, x1, y1, c) in [(620, 548, 700, 600, (1.0, 0.80, 0.30)), (630, 518, 660, 548, (0.95, 0.68, 0.25)),
                                (668, 524, 696, 548, (1.0, 0.86, 0.40)), (640, 500, 652, 518, (0.92, 0.36, 0.30))]:
        L[M([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) > 0.5] = new_label(c)
    L[M([(626, 518), (645, 496), (664, 518)]) > 0.5] = new_label((0.90, 0.40, 0.28))
    L[M([(664, 524), (682, 502), (700, 524)]) > 0.5] = new_label((0.90, 0.40, 0.28))
    # ---------- Clawd：几块橙色玻璃 ----------
    FX, FY, S = 470, 590, 12
    arms = [ARMS[0], rigid_arm("R", 25)]
    check_proportions(arms=arms)
    bm = M(to_px(rect_pts(BODY), FX, FY, S))
    xx = X / K
    for (cond, c) in [((yy < FY - 7 * S) & (xx < FX), (1.02, 0.60, 0.40)), ((yy < FY - 7 * S) & (xx >= FX), (0.96, 0.54, 0.36)),
                      ((yy >= FY - 7 * S) & (xx < FX - 2 * S), (0.92, 0.50, 0.33)), ((yy >= FY - 7 * S) & (xx >= FX - 2 * S), (0.86, 0.45, 0.30))]:
        L[(bm > 0.5) & cond] = new_label(c)
    for a in arms:
        pts = a if isinstance(a[0], tuple) else rect_pts(a)
        L[M(to_px(pts, FX, FY, S)) > 0.5] = new_label((0.95, 0.52, 0.34))
    for l in LEGS:
        L[M(to_px(rect_pts(l), FX, FY, S)) > 0.5] = new_label((0.88, 0.46, 0.30))
    # 举在右钳上的一块玻璃（要放上沙堡顶）
    L[M([(588, 512), (606, 508), (610, 524), (592, 528)]) > 0.5] = new_label((0.55, 0.85, 0.75))

    # 釉彩：眼睛、浪纹、贝壳纹、太阳的光芒
    paint = np.zeros((H_, W_), np.float32)
    for e in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        paint = np.maximum(paint, M(to_px(rect_pts(e), FX, FY, S)))
    for i in range(14):
        x, y = G.uniform(40, 1080), G.uniform(318, 410)
        paint = np.maximum(paint, SM([(x + t * 5, y + 3 * math.sin(t * 0.9)) for t in range(10)], 1.4, 1.0) * 0.7)
    for k in range(5):
        paint = np.maximum(paint, SM([(420, 604), (398 + k * 11, 576)], 1.0, 0, False) * 0.8)
    for k in range(10):
        a = math.pi + (k + 0.5) / 10 * math.pi
        paint = np.maximum(paint, SM([(800 + 66 * math.cos(a), HZ + 66 * math.sin(a)), (800 + 110 * math.cos(a), HZ + 110 * math.sin(a))], 1.2, 0, False) * 0.45)
    return glass_finish(L, color, nid[0], paint)


def glass_finish(L, color, n, paint, lead_w=2.2):
    """按标签图上色：每块玻璃的颜色 + 拉丝纹 + 气泡 + 靠近铅条变暗 + 灯箱亮度 + 釉彩 + 铅条 + 辉光。"""
    H_, W_ = L.shape
    nid = [n]
    # ---------- 上色：每块玻璃的颜色 + 拉丝纹 + 气泡 + 靠近铅条变暗 ----------
    ids = np.arange(nid[0] + 1)
    lut = np.zeros((nid[0] + 2, 3), np.float32)
    for k_, c in color.items():
        lut[k_] = c
    img = lut[np.clip(L, 0, None)]
    lead = np.zeros((H_, W_), bool)
    lead[:, 1:] |= L[:, 1:] != L[:, :-1]
    lead[1:, :] |= L[1:, :] != L[:-1, :]
    lead = ndimage.binary_dilation(lead, iterations=int(lead_w * K))
    lead = lead | (L < 0)
    dist = ndimage.distance_transform_edt(~lead)
    fall = 1 - 0.38 * np.exp(-dist / (7 * K))
    st = STREAK[0] * (L % 3 == 0) + STREAK[1] * (L % 3 == 1) + STREAK[2] * (L % 3 == 2)
    tex = 1 + 0.06 * st + 0.05 * (value_noise(H_, W_, 30, 91) - 0.5)
    bub = smooth(0.80, 0.84, value_noise(H_, W_, 2.2, 92)) * (fbm(H_, W_, 60, 93, 2) > 0.55)
    img = img * (fall * tex)[..., None] + bub[..., None] * 0.10
    # 灯箱：中间更亮，四角暗一点
    box = 1.12 - 0.28 * np.clip(((X / PW - 0.5) ** 2 + (Y / PH - 0.5) ** 2) * 2.5, 0, 1)
    img = img * box[..., None]
    img = img * (1 - 0.82 * gauss(paint, 0.5 * K))[..., None]
    # 铅条：深灰，带一点焊锡反光
    lead_f = gauss(lead.astype(np.float32), 0.6 * K)
    hl = np.clip(ishift(lead_f, -K, -K) - lead_f, 0, 1) * 0 + gauss(np.clip(lead_f - ishift(lead_f, 2, 2), 0, 1), 0.6) * 0.5
    img = img * (1 - lead_f[..., None]) + (np.array([0.10, 0.10, 0.11]) + hl[..., None] * 0.35) * lead_f[..., None]
    # 透光的辉光
    glow = gauss(np.clip(img - 0.7, 0, None), 8 * K)
    img = img + glow * 0.5
    return np.clip(img, 0, 1)


def render_b():
    """09-b：满台红紫色的星点灯；Clawd 穿银色亮片礼服——礼服切成许多小块银灰玻璃，就是亮片。"""
    L = np.full((PH, PW), -1, np.int32)
    color = {}
    nid = [0]

    def new_label(c):
        nid[0] += 1
        color[nid[0]] = np.asarray(c, np.float32)
        return nid[0]

    g = np.random.default_rng(19)
    yy = Y / K
    back = (yy < 500).astype(np.float32)
    floor = 1 - back
    for reg, n, st, sd, fn in [(back, 90, (1.0, 1.3), 21, lambda cx, cy: np.array([0.22, 0.08, 0.30]) * (0.8 + 0.4 * cy / 500)),
                               (floor, 40, (2.4, 0.8), 22, lambda cx, cy: np.array([0.16, 0.06, 0.20]))]:
        lab = voronoi_labels(reg, n, st, sd)
        for i in np.unique(lab[lab >= 0]):
            ys, xs = np.nonzero(lab == i)
            c = fn(xs.mean() / K, ys.mean() / K) * (0.8 + 0.4 * g.random())
            L[lab == i] = new_label(np.clip(c, 0.02, 1.2))
    # 星点灯：一颗颗圆形的亮玻璃
    for i in range(60):
        x, y, r = g.uniform(40, 1080), g.uniform(30, 470), g.uniform(7, 15)
        if abs(x - 560) < 160 and y > 230:
            continue
        c = [(1.3, 0.25, 0.45), (1.1, 0.30, 1.1), (1.3, 0.55, 0.85)][i % 3]
        L[M(circle(x, y, r, 24)) > 0.5] = new_label(c)
    # 地面上星点灯的倒影
    for i in range(16):
        x = g.uniform(60, 1060)
        if abs(x - 560) < 190:
            continue
        L[M(circle(x, g.uniform(540, 590), 5, 24, 16)) > 0.5] = new_label((0.62, 0.20, 0.45))
    L[M(circle(560, 520, 220, 80, 22)) > 0.5] = new_label((0.75, 0.45, 0.70))      # 脚下的追光
    # Clawd：银色亮片礼服
    FX, FY, S = 560, 500, 17
    arms = [rigid_arm("L", 0), rigid_arm("R", 0)]
    check_proportions(arms=arms)
    bm = M(to_px(rect_pts(BODY), FX, FY, S))
    L[bm > 0.5] = new_label((1.0, 0.56, 0.38))
    for a_ in arms:
        pts = a_ if isinstance(a_[0], tuple) else rect_pts(a_)
        L[M(to_px(pts, FX, FY, S)) > 0.5] = new_label((0.95, 0.52, 0.34))
    for l in LEGS:
        L[M(to_px(rect_pts(l), FX, FY, S)) > 0.5] = new_label((0.88, 0.46, 0.30))
    gown = garment([(-0.2, 5.4), (16.2, 5.4), (17.0, 8.6), (18.0, 12.0), (-2.0, 12.0), (-1.0, 8.6)], FX, FY, S, arms=arms, name="gown")
    glab = voronoi_labels(gown, 150, (1.0, 1.0), 23)
    for i in np.unique(glab[glab >= 0]):
        v = g.uniform(0.70, 1.25)
        L[glab == i] = new_label(np.array([0.78, 0.80, 0.88]) * v + np.array([0.10, 0.02, 0.10]) * g.random())
    paint = np.zeros((PH, PW), np.float32)
    for e in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        paint = np.maximum(paint, M(to_px(rect_pts(e), FX, FY, S)))
    img = glass_finish(L, color, nid[0], paint, lead_w=1.8)
    # 亮片的闪光：几颗四芒星
    for i in range(10):
        gx, gy = g.uniform(0, 16), g.uniform(6, 11.5)
        (px, py), = g2p([(gx, gy)], FX, FY, S)
        r = g.uniform(6, 12)
        st = M([(px, py - r), (px + r * 0.15, py - r * 0.15), (px + r, py), (px + r * 0.15, py + r * 0.15), (px, py + r),
                (px - r * 0.15, py + r * 0.15), (px - r, py), (px - r * 0.15, py - r * 0.15)])
        img = img + (st + gauss(st, 2 * K) * 0.6)[..., None] * 0.8
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = np.clip(downsample(render(), K), 0, 1)
    frame.save(frame.compose(painting, (0.17, 0.15, 0.30), "多少的光阴", spill=0.38), os.path.join(out_dir, "09_多少的光阴_彩色玻璃.png"))
    frame.save(painting, os.path.join(out_dir, "09_画心.png"))
    pb = np.clip(downsample(render_b(), K), 0, 1)
    frame.save(frame.compose(pb, (0.17, 0.15, 0.30), "多少的光阴", spill=0.38), os.path.join(out_dir, "09b_多少的光阴_星点灯舞台.png"))
    frame.save(pb, os.path.join(out_dir, "09b_画心.png"))
