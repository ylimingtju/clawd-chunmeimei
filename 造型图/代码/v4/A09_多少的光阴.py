"""09 多少的光阴 · 彩色玻璃花窗 · 101 BPM（1:28.45–1:37.95）。拍号 b 从 08 结束算起。画心是灯箱，彩光落在墙上。
E  b0–3   Clawd 在重拍上用力一踩，壁画的灰泥从它脚下裂开；裂缝变成玻璃之间的铅条，光从裂缝里透出来，
          被裂缝扫过的部分变成玻璃拼块。
a  b3–10  潮间带。背后的光从日出转到日落，再转到月夜，所有玻璃的颜色跟着变。Clawd 用玻璃块堆沙堡，
          潮水一拍一拍涨上来，把沙堡冲散；它留在沙滩上的脚印被抹平，一只贝壳被冲到它脚边。
M  b10–12 贝壳张开，玻璃碎片从 Clawd 脚下往上飞；一道上升的铅条线从脚扫到头，换上银色亮片礼服，
          飞起来的碎片变成舞台上的星点灯。
b  b12–16 满台红紫色的星点灯，亮片随拍点闪烁。慢慢推近。"""
import math

import numpy as np
from scipy import ndimage

from kit import (K, PW, PH, X, Y, M, SM, circle, catmull, clawd, garment, g2p, rigid_arm, check_proportions, gauss, fbm,
                 value_noise, smooth, over, downsample, ishift)
from common4 import to_px, rect_pts
from common import BODY, ARMS, LEGS
import anim_core as A
import frame
import S09_彩色玻璃 as S9
from S09_彩色玻璃 import voronoi_labels, glass_finish
import A08_还你茉莉 as A8

FPS = 30
T0 = A8.T0 + 15 * A8.BEAT
BEAT = 60 / 101
WALL8, LABEL8 = A8.WALL8, A8.LABEL8
WALL9, LABEL9 = (0.17, 0.15, 0.30), "多少的光阴"
SPILL = 0.38
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 16 * BEAT) * FPS)
FEET08 = (560, 522)
HERO = (470, 590, 12)
_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


# ---------- 玻璃：一整张的泰森多边形，按块的中心落在哪里决定它是天、海还是沙 ----------
def cells():
    if "cells" in _CACHE:
        return _CACHE["cells"]
    lab = voronoi_labels(np.ones((PH, PW), np.float32), 260, (1.7, 1.0), 91)
    n = lab.max() + 1
    idx = np.arange(n)
    ys = ndimage.mean(np.mgrid[0:PH, 0:PW][0], labels=lab, index=idx) / K
    xs = ndimage.mean(np.mgrid[0:PH, 0:PW][1], labels=lab, index=idx) / K
    g = np.random.default_rng(92)
    jit = g.uniform(0.88, 1.12, n)
    stage = voronoi_labels(np.ones((PH, PW), np.float32), 120, (1.0, 1.3), 93)
    sn = stage.max() + 1
    sys_ = ndimage.mean(np.mgrid[0:PH, 0:PW][0], labels=stage, index=np.arange(sn)) / K
    _CACHE["cells"] = (lab, xs, ys, jit, stage, sys_, g.uniform(0.8, 1.2, sn))
    return _CACHE["cells"]


def sky_col(cy, day):
    """day：0 日出 → 0.5 日落 → 1 月夜。"""
    dawn_top, dawn_bot = np.array([0.40, 0.30, 0.62]), np.array([1.0, 0.70, 0.55])
    noon_top, noon_bot = np.array([0.30, 0.55, 0.90]), np.array([0.75, 0.88, 0.98])
    dusk_top, dusk_bot = np.array([0.30, 0.18, 0.55]), np.array([1.0, 0.55, 0.22])
    nite_top, nite_bot = np.array([0.04, 0.06, 0.20]), np.array([0.12, 0.16, 0.38])
    keys = [(0.0, dawn_top, dawn_bot), (0.25, noon_top, noon_bot), (0.5, dusk_top, dusk_bot), (0.75, nite_top, nite_bot), (1.0, nite_top, nite_bot)]
    for (a, t0, b0), (bb, t1, b1) in zip(keys[:-1], keys[1:]):
        if a <= day <= bb:
            u = (day - a) / (bb - a)
            top, bot = t0 * (1 - u) + t1 * u, b0 * (1 - u) + b1 * u
            break
    v = np.clip(cy / 300, 0, 1) ** 1.6
    return top * (1 - v) + bot * v


def light(day):
    """整片玻璃的透光：白天亮，入夜偏冷偏暗。"""
    if day < 0.6:
        return np.array([1.0, 1.0, 1.0]) * (1.0 + 0.05 * math.sin(day * math.pi))
    u = A.seg(day, 0.6, 0.85)
    return np.array([1.0, 1.0, 1.0]) * (1 - u) + np.array([0.48, 0.55, 0.85]) * u


def scene_labels(b, day, tide, castle_n, wash, shell_u, hero_pose, gown_line=None):
    """拼出这一帧的标签图与颜色表。tide：海水下沿的 y；castle_n：已经堆上去的块数；wash：沙堡被冲散的程度。"""
    lab, xs, ys, jit, stage, sys_, sjit = cells()
    HZ = 300
    L = lab.copy() + 1
    color = {}
    lt = light(day)
    sea_line = lambda x: tide + 14 * np.sin(x / 70 + b * 0.8)
    for i in range(len(xs)):
        cx, cy = xs[i], ys[i]
        if cy < HZ:
            c = sky_col(cy, day)
        elif cy < sea_line(cx):
            d = (cy - HZ) / 260
            c = (np.array([0.10, 0.36, 0.62]) * (1 - d) + np.array([0.16, 0.56, 0.66]) * d) * (0.7 + 0.3 * (1 - abs(day - 0.25) * 2))
        else:
            c = np.array([0.95, 0.72, 0.36]) * (1 - (cy - 420) / 210 * 0.4)
        color[i + 1] = np.clip(c * jit[i] * lt, 0.02, 1.25)
    nid = [len(xs) + 1]

    def new(c, m):
        nid[0] += 1
        color[nid[0]] = np.clip(np.asarray(c, np.float32) * lt, 0, 1.3)
        L[m > 0.5] = nid[0]

    # 太阳 / 月亮：沿一道弧从左边升起、从右边落下；入夜后月亮升起来
    if day < 0.6:
        a = math.pi * (1 - day / 0.6)
        sx, sy = 560 + 420 * math.cos(a), HZ - 220 * math.sin(a) + 10
        new((1.25, 0.92, 0.45) if day > 0.35 or day < 0.1 else (1.25, 1.15, 0.80), M(circle(sx, sy, 50, 96)) * (Y / K < sea_line(X / K)))
    if day > 0.62:
        u = A.ease_io(A.seg(day, 0.62, 1.0))
        new((0.95, 0.96, 1.05), M(circle(820, A.lerp(330, 120, u), 40, 96)) * (Y / K < HZ + 2))
    # 潮水边（白色浪花）
    foam = np.clip(M([(x, sea_line(x) - 8) for x in np.linspace(-10, 1130, 60)] + [(x, sea_line(x) + 10) for x in np.linspace(1130, -10, 60)]), 0, 1)
    new((0.85, 0.94, 0.96), foam)
    # 脚印（被潮水抹平）
    for k, (x, y) in enumerate([(220, 520), (262, 540), (300, 516), (342, 538)]):
        if y > tide + 10:
            new(np.array([0.80, 0.56, 0.28]) * (1 - 0.1 * k), M(circle(x, y, 9, 24, 5)))
    # 沙堡：按拍点一块块堆上去，涨潮时被冲散
    blocks = [((620, 548, 700, 600), (1.0, 0.80, 0.30)), ((630, 518, 660, 548), (0.95, 0.68, 0.25)),
              ((668, 524, 696, 548), (1.0, 0.86, 0.40)), ((640, 500, 652, 518), (0.92, 0.36, 0.30))]
    g = np.random.default_rng(5)
    for k, ((x0, y0, x1, y1), c) in enumerate(blocks[:castle_n]):
        dx, dy, rot = g.uniform(-60, 120) * wash, g.uniform(10, 50) * wash, g.uniform(-40, 40) * wash
        cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
        pts = [(x0 + dx, y0 + dy), (x1 + dx, y0 + dy), (x1 + dx, y1 + dy), (x0 + dx, y1 + dy)]
        a = math.radians(rot)
        pts = [(cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a), cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)) for x, y in pts]
        new(c, M(pts))
    if castle_n >= 4 and wash < 0.3:
        new((0.90, 0.40, 0.28), M([(626, 518), (645, 496), (664, 518)]))
        new((0.90, 0.40, 0.28), M([(664, 524), (682, 502), (700, 524)]))
    # 贝壳：被冲到 Clawd 脚边
    if shell_u > 0:
        sx = A.lerp(700, 420, A.ease_out(shell_u))
        new((1.0, 0.82, 0.78), M([(sx - 28, 600), (sx - 16, 574), (sx, 568), (sx + 16, 574), (sx + 28, 600), (sx, 606)], 0.6))
    paint = np.zeros((PH, PW), np.float32)
    if hero_pose is not None:
        paint = hero_glass(L, new, *hero_pose, gown_line=gown_line)
    return L, color, nid[0], paint


def hero_glass(L, new, FX, FY, S, arms, ang=0.0, gown_line=None):
    """Clawd：身体切成 4 块橙色玻璃 + 手臂、腿；gown_line 以下（画心 y）换成银色亮片礼服。"""
    check_proportions(arms=arms)
    bm = M(to_px(rect_pts(BODY), FX, FY, S, ang))
    yy, xx = Y / K, X / K
    for (cond, c) in [((yy < FY - 7 * S) & (xx < FX), (1.02, 0.60, 0.40)), ((yy < FY - 7 * S) & (xx >= FX), (0.96, 0.54, 0.36)),
                      ((yy >= FY - 7 * S) & (xx < FX - 2 * S), (0.92, 0.50, 0.33)), ((yy >= FY - 7 * S) & (xx >= FX - 2 * S), (0.86, 0.45, 0.30))]:
        new(c, bm * cond)
    for a in arms:
        new((0.95, 0.52, 0.34), M(to_px(a if isinstance(a[0], (tuple, list)) else rect_pts(a), FX, FY, S, ang)))
    for l in LEGS:
        new((0.88, 0.46, 0.30), M(to_px(rect_pts(l), FX, FY, S, ang)))
    if gown_line is not None:
        gown = garment([(-0.2, 5.4), (16.2, 5.4), (17.0, 8.6), (18.0, 12.0), (-2.0, 12.0), (-1.0, 8.6)], FX, FY, S, ang, arms=arms, name="gown")
        gown = gown * (yy > gown_line)
        if gown.max() > 0.5:
            glab = voronoi_labels(gown, max(20, int(150 * (S / 17) ** 2 * float((gown > 0.5).mean() / max(1e-6, (garment([(-0.2, 5.4), (16.2, 5.4), (17.0, 8.6), (18.0, 12.0), (-2.0, 12.0), (-1.0, 8.6)], FX, FY, S, ang, arms=arms, name="gown") > 0.5).mean())))), (1.0, 1.0), 23)
            g = np.random.default_rng(24)
            for i in np.unique(glab[glab >= 0]):
                new(np.array([0.78, 0.80, 0.88]) * g.uniform(0.70, 1.25) + np.array([0.10, 0.02, 0.10]) * g.random(), (glab == i).astype(np.float32))
    paint = np.zeros((PH, PW), np.float32)
    for e in ((3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)):
        paint = np.maximum(paint, M(to_px(rect_pts(e), FX, FY, S, ang)))
    return paint


def stage_labels(b, L, color, nid, below):
    """09-b 舞台的玻璃（只填进 below 遮罩里）。返回新的 nid。"""
    lab, xs, ys, jit, stage, sys_, sjit = cells()
    beat = int(math.floor(b))
    base = nid + 1
    for i in range(stage.max() + 1):
        cy = sys_[i]
        c = np.array([0.22, 0.08, 0.30]) * (0.8 + 0.4 * cy / 500) if cy < 500 else np.array([0.16, 0.06, 0.20])
        color[base + i] = np.clip(c * sjit[i], 0.02, 1.2)
    sel = below > 0.5
    L[sel] = (stage + base)[sel]
    nid = base + stage.max() + 1
    g = np.random.default_rng(19)
    for i in range(60):
        x, y, r = g.uniform(40, 1080), g.uniform(30, 470), g.uniform(7, 15)
        if abs(x - 560) < 160 and y > 230:
            continue
        c = np.array([(1.3, 0.25, 0.45), (1.1, 0.30, 1.1), (1.3, 0.55, 0.85)][i % 3]) * (1.0 if (i + beat) % 4 else 0.65)   # 星点灯随拍点明暗
        nid += 1
        color[nid] = c
        L[(M(circle(x, y, r, 24)) > 0.5) & sel] = nid
    for i in range(16):
        x = g.uniform(60, 1060)
        if abs(x - 560) < 190:
            continue
        nid += 1
        color[nid] = np.array((0.62, 0.20, 0.45))
        L[(M(circle(x, g.uniform(540, 590), 5, 24, 16)) > 0.5) & sel] = nid
    nid += 1
    color[nid] = np.array((0.75, 0.45, 0.70))
    L[(M(circle(560, 520, 220, 80, 22)) > 0.5) & sel] = nid
    return nid


# ---------- 一帧 ----------
def a_state(b):
    day = A.ease_io(A.seg(b, 3.0, 10.0))
    tide = 420 + 26 * sum(A.back_out(A.seg(b, k, k + 0.3)) for k in (5.0, 6.0, 7.0, 8.0))
    castle_n = min(4, max(0, int((b - 3.0) / 0.5) + 1)) if b >= 3.0 else 0
    wash = A.ease_out(A.seg(b, 7.0, 7.6))
    shell_u = A.seg(b, 8.0, 9.0)
    # Clawd：b3–3.6 从踩裂的地方跳到沙滩上，然后蹲在沙堡边堆玻璃块
    if b < 3.6:
        u = A.seg(b, 3.0, 3.6)
        fx, fy = A.hop(FEET08, HERO[:2], u, 80)
        S = A.lerp(16, 12, A.ease_io(u))
    else:
        fx, fy, S = HERO
    put = math.sin(math.pi * ((b - 3.0) % 0.5) / 0.5) if 3.6 <= b < 5.0 else 0.0
    arms = [ARMS[0], rigid_arm("R", 25 + 20 * put)]
    return day, tide, castle_n, wash, shell_u, (fx, fy, S, arms)


def painting(t, b):
    if b < 3.0:
        # E：从脚下裂开的玻璃
        stomp = A.seg(b, 0.4, 0.6)
        R = A.ease_in(A.seg(b, 0.5, 2.9), 1.3) * 1300
        cx, cy = FEET08
        d = np.hypot(X / K - cx, (Y / K - cy) * 1.3)
        jag = 40 * (value_noise(PH, PW, 26, 95) - 0.5)
        inside = smooth(R + jag + 1, R + jag - 1, d)
        mural = A8.age(A8.b_scene(15.0))
        L, color, n, paint = scene_labels(3.0, 0.0, 420, 0, 0, 0, (cx, cy, 16, [rigid_arm("L", 30), rigid_arm("R", 30)]))
        glass = glass_finish(L, color, n, paint)
        img = mural * (1 - inside[..., None]) + glass * inside[..., None]
        # 裂缝的前沿：一道透光的细缝
        front = smooth(R + jag + 10, R + jag + 2, d) * smooth(R + jag - 6, R + jag + 2, d)
        img = img * (1 - 0.6 * front[..., None]) + (front * 0.8)[..., None] * np.array([1.0, 0.92, 0.70])
        if 0.4 <= b < 0.9:                                  # 一踩：画面跟着一震
            img = ishift(img, int(3 * K * math.sin(b * 60)), 0)
        return img, R > 260
    if b < 10.0:
        day, tide, cn, wash, su, hero = a_state(b)
        L, color, n, paint = scene_labels(b, day, tide, cn, wash, su, hero)
        return glass_finish(L, color, n, paint), True
    # M：上升的铅条线，从脚扫到头、再一路扫到画面顶
    day, tide, cn, wash, su, hero = a_state(10.0)
    u = A.ease_io(A.seg(b, 10.0, 11.8))
    hop = A.ease_io(A.seg(b, 10.2, 11.6))
    fx, fy = A.lerp(HERO[0], 560, hop), A.lerp(HERO[1], 500, hop) - 40 * math.sin(math.pi * hop)
    S = A.lerp(12, 17, hop)
    line_y = A.lerp(640, -10, u)
    below = (Y / K > line_y).astype(np.float32)
    L, color, n, paint = scene_labels(10.0, day, tide, cn, wash, 1.0, None)
    n = stage_labels(b, L, color, n, below)
    nid = [n]

    def new(c, m):
        nid[0] += 1
        color[nid[0]] = np.asarray(c, np.float32)
        L[m > 0.5] = nid[0]
    paint = hero_glass(L, new, fx, fy, S, [rigid_arm("L", 0), rigid_arm("R", 0)], gown_line=line_y)
    # 往上飞的玻璃碎片
    g = np.random.default_rng(97)
    for i in range(26):
        x0 = 420 + g.uniform(-30, 30)
        tx, ty = g.uniform(60, 1060), g.uniform(40, 460)
        k = A.ease_out(A.seg(b, 10.1 + 0.03 * i, 11.4 + 0.03 * i))
        x, y = A.lerp(x0, tx, k), A.lerp(600, ty, k)
        r = 5 + 7 * k
        new([(1.3, 0.25, 0.45), (1.1, 0.30, 1.1), (1.3, 0.55, 0.85)][i % 3], M(circle(x, y, r, 6 if k < 0.9 else 24)))
    img = glass_finish(L, color, nid[0], paint)
    lm = SM([(-10, line_y), (1130, line_y)], 3.0, 0, False)
    img = img * (1 - lm[..., None]) + np.array([0.10, 0.10, 0.11]) * lm[..., None] + (gauss(lm, 3 * K) * 0.4)[..., None] * np.array([1.0, 0.9, 1.0])
    return img, True


def b_frame(b):
    L = np.zeros((PH, PW), np.int32)
    color = {}
    n = stage_labels(b, L, color, 0, np.ones((PH, PW), np.float32))
    nid = [n]

    def new(c, m):
        nid[0] += 1
        color[nid[0]] = np.asarray(c, np.float32)
        L[m > 0.5] = nid[0]
    paint = hero_glass(L, new, 560, 500, 17, [rigid_arm("L", 0), rigid_arm("R", 0)], gown_line=-10)
    img = glass_finish(L, color, nid[0], paint, lead_w=1.8)
    g = np.random.default_rng(int(math.floor(b * 2)) + 300)      # 亮片的闪光：每半拍换一批位置
    for i in range(10):
        gx, gy = g.uniform(0, 16), g.uniform(6, 11.5)
        (px, py), = g2p([(gx, gy)], 560, 500, 17)
        r = g.uniform(6, 12)
        st = M([(px, py - r), (px + r * 0.15, py - r * 0.15), (px + r, py), (px + r * 0.15, py + r * 0.15), (px, py + r),
                (px - r * 0.15, py + r * 0.15), (px - r, py), (px - r * 0.15, py - r * 0.15)])
        img = img + (st + gauss(st, 2 * K) * 0.6)[..., None] * 0.8
    return np.clip(img, 0, 1)


def frame_at(i):
    t = i / FPS
    b = bt(A.twos(i))
    if b >= 12.0:
        p, sw = b_frame(b), True
    else:
        p, sw = painting(t, b)
    p = np.clip(downsample(p, K), 0, 1)
    bb = bt(t)
    if bb >= 12.0:
        p = A.zoom_painting(p, 1 + 0.08 * A.ease_io(A.seg(bb, 12.0, 16.0)), 560, 400)
    if sw:
        return A.compose(p, WALL9, LABEL9, spill=SPILL)
    return A.compose(p, WALL8, LABEL8)


def warm():
    A8.warm()
    cells()
    for w, l in ((WALL8, LABEL8), (WALL9, LABEL9)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a09_{i:04d}.png")
