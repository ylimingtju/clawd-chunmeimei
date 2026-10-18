"""01 还有什么更好的 · 瑞士国际主义海报 · 样张（01-a 网格上的纸壳小屋）。
画法：胶印专色——每种颜色是一层油墨，叠印处相乘变深；油墨有轻微的密度起伏和套色偏移；
网点照片质感用真实的网屏生成；文字只用抽象的灰条。Clawd 是一块橙色专色。
用法：python S01_瑞士海报.py 输出目录"""
import math
import os
import sys

import numpy as np

from kit import K, PW, PH, X, Y, M, SM, circle, rot_pts, clawd, garment, g2p, rigid_arm, check_proportions, ishift, gauss, fbm, value_noise, smooth, downsample
import frame

PAPER = np.array([0.95, 0.94, 0.90])
RED = np.array([0.87, 0.15, 0.12])
BLUE = np.array([0.46, 0.69, 0.86])
BLACK = np.array([0.10, 0.10, 0.11])
GREY = np.array([0.42, 0.42, 0.43])
ORANGE = np.array([202, 124, 94]) / 255

FIB = fbm(PH, PW, 4, 3, octaves=2) - 0.5
DENS = [fbm(PH, PW, 22, 10 + i, octaves=3) - 0.5 for i in range(4)]


class Press:
    def __init__(self):
        self.img = np.ones((PH, PW, 3), np.float32) * PAPER * (1 + 0.025 * FIB)[..., None]

    def ink(self, m, color, reg=(0, 0), density=1.0, k=0):
        """叠印一层专色：相乘混合，带套色偏移和密度起伏。"""
        m = ishift(m, int(reg[1] * K), int(reg[0] * K)) if reg != (0, 0) else m
        m = gauss(m, 0.35 * K)                                   # 油墨在纸上轻微扩散
        d = density * (0.96 + 0.05 * DENS[k % 4]) * (1 + 0.03 * FIB)
        a = np.clip(m * d, 0, 1)[..., None]
        self.img = self.img * (1 - a * (1 - np.asarray(color, np.float32)))

    def knockout(self, m):
        """镂空：露出纸色（相当于这一块所有油墨都不印）。"""
        a = gauss(m, 0.35 * K)[..., None]
        self.img = self.img * (1 - a) + PAPER * (1 + 0.025 * FIB)[..., None] * a


def halftone(value, cell=5.0, angle=15.0):
    """把 0..1 的油墨量变成网点（2 倍分辨率下计算）。"""
    a = math.radians(angle)
    u = (X * math.cos(a) + Y * math.sin(a)) / (cell * K)
    v = (-X * math.sin(a) + Y * math.cos(a)) / (cell * K)
    screen = (np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v) + 1) / 2     # 0..1
    return smooth(-0.06, 0.06, value - screen)


def render():
    Pr = Press()
    # 浅蓝底块与红色大圆（叠印处变成深紫）
    Pr.ink(M([(540, 300), (1130, 300), (1130, 640), (540, 640)]), BLUE, reg=(0, 0.6), k=0)
    Pr.ink(M(circle(850, 222, 200, 180)), RED, reg=(0.7, 0), k=1)
    # 网点照片质感的珍珠球
    cx, cy, r = 236, 556, 66
    d = np.hypot(X / K - cx, Y / K - cy) / r
    inside = smooth(1.0, 0.985, d)
    nz = np.sqrt(np.clip(1 - d ** 2, 0, 1))
    nx, ny = (X / K - cx) / r, (Y / K - cy) / r
    lam = np.clip(-0.5 * nx - 0.6 * ny + 0.62 * nz, 0, 1)
    spec = np.clip(-0.5 * nx - 0.6 * ny + 0.62 * nz, 0, 1) ** 30
    val = np.clip(0.92 - 0.75 * lam - 0.6 * spec, 0.03, 1) * inside
    Pr.ink(halftone(val, 4.2, 15) * inside, BLACK, k=2)
    # 抽象的文字灰条与细线
    for i, ln in enumerate([270, 236, 258, 190, 248, 120, 214]):
        Pr.ink(M([(60, 64 + i * 15), (60 + ln, 64 + i * 15), (60 + ln, 71 + i * 15), (60, 71 + i * 15)]), GREY, k=3)
    Pr.ink(M([(60, 196), (402, 196), (402, 238), (60, 238)]), BLACK, k=2)
    Pr.ink(M([(60, 252), (156, 252), (156, 262), (60, 262)]), RED, reg=(0.7, 0), k=1)
    Pr.ink(SM([(470, 60), (470, 300)], 1.4, 0, False), BLACK, k=2)
    Pr.ink(SM([(60, 300), (470, 300)], 1.4, 0, False), BLACK, k=2)
    # 黑色粗横条：Clawd 站在上面
    Pr.ink(M([(60, 452), (770, 452), (770, 472), (60, 472)]), BLACK, k=2)
    # 纸壳小屋：浅蓝方块 + 黑色三角屋顶，门是镂空
    Pr.ink(M([(150, 340), (300, 340), (300, 452), (150, 452)]), BLUE, reg=(0, 0.6), k=0)
    Pr.ink(M([(138, 340), (225, 270), (312, 340)]), BLACK, k=2)
    Pr.knockout(M([(204, 388), (246, 388), (246, 452), (204, 452)]))
    win = M([(258, 356), (284, 356), (284, 382), (258, 382)])
    Pr.knockout(win)
    Pr.ink(win, RED, reg=(0.7, 0), k=1)
    # 正在滑进网格的三个白圆（踏脚石）
    for (x, y, rr) in [(716, 528, 40), (838, 552, 28), (928, 572, 18)]:
        Pr.knockout(M(circle(x, y, rr, 72)))
    # 被推歪的红色方块，Clawd 正把它推回网格
    sq = rot_pts([(552, 398), (608, 398), (608, 452), (552, 452)], 552, 452, -8)
    Pr.knockout(M(sq))
    Pr.ink(M(sq), RED, reg=(0.7, 0), k=1)
    # Clawd：一块橙色专色
    cm, em = clawd(440, 452, 12)
    Pr.knockout(cm)
    Pr.ink(cm, ORANGE, k=3, density=1.0)
    Pr.ink(em, BLACK, k=2)
    # 画面右下角细小的套准标记
    for (x, y) in [(1090, 600)]:
        Pr.ink(SM([(x - 9, y), (x + 9, y)], 0.8, 0, False), BLACK, k=2)
        Pr.ink(SM([(x, y - 9), (x, y + 9)], 0.8, 0, False), BLACK, k=2)
        ring = np.clip(M(circle(x, y, 5.5, 40)) - M(circle(x, y, 4.6, 40)), 0, 1)
        Pr.ink(ring, BLACK, k=2)
    return np.clip(downsample(Pr.img, K), 0, 1)


def arch(cx, base, w, h):
    """拱门形：矩形 + 半圆顶。"""
    r = w / 2
    return [(cx - r, base), (cx - r, base - h + r)] + [(cx + r * math.cos(t), base - h + r - r * math.sin(t)) for t in np.linspace(math.pi, 0, 60)] + [(cx + r, base)]


def render_b():
    """01-b：同心拱门舞台，中央白色大圆当追光；Clawd 穿浅蓝长裙配披风袖，握白色麦克风。"""
    Pr = Press()
    CXs, BASE = 560, 560
    cols = [BLUE, None, RED, None, BLACK, None, BLUE]
    for i, (w, h) in enumerate([(980, 560), (860, 500), (740, 440), (630, 380), (520, 322), (420, 268), (330, 220)]):
        m = M(arch(CXs, BASE, w, h))
        c = cols[i]
        if c is None:
            Pr.knockout(m)
        else:
            Pr.knockout(m)
            Pr.ink(m, c, reg=(0.7 * (i % 2), 0.6 * ((i + 1) % 2)), k=i)
    # 追光：白色大圆（镂空），边缘是一圈网点
    cx, cy, r = CXs, 330, 150
    d = np.hypot(X / K - cx, Y / K - cy)
    ring = smooth(r + 22, r + 2, d) * smooth(r - 2, r + 2, d)
    Pr.knockout(M(circle(cx, cy, r, 160)))
    Pr.ink(halftone(0.55 - np.clip((d - r) / 22, 0, 1) * 0.45, 3.6, 15) * ring, BLACK, k=2, density=0.85)
    # 舞台：黑色横条与台口细线
    Pr.ink(M([(40, BASE), (1080, BASE), (1080, BASE + 20), (40, BASE + 20)]), BLACK, k=2)
    Pr.ink(SM([(40, BASE + 40), (1080, BASE + 40)], 1.2, 0, False), BLACK, k=2)
    for i, ln in enumerate([180, 140, 160]):
        Pr.ink(M([(60, 604 + i * 0), (60 + ln, 604), (60 + ln, 610), (60, 610)]) if i == 0 else M([(900, 590 + i * 12), (900 + ln * 0.8, 590 + i * 12), (900 + ln * 0.8, 596 + i * 12), (900, 596 + i * 12)]), GREY, k=3)
    # Clawd：浅蓝长裙 + 披风袖 + 白色麦克风
    FX, FY, S = CXs, BASE, 17
    arms = [rigid_arm("L", 0), rigid_arm("R", 34)]
    cm, em = clawd(FX, FY, S, arms)
    gown = [(-0.2, 5.4), (16.2, 5.4), (16.9, 8.4), (18.2, 12.0), (-2.2, 12.0), (-0.9, 8.4)]
    capeL = [(-1.0, 3.0), (0.2, 3.0), (0.4, 9.0), (-1.8, 11.6), (-3.6, 11.0)]
    capeR = [(15.8, 3.0), (17.2, 2.8), (19.8, 10.6), (18.0, 11.6), (15.6, 9.0)]
    Pr.knockout(cm)
    Pr.ink(cm, ORANGE, k=3)
    gm = garment(gown, FX, FY, S, arms=arms, name="gown")
    LB = np.array([0.62, 0.80, 0.93])
    Pr.knockout(gm)
    Pr.ink(gm, LB, reg=(0, 0.6), k=0)
    for cp in (capeL, capeR):
        Pr.ink(garment(cp, FX, FY, S, name="cape"), LB, reg=(0, 0.6), k=1, density=0.75)
    for gx in (3.0, 6.5, 9.5, 13.0):                       # 长裙的几道细褶线
        Pr.ink(SM(g2p([(gx, 7.0), (gx + (gx - 8) * 0.12, 11.8)], FX, FY, S), 1.0, 0, False), BLUE * 0.8, k=0)
    Pr.ink(em, BLACK, k=2)
    # 白色手持麦克风：手柄穿过右臂，网罩在脸侧；画完再把手臂块印上去，像握住
    from kit import mic_held
    from common4 import rect_pts as _rp
    mc = mic_held(arms[1], FX, FY, S, lean=12)
    for part in ("handle", "head"):
        Pr.knockout(mc[part])
        Pr.ink(outline_m(mc[part]), BLACK, k=2)
    Pr.ink(mc["grille"], BLACK, k=2, density=0.55)
    Pr.ink(mc["band"], BLACK, k=2)
    am = M(g2p(arms[1], FX, FY, S))
    Pr.knockout(am)
    Pr.ink(am, ORANGE, k=3)
    return np.clip(downsample(Pr.img, K), 0, 1)


def outline_m(m):
    from kit import outline
    return outline(m, int(1.2 * K))


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = render()
    frame.save(frame.compose(painting, (0.78, 0.82, 0.86), "还有什么更好的"), os.path.join(out_dir, "01_还有什么更好的_瑞士海报.png"))
    frame.save(painting, os.path.join(out_dir, "01_画心.png"))
    pb = render_b()
    frame.save(frame.compose(pb, (0.78, 0.82, 0.86), "还有什么更好的"), os.path.join(out_dir, "01b_还有什么更好的_同心拱门舞台.png"))
    frame.save(pb, os.path.join(out_dir, "01b_画心.png"))
