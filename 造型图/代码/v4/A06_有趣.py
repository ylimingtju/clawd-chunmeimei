"""06 有趣 · 波普网点漫画 · 151 BPM，大动作每 2 拍（0:59.23–1:08.77）。拍号 b 从 05 结束算起。
E  b0–4   一面镜子从右往左扫过《爵士》舞台，镜子里映出的是漫画网点画法。Clawd 往右走进镜子，
          跨过镜面的部分变成网点画法、换上黄裙和珍珠链；镜子扫完，整幅画变成镜面长廊。
a  b4–14  镜面长廊：镜子里每层镜像每 2 拍换一个动作（挥手、歪身、眨眼）。Clawd 走回镜前，和最近的镜像隔着镜面击掌；
          镜像们从远到近接力传来一颗珍珠，Clawd 接住。
M  b14–18 Clawd 跳过分格之间的白色分格线，跳进隔壁那一格；跳过去的部分换上灰色不对称上衣配长裙，
          隔壁那一格就是光管舞台，它一路张开成整幅。
b  b18–24 光管画成利希滕斯坦式的笔触，Clawd 在立式麦克风前张开双臂。画框里慢慢上摇。"""
import math

import numpy as np

from kit import (K, PW, PH, X, Y, M, SM, circle, clawd, garment, g2p, rigid_arm, check_proportions, gauss, smooth, over,
                 downsample, outline, dilate, mic_px)
import anim_core as A
import frame
import S06_波普网点 as S6
from S06_波普网点 import Comic, dots, pop_clawd, shine, brushstroke, PAPER, YEL, RED, BLUE, CYAN, INK, ORANGE, FIB
import A05_我管不住我自己 as A5

FPS = 30
T0 = A5.T0 + 13 * A5.BEAT
BEAT = 60 / 151
WALL5, LABEL5 = A5.WALL5, A5.LABEL5
WALL6, LABEL6 = (0.95, 0.80, 0.22), "有趣"
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 24 * BEAT) * FPS)
PX0, PY0, PX1, PY1 = 18, 18, 850, 612
QX0, QX1 = 872, 1102
MX0, MY0, MX1, MY1 = 52, 64, 420, 604
MVP = (250, 300)
PEARL_HAND = (424, 466)
_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


# ---------- 长廊静态层 ----------
def corridor_static():
    """不含任何 Clawd（主角和镜像）与珍珠的长廊：两格、红墙蓝地、镜框、镜中套镜的框。"""
    if "corr" in _CACHE:
        return _CACHE["corr"]
    C = Comic()
    VP = (440, 300)
    inside = M([(PX0, PY0), (PX1, PY0), (PX1, PY1), (PX0, PY1)])
    C.img = over(C.img, dots(RED, 9.0, 0.30, 45), inside)
    floor = M([(PX0, 470), (PX1, 470), (PX1, PY1), (PX0, PY1)])
    C.img = over(C.img, dots(BLUE, 9.0, 0.34, 45), floor)
    C.ink(SM([(PX0, 470), (PX1, 470)], 2.4, 0, False))
    for x in range(-600, 1500, 120):
        C.ink(SM([(x, PY1), (VP[0] + (x - VP[0]) * 0.36, 470)], 1.6, 0, False) * floor)
    C.fill(M([(MX0 - 16, MY0 - 16), (MX1 + 16, MY0 - 16), (MX1 + 16, MY1 + 16), (MX0 - 16, MY1 + 16)]), YEL, 3.6)
    mirror = M([(MX0, MY0), (MX1, MY0), (MX1, MY1), (MX0, MY1)])
    C.fill(mirror, dots(BLUE, 7.0, 0.22, 45, bg=CYAN * 0.6 + PAPER * 0.4), 3.0)
    levels = []
    for k in range(6):
        sk = 0.74 ** (k + 1)
        r = (MVP[0] + (MX0 - MVP[0]) * sk, MVP[1] + (MY0 - MVP[1]) * sk, MVP[0] + (MX1 - MVP[0]) * sk, MVP[1] + (MY1 - MVP[1]) * sk)
        levels.append((k, sk, r))
    for k, sk, (a, b, c, d) in levels[::-1]:
        fm = M([(a, b), (c, b), (c, d), (a, d)])
        C.img = over(C.img, YEL * 0.85 + CYAN * 0.15, np.clip(dilate(fm, int(10 * sk * K)) - fm, 0, 1) * mirror)
        C.ink(outline(fm, max(1, int(2.6 * sk * K))) * mirror)
    q = M([(QX0, PY0), (QX1, PY0), (QX1, PY1), (QX0, PY1)])
    C.img = over(C.img, dots(BLUE, 8.0, 0.40, 45, bg=np.array([0.08, 0.08, 0.12])), q)
    for i, (xa, xb) in enumerate([(900, 940), (950, 1010), (1030, 1060), (1070, 1090)]):
        tube = M([(xa, PY0), (xa + 14, PY0), (xb + 14, PY1), (xb, PY1)])
        C.fill(tube * q, np.array([1.0, 0.97, 0.80]), 2.0)
        C.ink(SM([(xa + 7, PY0 + 40), (xb + 7, PY1 - 60)], 1.2, 0, False) * q, 0.6)
    panels = np.clip(inside + q, 0, 1)
    C.img = over(C.img, PAPER, 1 - panels)
    for (a, b, c, d) in [(PX0, PY0, PX1, PY1), (QX0, PY0, QX1, PY1)]:
        C.ink(outline(M([(a, b), (c, b), (c, d), (a, d)]), int(3.2 * K)))
    _CACHE["corr"] = (C.img.copy(), levels, mirror)
    return _CACHE["corr"]


def stage_static():
    if "stage" in _CACHE:
        return _CACHE["stage"]
    C = Comic()
    inside = M([(18, 18), (1102, 18), (1102, 612), (18, 612)])
    C.img = over(C.img, dots(BLUE, 8.0, 0.42, 45, bg=np.array([0.07, 0.07, 0.12])), inside)
    for i, x in enumerate([120, 250, 380, 740, 870, 1000]):
        brushstroke(C, [(x - 30, 30), (x - 6, 220), (x + 10, 420), (x + 30, 600)], 34, np.array([1.0, 0.97, 0.80]) if i % 2 == 0 else YEL)
    fl = M([(18, 520), (1102, 520), (1102, 612), (18, 612)])
    C.img = over(C.img, dots(RED, 8.0, 0.30, 45, bg=PAPER), fl)
    C.ink(SM([(18, 520), (1102, 520)], 3.0, 0, False))
    C.fill(M(circle(560, 560, 240, 120, 40)), dots(YEL, 6.0, 0.42, 45, bg=PAPER), 2.4)
    C.fill(M(circle(742, 568, 40, 48, 8)), INK, 0)
    C.fill(M([(739, 446), (745, 446), (745, 566), (739, 566)]), INK, 0)
    C.fill(M([(734, 438), (750, 438), (750, 452), (734, 452)]), INK, 0)
    mc = mic_px(742, 444, -0.88, -0.30, 15)
    C.fill(mc["handle"], INK, 0)
    C.fill(mc["band"], np.array([0.55, 0.55, 0.58]), 2.0)
    C.fill(mc["head"], dots(INK, 4.0, 0.30, 45, bg=np.array([0.78, 0.78, 0.81])), 2.4)
    C.ink(mc["grille"] * 0.6)
    C.img = over(C.img, PAPER, 1 - inside)
    C.ink(outline(inside, int(3.2 * K)))
    _CACHE["stage"] = C.img.copy()
    return _CACHE["stage"]


def grey_clawd(C, FX, FY, S, arms, ang=0.0, mask=None):
    mm = 1.0 if mask is None else mask
    cm, em = clawd(FX, FY, S, arms, ang)
    shade = dots(RED * 0.85 + ORANGE * 0.15, 6.0, 0.30, 45, bg=ORANGE)
    rt = smooth(FX + 2 * S, FX + 4 * S, X / K)[..., None]
    C.fill(cm * mm, ORANGE[None, None] * (1 - rt) + shade * rt, 3.6 * S / 16)
    grey = dots(INK, 5.0, 0.26, 45, bg=np.array([0.80, 0.80, 0.82]))
    top = garment([(-0.2, 5.4), (16.2, 5.4), (16.2, 8.2), (-0.2, 7.0)], FX, FY, S, ang, arms=arms, name="top")
    sh = garment([(13.0, 5.4), (16.2, 2.0), (16.2, 3.8), (14.8, 5.4)], FX, FY, S, ang, name="shoulder")
    C.fill(np.clip(top + sh, 0, 1) * mm, grey, 3.0 * S / 16)
    skirt = garment([(-0.2, 7.0), (16.2, 8.2), (17.0, 12.0), (-1.0, 12.0)], FX, FY, S, ang, name="skirt") * mm
    C.fill(skirt, dots(INK, 5.0, 0.36, 45, bg=np.array([0.62, 0.62, 0.66])), 3.0 * S / 16)
    for gx in (2.5, 5.5, 8.5, 11.5, 14.5):
        C.ink(SM(g2p([(gx, 8.0), (gx + (gx - 8) * 0.12, 11.8)], FX, FY, S, ang), 2.2 * S / 16, 0, False) * mm)
    C.ink(em * mm)


def masked_pop(C, mask, *args, **kw):
    old = C.img.copy()
    pop_clawd(C, *args, **kw)
    C.img = old * (1 - mask[..., None]) + C.img * mask[..., None]


# ---------- 镜像的动作（每 2 拍换一次） ----------
POSES = [
    lambda: dict(arms=[(-1, 3, 0, 5), rigid_arm("R", 32)]),           # 挥手
    lambda: dict(arms=[rigid_arm("L", 40), rigid_arm("R", 30)]),
    lambda: dict(arms=None, eyes="left"),                              # 眨眼
    lambda: dict(arms=[rigid_arm("L", 30), (16, 3, 17, 5)]),
    lambda: dict(arms=[rigid_arm("L", 30), rigid_arm("R", 30)]),
    lambda: dict(arms=[rigid_arm("L", -30), rigid_arm("R", -30)]),
]


def reflections(C, b, levels, mirror):
    step = int(max(0.0, b - 4.0) // 2)
    lean_all = 0.0
    for k, sk, (a, bb, c, d) in levels[::-1]:
        fm = M([(a, bb), (c, bb), (c, d), (a, d)])
        S = 12.5 * sk
        fx = a + (c - a) * (0.27 if k % 2 == 0 else 0.73)
        fy = d - 4 * sk
        p_ = POSES[(k + step) % len(POSES)]()
        # 歪身：整只镜像左右歪一下（每 2 拍一次）
        old = C.img.copy()
        pop_clawd(C, fx, fy, S, p_.get("arms"), p_.get("eyes", "open"), p_.get("back", False), mirror=True, tint=(CYAN, 0.20 + 0.06 * k))
        C.img = old * (1 - fm[..., None]) + C.img * fm[..., None]


def pearl(C, pos, r=13):
    pm = M(circle(pos[0], pos[1], r, 48))
    C.fill(pm, np.ones(3) * 0.99, 2.4 * r / 13)
    C.img = over(C.img, dots(CYAN, 4.5, 0.32, 45, bg=PAPER), M(circle(pos[0] + 3 * r / 13, pos[1] + 4 * r / 13, 8 * r / 13, 32)) * pm)
    C.img = over(C.img, PAPER, M(circle(pos[0] - 4 * r / 13, pos[1] - 5 * r / 13, 3.5 * r / 13, 20)))


def relay_pos(b, levels):
    """珍珠从最深的一层镜像一层层往前跳，最后到镜面上主角钳子前。"""
    pts = []
    for k, sk, (a, bb, c, d) in levels[::-1]:
        fx = a + (c - a) * (0.27 if k % 2 == 0 else 0.73)
        pts.append(((fx + 9 * 12.5 * sk, d - 4 * sk - 9 * 12.5 * sk), 13 * sk + 3))
    pts.append((PEARL_HAND, 13))
    u = A.seg(b, 10.0, 12.0) * (len(pts) - 1)
    i = min(int(u), len(pts) - 2)
    f = u - i
    (p0, r0), (p1, r1) = pts[i], pts[i + 1]
    x, y = A.hop(p0, p1, A.ease_io(f), 30 + 20 * i)
    return (x, y), A.lerp(r0, r1, f)


# ---------- 一帧 ----------
def corridor_frame(b, hero=True, hero_pose=None):
    base, levels, mirror = corridor_static()
    C = Comic()
    C.img = base.copy()
    reflections(C, b, levels, mirror)
    # 最近的镜像：贴在镜面右边，和主角做同一个动作（镜像）
    hi5 = A.seg(b, 7.0, 8.0) if b < 10 else 1.0
    nr_arms = [(-1, 3, 0, 5), rigid_arm("R", 30)] if b >= 7.0 else [rigid_arm("L", 30), rigid_arm("R", 30)]
    masked_pop(C, mirror, 316, 596, 12.5, nr_arms, mirror=True, tint=(CYAN, 0.16))
    for (a_, b_, ww) in [(0.12, 0.48, 0.09), (0.28, 0.62, 0.035)]:
        w = MX1 - MX0
        C.img = over(C.img, PAPER, shine(MX0 + w * a_, MY0 + 4, MX0 + w * (a_ - 0.22), MY1 - 4, w * ww) * mirror * 0.9)
    if b >= 10.0:
        pos, r = relay_pos(b, levels)
        if b >= 12.0:
            pos = PEARL_HAND
        pearl(C, pos, r)
    if hero:
        fx, fy, S, arms = hero_pose or hero_at(b)
        pop_clawd(C, fx, fy, S, arms)
        if 7.8 <= b < 9.0:                                 # 击掌的冲击线
            for a in np.linspace(-1.0, 1.0, 5):
                C.ink(SM([(424 + 24 * math.cos(a - math.pi / 2), 466 + 24 * math.sin(a - math.pi / 2)),
                          (424 + 44 * math.cos(a - math.pi / 2), 466 + 44 * math.sin(a - math.pi / 2))], 2.6, 0, True))
    return C.img


def hero_at(b):
    """a 段主角：b4–6 从右边走回镜前，b8 击掌（左钳贴到镜面），b12 接住珍珠。"""
    S = 14
    x = A.lerp(700, 570, A.ease_io(A.seg(b, 4.0, 6.0)))
    fy = 596 - 6 * abs(math.sin(math.pi * 2 * A.seg(b, 4.0, 6.0)))
    if b < 7.0:
        arms = [rigid_arm("L", 30), rigid_arm("R", 30)]
    else:
        arms = [rigid_arm("L", 32), (16, 3, 17, 5)]
    return x, fy, S, arms


def painting(t, b):
    if b < 4.0:
        # E：镜子从右往左扫过《爵士》舞台
        edge = A.lerp(1135, -15, A.ease_io(A.seg(b, 0.3, 3.8)))
        jazz = A5.sticker(A5.jazz_static().copy())
        walk = A.ease_io(A.seg(b, 0.0, 2.6))
        fx = A.lerp(560, 700, walk)
        fy = A.lerp(548, 596, walk) - 5 * abs(math.sin(math.pi * 3 * walk))
        S = A.lerp(18, 14, walk)
        right = smooth(edge - 0.5, edge + 0.5, X / K)
        A5.sequin_clawd(jazz, fx, fy, S, 13.0 + b * BEAT / A5.BEAT, mask=(1 - right), mic=b < 1.0)
        jimg = jazz.img * (1 + 0.025 * A5.FIBER)[..., None]
        comic = corridor_frame(4.0, hero=False)
        C = Comic()
        C.img = comic
        pop_clawd(C, fx, fy, S, [rigid_arm("L", 30), rigid_arm("R", 30)])
        cimg = jimg * (1 - right[..., None]) + C.img * right[..., None]
        # 镜框：一条黄边 + 黑线，镜面一道斜向反光
        frame_m = np.clip(SM([(edge, -5), (edge, 640)], 6, 0, False), 0, 1)
        cimg = over(cimg, YEL, frame_m)
        cimg = over(cimg, INK, np.clip(SM([(edge - 3, -5), (edge - 3, 640)], 1.4, 0, False) + SM([(edge + 3, -5), (edge + 3, 640)], 1.4, 0, False), 0, 1))
        cimg = over(cimg, PAPER, shine(edge + 40, 0, edge - 90, 630, 30) * right * 0.6)
        x0, x1 = fx - 9 * S, fx + 9 * S
        return cimg, edge < x0
    if b < 14.0:
        return corridor_frame(b), True
    # M：跳过分格线，隔壁那一格张开成光管舞台
    ju = A.seg(b, 14.0, 15.6)
    fx, fy = A.hop((570, 596), (980, 596), A.ease_io(ju), 160)
    S = 14
    open_ = A.ease_io(A.seg(b, 15.2, 16.8))
    qx0 = A.lerp(QX0, -10, open_)
    back = A.ease_io(A.seg(b, 16.4, 17.8))
    fx, fy = A.lerp(fx, 560, back), A.lerp(fy, 560, back) - 50 * math.sin(math.pi * back)
    S = A.lerp(14, 16, back)
    stage = stage_static()
    reveal = smooth(qx0 - 0.5, qx0 + 0.5, X / K)
    img = corridor_frame(14.0, hero=False)
    img = img * (1 - reveal[..., None]) + stage * reveal[..., None]
    if open_ < 1:                                          # 正在张开的格子：左边一道白色分格线 + 黑框
        img = over(img, PAPER, np.clip(SM([(qx0 - 11, -5), (qx0 - 11, 640)], 11, 0, False), 0, 1) * (1 - reveal) * (qx0 > 20))
        img = over(img, INK, SM([(qx0, -5), (qx0, 640)], 3.2, 0, False))
    C = Comic()
    C.img = img
    gutter = 861
    right_m = smooth(gutter - 0.5, gutter + 0.5, X / K) if open_ <= 0 else np.clip(smooth(gutter - 0.5, gutter + 0.5, X / K) + reveal, 0, 1)
    arms = [rigid_arm("L", 32), rigid_arm("R", 32)]
    masked_pop(C, 1 - right_m, fx, fy, S, arms)
    grey_clawd(C, fx, fy, S, arms, mask=right_m)
    return C.img, True


def frame_at(i):
    t = i / FPS
    b = bt(t)
    if b >= 18.0:
        C = Comic()
        C.img = stage_static().copy()
        k = int((b - 18.0) // 2)                         # 每 2 拍张一次手臂：在 32° 和 0° 之间弹
        u = A.back_out(A.seg(b, 18.0 + 2 * k, 18.5 + 2 * k), 2.0)
        a0, a1 = (32, 0) if k % 2 == 0 else (0, 32)
        ang_arm = A.lerp(a0, a1, u)
        grey_clawd(C, 560, 560, 16, [rigid_arm("L", ang_arm), rigid_arm("R", ang_arm)])
        p, sw = C.img, True
    else:
        p, sw = painting(t, b)
    p = np.clip(downsample(p * (1 + 0.025 * FIB)[..., None], K), 0, 1)
    if b >= 18.0:
        u = A.ease_io(A.seg(b, 18.0, 24.0))
        p = A.zoom_painting(p, 1.08, 560, A.lerp(400, 330, u))
    return A.compose(p, WALL6 if sw else WALL5, LABEL6 if sw else LABEL5)


def warm():
    A5.warm()
    corridor_static()
    stage_static()
    for w, l in ((WALL5, LABEL5), (WALL6, LABEL6)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a06_{i:04d}.png")
