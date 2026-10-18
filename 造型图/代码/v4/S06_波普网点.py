"""06 有趣 · 波普网点漫画（利希滕斯坦）· 样张（06-a 镜面长廊）。
画法：漫画分格；原色平涂 + 本戴网点表现中间调；粗黑勾线（先铺黑再盖色，线宽稳定）；镜面用斜向的白色反光条。
镜子里的镜像各做各的动作，最近的那个和主角隔着镜面击掌，把接力传来的珍珠递过去。右边窄格露出下一格的光管舞台。
用法：python S06_波普网点.py 输出目录"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cel import SKIRT, pearls_shapes
from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, catmull, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, value_noise, smooth, over, downsample, ishift, outline, dilate)
from common4 import to_px
import frame

PAPER = np.array([0.97, 0.95, 0.90])
YEL = np.array([1.0, 0.85, 0.12])
RED = np.array([0.89, 0.14, 0.13])
BLUE = np.array([0.13, 0.34, 0.74])
CYAN = np.array([0.45, 0.75, 0.92])
INK = np.array([0.06, 0.06, 0.07])
ORANGE = np.array([202, 124, 94]) / 255
FIB = fbm(PH, PW, 4, 61, 2) - 0.5


def dots(color, spacing=9.0, frac=0.32, angle=45.0, bg=PAPER):
    """本戴网点：在 bg 上排出圆点。frac 为圆点半径 / 间距。"""
    a = math.radians(angle)
    u = (X * math.cos(a) + Y * math.sin(a)) / (spacing * K)
    v = (-X * math.sin(a) + Y * math.cos(a)) / (spacing * K)
    du, dv = u - np.round(u), v - np.round(v)
    d = np.sqrt(du ** 2 + dv ** 2)
    dm = smooth(frac + 0.03, frac - 0.03, d)[..., None]
    return np.asarray(bg, np.float32) * (1 - dm) + np.asarray(color, np.float32) * dm


class Comic:
    def __init__(self):
        self.img = np.ones((PH, PW, 3), np.float32) * PAPER

    def fill(self, m, color, line=3.0):
        """先铺黑（外扩 line），再盖颜色。color 可以是整幅的颜色场（网点）。"""
        if line:
            self.img = over(self.img, INK, dilate(m, int(line * K)))
        col = np.asarray(color, np.float32)
        col = col[None, None] if col.ndim == 1 else col
        self.img = over(self.img, col, m)

    def ink(self, m, op=1.0):
        self.img = over(self.img, INK, m * op)


def shine(x0, y0, x1, y1, w):
    return M([(x0, y0), (x0 + w, y0), (x1 + w, y1), (x1, y1)])


def pop_clawd(C, fx, fy, S, arms, eyes="open", back=False, mirror=False, skirt=True, tint=None):
    a_use = arms
    cm, em = clawd(fx, fy, S, a_use)
    shade = dots(RED * 0.85 + ORANGE * 0.15, 6.0, 0.30, 45, bg=ORANGE)
    rt = smooth(fx + 2 * S, fx + 4 * S, X / K)[..., None] if not mirror else smooth(fx - 2 * S, fx - 4 * S, X / K)[..., None]
    body_col = ORANGE[None, None] * (1 - rt) + shade * rt
    if tint is not None:
        body_col = body_col * (1 - tint[1]) + np.asarray(tint[0]) * tint[1]
    C.fill(cm, body_col, 3.4 * S / 14)
    if skirt:
        sm = garment(SKIRT, fx, fy, S, arms=a_use, name="skirt")
        sk = dots(np.array([0.95, 0.60, 0.05]), 6.0, 0.28, 45, bg=YEL)
        rt2 = smooth(fx + 3 * S, fx + 6 * S, X / K)[..., None] if not mirror else smooth(fx - 3 * S, fx - 6 * S, X / K)[..., None]
        skc = YEL[None, None] * (1 - rt2) + sk * rt2
        if tint is not None:
            skc = skc * (1 - tint[1]) + np.asarray(tint[0]) * tint[1]
        C.fill(sm, skc, 3.4 * S / 14)
        for sx in (3.0, 6.0, 10.0, 13.0):
            C.ink(SM(g2p([(sx, 7.2), (sx * 1.15 - 1.2, 11.6)], fx, fy, S), 2.0 * S / 14, 0, False))
        ps = pearls_shapes(3)
        check_proportions(extra={"pearls": ps})
        for sh in ps[::2]:
            pm = M(g2p(sh, fx, fy, S))
            C.fill(pm, np.array([0.99, 0.98, 0.95]), 1.0)
    if not back:
        if eyes == "open":
            C.ink(em)
        elif eyes == "left":
            C.ink(em * (smooth(fx - 1, fx + 1, X / K)))
            C.ink(M(g2p([(3.2, 3.8), (5.0, 3.8), (5.0, 4.3), (3.2, 4.3)], fx, fy, S)))
    return cm


def render():
    C = Comic()
    VP = (440, 300)
    PX0, PY0, PX1, PY1 = 18, 18, 850, 612            # 主格
    QX0, QX1 = 872, 1102                              # 右边窄格
    inside = M([(PX0, PY0), (PX1, PY0), (PX1, PY1), (PX0, PY1)])
    # ---------- 长廊：红网点墙、蓝网点地面、透视线 ----------
    wall = dots(RED, 9.0, 0.30, 45)
    C.img = over(C.img, wall, inside)
    floor = M([(PX0, 470), (PX1, 470), (PX1, PY1), (PX0, PY1)])
    C.img = over(C.img, dots(BLUE, 9.0, 0.34, 45), floor)
    C.ink(SM([(PX0, 470), (PX1, 470)], 2.4, 0, False))
    for x in range(-600, 1500, 120):
        C.ink(SM([(x, PY1), (VP[0] + (x - VP[0]) * 0.36, 470)], 1.6, 0, False) * floor)
    # ---------- 左边一面大镜子：镜中套镜，镜像一层层往里缩 ----------
    MX0, MY0, MX1, MY1 = 52, 64, 420, 604
    MVP = (250, 300)
    C.fill(M([(MX0 - 16, MY0 - 16), (MX1 + 16, MY0 - 16), (MX1 + 16, MY1 + 16), (MX0 - 16, MY1 + 16)]), YEL, 3.6)
    mirror = M([(MX0, MY0), (MX1, MY0), (MX1, MY1), (MX0, MY1)])
    C.fill(mirror, dots(BLUE, 7.0, 0.22, 45, bg=CYAN * 0.6 + PAPER * 0.4), 3.0)
    poses = [dict(arms=[(-1, 3, 0, 5), rigid_arm("R", 32)]),
             dict(arms=[rigid_arm("L", 40), rigid_arm("R", 30)]),
             dict(arms=None, eyes="left"),
             dict(arms=None, back=True),
             dict(arms=[rigid_arm("L", 30), (16, 3, 17, 5)]),
             dict(arms=[rigid_arm("L", 30), rigid_arm("R", 30)])]
    levels = []
    for k in range(6):
        sk = 0.74 ** (k + 1)
        r = (MVP[0] + (MX0 - MVP[0]) * sk, MVP[1] + (MY0 - MVP[1]) * sk, MVP[0] + (MX1 - MVP[0]) * sk, MVP[1] + (MY1 - MVP[1]) * sk)
        levels.append((k, sk, r))
    base = C.img.copy()
    for k, sk, (a, b, c, d) in levels[::-1]:
        fm = M([(a, b), (c, b), (c, d), (a, d)])
        C.img = over(C.img, YEL * 0.85 + CYAN * 0.15, np.clip(dilate(fm, int(10 * sk * K)) - fm, 0, 1) * mirror)
        C.ink(outline(fm, max(1, int(2.6 * sk * K))) * mirror)
        S = 12.5 * sk
        fx = a + (c - a) * (0.27 if k % 2 == 0 else 0.73)
        fy = d - 4 * sk
        p_ = poses[k % len(poses)]
        old = C.img.copy()
        pop_clawd(C, fx, fy, S, p_.get("arms"), p_.get("eyes", "open"), p_.get("back", False), mirror=True, tint=(CYAN, 0.20 + 0.06 * k))
        C.img = old * (1 - fm[..., None]) + C.img * fm[..., None]
    # 最近的镜像：贴在镜面右边，举起钳子
    old = C.img.copy()
    pop_clawd(C, 316, 596, 12.5, [(-1, 3, 0, 5), rigid_arm("R", 30)], mirror=True, tint=(CYAN, 0.16))
    C.img = old * (1 - mirror[..., None]) + C.img * mirror[..., None]
    for (a_, b_, ww) in [(0.12, 0.48, 0.09), (0.28, 0.62, 0.035)]:
        w = MX1 - MX0
        sh = shine(MX0 + w * a_, MY0 + 4, MX0 + w * (a_ - 0.22), MY1 - 4, w * ww) * mirror
        C.img = over(C.img, PAPER, sh * 0.9)
    # 接力传过来的珍珠，夹在两只钳子之间
    pc = (424, 466)
    pm = M(circle(*pc, 13, 48))
    C.fill(pm, np.ones(3) * 0.99, 2.4)
    C.img = over(C.img, dots(CYAN, 4.5, 0.32, 45, bg=PAPER), M(circle(pc[0] + 3, pc[1] + 4, 8, 32)) * pm)
    C.img = over(C.img, PAPER, M(circle(pc[0] - 4, pc[1] - 5, 3.5, 20)))
    # ---------- 主角：隔着镜面和最近的镜像击掌 ----------
    pop_clawd(C, 570, 596, 14, [rigid_arm("L", 32), (16, 3, 17, 5)])
    for a in np.linspace(-1.0, 1.0, 5):
        C.ink(SM([(424 + 24 * math.cos(a - math.pi / 2), 466 + 24 * math.sin(a - math.pi / 2)),
                  (424 + 44 * math.cos(a - math.pi / 2), 466 + 44 * math.sin(a - math.pi / 2))], 2.6, 0, True))
    # ---------- 右边窄格：下一格的光管舞台 ----------
    q = M([(QX0, PY0), (QX1, PY0), (QX1, PY1), (QX0, PY1)])
    C.img = over(C.img, dots(BLUE, 8.0, 0.40, 45, bg=INK * 0 + np.array([0.08, 0.08, 0.12])), q)
    for i, (xa, xb) in enumerate([(900, 940), (950, 1010), (1030, 1060), (1070, 1090)]):
        tube = M([(xa, PY0), (xa + 14, PY0), (xb + 14, PY1), (xb, PY1)])
        C.fill(tube * q, np.array([1.0, 0.97, 0.80]), 2.0)
        C.ink(SM([(xa + 7, PY0 + 40), (xb + 7, PY1 - 60)], 1.2, 0, False) * q, 0.6)
    # 分格的白色间隔与黑框
    panels = np.clip(inside + q, 0, 1)
    C.img = over(C.img, PAPER, 1 - panels)
    for (a, b, c, d) in [(PX0, PY0, PX1, PY1), (QX0, PY0, QX1, PY1)]:
        C.ink(outline(M([(a, b), (c, b), (c, d), (a, d)]), int(3.2 * K)))
    C.img *= (1 + 0.025 * FIB)[..., None]
    return C.img


def brushstroke(C, pts, w, color):
    """利希滕斯坦《笔触》：一笔粗笔触，黑色勾边，里面几道表示刷毛的细黑线。"""
    from common4 import taper_stroke
    m = M(taper_stroke(catmull(pts, 60), w, w * 0.8, w * 1.1), 0.0)
    C.fill(m, color, 3.0)
    for off in (-0.25, 0.1, 0.32):
        line = [(x + off * w * 0.0, y + off * w) for x, y in catmull(pts, 40)]
        C.ink(SM(line, 1.4, 0.8, False) * m, 0.8)
    return m


def render_b():
    """06-b：光管舞台——光管画成利希滕斯坦式的笔触；Clawd 穿灰色不对称上衣配长裙，站在立式麦克风前张开双臂。"""
    C = Comic()
    PX0, PY0, PX1, PY1 = 18, 18, 1102, 612
    inside = M([(PX0, PY0), (PX1, PY0), (PX1, PY1), (PX0, PY1)])
    C.img = over(C.img, dots(BLUE, 8.0, 0.42, 45, bg=np.array([0.07, 0.07, 0.12])), inside)
    for i, x in enumerate([120, 250, 380, 740, 870, 1000]):
        brushstroke(C, [(x - 30, 30), (x - 6, 220), (x + 10, 420), (x + 30, 600)], 34, np.array([1.0, 0.97, 0.80]) if i % 2 == 0 else YEL)
    # 舞台地面与追光
    fl = M([(PX0, 520), (PX1, 520), (PX1, PY1), (PX0, PY1)])
    C.img = over(C.img, dots(RED, 8.0, 0.30, 45, bg=PAPER), fl)
    C.ink(SM([(PX0, 520), (PX1, 520)], 3.0, 0, False))
    spot = M(circle(560, 560, 240, 120, 40))
    C.fill(spot, dots(YEL, 6.0, 0.42, 45, bg=PAPER), 2.4)
    # Clawd：灰色不对称上衣 + 长裙（灰色用黑网点表现）
    FX, FY, S = 560, 560, 16
    arms = [rigid_arm("L", 32), rigid_arm("R", 32)]
    cm, em = clawd(FX, FY, S, arms)
    shade = dots(RED * 0.85 + ORANGE * 0.15, 6.0, 0.30, 45, bg=ORANGE)
    rt = smooth(FX + 2 * S, FX + 4 * S, X / K)[..., None]
    C.fill(cm, ORANGE[None, None] * (1 - rt) + shade * rt, 3.6)
    grey = dots(INK, 5.0, 0.26, 45, bg=np.array([0.80, 0.80, 0.82]))
    top = garment([(-0.2, 5.4), (16.2, 5.4), (16.2, 8.2), (-0.2, 7.0)], FX, FY, S, arms=arms, name="top")
    sh = garment([(13.0, 5.4), (16.2, 2.0), (16.2, 3.8), (14.8, 5.4)], FX, FY, S, name="shoulder")
    C.fill(np.clip(top + sh, 0, 1), grey, 3.0)
    skirt = garment([(-0.2, 7.0), (16.2, 8.2), (17.0, 12.0), (-1.0, 12.0)], FX, FY, S, name="skirt")
    C.fill(skirt, dots(INK, 5.0, 0.36, 45, bg=np.array([0.62, 0.62, 0.66])), 3.0)
    for gx in (2.5, 5.5, 8.5, 11.5, 14.5):
        C.ink(SM(g2p([(gx, 8.0), (gx + (gx - 8) * 0.12, 11.8)], FX, FY, S), 2.2, 0, False))
    C.ink(em)
    # 立式麦克风：圆底座、直立杆、杆顶的夹子，话筒斜指向 Clawd 脸侧
    from kit import mic_px
    C.fill(M(circle(742, 568, 40, 48, 8)), INK, 0)
    C.fill(M([(739, 446), (745, 446), (745, 566), (739, 566)]), INK, 0)
    C.fill(M([(734, 438), (750, 438), (750, 452), (734, 452)]), INK, 0)
    mc = mic_px(742, 444, -0.88, -0.30, 15)
    C.fill(mc["handle"], INK, 0)
    C.fill(mc["band"], np.array([0.55, 0.55, 0.58]), 2.0)
    C.fill(mc["head"], dots(INK, 4.0, 0.30, 45, bg=np.array([0.78, 0.78, 0.81])), 2.4)
    C.ink(mc["grille"] * 0.6)
    C.ink(outline(inside, int(3.2 * K)))
    C.img *= (1 + 0.025 * FIB)[..., None]
    return np.clip(downsample(C.img, K), 0, 1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    painting = np.clip(downsample(render(), K), 0, 1)
    frame.save(frame.compose(painting, (0.95, 0.80, 0.22), "有趣"), os.path.join(out_dir, "06_有趣_波普网点.png"))
    frame.save(painting, os.path.join(out_dir, "06_画心.png"))
    pb = render_b()
    frame.save(frame.compose(pb, (0.95, 0.80, 0.22), "有趣"), os.path.join(out_dir, "06b_有趣_光管舞台.png"))
    frame.save(pb, os.path.join(out_dir, "06b_画心.png"))
