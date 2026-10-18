"""01 还有什么更好的 · 瑞士国际主义海报 · 82 BPM（0:08.00–0:17.51，帧 240–525）。
拍号 b 从 0:08.00 算起，一拍 = 60/82 秒。画心里的动作「一拍二」（每 2 帧换一次），画框里的推镜按 30fps。
E  b0–2   主角从万花筒跳下来：万花筒先收成圆盘，再一边缩小一边退到右上角，最后被红墨盖满，成了海报上的红色大圆；
          主角越出圆盘边缘的部分立刻变成海报里的扁平橙色、脱掉封面造型；它完全出圆的那一拍墙色和展签一起切换。
a  b2–8   扁平的纸盒按拍点打开（b2 盒盖翻开、b3 墙面立起）；Clawd 跳进盒子伸了个懒腰，跳出来后盒盖合成屋顶、
          墙升到整屋高；三个白圆按半拍滑进网格；Clawd 把歪掉的红色方块推回网格（b6 吸到位），踩着三个圆跳过去。
M  b8–10  Clawd 拉了一下挂在网格右侧的一根竖线（末端是白色的拉环，其实就是麦克风），竖线向左扫过整张画面，
          扫过的部分变成同心拱门舞台；Clawd 被扫到时换上浅蓝长裙配披风袖，两跳回到舞台中央，手里留着那只白色麦克风。
b  b10–13 同心拱门舞台，白色大圆追光；Clawd 握着麦克风轻轻摇晃，画框里慢慢推近。"""
import math

import numpy as np

from kit import K, PW, PH, X, Y, M, SM, circle, rot_pts, clawd, garment, g2p, rigid_arm, check_proportions, mic_px, smooth, downsample
import anim_core as A
import frame
import S01_瑞士海报 as S1
from S01_瑞士海报 import Press, halftone, arch, outline_m, PAPER, RED, BLUE, BLACK, GREY, ORANGE
import A00_开场 as A0

FPS = 30
T0 = 8.0
BEAT = 60 / 82
WALL0, LABEL0 = A0.WALL, A0.LABEL
WALL1, LABEL1 = (0.78, 0.82, 0.86), "还有什么更好的"
LB = np.array([0.62, 0.80, 0.93])

BAR_Y = 452
BOX = (110, 340)                        # 纸盒左右边
SQ0 = [(552, 398), (608, 398), (608, 452), (552, 452)]
CIRCLES = [(716, 528, 40), (838, 552, 28), (928, 572, 18)]
CORD_X, GRIP_Y = 1040, 440
RED_C, RED_R = (850, 222), 200
SLOT = (A0.CX + A0.SLOT_RF * math.cos(math.radians(120)), A0.CY + A0.SLOT_RF * math.sin(math.radians(120)))   # 00 结束时主角的脚（圆环转了 30°）
STAGE_C, STAGE_BASE = 560, 560

_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


# ---------- 静态层（每个进程只算一次；渲染前在主进程里预热，子进程直接继承） ----------
def poster_static():
    if "poster" not in _CACHE:
        Pr = Press()
        Pr.ink(M([(540, 300), (1130, 300), (1130, 640), (540, 640)]), BLUE, reg=(0, 0.6), k=0)
        cx, cy, r = 236, 556, 66
        d = np.hypot(X / K - cx, Y / K - cy) / r
        inside = smooth(1.0, 0.985, d)
        nz = np.sqrt(np.clip(1 - d ** 2, 0, 1))
        nx, ny = (X / K - cx) / r, (Y / K - cy) / r
        lam = np.clip(-0.5 * nx - 0.6 * ny + 0.62 * nz, 0, 1)
        val = np.clip(0.92 - 0.75 * lam - 0.6 * lam ** 30, 0.03, 1) * inside
        Pr.ink(halftone(val, 4.2, 15) * inside, BLACK, k=2)
        for i, ln in enumerate([270, 236, 258, 190, 248, 120, 214]):
            Pr.ink(M([(60, 64 + i * 15), (60 + ln, 64 + i * 15), (60 + ln, 71 + i * 15), (60, 71 + i * 15)]), GREY, k=3)
        Pr.ink(M([(60, 196), (402, 196), (402, 238), (60, 238)]), BLACK, k=2)
        Pr.ink(M([(60, 252), (156, 252), (156, 262), (60, 262)]), RED, reg=(0.7, 0), k=1)
        Pr.ink(SM([(470, 60), (470, 300)], 1.4, 0, False), BLACK, k=2)
        Pr.ink(SM([(60, 300), (470, 300)], 1.4, 0, False), BLACK, k=2)
        Pr.ink(M([(60, BAR_Y), (770, BAR_Y), (770, BAR_Y + 20), (60, BAR_Y + 20)]), BLACK, k=2)
        x, y = 1090, 600
        Pr.ink(SM([(x - 9, y), (x + 9, y)], 0.8, 0, False), BLACK, k=2)
        Pr.ink(SM([(x, y - 9), (x, y + 9)], 0.8, 0, False), BLACK, k=2)
        Pr.ink(np.clip(M(circle(x, y, 5.5, 40)) - M(circle(x, y, 4.6, 40)), 0, 1), BLACK, k=2)
        _CACHE["poster"] = Pr.img
    return _CACHE["poster"]


def stage_static():
    if "stage" not in _CACHE:
        Pr = Press()
        cols = [BLUE, None, RED, None, BLACK, None, BLUE]
        for i, (w, h) in enumerate([(980, 560), (860, 500), (740, 440), (630, 380), (520, 322), (420, 268), (330, 220)]):
            m = M(arch(STAGE_C, STAGE_BASE, w, h))
            Pr.knockout(m)
            if cols[i] is not None:
                Pr.ink(m, cols[i], reg=(0.7 * (i % 2), 0.6 * ((i + 1) % 2)), k=i)
        cx, cy, r = STAGE_C, 330, 150
        d = np.hypot(X / K - cx, Y / K - cy)
        ring = smooth(r + 22, r + 2, d) * smooth(r - 2, r + 2, d)
        Pr.knockout(M(circle(cx, cy, r, 160)))
        Pr.ink(halftone(0.55 - np.clip((d - r) / 22, 0, 1) * 0.45, 3.6, 15) * ring, BLACK, k=2, density=0.85)
        Pr.ink(M([(40, STAGE_BASE), (1080, STAGE_BASE), (1080, STAGE_BASE + 20), (40, STAGE_BASE + 20)]), BLACK, k=2)
        Pr.ink(SM([(40, STAGE_BASE + 40), (1080, STAGE_BASE + 40)], 1.2, 0, False), BLACK, k=2)
        Pr.ink(M([(60, 604), (240, 604), (240, 610), (60, 610)]), GREY, k=3)
        for i, ln in enumerate([140, 160]):
            Pr.ink(M([(900, 602 + i * 12), (900 + ln * 0.8, 602 + i * 12), (900 + ln * 0.8, 608 + i * 12), (900, 608 + i * 12)]), GREY, k=3)
        _CACHE["stage"] = Pr.img
    return _CACHE["stage"]


def warm():
    poster_static()
    stage_static()
    for w, l in ((WALL0, LABEL0), (WALL1, LABEL1)):
        A.plate(w, l)


# ---------- 海报里的动态元素 ----------
def draw_box(Pr, b):
    """纸盒 → 小屋。b2 盒盖两片翻开；b3 墙面立起到 62；b5 盒盖合成三角屋顶、墙升到 112，开出门窗。"""
    x0, x1 = BOX
    hw = 24 + 38 * A.back_out(A.seg(b, 3.0, 3.3)) + 50 * A.back_out(A.seg(b, 5.0, 5.35))
    top = BAR_Y - hw
    L = math.hypot((x1 - x0) / 2, 70)
    roof_a = math.degrees(math.atan2(70, (x1 - x0) / 2))
    opn = A.back_out(A.seg(b, 2.0, 2.35))
    cls = A.ease_io(A.seg(b, 5.0, 5.35))
    a = lerp_ang(0, 110, opn) if cls <= 0 else lerp_ang(110, roof_a, cls)
    flaps = []
    for side in (-1, 1):
        hx = x0 if side < 0 else x1
        aa = math.radians(a)
        dx, dy = -side * math.cos(aa) * L, -math.sin(aa) * L
        nx, ny = -dy / L * 6, dx / L * 6
        flaps.append([(hx - nx, top - ny), (hx + dx - nx, top + dy - ny), (hx + dx + nx, top + dy + ny), (hx + nx, top + ny)])
    return hw, top, flaps, cls


def lerp_ang(a, b, t):
    return a + (b - a) * t


def box_back(Pr, b):
    hw, top, flaps, cls = draw_box(Pr, b)
    if cls < 0.98:
        for f in flaps:
            Pr.ink(M(f), BLACK, k=2)
    return hw, top, cls


def box_front(Pr, b, hw, top, cls):
    x0, x1 = BOX
    wall = M([(x0, top), (x1, top), (x1, BAR_Y), (x0, BAR_Y)])
    Pr.knockout(wall)
    Pr.ink(wall, BLUE, reg=(0, 0.6), k=0)
    if cls >= 0.98:
        Pr.ink(M([(x0 - 12, top), ((x0 + x1) / 2, top - 70), (x1 + 12, top)]), BLACK, k=2)
    if hw > 100:
        Pr.knockout(M([(204, 388), (246, 388), (246, BAR_Y), (204, BAR_Y)]))
        win = M([(258, 356), (284, 356), (284, 382), (258, 382)])
        Pr.knockout(win)
        Pr.ink(win, RED, reg=(0.7, 0), k=1)


def square_pts(b):
    u = A.seg(b, 5.0, 6.0)
    ang = -8 * (1 - A.back_out(A.seg(b, 5.55, 6.0), 2.4)) if u > 0 else -8
    dx = 16 * A.ease_io(u)
    return rot_pts([(x + dx, y) for x, y in SQ0], 552 + dx, BAR_Y, ang)


def draw_square(Pr, b):
    sq = M(square_pts(b))
    Pr.knockout(sq)
    Pr.ink(sq, RED, reg=(0.7, 0), k=1)


def draw_circles(Pr, b):
    for k, (x, y, r) in enumerate(CIRCLES):
        u = A.ease_out(A.seg(b, 5.0 + k / 3, 5.0 + k / 3 + 0.45), 3)
        if u <= 0:
            continue
        xx = 1200 + r + (x - 1200 - r) * u
        Pr.knockout(M(circle(xx, y, r, 72)))


def draw_mic(Pr, grip, dir_deg, s):
    """白色手持麦克风（海报画法）：dir_deg 为网罩的朝向，0 = 朝上，正值朝画面右侧倾。"""
    a = math.radians(dir_deg)
    mc = mic_px(grip[0], grip[1], math.sin(a), -math.cos(a), s)
    for part in ("handle", "head"):
        Pr.knockout(mc[part])
        Pr.ink(outline_m(mc[part]), BLACK, k=2)
    Pr.ink(mc["grille"], BLACK, k=2, density=0.55)
    Pr.ink(mc["band"], BLACK, k=2)
    return mc


# ---------- Clawd ----------
def clawd_pose(b):
    """返回 (fx, fy, S, ang, arm_l, arm_r, lift)；fy 是脚底（画心坐标）。"""
    S, ang, al, ar, lift = 12.0, 0.0, 0.0, 0.0, 0.0
    if b < 1.6:                                              # E：从空位跳到黑横条上
        u = A.seg(b, 0.3, 1.6)
        x, y = A.hop(SLOT, (440, BAR_Y), A.ease_io(u) * 0.4 + u * 0.6, 150)
        S = A.lerp(A0.SLOT_S, 12.0, A.ease_io(u))
        ang = A.lerp(30, 0, A.ease_io(A.seg(b, 0.3, 1.4)))
        al = ar = 32.0
        if b < 0.3:
            x, y = SLOT
            y += 3 * math.sin(math.pi * A.seg(b, 0.0, 0.3))      # 起跳前往下一蹲（整体位移，不压扁）
        return x, y, S, ang, al, ar, lift
    if b < 2.5:
        al = ar = 32 * (1 - A.ease_io(A.seg(b, 1.6, 1.9)))
        ang = -4 * A.ease_io(A.seg(b, 2.0, 2.3))                   # 歪头看纸盒打开
        return 440, BAR_Y, S, ang, al, ar, lift
    if b < 3.0:                                                # 跳进盒子
        x, y = A.hop((440, BAR_Y), (225, BAR_Y), A.seg(b, 2.5, 3.0), 70)
        return x, y, S, -4 * (1 - A.seg(b, 2.5, 3.0)), 20, 20, lift
    if b < 4.4:                                                # 伸懒腰
        u = A.seg(b, 3.3, 4.3)
        st = math.sin(math.pi * u)
        lean = 6 * math.sin(2 * math.pi * u)
        return 225, BAR_Y - 8 * st, S, lean, 32 * st, 32 * st, lift
    if b < 5.0:                                                # 跳出来
        x, y = A.hop((225, BAR_Y), (440, BAR_Y), A.seg(b, 4.4, 4.9), 100)
        return x, y, S, 0, 20 * math.sin(math.pi * A.seg(b, 4.4, 4.9)), 20 * math.sin(math.pi * A.seg(b, 4.4, 4.9)), lift
    if b < 6.0:                                                # 推方块
        u = A.ease_io(A.seg(b, 5.0, 6.0))
        return 440 + 16 * u, BAR_Y, S, 3 * math.sin(math.pi * u), 0, 10, lift
    stones = [(456, BAR_Y)] + [(x, y - r) for x, y, r in CIRCLES]
    for k, (b0, b1, h) in enumerate([(6.0, 6.5, 120), (6.5, 7.0, 60), (7.0, 7.5, 50)]):
        if b < b1:
            x, y = A.hop(stones[k], stones[k + 1], A.seg(b, b0, b1), h)
            sw = math.sin(math.pi * A.seg(b, b0, b1))
            return x, y, S, -4 * sw, 18 * sw, 18 * sw, lift
    if b < 8.4:                                                # 伸手够拉环，往下一拉
        reach = A.ease_io(A.seg(b, 7.5, 8.0))
        pull = A.ease_io(A.seg(b, 8.0, 8.3)) * (1 - A.ease_io(A.seg(b, 8.3, 8.4)) * 0.3)
        x, y = stones[-1]
        return x, y + 10 * pull, S, 0, 0, 32 * reach, lift
    x3, y3 = stones[-1]
    if b < 10.0:                                               # 两跳回到舞台中央，越变越近
        if b < 9.25:
            u = A.seg(b, 8.6, 9.25)
            x, y = A.hop((x3, y3), (760, STAGE_BASE), u, 70)
            S = A.lerp(12, 14.5, A.ease_io(u))
        else:
            u = A.seg(b, 9.3, 9.95)
            x, y = A.hop((760, STAGE_BASE), (STAGE_C, STAGE_BASE), u, 60)
            S = A.lerp(14.5, 17, A.ease_io(u))
        return x, y, S, 0, 0, 34, lift
    sway = 3.5 * math.sin(math.pi * (b - 10.0))                 # b 段：每拍摇一下
    return STAGE_C, STAGE_BASE, 17, sway, 0, 34, lift


def mic_state(b, fx, fy, S, ang, ar):
    """拉环 / 麦克风：b<8.4 挂在竖线末端（网罩朝下）；之后留在 Clawd 右钳里，转成网罩朝上、略向身体倾。"""
    if b < 8.4:
        pull = A.ease_io(A.seg(b, 8.0, 8.3)) * (1 - A.ease_io(A.seg(b, 8.3, 8.4)) * 0.3)
        return (CORD_X, GRIP_Y + 10 * pull), 180.0, 12.0
    arm = rigid_arm("R", ar)
    cx = sum(p[0] for p in arm) / 4 + 0.15
    cy = sum(p[1] for p in arm) / 4
    (px, py), = g2p([(cx, cy)], fx, fy, S, ang)
    d = A.lerp(180.0, -12.0 + ang, A.ease_io(A.seg(b, 8.4, 8.9)))
    return (px, py), d, S


def draw_clawd(Pr, b, pose, region=None, dressed=False, in_disk=None):
    """region：只画在这块遮罩里（扫线左右分开画两种造型）；dressed：浅蓝长裙 + 披风袖；
    in_disk：E 段的万花筒圆盘遮罩（圆盘里的部分由万花筒层画，这里只画圆外）。"""
    fx, fy, S, ang, al, ar, lift = pose
    arms = [rigid_arm("L", al), rigid_arm("R", ar)]
    check_proportions(arms=arms)
    cm, em = clawd(fx, fy - lift, S, arms, ang)
    mm = 1.0
    if region is not None:
        mm = mm * region
    if in_disk is not None:
        mm = mm * (1 - in_disk)
    Pr.knockout(cm * mm)
    Pr.ink(cm * mm, ORANGE, k=3)
    if dressed:
        gown = [(-0.2, 5.4), (16.2, 5.4), (16.9, 8.4), (18.2, 12.0), (-2.2, 12.0), (-0.9, 8.4)]
        gm = garment(gown, fx, fy - lift, S, ang, arms=arms, name="gown") * mm
        Pr.knockout(gm)
        Pr.ink(gm, LB, reg=(0, 0.6), k=0)
        for cp in ([(-1.0, 3.0), (0.2, 3.0), (0.4, 9.0), (-1.8, 11.6), (-3.6, 11.0)], [(15.8, 3.0), (17.2, 2.8), (19.8, 10.6), (18.0, 11.6), (15.6, 9.0)]):
            Pr.ink(garment(cp, fx, fy - lift, S, ang, name="cape") * mm, LB, reg=(0, 0.6), k=1, density=0.75)
        for gx in (3.0, 6.5, 9.5, 13.0):
            Pr.ink(SM(g2p([(gx, 7.0), (gx + (gx - 8) * 0.12, 11.8)], fx, fy - lift, S, ang), 1.0, 0, False) * mm, BLUE * 0.8, k=0)
    Pr.ink(em * mm, BLACK, k=2)
    return arms


def cord_and_mic(Pr, b, pose, region_fn=None):
    fx, fy, S, ang, al, ar, lift = pose
    grip, d, s = mic_state(b, fx, fy - lift, S, ang, ar)
    if b < 8.4:                                                # 挂着拉环的竖线
        Pr.ink(SM([(CORD_X, 30), (grip[0], grip[1] - 2.25 * s)], 1.4, 0, False), BLACK, k=2)
    draw_mic(Pr, grip, d, s)
    if b >= 7.9:                                               # 握住：右钳最后再印一次，压在手柄上
        am = M(g2p(rigid_arm("R", ar), fx, fy - lift, S, ang))
        Pr.knockout(am)
        Pr.ink(am, ORANGE, k=3)


# ---------- 万花筒圆盘（E 段） ----------
def disk_state(b):
    R = A.lerp(760, 330, A.ease_out(A.seg(b, 0.0, 0.5)))
    u = A.ease_io(A.seg(b, 0.5, 1.5))
    c = A.lerp2((A0.CX, A0.CY), RED_C, u)
    R = A.lerp(R, RED_R, u)
    k = min(1.0, R / 330)
    flood = A.ease_in(A.seg(b, 1.15, 1.7), 1.4)                  # 红墨从圆边往里盖满
    return c, R, k, flood


def disk_layer(img, b, pose):
    """万花筒圆盘：内容随圆盘缩放；主角还在圆里的那部分保留封面造型。返回 (img, 圆盘遮罩)。"""
    c, R, k, flood = disk_state(b)
    dm = M(circle(c[0], c[1], R, 180))
    if flood >= 1:
        return img, dm
    kal = A0.kaleido(phi=1.5 + math.radians(30) + 0.4 * b, cx=c[0], cy=c[1], k=k, flow=0.25 + 0.05 * b)
    lay = kal
    for th in (90 + 72 * j for j in range(1, 5)):
        lay = A0.cover_clawd(lay, th + 30 + 12 * b, 32, cx=c[0], cy=c[1], k=k)
    fx, fy, S, ang, al, ar, lift = pose                        # 主角在圆里的部分：封面造型（只画在圆内）
    arms = [rigid_arm("L", al), rigid_arm("R", ar)]
    cm, em = clawd(fx, fy, S, arms, ang)
    lay = _cover_main(lay, fx, fy, S, ang, arms)
    pr = np.hypot(X / K - c[0], Y / K - c[1])
    lay = lay + np.exp(-(pr / (60 * k)) ** 2)[..., None] * 0.35
    rim = np.clip(M(circle(c[0], c[1], R + 2, 180)) - M(circle(c[0], c[1], R - 2, 180)), 0, 1)
    img = img * (1 - dm[..., None]) + lay * dm[..., None]
    img = img * (1 - 0.25 * rim[..., None])
    if flood > 0:
        rf = R * (1 - flood)
        red = np.clip(dm - M(circle(c[0], c[1], max(rf, 0.5), 180)), 0, 1)
        img = img * (1 - red[..., None]) + (PAPER * RED)[None, None] * red[..., None]
    return img, dm


def _cover_main(img, fx, fy, S, ang, arms):
    cm, em = clawd(fx, fy, S, arms, ang)
    img = over(img, ORANGE * 1.02, cm)
    sm = garment([(-0.6, 6.8), (16.6, 6.8), (17.6, 11.9), (-1.6, 11.9)], fx, fy, S, ang, arms=arms, name="skirt")
    img = over(img, np.array([0.56, 0.50, 0.20]), sm)
    from kit import circle as _c
    pearls = np.zeros((PH, PW), np.float32)
    for j, gy in enumerate(np.arange(5.3, 7.2, 0.48)):
        for gx in np.arange(0.5, 16.0, 0.62):
            (px, py), = g2p([(gx + (0.31 if j % 2 else 0), gy)], fx, fy, S, ang)
            pearls = np.maximum(pearls, M(_c(px, py, 0.21 * S, 12)))
    img = over(img, (0.99, 0.97, 0.93), pearls)
    return over(img, (0.06, 0.05, 0.06), em)


from common4 import over  # noqa: E402


def clawd_outside(b, pose):
    """主角是否已经完全越出圆盘（墙色和展签在这一拍切换）。"""
    if b >= 1.7:
        return True
    c, R, k, flood = disk_state(b)
    fx, fy, S, ang, *_ = pose
    pts = g2p([(-1, 0), (17, 0), (17, 12), (-1, 12)], fx, fy, S, ang)
    return all(math.hypot(x - c[0], y - c[1]) > R for x, y in pts)


# ---------- 一帧 ----------
def painting(b):
    pose = clawd_pose(b)
    Pr = Press()
    Pr.img = poster_static().copy()
    if b >= 1.7:
        Pr.ink(M(circle(*RED_C, RED_R, 180)), RED, reg=(0.7, 0), k=1)
    hw, top, cls = box_back(Pr, b)
    draw_square(Pr, b)
    draw_circles(Pr, b)
    in_box = 2.75 <= b < 4.6
    if b < 1.7:
        draw_clawd(Pr, b, pose, in_disk=M(circle(*disk_state(b)[0], disk_state(b)[1], 180)))
    elif b < 8.4:
        if not in_box:
            draw_clawd(Pr, b, pose)
    if in_box:
        draw_clawd(Pr, b, pose)
    box_front(Pr, b, hw, top, cls)
    if b >= 1.6 and b < 8.4:
        cord_and_mic(Pr, b, pose)
    elif b < 1.6:
        Pr.ink(SM([(CORD_X, 30), (CORD_X, GRIP_Y - 27)], 1.4, 0, False), BLACK, k=2)
        draw_mic(Pr, (CORD_X, GRIP_Y), 180.0, 12.0)
    img = Pr.img
    if b < 1.7:
        img, _ = disk_layer(img, b, pose)
    if b >= 8.4:                                               # M：竖线向左扫过画面，右边是舞台
        lx = A.lerp(CORD_X, -12, A.ease_io(A.seg(b, 8.4, 9.9)))
        right = smooth(lx - 0.5, lx + 0.5, X / K)
        St = Press()
        St.img = stage_static().copy()
        draw_clawd(St, b, pose, dressed=True)
        cord_and_mic(St, b, pose)
        if lx > -10:
            Pl = Press()
            Pl.img = img.copy()
            draw_clawd(Pl, b, pose)
            cord_and_mic(Pl, b, pose)
            img = Pl.img * (1 - right[..., None]) + St.img * right[..., None]
            img = img * (1 - 0.9 * np.clip(SM([(lx, -5), (lx, 640)], 1.4, 0, False), 0, 1))[..., None]   # 扫线本身
        else:
            img = St.img
    return np.clip(downsample(img, K), 0, 1)


def frame_at(i):
    t = i / FPS
    b2 = bt(A.twos(i))
    p = painting(b2)
    b = bt(t)
    if b >= 10.0:
        z = 1 + 0.08 * A.ease_io(A.seg(b, 10.0, 13.0))
        p = A.zoom_painting(p, z, STAGE_C, 400)
    if clawd_outside(b2, clawd_pose(b2)):
        return A.compose(p, WALL1, LABEL1)
    return A.compose(p, WALL0, LABEL0)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a01_{i:04d}.png")
