"""10 珠玉 · 装饰艺术 · 按 90 BPM（1:37.95–1:47.29）+ 结尾（1:47.29–1:53.29）。拍号 b 从 09 结束算起。
E  b0–3   亮片从裙子上脱落，变成一颗珍珠往左滚向台口。Clawd 追过去，从舞台边缘跳下；越过边缘的部分没有掉下去，
          而是飞了起来，变成装饰艺术画法、换上白色钩针镂空流苏长裙。随后那道边缘一路扫过整幅画。
a  b3–9   夜海上空浮着光之幻影。Clawd 踩着程式化的浪花一跳一跳追着珍珠，身后的薄纱拖出金线，最后飞向幻影。
M  b9–11  Clawd 飞进幻影身后的放射扇纹；扇纹的光从里面张开，进去的部分换上白色垂坠礼服、戴上珍珠耳饰。
b  b11–14 几束探照灯在 Clawd 身后交汇。几乎不动。
结尾 b14–29（15 拍，2026-10-17 按用户意见改：不要光束收珍珠、不要珍珠链）：
  1–3 拍  舞台身后的放射扇纹从中心张开成旋转的专辑封面万花筒，Clawd 落回正下方的空位；墙色和展签变回开场。
  4–6 拍  封面越转越快，转满一圈、主角回到正下方时把它甩出来。
  6–7 拍  Clawd 飞出画框，越过画框平面时从 2D 变回三维，翻着身落到看展的位置，正面对着我们。
  8–9 拍  站定，略停一下。
  10–13 拍 原地转身 90°，侧着身子从画面右侧走出镜头；封面减速停下，正下方空着。
  14–15 拍 只剩墙和画框；最后 1 拍淡出到黑。
"""
import inspect
import math
import os
import sys
import textwrap

import numpy as np
from scipy.ndimage import map_coordinates

from kit import (K, PW, PH, X, Y, M, SM, circle, catmull, clawd, garment, g2p, rigid_arm, check_proportions, gauss, smooth,
                 over, downsample, outline)
import anim_core as A
import frame
import S10_装饰艺术 as S10
from S10_装饰艺术 import gold, NAVY, NAVY2, CREAM, HZ, HEAD
import A09_多少的光阴 as A9
import A00_开场 as A0
import kit3d

FPS = 30
T0 = A9.T0 + 16 * A9.BEAT
BEAT = 60 / 90
WALL9, LABEL9 = A9.WALL9, A9.LABEL9
WALL10, LABEL10 = (0.90, 0.86, 0.78), "珠玉"
FRAME0 = math.ceil(T0 * FPS)
END = 29.0
FRAME1 = math.ceil((T0 + END * BEAT) * FPS)
STAGE_CL = (560, 470, 16)
EDGE_X = 190
SAMPLES = "/Users/limingyang/Desktop/单依纯二专/关键帧/v4样张"
_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


# ---------- 从样张的 render 函数里截取图层与 Clawd 的画法 ----------
class _Stop(Exception):
    pass


def _grab(fn, nth_check):
    """运行 fn，在第 nth_check 次调用 check_proportions 时停下，取出调用者（render 函数）里的 img。"""
    cnt = [0]
    orig = S10.check_proportions
    grabbed = {}

    def chk(*a, **k):
        cnt[0] += 1
        if cnt[0] == nth_check:
            f = inspect.currentframe().f_back
            grabbed["img"] = f.f_locals["img"].copy()
            raise _Stop()
        return orig(*a, **k)
    S10.check_proportions = chk
    S10.G = np.random.default_rng(10)
    try:
        fn()
    except _Stop:
        pass
    finally:
        S10.check_proportions = orig
    return grabbed["img"].astype(np.float32)


def layers():
    if "lay" in _CACHE:
        return _CACHE["lay"]
    sky_sea = _grab(S10.render, 2)                  # 第 1 次在 apparition() 里，第 2 次在 Clawd 之前
    # 天空（不含海）：在海之前截
    orig_app = S10.apparition
    keep = {}

    def app(img, *a, **k):
        out = orig_app(img, *a, **k)
        keep["sky"] = out.copy()
        return out
    S10.apparition = app
    try:
        _grab(S10.render, 2)
    finally:
        S10.apparition = orig_app
    sky = keep["sky"].astype(np.float32)
    sea_a = np.clip(np.abs(sky_sea - sky).max(-1) * 30, 0, 1) * (Y / K > HZ - 30)
    stage = _grab(S10.render_b, 1)
    _CACHE["lay"] = (sky, sky_sea, sea_a.astype(np.float32), stage)
    return _CACHE["lay"]


def _snippet(fn, start, end, drop):
    src = textwrap.dedent(inspect.getsource(fn))
    body = src[src.index(start):src.index(end)]
    lines = [ln for ln in body.splitlines() if not any(d in ln for d in drop)]
    return textwrap.dedent("\n".join(lines))


FLY_SRC = None
GOWN_SRC = None


def fly_clawd(img, FX, FY, S, ANG, mask=None):
    """样张 10 里那只穿白色钩针镂空流苏长裙、拖着薄纱的 Clawd（代码直接取自样张，位置可变）。"""
    global FLY_SRC
    if FLY_SRC is None:
        FLY_SRC = compile(_snippet(S10.render, "    # ---------- Clawd：白色钩针", "    # 前面那颗被追着的珍珠", ["FX, FY, S, ANG ="]), "fly", "exec")
    ns = dict(vars(S10))
    ns.update(img=img.copy(), FX=FX, FY=FY, S=S, ANG=ANG)
    exec(FLY_SRC, ns)
    out = ns["img"]
    if mask is not None:
        out = img * (1 - mask[..., None]) + out * mask[..., None]
    return out


def gown_clawd(img, FX, FY, S, mask=None):
    global GOWN_SRC
    if GOWN_SRC is None:
        GOWN_SRC = compile(_snippet(S10.render_b, "    # Clawd：白色垂坠礼服", "    # 阶梯角的细金线边框", ["FX, FY, S = cx"]), "gown", "exec")
    ns = dict(vars(S10))
    ns.update(img=img.copy(), FX=FX, FY=FY, S=S, cx=560)
    exec(GOWN_SRC, ns)
    out = ns["img"]
    if mask is not None:
        out = img * (1 - mask[..., None]) + out * mask[..., None]
    return out


def border(img):
    def stepped(inset, step):
        a_, b_ = inset, inset + step
        return [(b_, a_), (1120 - b_, a_), (1120 - b_, b_), (1120 - a_, b_), (1120 - a_, 630 - b_), (1120 - b_, 630 - b_), (1120 - b_, 630 - a_),
                (b_, 630 - a_), (b_, 630 - b_), (a_, 630 - b_), (a_, b_), (b_, b_)]
    for inset, w in ((14, 1.6), (22, 0.8)):
        pts = stepped(inset, 18)
        img = over(img, gold("x", 0, 1120, 10), SM(pts + [pts[0]], w, 0, False))
    return img


def pearl(img, x, y, r, glow=0.5):
    img = img + (np.exp(-(np.hypot(X / K - x, Y / K - y) / (r * 2.2)) ** 2) * glow)[..., None] * np.array([1.0, 0.96, 0.88])
    pr = np.hypot(X / K - x + r * 0.35, Y / K - y + r * 0.35) / r
    return over(img, np.array([1.0, 0.98, 0.94])[None, None] * (1 - 0.25 * smooth(0, 1.4, pr))[..., None], M(circle(x, y, r, 40)))


def warp(src, z, c_src, c_dst):
    qx = c_src[0] + (X / K - c_dst[0]) / z
    qy = c_src[1] + (Y / K - c_dst[1]) / z
    return np.stack([map_coordinates(src[..., c], [qy * K, qx * K], order=1, mode="nearest") for c in range(3)], -1)


# ---------- 一帧 ----------
def sea_scene(b):
    """夜海：天空静止，海面向左卷动（Clawd 在往右追）。"""
    sky, sky_sea, sea_a, stage = layers()
    shift = int(round(-38 * K * max(0.0, b - 2.0)))
    sea = np.roll(sky_sea, shift, axis=1)
    sa = np.roll(sea_a, shift, axis=1)
    img = sky * (1 - sa[..., None]) + sea * sa[..., None]
    return img


def chase_pose(b):
    """a 段：踩着浪花一跳一跳往右追，b8 之后升起飞向幻影。"""
    u = A.seg(b, 3.0, 9.0)
    x = A.lerp(240, 520, A.ease_io(u))
    bounce = abs(math.sin(math.pi * (b - 3.0)))
    y = 392 - 64 * bounce
    rise = A.ease_in(A.seg(b, 8.0, 9.0), 1.6)
    x, y = A.lerp(x, 600, rise), A.lerp(y, 300, rise)
    return x, y, 12, -10 - 8 * math.cos(math.pi * (b - 3.0))


def painting(t, b):
    sky, sky_sea, sea_a, stage = layers()
    if b < 3.0:
        # E：09 的星点灯舞台；一片亮片脱落成珍珠，往左滚向台口；Clawd 追过去、跳下
        glass = A9.b_frame(16.0 + b * BEAT / A9.BEAT)
        roll = A.ease_in(A.seg(b, 0.3, 1.8), 1.4)
        px = A.lerp(620, EDGE_X - 30, roll)
        py = 500 - 2 + (60 * A.ease_in(A.seg(b, 1.6, 2.0)) if b > 1.6 else 0) - 40 * math.sin(math.pi * A.seg(b, 0.2, 0.5))
        chase = A.ease_io(A.seg(b, 0.7, 2.0))
        fx = A.lerp(560, EDGE_X + 40, chase)
        fy = 500 - 6 * abs(math.sin(math.pi * 4 * chase))
        S = A.lerp(17, 13, chase)
        jump = A.seg(b, 2.0, 2.9)
        if jump > 0:
            fx, fy = A.hop((EDGE_X + 40, 500), (240, 392), jump, 120)
            S = A.lerp(13, 12, jump)
        edge = EDGE_X if b < 2.2 else A.lerp(EDGE_X, 1135, A.ease_in(A.seg(b, 2.2, 3.0), 1.5))
        left = smooth(edge + 0.5, edge - 0.5, X / K)
        deco = sea_scene(3.0)
        deco = pearl(deco, max(px, 60), min(py, 380) if b > 1.8 else py, 8)
        deco = fly_clawd(deco, fx, fy, S, -10)
        deco = border(deco)
        # 舞台上的 Clawd（玻璃）由 09 的 b_frame 画在中央；离开后只保留背景：直接用 09 的画面并在右边盖住原位
        img = glass * (1 - left[..., None]) + deco * left[..., None]
        if chase > 0:                                  # 玻璃世界里的 Clawd 跟着移动：重画一次玻璃（只在 edge 右边）
            L = np.zeros((PH, PW), np.int32)
            color = {}
            n = A9.stage_labels(16.0, L, color, 0, np.ones((PH, PW), np.float32))
            nid = [n]

            def new(c, m):
                nid[0] += 1
                color[nid[0]] = np.asarray(c, np.float32)
                L[m > 0.5] = nid[0]
            paint = A9.hero_glass(L, new, fx, fy, S, [rigid_arm("L", 0), rigid_arm("R", 0)], gown_line=-10)
            if px > edge:
                new((1.25, 1.20, 1.15), M(circle(px, py, 9, 24)))
            g2 = A9.glass_finish(L, color, nid[0], paint, lead_w=1.8)
            img = g2 * (1 - left[..., None]) + deco * left[..., None]
        elif b < 0.3 or px > edge:
            pass
        return img, edge > fx + 9 * S or b >= 2.3
    if b < 9.0:
        img = sea_scene(b)
        x, y, S, ang = chase_pose(b)
        q = (x + 170 + 20 * math.sin(b * 2), 300 + 18 * math.sin(b * 3))
        img = pearl(img, *q, 8)
        img = fly_clawd(img, x, y, S, ang)
        return border(img), True
    # M：飞进放射扇纹，扇纹的光从中心张开成 10-b
    u = A.seg(b, 9.0, 11.0)
    img = sea_scene(9.0)
    fl = A.ease_in(A.seg(b, 9.0, 10.0), 1.4)
    x0, y0, S0, a0 = chase_pose(9.0)
    fx, fy = A.lerp(x0, HEAD[0], fl), A.lerp(y0, HEAD[1] + 60, fl)
    S = A.lerp(S0, 6, fl)
    open_ = A.ease_in(A.seg(b, 9.6, 11.0), 1.6)
    R = 240 * (1 - open_) * 0 + A.lerp(30, 1400, open_)
    c_now = A.lerp2(HEAD, (560, 315), A.ease_io(A.seg(b, 9.6, 11.0)))
    s_st = A.lerp(0.25, 1.0, A.ease_io(A.seg(b, 9.6, 11.0)))
    st = warp(stage, s_st, (560, 315), c_now)
    disk = M(circle(c_now[0], c_now[1], R, 200)) if open_ > 0 else np.zeros((PH, PW), np.float32)
    img = img * (1 - disk[..., None]) + st * disk[..., None]
    fin = A.ease_io(A.seg(b, 10.2, 11.0))
    fx, fy, S = A.lerp(fx, STAGE_CL[0], fin), A.lerp(fy, STAGE_CL[1], fin), A.lerp(S, STAGE_CL[2], fin)
    img = fly_clawd(img, fx, fy, S, -10 * (1 - fin), mask=(1 - disk))
    img = gown_clawd(img, fx, fy, S, mask=disk)
    return border(img), True


def stage_frame(b, beams_to=None, pearl_at=None):
    sky, sky_sea, sea_a, stage = layers()
    img = stage.copy()
    if beams_to is not None:                          # 光束收拢：从台口两侧斜打上去，汇到一点
        for (x0, x1) in [(60, 760), (220, 700), (380, 640), (740, 480), (900, 420), (1060, 360)]:
            x1b = A.lerp(x1, beams_to[0], beams_to[2])
            yt = A.lerp(-10, beams_to[1], beams_to[2])
            beam = M([(x0 - 30, 640), (x0 + 30, 640), (x1b + 4, yt), (x1b - 4, yt)])
            img = img + (gauss(beam, 4 * K) * 0.18 * beams_to[2])[..., None] * np.array([1.0, 0.93, 0.75])
    img = gown_clawd(img, *STAGE_CL)
    if pearl_at is not None:
        img = pearl(img, *pearl_at)
    return border(img)


def kaleido_k(t, phi, ring_rot, swirl=0.0):
    """K 倍分辨率的封面万花筒（不含正下方的主角），与 A00_开场.painting 同一画法。"""
    img = A0.kaleido(phi, swirl, flow=0.05 * t)
    for (a, r, s_, rot, ph) in A0.SPARKS:
        a2 = a + math.radians(ring_rot)
        img = A0.sparkle(img, A0.CX + r * math.cos(a2), A0.CY + r * math.sin(a2) * 0.62, s_ * (0.65 + 0.35 * math.sin(t * 2.6 + ph)), rot + math.radians(ring_rot))
    for th in (90 + 72 * k for k in range(1, 5)):
        img = A0.cover_clawd(img, th + ring_rot, 32.0)
    img = img + np.exp(-(np.hypot(X / K - A0.CX, Y / K - A0.CY) / 60) ** 2)[..., None] * 0.35
    pr = np.hypot(X / K - A0.CX + 6, Y / K - A0.CY + 6) / 28
    return over(img, np.array([1.0, 0.98, 0.95])[None, None] * (1 - 0.2 * smooth(0, 1.4, pr))[..., None], M(circle(A0.CX, A0.CY, 26, 64)))


def spin(b):
    """封面整体转角（度）：b16.2 起越转越快，b19 转满一圈（主角回到正下方）甩出去，之后再转一圈慢慢停下。"""
    if b < 16.2:
        return 0.0
    if b < 19.0:
        return 360.0 * A.ease_in(A.seg(b, 16.2, 19.0), 2.2)
    return 360.0 + 360.0 * A.ease_out(A.seg(b, 19.0, 22.5), 2.2)


def ending(t, b):
    """结尾：返回 (全幅画面, 淡出系数)。"""
    if b < 16.2:
        # 放射扇纹从中心张开成旋转的专辑封面；Clawd 落回正下方的空位
        sky, sky_sea, sea_a, stage = layers()
        st = border(stage.copy())
        op = A.ease_in(A.seg(b, 14.3, 16.0), 1.5)
        R = A.lerp(20, 1300, op)
        phi = 1.4 * A.ease_in(A.seg(b, 14.0, 16.2), 1.4)          # 液体先转起来
        kal = kaleido_k(t, phi, 0.0, swirl=0.8 * op)
        disk = M(circle(A0.CX, A0.CY, R, 200))
        img = st * (1 - disk[..., None]) + kal * disk[..., None]
        u = A.ease_io(A.seg(b, 14.6, 16.0))
        fx, fy, S = 560, A.lerp(STAGE_CL[1], A0.CY + A0.SLOT_RF, u), A.lerp(STAGE_CL[2], A0.SLOT_S, u)
        img = gown_clawd(img, fx, fy, S, mask=(1 - disk))
        import A01_还有什么更好的 as A1
        arms = [rigid_arm("L", 32 * u), rigid_arm("R", 32 * u)]
        cov = A1._cover_main(img.copy(), fx, fy, S, 0.0, arms)
        img = img * (1 - disk[..., None]) + cov * disk[..., None]
        p = np.clip(downsample(img, K), 0, 1)
        back = op > 0.85
        return A.compose(p, A0.WALL if back else WALL10, A0.LABEL if back else LABEL10), 0.0
    rot = spin(b)
    sw = 1.4 * math.sin(math.pi * A.seg(b, 16.2, 22.5))
    filled = b < 19.0
    p = A0.painting(t, filled, math.radians(rot), sw, rot, 32.0)
    full = A.compose(p, A0.WALL, A0.LABEL)
    if filled:
        return full, 0.0
    if b < 20.6:                                       # 被甩出来：翻着身飞出画框、落地
        e = A.ease_out(A.seg(b, 19.0, 20.6), 1.6)
        z = (A0.Z_WALL - 4.0) * (1 - e)
        lift = A0.SLOT_FEET_Y * (1 - e) + 6.0 * math.sin(math.pi * e)
        yaw = 180.0 + 360.0 * (1 - A.ease_out(A.seg(b, 19.0, 20.4), 2.0))
        lean = 22 * math.sin(2 * math.pi * e) * (1 - e)
        flail = 50 * math.sin(math.pi * e)
        flat = 1 - A.ease_out(A.seg(b, 19.0, 20.0), 1.5)
        boxes = A0.posed_boxes(lean, flail, flail, lift)
        return A0.viewer(full, p, boxes, yaw, 8.0, z, flat=flat), 0.0
    if b < 22.6:                                       # 落地后略停一下
        land = math.sin(math.pi * A.seg(b, 20.6, 20.95)) * 0.8
        breathe = 1.5 * math.sin(math.pi * A.seg(b, 21.2, 22.4))
        boxes = A0.posed_boxes(breathe, 0, 0, land)
        return A0.viewer(full, p, boxes, 180.0, 8.0, 0.0), 0.0
    if b < 26.6:                                       # 转身 90°，侧着走出右边
        turn = A.ease_io(A.seg(b, 22.6, 23.3))
        walk = A.ease_in(A.seg(b, 23.2, 26.4), 1.3)
        x = A.lerp(8.0, 46.0, walk)
        step = abs(math.sin(math.pi * 5 * walk)) if walk > 0 else 0
        boxes = A0.posed_boxes(2 * math.sin(math.pi * 5 * walk) if walk > 0 else 0, 0, 0, 0.35 * step)
        return A0.viewer(full, p, boxes, A.lerp(180.0, 90.0, turn), x, 0.0), 0.0
    return full, A.ease_io(A.seg(b, END - 1.0, END))


def _up(p1):
    """1 倍画心 → K 倍（最近邻放大，用于与 K 倍图层拼接）。"""
    return np.repeat(np.repeat(p1, K, 0), K, 1)


def frame_at(i):
    t = i / FPS
    bb = bt(t)
    if bb >= 14.0:
        full, fade = ending(t, bb)
        return full * (1 - fade)
    b = bt(A.twos(i)) if bb < 3.0 else bb
    if b >= 11.0:
        p, sw = stage_frame(b), True
    else:
        p, sw = painting(t, b)
    p = np.clip(downsample(p, K), 0, 1)
    if sw:
        return A.compose(p, WALL10, LABEL10)
    return A.compose(p, WALL9, LABEL9, spill=A9.SPILL)


def warm():
    A9.warm()
    layers()
    for w, l in ((WALL9, LABEL9), (WALL10, LABEL10), (A0.WALL, A0.LABEL)):
        A.plate(w, l)


if __name__ == "__main__":
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a10_{i:04d}.png")
