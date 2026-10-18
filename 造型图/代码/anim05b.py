"""05｜我管不住我自己 · 动画样片 v2（Clawd 穿越式转场；80 BPM，每拍 0.75s；30fps，10.5s = 315 帧）

T1  0.00–2.25  04 → 05-a：穿金色长裙的 Clawd 从门里的金色世界（04）走出；跨过门框的那一刻，
               跨过去的部分变成水彩画法（05），手里出现水壶。
    2.25–4.50  05-a：边摇摆边浇花，歌名逐字晕开、跟拍跳动；音符小花飘出；一朵大水彩花飘到身旁慢慢盛开。
T2  4.50–6.00  05-a → 05-b：Clawd 在拍点上跳进花心；镜头随它冲进花里，花心里就是南京舞台；
               穿过花心边缘的部分换上银色亮片裙，落地时已经站在舞台上。
    6.00–9.00  05-b：追光落下，跟拍轻晃，亮片闪烁。
T3  9.00–10.5  05-b → 06：Clawd 往右走，一面镜子从右向左扫过画面，镜子那边是 06 的赛璐璐镜面长廊；
               跨过镜面的部分变成 06 的造型（黄裙 + 珍珠链），最后整幅变成镜面长廊。
用法：python3 anim05b.py 输出.mp4 静帧目录 [帧号,帧号…]"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
import watercolor as wc
import cel
from anim05 import (poly_mask, ell_mask, line_mask, place_mask, camera, flower5, bg_a, bg_b, FLOWER_COLS, ORANGE_WC)
from video import render_video, ease, FPS

BEAT = 60 / 80                    # 05 的 BPM = 80（分镜/BPM清单.md）
T1_END, T2_START, T2_END, T3_START, END = 2.25, 4.5, 6.0, 9.0, 10.5
N = int(END * FPS)
x, y = grid()
ORANGE = np.array(CLAWD_ORANGE) / 255
BBOX = (-5, -5, 27, 14)           # 所有画法共用的格坐标画布，保证换画法时位置一致
BB_C = ((BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2)
FLOWER = (1080, 430)              # 大水彩花（世界坐标）
A_GROUND = (855, 752)             # 05-a 中 Clawd 脚底中心（世界坐标）
B_GROUND = (908, 944)             # 05-b 中 Clawd 脚底中心
SEAM_DOOR = 215                   # 门框内沿（世界坐标 x）
_cache = {}


def rot(vx, vy, ang):
    a = math.radians(ang)
    return vx * math.cos(a) + vy * math.sin(a), -vx * math.sin(a) + vy * math.cos(a)


def anchor_to_center(ax, ay, S, ang):
    """脚底中心（格坐标 (8,12)）的屏幕位置 → 遮罩画布中心的屏幕位置。"""
    dx, dy = rot((BB_C[0] - 8) * S, (BB_C[1] - 12) * S, ang)
    return ax + dx, ay + dy


def grid_pt(gx_, gy_, ax, ay, S, ang):
    dx, dy = rot((gx_ - 8) * S, (gy_ - 12) * S, ang)
    return ax + dx, ay + dy


def walk_legs(phase, lift=0.6):
    """走路：两组腿交替整块抬起（不改变腿的尺寸）。phase 以“步”为单位。"""
    la = lift * max(0.0, math.sin(math.pi * phase))
    lb = lift * max(0.0, -math.sin(math.pi * phase))
    out = []
    for i, (x0, y0, x1, y1) in enumerate(LEGS):
        l = la if i in (0, 2) else lb
        out.append((x0, y0 - l, x1, y1 - l))
    return out


def masks(kind, S, legs=None, arms=None):
    S = round(S * 2) / 2
    key = (kind, S, tuple(map(tuple, legs)) if legs else None, str(arms))
    if key in _cache:
        return _cache[key]
    extra = {}
    if kind == "a":
        extra["can"] = [[(16.8, 4.4), (20.6, 4.4), (21.0, 8.0), (16.4, 8.0)], [(20.6, 5.2), (24.4, 3.0), (24.8, 3.6), (21.0, 6.4)],
                        [(16.8, 4.4), (17.6, 2.8), (19.6, 2.8), (20.4, 4.4), (19.6, 4.4), (19.2, 3.6), (18.0, 3.6), (17.6, 4.4)]]
    elif kind == "gold":
        extra["gown"] = [[(0.2, 5.4), (15.8, 5.4), (15.6, 8.2), (16.4, 10.4), (18.0, 11.8), (8.0, 12.0), (-2.0, 11.8), (-0.4, 10.4), (0.4, 8.2)]]
    elif kind == "b":
        def ruffle(x0, x1, ytop, ybot, waves, amp, flare):
            top = [(x0 + flare * 0.3, ytop), (x1 - flare * 0.3, ytop)]
            return top + [(x1 + flare - (x1 - x0 + 2 * flare) * (i / (waves * 8)), ybot + amp * math.sin(i / (waves * 8) * waves * 2 * math.pi)) for i in range(waves * 8 + 1)]
        tiers = [ruffle(-0.4, 16.4, 7.8, 9.0, 6, 0.22, 0.6), ruffle(-0.8, 16.8, 8.8, 10.0, 7, 0.22, 1.2), ruffle(-1.2, 17.2, 9.8, 10.9, 8, 0.22, 1.8)]
        extra["dress"] = [[(0.0, 5.6), (16.0, 5.6), (16.2, 8.4), (-0.2, 8.4)]] + tiers
        extra["tiers"] = tiers[1:]
        extra["trail"] = [[(16.6, 8.4), (18.8, 9.0), (20.8, 10.2), (22.2, 11.4), (21.4, 11.9), (20.6, 11.5), (19.8, 12.0), (19.0, 11.5), (18.2, 11.8), (17.4, 10.8)]]
    check_proportions(arms=[a for a in (arms or []) if isinstance(a[0], (tuple, list))], extra=extra)
    cm, _, _ = clawd_masks(S, int(S * 1.5) + 4, extra=extra, legs=legs, arms=arms, bbox=BBOX)
    _cache[key] = cm
    return cm


def placed(cm, ax, ay, S, ang):
    cx, cy = anchor_to_center(ax, ay, S, ang)
    return {k: place_mask(v, cx, cy, ang) for k, v in cm.items()}

# ---------------------------------------------------------------- 各画法的 Clawd（屏幕坐标）


def draw_a(img, ax, ay, S, ang, legs, TEX, shadow=1.0):
    """05-a 水彩：无服装 + 水壶。"""
    P = placed(masks("a", S, legs), ax, ay, S, ang)
    body = np.clip(P["body"] + P["arms"] + P["legs"], 0, 1)
    if shadow > 0:
        img = wc.wash(img, shift(body, int(S * 1.1), int(S * 1.3)) * (1 - body), (0.75, 0.68, 0.72), 0.45 * shadow, seed=70, bleed=8, tex=TEX)
    img = wc.wash(img, body, ORANGE_WC, 0.85, seed=71, bleed=1.2, edge=0.5, tex=TEX)
    cx, _ = anchor_to_center(ax, ay, S, ang)
    img = wc.wash(img, body * np.clip((x - cx + 60) / 200, 0, 1), (0.8, 0.45, 0.35), 0.35, seed=72, bleed=4, tex=TEX)
    img = wc.wash(img, P["eyes"], (0.12, 0.1, 0.12), 0.95, seed=73, bleed=0.6, edge=0.2, tex=TEX)
    img = wc.wash(img, P["can"], (0.45, 0.78, 0.66), 0.85, seed=74, bleed=1, tex=TEX)
    return img, P


def draw_gold(img, ax, ay, S, ang, legs, t):
    """04 胶片金色舞台的画法：平滑渐变、亮片金裙，不带水彩肌理。"""
    P = placed(masks("gold", S, legs), ax, ay, S, ang)
    body = np.clip(P["body"] + P["arms"] + P["legs"], 0, 1)
    shade = np.clip(1 - (y - ay + 12 * S) / (12 * S), 0, 1)
    img = over(img, ORANGE * (0.85 + 0.3 * shade)[..., None] * np.array([1.05, 0.95, 0.82]), body)
    flow = 0.5 + 0.5 * np.sin((x - ax) / (0.45 * S) + np.sin((y - ay) / (0.8 * S)) * 0.6)
    gold = np.array([0.86, 0.68, 0.30])[None, None] * (0.65 + 0.4 * flow)[..., None]
    img = over(img, gold, P["gown"])
    spark = (rng(int(t * 10) + 5).random((H, W)) > 0.992) * (P["gown"] > 0.5)
    img = img + np.clip(blur(spark.astype(np.float32), 0.9) * 4, 0, 1)[..., None] * 0.8
    img = over(img, np.array([0.06, 0.04, 0.03]), P["eyes"])
    return img, P


def draw_b(img, ax, ay, S, ang, legs, TEX, tw_seed, lit=1.0):
    """05-b 水彩：银色亮片层叠短裙 + 粉色玫瑰。"""
    arms = [(-1, 3, 0, 5), rigid_arm('R', -35)]
    P = placed(masks("b", S, legs, arms), ax, ay, S, ang)
    fig = np.clip(P["body"] + P["arms"] + P["dress"] + P["trail"] + P["legs"], 0, 1)
    img = img + blur(fig, 2)[..., None] * (np.array([0.98, 0.97, 0.97]) - img) * 0.85 * lit
    img = wc.wash(img, np.clip(P["body"] + P["arms"] - P["dress"], 0, 1), ORANGE_WC, 0.85, seed=550, bleed=1.2, tex=TEX)
    img = wc.wash(img, P["eyes"], (0.12, 0.1, 0.12), 0.95, seed=551, bleed=0.6, edge=0.2, tex=TEX)
    img = wc.wash(img, P["dress"], (0.80, 0.78, 0.88), 0.7, seed=552, bleed=1.5, edge=0.8, tex=TEX)
    img = wc.wash(img, P["tiers"], (0.74, 0.72, 0.86), 0.45, seed=553, bleed=1.2, edge=1.0, tex=TEX)
    img = wc.wash(img, np.clip(P["tiers"] - shift(P["tiers"], -int(S * 0.4), 0), 0, 1), (0.6, 0.58, 0.76), 0.6, seed=5531, bleed=1.5, tex=TEX)
    img = wc.wash(img, P["trail"], (0.86, 0.84, 0.92), 0.55, seed=554, bleed=2, edge=0.9, tex=TEX)
    img = wc.wash(img, P["legs"], (0.78, 0.78, 0.86), 0.75, seed=555, bleed=1.2, edge=0.8, tex=TEX)
    sq = np.clip(P["dress"] + P["legs"], 0, 1)
    k = (S / 26) ** 2
    img = wc.gouache_dots(img, sq, max(1, int(520 * lit * k)), seed=556 + tw_seed, r=(1.2 * S / 26, 3.0 * S / 26))
    img = wc.gouache_dots(img, sq, max(1, int(120 * lit * k)), seed=7557 + tw_seed, r=(1.0 * S / 26, 2.2 * S / 26), color=(0.72, 0.74, 0.95))
    for i, (rgx, rgy, rr) in enumerate([(-0.4, 5.8, 1.7), (15.2, 8.2, 1.5), (1.2, 10.0, 1.1), (20.4, 10.8, 1.1), (1.4, 5.0, 0.9), (16.6, 9.6, 0.9)]):
        px_, py_ = grid_pt(rgx, rgy, ax, ay, S, ang)
        r_ = rr * S
        img = wc.wet_bloom(img, px_, py_, r_, (0.92, 0.55, 0.66), seed=560 + i, density=0.85, tex=TEX)
        spiral = [[(px_ + r_ * 0.75 * (1 - s_ / 30) * math.cos(s_ * 0.55), py_ + r_ * 0.75 * (1 - s_ / 30) * math.sin(s_ * 0.55)) for s_ in range(30)]]
        img = wc.wash(img, line_mask(spiral, max(1.5, 3 * S / 26)), (0.7, 0.3, 0.42), 0.7, seed=570 + i, bleed=0.6, tex=TEX)
    img = img + blur(fig, 30)[..., None] * 0.04 * lit
    return img, P


def draw_cel(img, ax, ay, S, ang, legs):
    """06 赛璐璐：黄裙 + 珍珠链，粗描边。"""
    spr = cel.cel_clawd(S, cel.outfit_parts(), legs=legs, bbox=BBOX)
    cx, cy = anchor_to_center(ax, ay, S, ang)
    refl = spr[::-1].copy()
    refl[..., 3] *= 0.22
    img = paste_rgba(img, refl, cx, ay + (ay - cy) - 0.2 * S, -ang)
    return paste_rgba(img, spr, cx, cy, ang)

# ---------------------------------------------------------------- 背景（世界坐标）


def gold_world():
    """04-b 金色舞台的烟雾与光斑（门里那一侧的世界）。"""
    if "gold" not in _cache:
        img = np.zeros((H, W, 3), np.float32) + np.array([0.20, 0.13, 0.05])
        haze = fbm(H, W, 61, octaves=5, base=5)
        img += (haze - 0.4)[..., None] * np.array([0.25, 0.17, 0.06])
        img += (np.exp(-((x - 110) / 160) ** 2) * np.clip(1 - y / 900, 0, 1))[..., None] * np.array([0.6, 0.45, 0.2])
        g = rng(62)
        bok = np.zeros((H, W), np.float32)
        for _ in range(26):
            bx, by, r = g.uniform(30, 200), g.uniform(60, 900), g.uniform(8, 22)
            dd = np.sqrt((x - bx) ** 2 + (y - by) ** 2)
            bok += np.clip(r - dd, 0, 1) * g.uniform(0.2, 0.6)
        _cache["gold"] = img + blur(bok, 2)[..., None] * np.array([1.0, 0.8, 0.4])
    return _cache["gold"]


def world_a(t):
    """05-a 世界（不含 Clawd）。"""
    B = bg_a()
    img, TEX = B["img"].copy(), B["TEX"]
    beat = t / BEAT
    # 门里：04 的金色世界
    glass = poly_mask([[(40, 60), (190, 60), (190, 1000), (40, 1000)]])
    k = 1 - ease((t - 2.4) / 1.0)                 # Clawd 走出来后，门里的金光慢慢退去
    img = img * (1 - glass[..., None] * k) + gold_world() * glass[..., None] * k
    a = t * 2 * math.pi / (BEAT * 2)
    img = img + (ell_mask([(1545 + 70 * math.cos(a), 700 + 18 * math.sin(a), 26, 5)]) * 0.22)[..., None]
    steam = line_mask([[(1750 + 12 * math.sin(s_ / 9 - t * 3), 690 - s_ - (t * 20) % 12) for s_ in range(0, 90, 3)],
                       [(1772 + 10 * math.sin(s_ / 8 + 1 - t * 2.6), 690 - s_ - (t * 16) % 10) for s_ in range(0, 80, 3)]], 4)
    img = wc.wash(img, steam, (0.82, 0.82, 0.88), 0.4, seed=27, bleed=3, tex=TEX)
    for i in range(16):
        u = ((i + t * 1.6) / 16) % 1.0
        fx = 1450 - u * 760 + 60 * math.sin(u * 7)
        fy = 640 - u * 330 - 50 * math.cos(u * 5) - 6 * math.sin(beat * math.pi * 2 + i)
        r = (14 + 14 * math.sin(math.pi * u)) * min(1, u * 8)
        fade = min(1.0, u * 6, (1 - u) * 5) * ease(t / 0.8)
        if r < 2 or fade <= 0:
            continue
        img = wc.wash(img, poly_mask([flower5(fx, fy, r, i * 0.7 + t * 1.5)]), FLOWER_COLS[i % 4], 0.7 * fade, seed=40 + i, bleed=2.5, edge=0.7, tex=TEX)
        img = wc.wash(img, ell_mask([(fx, fy, r * 0.22, r * 0.22)]), (0.98, 0.8, 0.3), 0.8 * fade, seed=60 + i, bleed=1, tex=TEX)
    bob = 8 * math.sin(beat * math.pi * 2)
    notes = []
    for kk, (nx, ny, s_) in enumerate([(1380, 560, 1.0), (1180, 470, 0.8), (980, 380, 0.9), (1290, 400, 0.7)]):
        ny = ny + bob * (1 if kk % 2 else -1)
        notes.append([(nx, ny), (nx, ny - 50 * s_), (nx + 26 * s_, ny - 40 * s_)])
        notes.append([(nx - 10 * s_, ny + 4), (nx, ny + 8 * s_), (nx + 4, ny), (nx - 8 * s_, ny - 4 * s_), (nx - 10 * s_, ny + 4)])
    img = wc.pencil(img, notes, seed=31, alpha=0.5, width=2)
    # 大水彩花：3.4s 起飘到 Clawd 身边，盛开；花心就是之后的入口
    if t > 3.4:
        g = ease((t - 3.4) / 1.0)
        R = 150 * g
        fxx = FLOWER[0] + 160 * (1 - g)
        fyy = FLOWER[1] - 60 * (1 - g) + 6 * math.sin(beat * math.pi)
        if R > 3:
            img = wc.wash(img, poly_mask([flower5(fxx, fyy, R, 0.3 + t * 0.6)]), (0.96, 0.62, 0.72), 0.75, seed=995, bleed=3, edge=0.9, tex=TEX)
            img = wc.wash(img, poly_mask([flower5(fxx, fyy, R * 0.62, 0.9 + t * 0.6)]), (0.95, 0.5, 0.62), 0.5, seed=996, bleed=2.5, edge=0.8, tex=TEX)
            img = wc.wash(img, ell_mask([(fxx, fyy, R * 0.3, R * 0.3)]), (0.99, 0.84, 0.4), 0.85, seed=997, bleed=1.5, tex=TEX)
    # 歌名
    title, boxes = B["title"], B["boxes"]
    th, tw = title.shape[:2]
    tmask = np.zeros((H, W), np.float32)
    tcol = np.zeros((H, W, 3), np.float32)
    TX, TY = 560, -20
    out_fade = 1 - ease((t - 4.0) / 0.5)
    for kk, (bx0, by0, bx1, by1) in enumerate(boxes):
        p = ease((t - (1.0 + kk * BEAT / 4)) / 0.45)
        if p <= 0 or out_fade <= 0:
            continue
        cxk, cyk = (bx0 + bx1) / 2, (by0 + by1) / 2
        X0, X1 = int(max(bx0 - 14, 0)), int(min(bx1 + 14, tw))
        sub = title[:, X0:X1]
        yy_, xx_ = np.mgrid[0:th, X0:X1].astype(np.float32)
        rr = np.sqrt((xx_ - cxk) ** 2 + (yy_ - cyk) ** 2)
        reveal = np.clip((p * 170 - rr + 25 * (fbm(th, X1 - X0, 990 + kk, octaves=2, base=4) - 0.5)) / 12, 0, 1)
        jump = int(round(-10 * max(0.0, math.sin(math.pi * (beat + kk * 0.25))) * p))
        ys0 = TY + jump
        Y0, Y1 = max(ys0, 0), min(ys0 + th, H)
        a_ = sub[Y0 - ys0:Y1 - ys0, :, 3] * reveal[Y0 - ys0:Y1 - ys0] * out_fade
        tmask[Y0:Y1, TX + X0:TX + X1] = np.maximum(tmask[Y0:Y1, TX + X0:TX + X1], a_)
        tcol[Y0:Y1, TX + X0:TX + X1] = np.where(a_[..., None] > 0, sub[Y0 - ys0:Y1 - ys0, :, :3], tcol[Y0:Y1, TX + X0:TX + X1])
    img = img * (1 - tmask[..., None] * (1 - tcol))
    img = img + (np.exp(-(((x - 1560) / 700) ** 2 + ((y - 170) / 500) ** 2)))[..., None] * np.array([0.05, 0.04, 0.0])
    return img * (0.97 + 0.03 * TEX[..., None]), TEX


def world_b(tb):
    """05-b 世界（不含 Clawd）。tb 为 05-b 本地时间（落地 = 0）。"""
    B = bg_b()
    img, TEX = B["img"].copy(), B["TEX"]
    p = ease((tb - 0.1) / 0.75)
    cone = B["cone"] * np.clip((-200 + 1350 * p - y) / 120, 0, 1)
    img = img + cone[..., None] * (np.array([0.98, 0.95, 0.96]) - img) * 0.55
    q = ease((tb - 0.6) / 0.4)
    img = img + (np.exp(-(((x - 960) / 360) ** 2 + ((y - 980) / 55) ** 2)) * q)[..., None] * (np.array([1, 0.97, 0.98]) - img) * 0.6
    img = img + (np.exp(-(((x - 960) / 330) ** 2 + ((y - 965) / 40) ** 2)) * q)[..., None] * (np.array([1, 0.97, 0.98]) - img) * 0.6
    return img * (0.97 + 0.03 * TEX[..., None]), TEX

# ---------------------------------------------------------------- 相机


def cam_a(t):
    """05-a 相机：缓慢右移；T2 中以花心为不动点冲进去。返回 (zoom, dx, 中心)。"""
    tp = min(t, T2_START)
    z = 1.04 + 0.01 * tp / T2_START
    dx = -38 + 76 * ease(tp / T2_START)
    if t > 4.95:
        u = ease((t - 4.95) / (T2_END - 4.95))
        z = z * math.exp(math.log(34) * u ** 1.4)
        return z, dx, FLOWER
    return z, dx, (W / 2, H / 2)


def camera_fill(img, zoom, dx, dy, cx, cy):
    """同 camera()，但画面外用该图边缘的平均色填充，避免缩小时露出黑边。"""
    fill = tuple(int(v * 255) for v in np.clip(np.concatenate([img[:20].reshape(-1, 3), img[-20:].reshape(-1, 3)]).mean(0), 0, 1))
    a = 1 / zoom
    coeffs = (a, 0, cx - (cx + dx) * a, 0, a, cy - (cy + dy) * a)
    return to_arr(to_img(img).transform((W, H), Image.AFFINE, coeffs, resample=Image.BICUBIC, fillcolor=fill))


def to_screen(px, py, z, dx, c):
    return (px - c[0]) * z + c[0] + dx, (py - c[1]) * z + c[1]


def cam_b(tb):
    return 1.0 + 0.06 * ease(min(tb, 3.0) / 3.0), (960, 700)

# ---------------------------------------------------------------- 每帧


def frame(i):
    t = i / FPS
    beat = t / BEAT
    if t < T2_START:
        return frame_t1_a(t, beat)
    if t < T2_END:
        return frame_t2(t, beat)
    if t < T3_START:
        return frame_b(t, beat)
    return frame_t3(t, beat)


def frame_t1_a(t, beat):
    w, TEX = world_a(t)
    z, dx, c = cam_a(t)
    base = camera(w, z, dx, 0, *c)
    S = 17 * z
    if t < T1_END:
        u = t / T1_END
        wx = -170 + (A_GROUND[0] + 170) * (1 - (1 - u) ** 1.6)       # 走进来，最后几步放慢
        legs = walk_legs(beat * 2 * (1 - u * 0.3))
        ang = 3 * math.sin(math.pi * beat * 2)
        hop = 6 * abs(math.sin(math.pi * beat * 2))
    else:
        wx = A_GROUND[0]
        legs = None
        ang = -6 + 8 * math.sin(math.pi * (beat - T1_END / BEAT))
        hop = 14 * abs(math.sin(math.pi * (beat - T1_END / BEAT)))
    ax, ay = to_screen(wx, A_GROUND[1], z, dx, c)
    ay -= hop
    img_wc, P = draw_a(base.copy(), ax, ay, S, ang, legs, TEX)
    img_wc = water_drops(img_wc, t, ax, ay, S, ang, TEX, on=t > 2.0)
    seam_x = to_screen(SEAM_DOOR, 0, z, dx, c)[0]
    if ax - 10 * S < seam_x:        # Clawd 还有一部分在门里：门里那部分用 04 的金色画法
        img_g, _ = draw_gold(base.copy(), ax, ay, S, ang, legs, t)
        m = np.clip((seam_x - x) / 8 + 0.5, 0, 1)[..., None]
        img = img_g * m + img_wc * (1 - m)
        # 门框投下的光带：Clawd 正在穿过的那条线
        band = np.exp(-((x - seam_x) / 26) ** 2) * (0.6 + 0.4 * ease(min(1, (seam_x - (ax - 10 * S)) / (4 * S))))
        img = img + band[..., None] * (np.array([1.0, 0.95, 0.8]) - img) * 0.55
    else:
        img = img_wc
    if t < 0.5:
        img = img + (1 - ease(t / 0.5)) * (np.array([1.0, 0.92, 0.72]) - img) * 0.8   # 承接 04 的金色光
    return img


def water_drops(img, t, ax, ay, S, ang, TEX, on=True):
    if not on:
        return img
    sx, sy = grid_pt(24.6, 3.3, ax, ay, S, ang)
    dirx = math.cos(math.radians(-ang - 30))
    g = rng(75)
    jit = g.uniform(-6, 6, 14)
    drops = []
    for k in range(14):
        u = ((k / 14) + t * 1.3) % 1.0
        drops.append((sx + 20 + 130 * u * dirx * S / 17 + jit[k], sy + 10 + 230 * u * u * S / 17, 5 * S / 17, 7 * S / 17))
    return wc.wash(img, ell_mask(drops), (0.6, 0.8, 0.95), 0.75, seed=76, bleed=1, tex=TEX)


def frame_t2(t, beat):
    """跳进花心：花外是 05-a（被镜头冲进去），花心里是 05-b（从小变大），Clawd 穿过花心边缘时换装。"""
    wa, TEXa = world_a(t)
    z, dx, c = cam_a(t)
    A = camera(wa, z, dx, 0, *c)
    fx, fy = to_screen(*FLOWER, z, dx, c)
    core_r = 150 * 0.3 * z                         # 花心半径（屏幕）
    r_full = math.hypot(max(fx, W - fx), max(fy, H - fy)) + 40
    sb = min(1.0, core_r / r_full)
    wb, TEXb = world_b(0.0)
    # 花心里的 05-b：缩放到刚好盖住花心（不露边），随入口变大逐渐恢复原大小
    sc = min(1.0, max(sb, 2.6 * core_r / H))
    tgt = (fx + (960 - fx) * sc, fy + (540 - fy) * sc)
    Bimg = camera_fill(wb, sc, tgt[0] - 960, tgt[1] - 540, 960, 540)
    # 入口遮罩：花心的不规则圆
    ang_ = np.arctan2(y - fy, x - fx)
    edge = core_r * (1 + 0.06 * np.sin(ang_ * 5 + t * 2) + 0.03 * np.sin(ang_ * 11))
    portal = np.clip((edge - np.sqrt((x - fx) ** 2 + (y - fy) ** 2)) / 6, 0, 1) if sb < 1 else np.ones((H, W), np.float32)
    # Clawd 的轨迹：起跳（一拍）→ 冲进花心 → 落在舞台
    u = (t - T2_START) / (T2_END - T2_START)
    if u > 0.97:
        sb, portal = 1.0, np.ones((H, W), np.float32)
        Bimg = wb
    S = 17 * 1.04 + (26 - 17 * 1.04) * ease(u)
    g0 = to_screen(*A_GROUND, *cam_a(T2_START))
    land = B_GROUND
    mid = (fx, fy + 6 * S)                          # 身体中心对准花心
    if u < 0.5:
        v = ease(u / 0.5)
        ax = g0[0] + (mid[0] - g0[0]) * v
        ay = g0[1] + (mid[1] - g0[1]) * v - 260 * math.sin(math.pi * v * 0.9)
    else:
        v = ease((u - 0.5) / 0.5)
        ax = mid[0] + (land[0] - mid[0]) * v
        ay = mid[1] + (land[1] - mid[1]) * v - 120 * math.sin(math.pi * v)
    ang = -10 * math.sin(math.pi * u)               # 空中轻轻一转身体的角度
    legs = walk_legs(0.5) if 0.1 < u < 0.9 else None   # 空中收腿（整块抬起）
    imgA, _ = draw_a(A.copy(), ax, ay, S, ang, legs, TEXa, shadow=0.0)
    if portal.max() > 0:
        imgB, _ = draw_b(Bimg.copy(), ax, ay, S, ang, legs, TEXb, int(t * 10), lit=0.4)
        img = imgA * (1 - portal[..., None]) + imgB * portal[..., None]
        # 花心边缘的一圈亮光（入口）
        ring = np.exp(-((np.sqrt((x - fx) ** 2 + (y - fy) ** 2) - edge) / 10) ** 2) * (1 - sb)
        img = img + ring[..., None] * (np.array([1.0, 0.95, 0.85]) - img) * 0.7
    else:
        img = imgA
    return img


def frame_b(t, beat):
    tb = t - T2_END
    w, TEX = world_b(tb)
    z, c = cam_b(tb)
    base = camera(w, z, 0, 0, *c)
    ang = 5 + 4 * math.sin(math.pi * beat)
    ax, ay = to_screen(B_GROUND[0] + 5 * math.sin(math.pi * beat), B_GROUND[1], z, 0, c)
    ay -= 4 * abs(math.sin(math.pi * beat))
    # 落地后的一个小回弹
    if tb < 0.3:
        ay += 10 * math.sin(math.pi * tb / 0.3)
    lit = 0.4 + 0.6 * ease((tb - 0.4) / 0.6)
    img, _ = draw_b(base, ax, ay, 26 * z, ang, None, TEX, int(tb * 10), lit)
    return img


def frame_t3(t, beat):
    """往右走进镜子：镜面从右向左扫过，镜子那边是 06 的镜面长廊，Clawd 跨过镜面的部分换成 06 造型。"""
    tb = t - T2_END
    w, TEX = world_b(tb)
    z, c = cam_b(tb)
    Bimg = camera(w, z, 0, 0, *c)
    u = (t - T3_START) / (END - T3_START)
    wx = B_GROUND[0] + 420 * ease(min(1, u / 0.8))
    ax, ay = to_screen(wx, B_GROUND[1], z, 0, c)
    S = 26 * z
    walking = u < 0.8
    legs = walk_legs(beat * 2) if walking else None
    ang = (3 * math.sin(math.pi * beat * 2)) if walking else 0
    ay -= (5 * abs(math.sin(math.pi * beat * 2))) if walking else 0
    seam = W + 60 - (W + 160) * ease(min(1, max(0, (u - 0.12) / 0.75)))
    imgL, _ = draw_b(Bimg.copy(), ax, ay, S, ang, legs, TEX, int(tb * 10), 1.0)
    C = cel.corridor(True).copy()
    # 镜子那边的 Clawd：06 的赛璐璐造型，地面高度对到长廊里最前排的位置
    imgR = draw_cel(C, ax, ay, S, ang, legs)        # 与镜子这边的 Clawd 完全重合，接缝两侧不跳
    m = np.clip((x - seam) / 4 + 0.5, 0, 1)[..., None]
    img = imgL * (1 - m) + imgR * m
    # 镜面边缘：一道竖亮线 + 星芒
    glow = np.exp(-((x - seam) / 5) ** 2) + 0.35 * np.exp(-((x - seam) / 40) ** 2)
    img = img + glow[..., None] * (np.array([0.95, 0.97, 1.0]) - img) * 0.85
    if 0 < seam < W:
        img = cel.star4(img, seam, 140 + 40 * math.sin(t * 4), 46)
    return img


if __name__ == "__main__":
    out = sys.argv[1]
    stills = sys.argv[2] if len(sys.argv) > 2 else None
    if len(sys.argv) > 3:
        os.makedirs(stills, exist_ok=True)
        for i in map(int, sys.argv[3].split(",")):
            save(frame(i), os.path.join(stills, f"测试帧_{i:04d}.png"))
            print("frame", i, flush=True)
    else:
        render_video(frame, N, out, stills=(10, 30, 50, 90, 140, 150, 160, 172, 200, 250, 282, 296, 314), stills_dir=stills)
