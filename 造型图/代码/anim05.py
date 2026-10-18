"""05｜我管不住我自己 · 动画样片（30fps，9 秒 = 270 帧）
0.0–5.0s  05-a 晨光阳台：晨光从过曝中淡入；歌名逐字晕开、跟着节拍跳；Clawd 每拍左右摇摆、轻跳着浇花，水滴洒落；
          音符化成的水彩小花从唱机一路飘出；蒸汽升起；镜头缓慢右移；4.4s 起一朵水彩花晕开到满屏。
5.0–9.0s  05-b 南京现场：花色退去露出舞台；追光从上方落下；Clawd 跟着节拍轻轻摇晃，亮片闪烁；最后亮片反光闪成一片白（接 06 的镜面）。
用法：python3 anim05.py 输出.mp4 [静帧目录]"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from titles import watercolor_title, text_mask
import watercolor as wc
from video import render_video, ease, FPS

SS = 2
BEAT = 0.6          # 每拍 0.6 秒（约 100 BPM）
DUR_A, DUR_B = 5.0, 4.0
N = int((DUR_A + DUR_B) * FPS)
ORANGE_WC = (0.86, 0.52, 0.38)
x, y = grid()


def poly_mask(polys):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([(px * SS, py * SS) for px, py in p], fill=255)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def ell_mask(items):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for (cx, cy, rx, ry) in items:
        d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS], fill=255)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def line_mask(lines, width):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for pl in lines:
        d.line([(px * SS, py * SS) for px, py in pl], fill=255, width=int(width * SS), joint="curve")
    return to_arr(im.resize((W, H), Image.LANCZOS))


def place_mask(m, cx, cy, angle=0.0):
    im = to_img(m)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    a = to_arr(im)
    out = np.zeros((H, W), np.float32)
    h, w = a.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 > X0 and Y1 > Y0:
        out[Y0:Y1, X0:X1] = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    return out


def grid_to_canvas(gx_, gy_, S, ox, oy, pw, ph, cx, cy, ang):
    """格坐标 → 画布坐标，与 place_mask 的旋转（PIL 逆时针）一致。"""
    vx, vy = ox + gx_ * S - pw / 2, oy + gy_ * S - ph / 2
    a = math.radians(ang)
    return cx + vx * math.cos(a) + vy * math.sin(a), cy - vx * math.sin(a) + vy * math.cos(a)


def flower5(cx, cy, r, rot=0.0):
    pts = []
    for i in range(60):
        a = rot + 2 * math.pi * i / 60
        rr = r * (0.55 + 0.45 * abs(math.cos(a * 2.5 - rot * 2.5)))
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def camera(img, zoom, dx, dy, cx=W / 2, cy=H / 2):
    """以 (cx,cy) 为不动点放大 zoom 倍并平移 (dx,dy)，输出仍为 W×H。"""
    im = to_img(img)
    a = 1 / zoom
    # 以 (cx,cy) 为不动点缩放：输出像素 u 取自源 (u - cx - dx)/zoom + cx
    coeffs = (a, 0, cx - (cx + dx) * a, 0, a, cy - (cy + dy) * a)
    return to_arr(im.transform((W, H), Image.AFFINE, coeffs, resample=Image.BICUBIC))

# =====================================================================
# 静态背景（每个进程首次调用时生成）
# =====================================================================
_cache = {}


def bg_a():
    if "a" in _cache:
        return _cache["a"]
    img, TEX = wc.paper()
    skym = (y < 640).astype(np.float32)
    img = wc.wash(img, skym * np.clip(1 - y / 640, 0, 1) ** 0.8, (0.62, 0.80, 0.95), 0.55, seed=1, bleed=10, edge=0.1, tex=TEX)
    img = wc.wash(img, skym * np.clip((y - 250) / 390, 0, 1), (0.99, 0.86, 0.58), 0.45, seed=2, bleed=14, edge=0.05, tex=TEX)
    sun = np.exp(-(((x - 1560) ** 2 + (y - 170) ** 2) / (2 * 140 ** 2)))
    img = img + sun[..., None] * (np.array([1, 0.99, 0.95]) - img) * 0.85
    roofs, gg, xx = [], rng(3), -40
    while xx < W:
        w_, h_ = gg.uniform(90, 200), gg.uniform(50, 130)
        roofs.append([(xx, 640), (xx, 640 - h_), (xx + w_ * 0.5, 640 - h_ - gg.uniform(20, 50)), (xx + w_, 640 - h_), (xx + w_, 640)])
        xx += w_ + gg.uniform(10, 40)
    img = wc.wash(img, poly_mask(roofs), (0.78, 0.74, 0.82), 0.35, seed=4, bleed=3, tex=TEX)
    trees = ell_mask([(gg.uniform(0, W), 620 - gg.uniform(0, 40), gg.uniform(30, 70), gg.uniform(25, 50)) for _ in range(14)])
    img = wc.wash(img, trees, (0.66, 0.82, 0.62), 0.4, seed=5, bleed=4, tex=TEX)
    LDY = 75
    line_y = lambda xv: 175 + LDY + 0.00012 * (xv - 1000) ** 2
    img = img * (1 - line_mask([[(xv, line_y(xv)) for xv in range(300, 1500, 20)]], 2)[..., None] * 0.55)
    shirt = [(560, line_y(560)), (720, line_y(720)), (760, 250), (730, 268), (720, 400), (560, 400), (550, 268), (520, 250)]
    towel = [(860, line_y(860)), (1010, line_y(1010)), (1018, 350), (1000, 362), (980, 348), (960, 362), (940, 348), (920, 362), (900, 348), (880, 362), (852, 350)]
    sock = [(1140, line_y(1140)), (1180, line_y(1180)), (1186, 260), (1210, 280), (1196, 300), (1150, 276)]
    dyp = lambda P: [(a, b + (LDY if abs(b - line_y(a)) > 0.01 else 0)) for a, b in P]
    shirt, towel, sock = dyp(shirt), dyp(towel), dyp(sock)
    img = wc.wash(img, poly_mask([shirt]), (0.96, 0.66, 0.72), 0.65, seed=6, tex=TEX)
    img = wc.wash(img, poly_mask([towel]), (0.62, 0.86, 0.78), 0.65, seed=7, tex=TEX)
    img = wc.wash(img, poly_mask([towel]) * (np.sin(x / 9) > 0.5), (0.5, 0.78, 0.72), 0.4, seed=8, bleed=1.5, tex=TEX)
    img = wc.wash(img, poly_mask([sock]), (0.98, 0.84, 0.5), 0.65, seed=9, tex=TEX)
    pins = poly_mask([[(px - 5, line_y(px) - 12), (px + 5, line_y(px) - 12), (px + 4, line_y(px) + 14), (px - 4, line_y(px) + 14)]
                      for px in (575, 705, 875, 995, 1150, 1176)])
    img = wc.wash(img, pins, (0.75, 0.6, 0.45), 0.8, seed=10, bleed=1, tex=TEX)
    floor = (y > 790).astype(np.float32)
    img = wc.wash(img, floor, (0.93, 0.72, 0.58), 0.6, seed=11, bleed=3, tex=TEX)
    tile_lines = [[(960 + k * 40, 790), (960 + k * 220, H + 10)] for k in range(-14, 15)] + [[(0, yy), (W, yy)] for yy in (830, 885, 960, 1060)]
    img = img * (1 - line_mask(tile_lines, 2)[..., None] * 0.1)
    rail = [[(0, 600), (W, 600)], [(0, 780), (W, 780)]]
    for k in range(0, W, 64):
        rail.append([(k, 600), (k, 780)])
        rail.append([(k + 32, 700), (k + 20, 688), (k + 32, 676), (k + 44, 688), (k + 32, 700)])
    img = img * (1 - line_mask(rail, 4)[..., None] * np.array([0.55, 0.5, 0.5]))
    door = poly_mask([[(0, 0), (230, 0), (230, H), (0, H)]]) - poly_mask([[(40, 60), (190, 60), (190, 1000), (40, 1000)]])
    img = wc.wash(img, np.clip(door, 0, 1), (0.60, 0.84, 0.76), 0.7, seed=12, tex=TEX)
    img = wc.wash(img, poly_mask([[(40, 60), (190, 60), (190, 1000), (40, 1000)]]), (0.85, 0.93, 0.96), 0.3, seed=13, tex=TEX)
    leaves = []
    for (lx, ly, r, a) in [(330, 560, 110, -0.6), (440, 470, 120, -0.2), (250, 690, 100, -1.0), (520, 610, 90, 0.4), (360, 400, 90, -0.9)]:
        pts = []
        for i in range(40):
            t_ = 2 * math.pi * i / 40
            rr = r * (0.75 + 0.25 * math.cos(t_)) * (0.82 if math.sin(t_ * 6) > 0.6 else 1.0)
            pts.append((lx + rr * math.cos(t_ + a), ly + rr * math.sin(t_ + a) * 0.72))
        leaves.append(pts)
    stems = line_mask([[(380, 880), (330, 560)], [(390, 880), (440, 470)], [(370, 880), (250, 690)], [(400, 880), (520, 610)], [(375, 880), (360, 400)]], 6)
    img = wc.wash(img, stems, (0.5, 0.68, 0.45), 0.45, seed=16, bleed=1, tex=TEX)
    img = wc.wash(img, poly_mask(leaves), (0.35, 0.66, 0.42), 0.72, seed=14, tex=TEX)
    img = wc.wash(img, poly_mask(leaves[::2]), (0.3, 0.55, 0.36), 0.35, seed=15, bleed=6, tex=TEX)
    pots = poly_mask([[(300, 860), (470, 860), (450, 990), (320, 990)], [(1240, 740), (1330, 740), (1320, 800), (1250, 800)],
                      [(640, 745), (710, 745), (702, 795), (648, 795)]])
    img = wc.wash(img, pots, (0.86, 0.5, 0.38), 0.75, seed=17, tex=TEX)
    for i, (bx, by, br) in enumerate([(1270, 715, 22), (1305, 700, 18), (1290, 690, 15), (660, 720, 16), (690, 712, 14)]):
        img = wc.wet_bloom(img, bx, by, br, [(0.95, 0.5, 0.62), (0.98, 0.78, 0.4)][i % 2], seed=20 + i, density=0.8, tex=TEX)
    img = wc.wash(img, ell_mask([(1580, 770, 230, 46)]), (0.95, 0.92, 0.86), 0.5, seed=18, tex=TEX)
    img = wc.wash(img, line_mask([[(1580, 790), (1580, 1000)], [(1500, 1000), (1660, 1000)]], 10), (0.55, 0.45, 0.4), 0.7, seed=19, tex=TEX)
    img = wc.wash(img, poly_mask([[(1420, 690), (1690, 690), (1710, 760), (1400, 760)]]), (0.72, 0.5, 0.35), 0.8, seed=21, tex=TEX)
    img = wc.wash(img, ell_mask([(1545, 700, 105, 30)]), (0.15, 0.13, 0.16), 0.9, seed=22, bleed=1.5, tex=TEX)
    img = wc.wash(img, ell_mask([(1545, 700, 22, 7)]), (0.95, 0.45, 0.5), 0.9, seed=23, bleed=1, tex=TEX)
    img = wc.wash(img, line_mask([[(1660, 690), (1640, 660), (1580, 700)]], 4), (0.5, 0.5, 0.52), 0.85, seed=24, bleed=0.8, tex=TEX)
    img = wc.wash(img, poly_mask([[(1730, 700), (1790, 700), (1784, 760), (1736, 760)]]), (0.98, 0.98, 0.98), 0.4, seed=25, tex=TEX)
    img = wc.wash(img, line_mask([[(1790, 712), (1808, 722), (1790, 742)]], 5), (0.85, 0.85, 0.88), 0.6, seed=26, bleed=1, tex=TEX)
    outline = [p + [p[0]] for p in [shirt, towel, [(1420, 690), (1690, 690), (1710, 760), (1400, 760)]]]
    img = wc.pencil(img, outline, seed=78, alpha=0.3)
    # Clawd 的几何（只算一次）
    S = 17
    can = [(16.8, 4.4), (20.6, 4.4), (21.0, 8.0), (16.4, 8.0)]
    spout = [(20.6, 5.2), (24.4, 3.0), (24.8, 3.6), (21.0, 6.4)]
    handle = [(16.8, 4.4), (17.6, 2.8), (19.6, 2.8), (20.4, 4.4), (19.6, 4.4), (19.2, 3.6), (18.0, 3.6), (17.6, 4.4)]
    cm, (ox, oy), (pw, ph) = clawd_masks(S, 40, bbox=(-2, -1, 27, 14), extra={"can": [can, spout, handle]})
    # 歌名：整行水彩字 + 每个字的外框（与 watercolor_title 同参数）
    title = watercolor_title("我管不住我自己", size=150, seed=4, colors=((0.92, 0.42, 0.52), (0.95, 0.66, 0.2)))
    _, boxes = text_mask("我管不住我自己", "wenkai", 150, pad=80, jitter=(14, 4), seed=4)
    _cache["a"] = dict(img=img, TEX=TEX, S=S, cm=cm, geo=(ox, oy, pw, ph), title=title, boxes=boxes)
    return _cache["a"]


def bg_b():
    if "b" in _cache:
        return _cache["b"]
    img, TEX = wc.paper(seed=520)
    for k, (c, dns) in enumerate([((0.30, 0.30, 0.55), 0.9), ((0.42, 0.28, 0.5), 0.7), ((0.25, 0.25, 0.45), 0.8), ((0.5, 0.3, 0.45), 0.5)]):
        m = np.clip(fbm(H, W, 530 + k, octaves=3, base=3) * 1.4 - 0.1, 0, 1)
        img = wc.wash(img, m, c, dns, seed=531 + k, bleed=20, edge=0.3, tex=TEX)
    gg = rng(540)
    for i in range(22):
        bx, by = gg.uniform(0, W), gg.uniform(0, 760)
        if abs(bx - 960) < 380:
            continue
        img = wc.wet_bloom(img, bx, by, gg.uniform(18, 46), [(0.95, 0.75, 0.85), (0.8, 0.8, 1.0), (1.0, 0.9, 0.8)][i % 3], seed=541 + i, density=0.35, tex=TEX)
        img = img + (ell_mask([(bx, by, 6, 6)]) * 0.35)[..., None]
    img = wc.wash(img, np.clip((y - 930) / 30, 0, 1), (0.25, 0.22, 0.4), 0.75, seed=545, bleed=6, edge=0.1, tex=TEX)
    cone = blur(poly_mask([[(860, -20), (1060, -20), (1300, 1000), (620, 1000)]]), 50) * np.clip(1 - y / 1300, 0.3, 1)

    def ruffle(x0, x1, ytop, ybot, waves, amp=0.35, flare=0.0):
        top = [(x0 + flare * 0.3, ytop), (x1 - flare * 0.3, ytop)]
        bot = [(x1 + flare - (x1 - x0 + 2 * flare) * (i / (waves * 8)), ybot + amp * math.sin(i / (waves * 8) * waves * 2 * math.pi)) for i in range(waves * 8 + 1)]
        return top + bot
    bodice = [(0.0, 5.6), (16.0, 5.6), (16.2, 8.4), (-0.2, 8.4)]
    tiers = [ruffle(-0.4, 16.4, 7.8, 9.0, 6, 0.22, 0.6), ruffle(-0.8, 16.8, 8.8, 10.0, 7, 0.22, 1.2), ruffle(-1.2, 17.2, 9.8, 10.9, 8, 0.22, 1.8)]
    trail = [(16.6, 8.4), (18.8, 9.0), (20.8, 10.2), (22.2, 11.4), (21.4, 11.9), (20.6, 11.5), (19.8, 12.0), (19.0, 11.5), (18.2, 11.8), (17.4, 10.8)]
    arm_chest = rigid_arm('R', -35)
    check_proportions(arms=[arm_chest], extra={"dress": [bodice] + tiers, "trail": [trail]})
    S = 26
    cm, (ox, oy), (pw, ph) = clawd_masks(S, 40, bbox=(-4, -1, 24, 14), arms=[(-1, 3, 0, 5), arm_chest],
                                         extra={"dress": [bodice] + tiers, "tiers": tiers[1:], "trail": [trail]})
    _cache["b"] = dict(img=img, TEX=TEX, cone=cone, S=S, cm=cm, geo=(ox, oy, pw, ph))
    return _cache["b"]

# =====================================================================
# 05-a 每帧
# =====================================================================
FLOWER_COLS = [(0.95, 0.5, 0.62), (0.98, 0.78, 0.36), (0.52, 0.8, 0.72), (0.72, 0.6, 0.9)]


def frame_a(t):
    B = bg_a()
    img, TEX, S, cm = B["img"].copy(), B["TEX"], B["S"], B["cm"]
    ox, oy, pw, ph = B["geo"]
    beat = t / BEAT
    # 唱片反光绕圈
    a = t * 2 * math.pi / 1.8
    img = img + (ell_mask([(1545 + 70 * math.cos(a), 700 + 18 * math.sin(a), 26, 5)]) * 0.22)[..., None]
    # 蒸汽
    steam = line_mask([[(1750 + 12 * math.sin(s_ / 9 - t * 3), 690 - s_ - (t * 20) % 12) for s_ in range(0, 90, 3)],
                       [(1772 + 10 * math.sin(s_ / 8 + 1 - t * 2.6), 690 - s_ - (t * 16) % 10) for s_ in range(0, 80, 3)]], 4)
    img = wc.wash(img, steam, (0.82, 0.82, 0.88), 0.4, seed=27, bleed=3, tex=TEX)
    # 音符小花：沿路径从唱机持续飘出
    for i in range(16):
        u = ((i + t * 2.0) / 16) % 1.0
        fx = 1450 - u * 760 + 60 * math.sin(u * 7)
        fy = 640 - u * 330 - 50 * math.cos(u * 5) - 6 * math.sin(beat * math.pi * 2 + i)
        r = (14 + 14 * math.sin(math.pi * u)) * min(1, u * 8)
        fade = min(1.0, u * 6, (1 - u) * 5)
        if r < 2 or fade <= 0:
            continue
        img = wc.wash(img, poly_mask([flower5(fx, fy, r, i * 0.7 + t * 1.5)]), FLOWER_COLS[i % 4], 0.7 * fade, seed=40 + i, bleed=2.5, edge=0.7, tex=TEX)
        img = wc.wash(img, ell_mask([(fx, fy, r * 0.22, r * 0.22)]), (0.98, 0.8, 0.3), 0.8 * fade, seed=60 + i, bleed=1, tex=TEX)
    # 铅笔音符：跟节拍上下点
    bob = 8 * math.sin(beat * math.pi * 2)
    notes = []
    for k, (nx, ny, s_) in enumerate([(1380, 560, 1.0), (1180, 470, 0.8), (980, 380, 0.9), (1290, 400, 0.7)]):
        ny = ny + bob * (1 if k % 2 else -1)
        notes.append([(nx, ny), (nx, ny - 50 * s_), (nx + 26 * s_, ny - 40 * s_)])
        notes.append([(nx - 10 * s_, ny + 4), (nx, ny + 8 * s_), (nx + 4, ny), (nx - 8 * s_, ny - 4 * s_), (nx - 10 * s_, ny + 4)])
    img = wc.pencil(img, notes, seed=31, alpha=0.5, width=2)
    # Clawd：每两拍左右摆一次，每拍轻跳
    ang = -6 + 8 * math.sin(math.pi * beat)
    hop = 14 * abs(math.sin(math.pi * beat))
    cx, cy = 930 + 6 * math.sin(math.pi * beat), 657 - hop
    bodyM = place_mask(np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1), cx, cy, ang)
    eyeM = place_mask(cm["eyes"], cx, cy, ang)
    canM = place_mask(cm["can"], cx, cy, ang)
    shadow = place_mask(np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1), cx + 22, 657 + 18, ang)
    img = wc.wash(img, shadow * (1 - bodyM), (0.75, 0.68, 0.72), 0.45 * (1 - hop / 40), seed=70, bleed=8, tex=TEX)
    img = wc.wash(img, bodyM, ORANGE_WC, 0.85, seed=71, bleed=1.2, edge=0.5, tex=TEX)
    img = wc.wash(img, bodyM * np.clip((x - cx + 60) / 200, 0, 1), (0.8, 0.45, 0.35), 0.35, seed=72, bleed=4, tex=TEX)
    img = wc.wash(img, eyeM, (0.12, 0.1, 0.12), 0.95, seed=73, bleed=0.6, edge=0.2, tex=TEX)
    img = wc.wash(img, canM, (0.45, 0.78, 0.66), 0.85, seed=74, bleed=1, tex=TEX)
    # 水滴：从壶嘴沿抛物线持续落下
    tx_, ty_ = grid_to_canvas(24.6, 3.3, S, ox, oy, pw, ph, cx, cy, ang)
    dirx = math.cos(math.radians(-ang - 30))
    drops = []
    g = rng(75)
    jit = g.uniform(-6, 6, 14)
    for k in range(14):
        u = ((k / 14) + t * 1.6) % 1.0
        drops.append((tx_ + 20 + 130 * u * dirx + jit[k], ty_ + 10 + 230 * u * u, 5, 7))
    img = wc.wash(img, ell_mask(drops), (0.6, 0.8, 0.95), 0.75, seed=76, bleed=1, tex=TEX)
    # 摇摆动感线：在摆动方向的另一侧
    side = 1 if math.cos(math.pi * beat) > 0 else -1
    img = wc.pencil(img, [[(cx - side * 200, cy - 120 + k * 30), (cx - side * 240, cy - 100 + k * 30)] for k in range(3)], seed=77, alpha=0.4)
    # 歌名：0.3s 起逐字晕开，跟拍跳动，3.4s 后渐淡
    title, boxes = B["title"], B["boxes"]
    th, tw = title.shape[:2]
    tmask = np.zeros((H, W), np.float32)
    tcol = np.zeros((H, W, 3), np.float32)
    TX, TY = 560, -20
    out_fade = 1 - ease((t - 3.4) / 0.6)
    for k, (bx0, by0, bx1, by1) in enumerate(boxes):
        t0 = 0.3 + k * 0.09
        p = ease((t - t0) / 0.45)
        if p <= 0 or out_fade <= 0:
            continue
        cxk, cyk = (bx0 + bx1) / 2, (by0 + by1) / 2
        X0, X1 = int(max(bx0 - 14, 0)), int(min(bx1 + 14, tw))
        sub = title[:, X0:X1]
        yy_, xx_ = np.mgrid[0:th, X0:X1].astype(np.float32)
        rr = np.sqrt((xx_ - cxk) ** 2 + (yy_ - cyk) ** 2)
        reveal = np.clip((p * 170 - rr + 25 * (fbm(th, X1 - X0, 990 + k, octaves=2, base=4) - 0.5)) / 12, 0, 1)
        jump = int(round(-10 * max(0.0, math.sin(math.pi * (beat + k * 0.25))) * p))
        ys0 = TY + jump
        Y0, Y1 = max(ys0, 0), min(ys0 + th, H)
        a_ = sub[Y0 - ys0:Y1 - ys0, :, 3] * reveal[Y0 - ys0:Y1 - ys0] * out_fade
        tmask[Y0:Y1, TX + X0:TX + X1] = np.maximum(tmask[Y0:Y1, TX + X0:TX + X1], a_)
        tcol[Y0:Y1, TX + X0:TX + X1] = np.where(a_[..., None] > 0, sub[Y0 - ys0:Y1 - ys0, :, :3], tcol[Y0:Y1, TX + X0:TX + X1])
    img = img * (1 - tmask[..., None] * (1 - tcol))
    img = img + (np.exp(-(((x - 1560) / 700) ** 2 + ((y - 170) / 500) ** 2)))[..., None] * np.array([0.05, 0.04, 0.0])
    img = img * (0.97 + 0.03 * TEX[..., None])
    # 结尾：一朵水彩花晕开到满屏
    if t > 4.4:
        p = ease((t - 4.4) / 0.6)
        img = pink_bloom(img, p, TEX)
    # 镜头：缓慢右移 + 轻微推近
    img = camera(img, 1.04 + 0.01 * t / DUR_A, -38 + 76 * ease(t / DUR_A), 0)
    # 开头：晨光过曝淡入（承接 04 的金色光点）
    if t < 0.7:
        k = 1 - ease(t / 0.7)
        img = img + k * (np.array([1.0, 0.93, 0.75]) - img)
    return img

# =====================================================================
# 05-b 每帧
# =====================================================================

def frame_b(tb):
    B = bg_b()
    img, TEX, S, cm = B["img"].copy(), B["TEX"], B["S"], B["cm"]
    ox, oy, pw, ph = B["geo"]
    beat = tb / BEAT
    # 追光从上往下落下
    p = ease((tb - 0.15) / 0.8)
    ylim = -200 + 1350 * p
    cone = B["cone"] * np.clip((ylim - y) / 120, 0, 1)
    img = img + cone[..., None] * (np.array([0.98, 0.95, 0.96]) - img) * 0.55
    pool = np.exp(-(((x - 960) / 360) ** 2 + ((y - 980) / 55) ** 2)) * ease((tb - 0.7) / 0.4)
    img = img + pool[..., None] * (np.array([1, 0.97, 0.98]) - img) * 0.6
    img = img + (np.exp(-(((x - 960) / 330) ** 2 + ((y - 965) / 40) ** 2)) * ease((tb - 0.7) / 0.4))[..., None] * (np.array([1, 0.97, 0.98]) - img) * 0.6
    # Clawd：享受地轻轻摇晃
    ang = 5 + 4 * math.sin(math.pi * beat)
    cx, cy = 960 + 5 * math.sin(math.pi * beat), 801 - 4 * abs(math.sin(math.pi * beat))
    lit = 0.25 + 0.75 * ease((tb - 0.5) / 0.6)     # 追光落到身上之前人物偏暗
    Bm = place_mask(cm["body"], cx, cy, ang)
    Am = place_mask(cm["arms"], cx, cy, ang)
    E = place_mask(cm["eyes"], cx, cy, ang)
    D = place_mask(cm["dress"], cx, cy, ang)
    T = place_mask(cm["tiers"], cx, cy, ang)
    TR = place_mask(cm["trail"], cx, cy, ang)
    L = place_mask(cm["legs"], cx, cy, ang)
    fig = np.clip(Bm + Am + D + TR + L, 0, 1)
    img = img + blur(fig, 2)[..., None] * (np.array([0.98, 0.97, 0.97]) - img) * 0.85 * lit
    img = wc.wash(img, np.clip(Bm + Am - D, 0, 1), ORANGE_WC, 0.85, seed=550, bleed=1.2, tex=TEX)
    img = wc.wash(img, E, (0.12, 0.1, 0.12), 0.95, seed=551, bleed=0.6, edge=0.2, tex=TEX)
    img = wc.wash(img, D, (0.80, 0.78, 0.88), 0.7, seed=552, bleed=1.5, edge=0.8, tex=TEX)
    img = wc.wash(img, T, (0.74, 0.72, 0.86), 0.45, seed=553, bleed=1.2, edge=1.0, tex=TEX)
    img = wc.wash(img, np.clip(T - shift(T, -10, 0), 0, 1), (0.6, 0.58, 0.76), 0.6, seed=5531, bleed=1.5, tex=TEX)
    img = wc.wash(img, TR, (0.86, 0.84, 0.92), 0.55, seed=554, bleed=2, edge=0.9, tex=TEX)
    img = wc.wash(img, L, (0.78, 0.78, 0.86), 0.75, seed=555, bleed=1.2, edge=0.8, tex=TEX)
    # 亮片闪烁：每 0.1 秒换一批亮点
    tw_seed = int(tb * 10)
    img = wc.gouache_dots(img, np.clip(D + L, 0, 1), int(520 * lit), seed=556 + tw_seed, r=(1.2, 3.0))
    img = wc.gouache_dots(img, np.clip(D + L, 0, 1), int(120 * lit), seed=7557 + tw_seed, r=(1.0, 2.2), color=(0.72, 0.74, 0.95))
    for i, (rgx, rgy, rr) in enumerate([(-0.4, 5.8, 1.7), (15.2, 8.2, 1.5), (1.2, 10.0, 1.1), (20.4, 10.8, 1.1), (1.4, 5.0, 0.9), (16.6, 9.6, 0.9)]):
        px_, py_ = grid_to_canvas(rgx, rgy, S, ox, oy, pw, ph, cx, cy, ang)
        r_ = rr * S
        img = wc.wet_bloom(img, px_, py_, r_, (0.92, 0.55, 0.66), seed=560 + i, density=0.85, tex=TEX)
        spiral = [[(px_ + r_ * 0.75 * (1 - s_ / 30) * math.cos(s_ * 0.55), py_ + r_ * 0.75 * (1 - s_ / 30) * math.sin(s_ * 0.55)) for s_ in range(30)]]
        img = wc.wash(img, line_mask(spiral, 3), (0.7, 0.3, 0.42), 0.7, seed=570 + i, bleed=0.6, tex=TEX)
        leaf = [(px_ - r_ * 1.2, py_ + r_ * 0.4), (px_ - r_ * 0.4, py_ + r_ * 0.2), (px_ - r_ * 0.6, py_ + r_ * 0.8)]
        img = wc.wash(img, poly_mask([leaf]), (0.55, 0.72, 0.6), 0.6, seed=580 + i, bleed=1, tex=TEX)
    img = img + blur(fig, 30)[..., None] * 0.04 * lit
    img = img * (0.97 + 0.03 * TEX[..., None])
    # 结尾：亮片反光放大成十字星芒，整幅闪白（接 06 的镜面）
    if tb > 3.3:
        q = ease((tb - 3.3) / 0.7)
        g = rng(4242)
        ys, xs = np.nonzero(np.clip(D + L, 0, 1) > 0.6)
        glint = np.zeros((H, W), np.float32)
        pts = []
        for j in g.choice(len(xs), 6, replace=False):
            sx, sy = xs[j], ys[j]
            Lr = 30 + 380 * q * g.uniform(0.4, 1.0)
            w_ = 3 + 6 * q
            rot = g.uniform(-0.3, 0.3)
            star = []
            for k in range(8):
                a = rot + k * math.pi / 4
                rr = Lr if k % 2 == 0 else w_
                star.append((sx + rr * math.cos(a), sy + rr * math.sin(a)))
            pts.append(star)
        glint = poly_mask(pts)
        img = img + (blur(glint, 1.5) + blur(glint, 12) * 0.6)[..., None] * (1 - img) * 0.9
        img = img + ease((tb - 3.75) / 0.25) * (1 - img)
    # 镜头：缓慢推近人物
    img = camera(img, 1.0 + 0.06 * ease(tb / DUR_B), 0, 0, 960, 700)
    # 开头：从 05-a 的满屏花色过渡
    if tb < 0.5:
        img = img + (1 - ease(tb / 0.5)) * (pink_paper() - img)
    return img


PINK = np.array([0.96, 0.70, 0.76])


def pink_paper():
    if "pink" not in _cache:
        paper, TEX = wc.paper(seed=990)
        pp = paper * PINK[None, None]
        _cache["pink"] = pp * (0.95 + 0.08 * fbm(H, W, 991, octaves=4, base=5)[..., None])
    return _cache["pink"]


def pink_bloom(img, p, TEX):
    """一朵水彩花从 (1000,480) 晕开，花色逐渐铺满整幅：带不规则湿边，最后完全变成粉色纸面。"""
    R = 30 + 1700 * p ** 1.5
    ang = np.arctan2(y - 480, x - 1000)
    edge = R * (1 + 0.08 * np.sin(ang * 5 + 1) + 0.05 * np.sin(ang * 11 + 2))
    d = np.sqrt((x - 1000) ** 2 + (y - 480) ** 2)
    inside = np.clip((edge - d) / 30, 0, 1)
    ring = np.exp(-((d - edge) / 14) ** 2) * (1 - p * 0.6)
    dens = np.clip(inside * (0.45 + 0.5 * p) + ring * 0.7, 0, 1)[..., None]
    img = img * (1 - dens * (1 - PINK))
    return img + (p ** 2.2) * (pink_paper() - img)


def frame(i):
    t = i / FPS
    return frame_a(t) if t < DUR_A else frame_b(t - DUR_A)


if __name__ == "__main__":
    out = sys.argv[1]
    stills = sys.argv[2] if len(sys.argv) > 2 else None
    if len(sys.argv) > 3:                       # 只渲染指定帧做检查
        for i in map(int, sys.argv[3].split(",")):
            save(frame(i), os.path.join(stills, f"测试帧_{i:04d}.png"))
            print("frame", i)
    else:
        render_video(frame, N, out, stills=(15, 45, 75, 105, 140, 160, 175, 200, 230, 262), stills_dir=stills)
