# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
"""01｜还有什么更好的 · 纸艺立体书
01-a：立体书里弹出的清晨草地、山林与歪斜的纸壳小屋；Clawd 披粉色薄纱伸懒腰，泡泡飘过。
01-b：纸雕剧场舞台；Clawd 穿浅蓝飘逸长裙（披风式袖、散落叶片贴花、金色胸针），握白色麦克风。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha

from titles import paper_title
place = no_actor

OUT = str(ASSETS)
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()
FIB = fbm(H, W, 301, octaves=3, base=140)          # 纸纤维
FIB2 = fbm(H, W, 302, octaves=4, base=12)          # 纸张轻微不匀


def poly_mask(polys, ss=SS):
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([(px * ss, py * ss) for px, py in p], fill=255)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def ellipse_mask(items, ss=SS):
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for (cx, cy, rx, ry) in items:
        d.ellipse([(cx - rx) * ss, (cy - ry) * ss, (cx + rx) * ss, (cy + ry) * ss], fill=255)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def paper(img, mask, color, sh=(8, 12, 9, 0.35), tex=1.0, light_dir=(-1, -1), rim=0.18):
    """一层纸：先在下层投影，再铺带纤维纹理的纸色，并给受光边缘一点亮边。"""
    if sh:
        dx, dy, r, k = sh
        s = shift(blur(mask, r), dy, dx)
        img = img * (1 - k * s * (1 - mask))[..., None]
    c = np.array(color, np.float32) / 255 if max(color) > 1 else np.array(color, np.float32)
    col = c[None, None] * (0.93 + 0.07 * FIB * tex + 0.05 * (FIB2 - 0.5) * tex)[..., None]
    lx, ly = light_dir
    edge = np.clip(mask - shift(mask, -3 * ly, -3 * lx), 0, 1)
    col = col + edge[..., None] * rim
    return over(img, col, mask)


def corrugated(mask, base=(0.76, 0.6, 0.42), angle=0.0, period=7.0):
    """瓦楞纸：楞纹 + 切口边缘露出的波浪芯。"""
    xr = x * math.cos(angle) + y * math.sin(angle)
    flute = 0.5 + 0.5 * np.sin(xr / period * 2 * math.pi)
    col = np.array(base)[None, None] * (0.9 + 0.08 * flute + 0.06 * FIB)[..., None]
    inner = np.clip(blur(mask, 2.5) * 1.25 - 0.25, 0, 1)
    edge = np.clip(mask - inner, 0, 1)
    col = col * (1 - 0.25 * edge[..., None]) + (edge * (flute > 0.5))[..., None] * np.array([0.1, 0.07, 0.03])
    return col


# =====================================================================
# 01-a
# =====================================================================
img = np.ones((H, W, 3), np.float32)
t = y / H
sky = np.stack([0.99 - 0.06 * t, 0.93 - 0.02 * t, 0.86 + 0.06 * t], -1)
img = sky * (0.95 + 0.05 * FIB2[..., None])
# 太阳：两层纸圆 + 柔光
sx, sy = 1540, 210
img = screen(img, (np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 260 ** 2)))[..., None] * np.array([0.25, 0.18, 0.05]))
img = paper(img, ellipse_mask([(sx, sy, 128, 128)]), (252, 226, 160), sh=(6, 9, 8, 0.18))
img = paper(img, ellipse_mask([(sx, sy, 92, 92)]), (255, 238, 186), sh=(4, 6, 5, 0.2))
# 纸云
gg = rng(11)
for (cx, cy, s_) in [(380, 330, 1.0), (980, 220, 0.75), (1800, 380, 0.8), (720, 400, 0.6)]:
    blobs = [(cx + dx * s_, cy + dy * s_, r * s_, r * s_ * 0.8) for dx, dy, r in
             [(-70, 10, 55), (-10, -15, 70), (60, 5, 58), (110, 20, 40), (-120, 25, 35)]]
    base = [(cx - 160 * s_, cy + 20 * s_, 170 * s_, 30 * s_)]
    m = np.clip(ellipse_mask(blobs) + ellipse_mask(base), 0, 1)
    m = m * (y < cy + 40 * s_)
    img = paper(img, m, (255, 252, 246), sh=(7, 10, 8, 0.22))

# 远山 3 层
def ridge(base_y, amp, seed, freq):
    gr = rng(seed)
    ph = gr.uniform(0, 10, 4)
    xs = np.arange(0, W + 41, 40)
    ys = base_y - amp * (0.55 * np.sin(xs / freq + ph[0]) + 0.3 * np.sin(xs / (freq * 0.43) + ph[1]) + 0.15 * np.sin(xs / (freq * 0.2) + ph[2]))
    return [(float(a), float(b)) for a, b in zip(xs, ys)] + [(W, H), (0, H)]

for base_y, amp, col, seed, fq in [(470, 70, (184, 208, 182), 1, 260), (535, 75, (140, 182, 140), 2, 190), (600, 60, (100, 152, 104), 3, 150)]:
    img = paper(img, poly_mask([ridge(base_y, amp, seed, fq)]), col, sh=(-8, 10, 10, 0.3))

# 树林（纸剪的圆锥与棒棒糖树）
trees = []
gg = rng(21)
for i in range(46):
    tx = gg.uniform(-20, W + 20)
    ty = 640 + gg.uniform(-14, 12)
    th = gg.uniform(70, 130)
    if gg.random() < 0.6:
        trees.append([(tx - th * 0.28, ty), (tx, ty - th), (tx + th * 0.28, ty)])
    else:
        r = th * 0.32
        trees.append([(tx + r * math.cos(a), ty - th * 0.62 + r * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)])
img = paper(img, poly_mask(trees), (70, 118, 80), sh=(-6, 8, 6, 0.3))

# 草地：顶部剪成草尖锯齿
grass_top = [(0.0, 700.0)]
gg = rng(31)
for i in range(0, W + 1, 18):
    grass_top.append((i + 9, 700 - gg.uniform(10, 26)))
    grass_top.append((i + 18, 700 + gg.uniform(-3, 4)))
meadow = grass_top + [(W, H), (0, H)]
mm = poly_mask([meadow])
img = paper(img, mm, (134, 190, 104), sh=(-7, 9, 9, 0.32))
# 草地上的小纸花
fl = []
gg = rng(41)
for i in range(140):
    fx, fy = gg.uniform(0, W), gg.uniform(712, 840)
    r = gg.uniform(4, 9) * (0.6 + (fy - 700) / 300)
    fl.append((fx, fy, r, r))
fmask = ellipse_mask(fl)
img = paper(img, fmask * (FIB2 > 0.0), (255, 250, 245), sh=(2, 3, 2, 0.25))
pink = ellipse_mask(fl[::3])
img = paper(img, pink, (246, 176, 196), sh=None)

# ---------- 歪斜的纸壳小屋（两段错位，参照 MV） ----------
def house(cx, by, w, h, roof, ang):
    pts = [(-w / 2, 0), (w / 2, 0), (w / 2, -h), (0, -h - roof), (-w / 2, -h)]
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return [(cx + px * c - py * s, by + px * s + py * c) for px, py in pts]

def rot_pts(pts, cx, by, ang):
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return [(cx + px * c - py * s, by + px * s + py * c) for px, py in pts]

H1 = house(1110, 780, 300, 260, 110, -9)
H2 = house(1360, 760, 250, 230, 100, 13)
hm1, hm2 = poly_mask([H1]), poly_mask([H2])
# 窗洞与门
win1 = poly_mask([rot_pts([(-95, -200), (-40, -200), (-40, -145), (-95, -145)], 1110, 780, -9),
                  rot_pts([(40, -120), (95, -120), (95, -65), (40, -65)], 1110, 780, -9)])
door1 = poly_mask([rot_pts([(-40, 0), (30, 0), (30, -150), (-40, -150)], 1110, 780, -9)])
win2 = poly_mask([rot_pts([(-70, -180), (-20, -180), (-20, -130), (-70, -130)], 1360, 760, 13),
                  rot_pts([(25, -150), (75, -150), (75, -100), (25, -100)], 1360, 760, 13)])
# 投影
sh = blur(shift(np.clip(hm1 + hm2, 0, 1), 16, -26), 14)
img = img * (1 - 0.35 * sh * (1 - np.clip(hm1 + hm2, 0, 1)))[..., None]
house2 = corrugated(hm2, base=(0.82, 0.68, 0.5), angle=math.radians(13))
house2 = house2 * (1 - 0.12)  # 后面那段稍暗
img = over(img, house2, hm2 * (1 - win2))
img = over(img, np.array([0.18, 0.13, 0.1]), hm2 * win2)
house1 = corrugated(hm1, angle=math.radians(-9))
img = over(img, house1, hm1 * (1 - win1))
img = over(img, np.array([0.16, 0.11, 0.08]), hm1 * win1)
# 门洞里的暗部
img = over(img, np.array([0.22, 0.15, 0.1]), door1 * 0.85)
# 屋檐的橙色纸条
trim = []
for (H_, ang, cx, by, w, h, roof) in [(H1, -9, 1110, 780, 300, 260, 110), (H2, 13, 1360, 760, 250, 230, 100)]:
    pts = [(-w / 2 - 14, -h + 6), (0, -h - roof - 10), (w / 2 + 14, -h + 6), (w / 2 + 14, -h - 14), (0, -h - roof - 32), (-w / 2 - 14, -h - 14)]
    trim.append(rot_pts(pts, cx, by, ang))
img = paper(img, poly_mask(trim), (226, 120, 60), sh=(5, 8, 5, 0.35))

# ---------- Clawd：伸懒腰 + 粉色薄纱 ----------
S = 19
arms_up = [rigid_arm('L', 65), rigid_arm('R', 65)]   # 整块抬起，长度不变
# 薄纱：从头顶披下，被风吹向右侧
veil = [(1.0, 0.6), (4.5, -0.5), (9.5, -0.6), (14.5, -0.2), (18.6, 0.8), (22.6, 2.4), (26.4, 4.8), (28.6, 8.0),
        (26.2, 8.6), (24.4, 7.2), (22.6, 10.8), (20.4, 9.4), (18.6, 6.4), (16.4, 4.6), (15.6, 2.4), (11.0, 1.4), (5.0, 1.4)]
check_proportions(arms=arms_up, extra={"veil": [veil]})
cm, (ox, oy), (pw, ph) = clawd_masks(S, 30, arms=arms_up, bbox=(-4, -3, 29, 13), extra={"veil": [veil]})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
orange = np.array(CLAWD_ORANGE) / 255
body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
fib_s = fbm(ph, pw, 51, octaves=3, base=60)
col = orange[None, None] * (0.95 + 0.06 * fib_s)[..., None]
col = col * (1 - 0.14 * np.clip((gx - 8) / 8, 0, 1))[..., None]   # 右侧背光稍暗（阳光来自右上：纸卡左侧更亮）
col = over(col, np.array([0.08, 0.06, 0.06]), cm["eyes"])
spr = np.dstack([col, body])
# 薄纱：半透明粉 + 细网格 + 褶皱
vm = cm["veil"]
folds = 0.5 + 0.5 * np.sin(gx * 1.6 + gy * 0.9 + np.sin(gy * 0.7) * 1.4)
mesh = 0.85 + 0.15 * ((np.sin(gx_ * 1.9) * np.sin(gy_ * 1.9)) > 0.2)
tulle = np.array([0.98, 0.72, 0.8])[None, None] * (0.85 + 0.2 * folds)[..., None]
vedge = np.clip(vm - blur(vm, 2.5), 0, 1) * 2
va = np.clip(vm * (0.3 + 0.25 * folds) * mesh + vedge * 0.5, 0, 0.85)
va = va * (1 - 0.6 * np.clip(blur(cm["eyes"], 1.5) * 2, 0, 1))  # 不遮住眼睛
spr[..., :3] = over(spr[..., :3], tulle, va)
spr[..., 3] = np.maximum(spr[..., 3], va)
export_sprite("01a", spr, S, (ox,oy), arms=arms_up, extra={"veil":[veil]})
CX, CY = 1010, 659
# Clawd 投影到小屋上
alpha_full = sprite_alpha_layer(spr, CX, CY, 3)
img = img * (1 - 0.3 * blur(shift(alpha_full, 12, -18), 10) * (1 - alpha_full))[..., None]
img = paste_rgba(img, spr, CX, CY, 3)

# ---------- 泡泡（玻璃纸质感） ----------
def bubble(img, bx, by, r, seed):
    X0, X1, Y0, Y1 = int(max(bx - r - 4, 0)), int(min(bx + r + 4, W)), int(max(by - r - 4, 0)), int(min(by + r + 4, H))
    xs, ys = x[Y0:Y1, X0:X1] - bx, y[Y0:Y1, X0:X1] - by
    d = np.sqrt(xs ** 2 + ys ** 2) / r
    ang = np.arctan2(ys, xs)
    inside = np.clip((1 - d) * r, 0, 1)
    rimw = np.exp(-((d - 0.94) / 0.06) ** 2)
    hue = (ang / (2 * np.pi) + 0.5 + seed * 0.13 + d * 0.4) % 1
    iri = hsv2rgb(hue, np.full_like(hue, 0.45), np.ones_like(hue))
    sub = img[Y0:Y1, X0:X1]
    sub = sub * (1 - 0.05 * inside[..., None]) + (iri * rimw[..., None] * 0.55 + inside[..., None] * 0.04) * inside[..., None]
    hl = np.exp(-(((xs + r * 0.38) / (r * 0.16)) ** 2 + ((ys + r * 0.42) / (r * 0.1)) ** 2))
    sub = screen(sub, hl[..., None] * 0.9)
    img[Y0:Y1, X0:X1] = sub
    return img

gg = rng(61)
for (bx, by, r) in [(860, 520, 46), (930, 420, 30), (1480, 470, 56), (1610, 600, 26), (760, 640, 22),
                    (1290, 300, 34), (1700, 300, 40), (600, 470, 28), (1030, 240, 20)]:
    img = bubble(img, bx, by, r, gg.integers(10))

# ---------- 立体书页 ----------
book_top = 860
pages = poly_mask([[(-40, book_top + 10), (W / 2, book_top - 6), (W + 40, book_top + 10), (W + 40, H), (-40, H)]])
gut = np.exp(-((x - W / 2) / 60) ** 2)
pcol = np.array([0.97, 0.94, 0.88])[None, None] * (0.95 + 0.05 * FIB[..., None]) * (1 - 0.25 * gut[..., None])
# 书页向两侧微微弯曲的明暗
pcol = pcol * (1 - 0.08 * np.clip((y - book_top) / 220, 0, 1))[..., None]
# 下层书页边（纸叠）
stack_lines = (np.sin((y - book_top) * 1.6) > 0.6) * np.clip((y - (H - 60)) / 60, 0, 1)
pcol = pcol * (1 - 0.18 * stack_lines[..., None])
img = img * (1 - 0.3 * blur((pages > 0.5).astype(np.float32), 10) * (1 - pages))[..., None]
img = over(img, pcol, pages)
# 弹出结构的折叠底座：从书缝里伸出的纸片支架（草地与书页之间的投影带）
fold_sh = np.exp(-((y - book_top) / 14) ** 2) * (y > book_top)
img = img * (1 - 0.28 * fold_sh)[..., None]

# 书页前景的纸草丛（左右两角）
fg = []
gg = rng(141)
for cx0 in (130, 330, 1600, 1810):
    for k in range(9):
        bx = cx0 + gg.uniform(-70, 70)
        h_ = gg.uniform(70, 150)
        lean = gg.uniform(-25, 25)
        fg.append([(bx - 9, 1080), (bx + lean, 1080 - h_), (bx + 9, 1080)])
fgm = poly_mask(fg)
img = paper(img, fgm, (118, 176, 96), sh=(-8, 6, 8, 0.3))
fgf = ellipse_mask([(150, 960, 16, 16), (360, 990, 13, 13), (1620, 975, 15, 15), (1790, 955, 17, 17)])
img = paper(img, fgf, (247, 180, 198), sh=(2, 3, 3, 0.3))

# ---------- 标题 ----------
title = paper_title("还有什么更好的", size=150, seed=3)
place(img, title, 70, 40)

# ---------- 后期：晨光、景深、纸感 ----------
img = screen(img, (np.exp(-(((x - 1650) / 900) ** 2 + ((y + 50) / 700) ** 2)))[..., None] * np.array([0.22, 0.16, 0.06]))
bl = blur(img, 5)
dof = np.clip((y - 900) / 120, 0, 1)
img = img * (1 - dof[..., None]) + bl * dof[..., None]
img = img * (0.97 + 0.03 * FIB[..., None])
img = vignette(img, 0.25, 2.5)
img = grain(img, 0.01, 5)
export_bg("01a", img)


# =====================================================================
# 01-b 纸雕剧场
# =====================================================================
img = np.zeros((H, W, 3), np.float32)
img[:] = np.array([0.07, 0.09, 0.22])
img = img * (0.9 + 0.1 * FIB2[..., None])
# 镂空星点（背后透出浅蓝）
gg = rng(71)
stars = [(gg.uniform(0, W), gg.uniform(0, 620), r, r) for r in gg.uniform(2, 6, 110)]
sm = ellipse_mask(stars)
img = over(img, np.array([0.55, 0.65, 0.95]), sm * 0.85)
img = screen(img, blur(sm, 4)[..., None] * np.array([0.2, 0.25, 0.5]))
# 层叠拱门（3 层纸雕）
# 用更直接的方法：每层 = 满屏 - 拱形开口
def arch_layer(cx, top, half_w, bottom):
    opening = [(cx - half_w, bottom), (cx - half_w, top + half_w)]
    for a in np.linspace(math.pi, 2 * math.pi, 60):
        opening.append((cx + half_w * math.cos(a), top + half_w + half_w * math.sin(a)))
    opening.append((cx + half_w, bottom))
    full = np.ones((H, W), np.float32)
    return np.clip(full - poly_mask([opening]), 0, 1)

# 由远及近：最远的开口最小
for (hw, top, col) in [(520, 210, (40, 58, 126)), (640, 130, (30, 44, 104)), (780, 50, (20, 30, 76))]:
    m = arch_layer(W / 2, top, hw, H + 10)
    img = paper(img, m, col, sh=(0, 14, 16, 0.5), rim=0.08, light_dir=(0, -1))
# 地板：纸条拼成的木板（透视）
floor_top = 840
fm = (y > floor_top).astype(np.float32)
fm = blur(fm, 0.8)
vx = W / 2
u = (x - vx) / np.maximum(y - floor_top + 60, 1)
planks = (np.sin(u * 22) > 0.92).astype(np.float32)
fcol = np.array([0.24, 0.3, 0.5])[None, None] * (0.85 + 0.1 * FIB[..., None]) * (1 - 0.3 * planks[..., None])
fcol = fcol * (0.7 + 0.5 * np.clip((y - floor_top) / 240, 0, 1))[..., None]
img = over(img, fcol, fm)
img = img * (1 - 0.4 * np.exp(-((y - floor_top) / 10) ** 2) * (y > floor_top))[..., None]

# 硫酸纸光束（半透明三角）
beam = np.zeros((H, W), np.float32)
for (bx, spread) in [(640, 120), (1280, 120), (960, 90)]:
    beam += poly_mask([[(bx - 24, -10), (bx + 24, -10), (960 + spread * 1.6, 980), (960 - spread * 1.6, 980)]]) * 0.16
img = screen(img, np.clip(beam, 0, 0.45)[..., None] * np.array([0.85, 0.92, 1.0]))
# 光斑
pool = np.exp(-(((x - 960) / 330) ** 2 + ((y - 920) / 60) ** 2))
img = screen(img, pool[..., None] * np.array([0.45, 0.55, 0.8]) * 0.7)

# ---------- Clawd 浅蓝长裙 ----------
S = 24
gown = [(0.0, 5.7), (16.0, 5.7), (16.6, 8.4), (17.8, 10.8), (18.6, 11.7), (16.2, 12.0), (13.6, 11.6), (10.8, 12.0),
        (8.0, 11.6), (5.2, 12.0), (2.4, 11.6), (-0.2, 12.0), (-2.6, 11.7), (-1.8, 10.8), (-0.6, 8.4)]
cape_l = [(-1.0, 2.9), (0.1, 2.9), (0.3, 6.2), (-1.0, 10.4), (-3.0, 11.9), (-4.8, 11.0), (-3.4, 7.6)]
cape_r = [(15.9, 2.9), (17.0, 2.9), (19.4, 7.6), (20.8, 11.0), (19.0, 11.9), (17.0, 10.4), (15.7, 6.2)]
neck_v = [(0, 0), (0, 0), (0, 0)]
mic = [(15.6, 4.6), (16.8, 4.0), (14.0, 1.2), (13.0, 1.8)]
check_proportions(extra={"gown": [gown], "cape": [cape_l, cape_r]})
cm, (ox, oy), (pw, ph) = clawd_masks(S, 30, bbox=(-6, -1, 22, 14),
                                     extra={"gown": [gown], "cape": [cape_l, cape_r], "neck": [neck_v], "mic": [mic]})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
fib_s = fbm(ph, pw, 81, octaves=3, base=60)
body = np.clip(cm["body"] + cm["arms"], 0, 1)
col = orange[None, None] * (0.92 + 0.08 * fib_s)[..., None] * (1 + 0.12 * np.exp(-((gx - 8) / 6) ** 2))[..., None]
col = over(col, np.array([0.06, 0.05, 0.06]), cm["eyes"])
spr = np.dstack([col, body])
blue = np.array([0.68, 0.82, 0.92])
# 长裙：纸片层叠 + 竖向折痕
gm = np.clip(cm["gown"] - cm["neck"], 0, 1)
crease = 0.5 + 0.5 * np.sin((gx - 8) * 2.0 / (1 + np.clip(gy - 5, 0, 15) * 0.05))
gcol = blue[None, None] * (0.86 + 0.12 * crease + 0.05 * fib_s)[..., None]
gcol = gcol * (1 - 0.15 * np.clip((gy - 8) / 5, 0, 1))[..., None]
spr[..., :3] = over(spr[..., :3], gcol, gm)
spr[..., 3] = np.maximum(spr[..., 3], gm)
# 披风袖：半透明硫酸纸
cedge = np.clip(cm["cape"] - blur(cm["cape"], 3), 0, 1) * 2
ca = np.clip(cm["cape"] * 0.85 + cedge * 0.3, 0, 0.97)
ccol = np.array([0.74, 0.88, 1.0])[None, None] * (0.9 + 0.1 * np.sin(gy * 2.2 + gx)[..., None])
spr[..., :3] = over(spr[..., :3], ccol, ca)
spr[..., 3] = np.maximum(spr[..., 3], ca)
# 叶片/羽毛贴花：深浅不一的蓝绿纸叶
leaf_im = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
ld = ImageDraw.Draw(leaf_im)
gg = rng(91)
anyc = np.clip(gm + cm["cape"], 0, 1)
cnt = 0
while cnt < 32:
    lx, ly = gg.uniform(-5, 21), gg.uniform(5.8, 11.8)
    px_, py_ = int(ox + lx * S), int(oy + ly * S)
    if not (0 <= py_ < ph and 0 <= px_ < pw) or anyc[py_, px_] < 0.5:
        continue
    cnt += 1
    L = gg.uniform(0.7, 1.15) * S
    a = gg.uniform(-0.9, 0.9) + math.pi / 2
    w_ = L * 0.28
    pts = []
    for k in range(13):
        u = k / 12
        ww = math.sin(math.pi * u) * w_
        pts.append((px_ + math.cos(a) * L * (u - 0.5) - math.sin(a) * ww, py_ + math.sin(a) * L * (u - 0.5) + math.cos(a) * ww))
    for k in range(12, -1, -1):
        u = k / 12
        ww = -math.sin(math.pi * u) * w_
        pts.append((px_ + math.cos(a) * L * (u - 0.5) - math.sin(a) * ww, py_ + math.sin(a) * L * (u - 0.5) + math.cos(a) * ww))
    c = [(44, 104, 132), (70, 138, 160), (30, 72, 108), (120, 170, 190)][gg.integers(4)]
    ld.polygon([(p[0] * SS, p[1] * SS) for p in pts], fill=c + (255,))
leaves = to_arr(leaf_im.resize((pw, ph), Image.LANCZOS))
lsh = shift(blur(leaves[..., 3], 1.5), 2, 2) * 0.35
spr[..., :3] = spr[..., :3] * (1 - lsh[..., None])
spr[..., :3] = over(spr[..., :3], leaves[..., :3], leaves[..., 3])
# 金色胸针（叶形）
bim = Image.new("RGBA", (pw * SS, ph * SS), (0, 0, 0, 0))
bd = ImageDraw.Draw(bim)
bx_, by_ = ox + 8.0 * S, oy + 6.3 * S
for k, (dx, dy, rr) in enumerate([(0, 0, 0.38), (0.32, -0.25, 0.26), (-0.3, -0.22, 0.24), (0.05, -0.5, 0.22)]):
    bd.ellipse([(bx_ + (dx - rr) * S) * SS, (by_ + (dy - rr * 0.7) * S) * SS, (bx_ + (dx + rr) * S) * SS, (by_ + (dy + rr * 0.7) * S) * SS],
               fill=(222, 176, 84, 255))
bro = to_arr(bim.resize((pw, ph), Image.LANCZOS))
spr[..., :3] = over(spr[..., :3], bro[..., :3] * (0.9 + 0.2 * fib_s[..., None]), bro[..., 3])
# 白色麦克风
mm_ = cm["mic"]
spr[..., :3] = over(spr[..., :3], np.array([0.95, 0.95, 0.96]), mm_)
spr[..., 3] = np.maximum(spr[..., 3], mm_)
mh = np.zeros_like(mm_)
hx, hy = ox + 13.3 * S, oy + 1.3 * S
mh = ((gx_ - hx) ** 2 + (gy_ - hy) ** 2) < (0.62 * S) ** 2
mh = blur(mh.astype(np.float32), 1)
spr[..., :3] = over(spr[..., :3], np.array([0.86, 0.87, 0.9]), mh)
spr[..., 3] = np.maximum(spr[..., 3], mh)
# 握麦克风的那只钳子前移
spr[..., :3] = over(spr[..., :3], orange * 1.05, cm["arms"] * (gx > 15.5))

export_sprite("01b", spr, S, (ox,oy))
CX, CY = 960, 776
alpha_full = sprite_alpha_layer(spr, CX, CY)
img = img * (1 - 0.45 * blur(shift(alpha_full, 18, 0), 16) * (1 - alpha_full))[..., None]
img = paste_rgba(img, spr, CX, CY)
# 顶光：主体受光
topl = np.exp(-(((x - 960) / 520) ** 2 + ((y - 520) / 460) ** 2))
img = img * (0.86 + 0.28 * topl)[..., None]
img = screen(img, (alpha_full * topl)[..., None] * np.array([0.05, 0.07, 0.1]))

# 飘落的纸叶片
lim = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
ld = ImageDraw.Draw(lim)
gg = rng(97)
for i in range(26):
    px_, py_ = gg.uniform(200, 1720), gg.uniform(60, 820)
    if abs(px_ - 960) < 330 and 260 < py_ < 860:
        continue
    L, a = gg.uniform(22, 40), gg.uniform(0, math.pi)
    w_ = L * 0.3
    pts = []
    for k in range(25):
        u = k / 24 * 2
        uu = u if u <= 1 else 2 - u
        sg = 1 if u <= 1 else -1
        ww = math.sin(math.pi * uu) * w_ * sg
        pts.append((px_ + math.cos(a) * L * (uu - 0.5) - math.sin(a) * ww, py_ + math.sin(a) * L * (uu - 0.5) + math.cos(a) * ww))
    c = [(70, 138, 160), (120, 170, 190), (44, 104, 132)][gg.integers(3)]
    ld.polygon([(p[0] * SS, p[1] * SS) for p in pts], fill=c + (255,))
fl_ = to_arr(lim.resize((W, H), Image.LANCZOS))
img = img * (1 - 0.3 * shift(blur(fl_[..., 3], 3), 8, 6))[..., None]
img = over(img, fl_[..., :3], fl_[..., 3])

img = img * (0.96 + 0.04 * FIB[..., None])
img = vignette(img, 0.4, 2.2)
img = grain(img, 0.012, 6)
export_bg("01b", img)

