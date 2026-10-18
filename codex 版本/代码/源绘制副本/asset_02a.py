# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
"""B｜02 纯妹妹 · 巨像花园
羊毛毡 + 黏土手作质感，移轴微缩；巨型 Clawd 趴睡在云海仙岛上，身穿白色立体廓形钉珠礼服。"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha


OUT = str(ASSETS)
g = rng(2)
x, y = grid()
SS = 2  # 图层绘制超采样

_n = rng(21).normal(0, 1, (H, W)).astype(np.float32)
FINE = np.clip(blur(np.clip(_n * 0.18 + 0.5, 0, 1), 0.8) * 1.6 - 0.3, 0, 1)   # 毛毡纤维（细）
FINE = 0.6 * FINE + 0.4 * fbm(H, W, 23, octaves=2, base=90)
MID = fbm(H, W, 22, octaves=4, base=24)


def felt(rgba, seed=0, puff=10, fuzz=0.25, light=0.35, shift=6):
    """把平面色块变成羊毛毡/黏土：边缘毛糙、鼓起的明暗、细纤维噪声。"""
    rgb, a = rgba[..., :3], rgba[..., 3]
    n = np.roll(FINE, (seed * 37) % H, axis=0)
    a2 = np.clip(blur(a, 1.4) + (n - 0.5) * fuzz * 2.2 * (1 - np.abs(a * 2 - 1)), 0, 1)
    ab = blur(a, puff)
    emb = ab - np.roll(np.roll(ab, shift, axis=0), shift, axis=1)  # 左上受光
    shade = 0.82 + 0.25 * ab + light * emb
    rgb = rgb * shade[..., None] * (0.92 + 0.16 * n[..., None])
    return np.concatenate([rgb, a2[..., None]], -1)


def layer():
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def done(im):
    return to_arr(im.resize((W, H), Image.LANCZOS))


def comp(dst, rgba):
    return over(dst, rgba[..., :3], rgba[..., 3])


def C(c):
    return tuple(int(v) for v in c) + (255,)

# ---------- 天空 ----------
t = y / H
sky = np.stack([0.62 + 0.30 * t, 0.78 + 0.16 * t, 0.95 - 0.02 * t], -1)
img = sky.astype(np.float32)


def cloud_layer(seed, centers, scale, top=(252, 250, 255), bottom=(214, 205, 228)):
    gg = rng(seed)
    im, d = layer()
    for (cx, cy, w) in centers:
        k = int(w / 40)
        for i in range(k):
            px = cx + gg.uniform(-w / 2, w / 2)
            rr = gg.uniform(0.35, 1.0) * scale * (1 - abs(px - cx) / (w / 1.6))
            rr = max(rr, scale * 0.25)
            py = cy - gg.uniform(0, rr * 0.6)
            d.ellipse([(px - rr) * SS, (py - rr) * SS, (px + rr) * SS, (py + rr * 0.8) * SS], fill=C(top))
    arr = done(im)
    a = arr[..., 3]
    # 底部偏紫灰、顶部亮白
    ab = blur(a, 18)
    tint = np.clip((ab - 0.5) * 2, 0, 1)
    col = np.array(bottom) / 255 * (1 - tint[..., None]) + np.array(top) / 255 * tint[..., None]
    arr = np.concatenate([col, a[..., None]], -1)
    return felt(arr, seed, puff=14, fuzz=0.35, light=0.6)

# 远处云海（地平线）
img = comp(img, cloud_layer(5, [(200, 560, 700), (900, 540, 900), (1700, 560, 700)], 80,
                            top=(246, 240, 250), bottom=(200, 196, 226)))
# 天空中的积云
img = comp(img, cloud_layer(6, [(260, 170, 380), (1500, 130, 460), (950, 250, 240)], 70))

# ---------- 浮岛 ----------
ICX, ITOP = 960, 690
im, d = layer()
pts_top, pts_under = [], []
for i in range(181):
    a = math.pi * 2 * i / 180
    rx, ry = 840, 120
    jit = 1 + 0.05 * math.sin(a * 7) + 0.03 * math.sin(a * 13 + 1)
    pts_top.append(((ICX + rx * math.cos(a) * jit) * SS, (ITOP + ry * math.sin(a) * jit) * SS))
# 下方的倒锥形岩体
under = [(ICX - 840, ITOP)]
for i in range(0, 41):
    s_ = i / 40
    xx = ICX - 840 + 1680 * s_
    depth = 330 * math.sin(math.pi * s_) ** 0.7 + 25 * math.sin(s_ * 31)
    under.append((xx, ITOP + depth))
d.polygon([(px * SS, py * SS) for px, py in under], fill=C((222, 176, 170)))
rock = done(im)
# 岩层条纹
strata = 0.85 + 0.15 * np.sin((y - ITOP) / 9 + MID * 6)
rock[..., :3] *= strata[..., None]
rock[..., :3] *= (1 - 0.35 * np.clip((y - ITOP) / 330, 0, 1))[..., None]
img = comp(img, felt(rock, 3, puff=20, fuzz=0.3))

im, d = layer()
d.polygon(pts_top, fill=C((126, 178, 112)))
top = done(im)
patch = fbm(H, W, 31, octaves=4, base=16)
grass = np.stack([0.42 + 0.18 * patch, 0.64 + 0.14 * patch, 0.40 + 0.08 * patch], -1)
top[..., :3] = grass
img = comp(img, felt(top, 4, puff=26, fuzz=0.3, light=0.25))

# ---------- 微缩景物 ----------

def pagoda(d, px, base, sc, roof=(196, 82, 78), wall=(244, 232, 214), tiers=3):
    yy = base
    for k in range(tiers):
        w = sc * (1.0 - k * 0.18)
        h = sc * 0.45
        d.rectangle([(px - w * 0.36) * SS, (yy - h) * SS, (px + w * 0.36) * SS, yy * SS], fill=C(wall))
        # 门窗
        d.rectangle([(px - w * 0.08) * SS, (yy - h * 0.7) * SS, (px + w * 0.08) * SS, yy * SS], fill=C((150, 70, 60)))
        yy -= h
        rh = sc * 0.22
        # 翘角屋檐
        d.polygon([((px - w * 0.62) * SS, (yy - rh * 0.1) * SS), ((px - w * 0.5) * SS, (yy + rh * 0.25) * SS),
                   ((px + w * 0.5) * SS, (yy + rh * 0.25) * SS), ((px + w * 0.62) * SS, (yy - rh * 0.1) * SS),
                   ((px + w * 0.3) * SS, (yy - rh) * SS), ((px - w * 0.3) * SS, (yy - rh) * SS)], fill=C(roof))
        yy -= rh * 0.85
    d.polygon([((px - 3) * SS, yy * SS), ((px + 3) * SS, yy * SS), (px * SS, (yy - sc * 0.3) * SS)], fill=C((214, 170, 70)))


def pine(d, px, base, sc, col=(58, 104, 72)):
    d.line([(px * SS, base * SS), ((px + sc * 0.1) * SS, (base - sc * 0.9) * SS)], fill=C((112, 78, 58)), width=int(sc * 0.12 * SS))
    for k, (dx, dy, w) in enumerate([(-0.25, -0.55, 0.6), (0.25, -0.8, 0.55), (0.0, -1.05, 0.45)]):
        cx_, cy_ = px + dx * sc, base + dy * sc
        c = tuple(int(v * (1 + 0.08 * k)) for v in col)
        d.ellipse([(cx_ - w * sc) * SS, (cy_ - w * sc * 0.35) * SS, (cx_ + w * sc) * SS, (cy_ + w * sc * 0.35) * SS], fill=C(c))


def flowers(d, seed, region, count, palette):
    gg = rng(seed)
    x0, y0, x1, y1 = region
    for _ in range(count):
        px, py = gg.uniform(x0, x1), gg.uniform(y0, y1)
        # 只在岛面椭圆内
        if ((px - ICX) / 820) ** 2 + ((py - ITOP) / 112) ** 2 > 1:
            continue
        r = gg.uniform(3, 7)
        c = palette[gg.integers(len(palette))]
        d.ellipse([(px - r) * SS, (py - r) * SS, (px + r) * SS, (py + r) * SS], fill=C(c))
        d.ellipse([(px - r * 0.35) * SS, (py - r * 0.35) * SS, (px + r * 0.35) * SS, (py + r * 0.35) * SS], fill=C((250, 220, 120)))

FLOWER_PAL = [(242, 150, 180), (250, 200, 210), (255, 245, 240), (190, 160, 230), (250, 190, 120), (240, 110, 130)]

# 后景：亭台、松树、花
im, d = layer()
flowers(d, 40, (160, 580, 1760, 690), 260, FLOWER_PAL)
for px, base, sc, roof, tiers in [(300, 652, 52, (196, 82, 78), 2), (470, 618, 40, (110, 168, 170), 3), (1320, 612, 46, (214, 120, 140), 2),
                                  (1540, 640, 58, (196, 82, 78), 3), (1190, 598, 34, (214, 176, 90), 2), (640, 600, 30, (214, 120, 140), 1)]:
    pagoda(d, px, base, sc, roof, tiers=tiers)
for px, base, sc in [(420, 650, 70), (640, 615, 55), (1450, 625, 65), (1700, 668, 60), (260, 676, 50), (1250, 610, 45)]:
    pine(d, px, base, sc)
img = comp(img, felt(done(im), 7, puff=6, fuzz=0.3))

# ---------- 巨型 Clawd（趴睡） + 礼服 ----------
S = 27
PAD = 60
gown = [(0.4, 6.4), (15.6, 6.4), (16.6, 9.6), (17.8, 12.0), (-1.8, 12.0), (-0.6, 9.6)]
flare_l = [(0.6, 6.3), (-1.6, 5.6), (-1.3, 6.8), (-0.9, 8.0), (0.4, 8.2)]
flare_r = [(15.4, 6.3), (17.6, 5.6), (17.3, 6.8), (16.9, 8.0), (15.6, 8.2)]
hip_l = [(-0.6, 8.6), (-3.2, 9.8), (-2.8, 11.9), (-0.4, 11.4)]
hip_r = [(16.6, 8.6), (19.2, 9.8), (18.8, 11.9), (16.4, 11.4)]
closed = [(3.2, 4.2, 5.0, 4.75), (11.0, 4.2, 12.8, 4.75)]
masks, (ox, oy), (pw, ph) = clawd_masks(S, PAD, extra={
    "gown": [gown], "flares": [flare_l, flare_r, hip_l, hip_r], "closed": closed})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx = (gx_ - ox) / S
gy = (gy_ - oy) / S

orange = np.array(CLAWD_ORANGE) / 255 * 1.05
spr = np.zeros((ph, pw, 4), np.float32)
body_a = np.clip(masks["body"] + masks["arms"], 0, 1)
spr[..., :3] = orange
spr[..., 3] = body_a
# 礼服：白色 + 硬挺廓形
gown_a = np.clip(masks["gown"] + masks["flares"], 0, 1)
white = np.array([0.97, 0.95, 0.93])
seams = 0.5 + 0.5 * np.cos((gx - 8) * 1.6 + (gy - 6) * 0.25 * np.sign(gx - 8))
gcol = white[None, None] * (0.9 + 0.1 * seams)[..., None]
# 廓形部分稍暗，产生体积
gcol = gcol * (1 - 0.1 * masks["flares"])[..., None]
spr[..., :3] = over(spr[..., :3], gcol, gown_a)
spr[..., 3] = np.maximum(spr[..., 3], gown_a)
# 闭着的眼睛
spr[..., :3] = over(spr[..., :3], np.array([0.12, 0.08, 0.08]), masks["closed"])

# 放到画布全尺寸以便统一做毛毡
big = np.zeros((H, W, 4), np.float32)
check_proportions(extra={"gown": [gown], "flares": [flare_l, flare_r, hip_l, hip_r]})
CX, CY = 905, 539
rot = 0.0
tmp = to_img(spr).convert("RGBa").rotate(rot, resample=Image.BICUBIC, expand=True).convert("RGBA")
ta = to_arr(tmp)
bx, by = int(CX - ta.shape[1] / 2), int(CY - ta.shape[0] / 2)
big[by:by + ta.shape[0], bx:bx + ta.shape[1]] = ta

# 阴影
sh = blur(np.roll(np.roll(big[..., 3], 22, axis=0), 18, axis=1), 22)

clawd = felt(big, 9, puff=30, fuzz=0.22, light=1.1, shift=18)
clawd[..., :3] *= (1.0 - 0.12 * np.clip((y - 380) / 300, 0, 1))[..., None]

# 钉珠：沿礼服的竖向线条排列的小珠子 + 少量水晶闪光
bead = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
bd = ImageDraw.Draw(bead)
gown_full = np.zeros((H, W), np.float32)
gm = to_img(np.stack([gown_a] * 3 + [gown_a], -1)).convert("RGBa").rotate(rot, resample=Image.BICUBIC, expand=True).convert("RGBA")
gma = to_arr(gm)[..., 3]
gown_full[by:by + gma.shape[0], bx:bx + gma.shape[1]] = gma
cosr, sinr = math.cos(math.radians(-rot)), math.sin(math.radians(-rot))
for lane in np.linspace(-1.6, 17.6, 17):
    for k in range(40):
        yy_c = 6.6 + k * 0.16
        xx_c = 8 + (lane - 8) * (1 + (yy_c - 6) * 0.06)
        # 格坐标 -> 精灵像素 -> 画布（含旋转）
        sx_, sy_ = ox + xx_c * S - pw / 2, oy + yy_c * S - ph / 2
        px = CX + sx_ * cosr - sy_ * sinr
        py = CY + sx_ * sinr + sy_ * cosr
        if 0 <= int(py) < H and 0 <= int(px) < W and gown_full[int(py), int(px)] > 0.5:
            r = 2.6 + (k % 3 == 0) * 1.0
            c = (214, 204, 214) if k % 4 else (186, 200, 232)
            bd.ellipse([(px - r) * SS, (py - r) * SS, (px + r) * SS, (py + r) * SS], fill=c + (255,))
            bd.ellipse([(px - r * 0.4 - 0.6) * SS, (py - r * 0.5) * SS, (px) * SS, (py - r * 0.1) * SS], fill=(255, 255, 255, 255))
beads = done(bead)
export_sprite("02a", rgba_over(clawd, beads), S, (bx+ox, by+oy), extra={"gown":[gown], "flares":[flare_l,flare_r,hip_l,hip_r]})
# 前景：花丛、松树、亭子
im, d = layer()
flowers(d, 41, (120, 700, 1800, 815), 340, FLOWER_PAL)
for px, base, sc, roof, tiers in [(250, 760, 74, (214, 120, 140), 2), (1640, 752, 82, (110, 168, 170), 3), (980, 800, 44, (196, 82, 78), 1)]:
    pagoda(d, px, base, sc, roof, tiers=tiers)
for px, base, sc in [(420, 795, 95), (1460, 790, 90), (1840, 745, 70), (120, 745, 70), (720, 800, 60)]:
    pine(d, px, base, sc)
img = comp(img, felt(done(im), 12, puff=7, fuzz=0.3))

# 瀑布（岛右缘）
im, d = layer()
for k in range(14):
    xx = 1700 + k * 4 + rng(k).uniform(-2, 2)
    d.line([(xx * SS, 745 * SS), ((xx + 6) * SS, 1000 * SS)], fill=(225, 240, 255, 255), width=6 * SS)
wf = done(im)
wf[..., 3] *= (0.75 + 0.25 * np.sin(y / 7 + x))
img = comp(img, felt(wf, 13, puff=4, fuzz=0.4, light=0.1))

# 前景云雾（下方云海，盖住岩体底部）
img = comp(img, cloud_layer(15, [(150, 1010, 600), (760, 1050, 900), (1500, 1020, 800), (1880, 990, 380)], 110))
mist = blur(np.clip((y - 760) / 260, 0, 1) * (0.6 + 0.4 * MID), 10)
img = over(img, np.array([0.97, 0.95, 0.99]), mist * 0.35)

# ---------- 移轴模糊 + 暖调日光 ----------
focus_c = 590
amt = np.clip(np.abs(y - focus_c) / 360, 0, 1) ** 1.6
b1, b2 = blur(img, 4), blur(img, 10)
img = img * (1 - amt[..., None]) + (b1 * (1 - amt[..., None]) + b2 * amt[..., None]) * amt[..., None]
sun = np.exp(-(((x - 300) / 1100) ** 2 + ((y + 200) / 900) ** 2))
img = screen(img, sun[..., None] * np.array([1.0, 0.93, 0.8]) * 0.25)
lum = img.mean(-1, keepdims=True)
img = lum + (img - lum) * 1.25
img = img * 0.96 + 0.04  # 粉彩：抬黑
img = grain(img, 0.015, 3)
export_bg("02a", img)

