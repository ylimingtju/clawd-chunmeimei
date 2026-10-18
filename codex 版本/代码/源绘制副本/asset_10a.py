# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
"""D｜10 珠玉 · 飞向巨像
电影级写实；小 Clawd（磨砂陶瓷）身穿白色钩针镂空流苏长裙飞向海上升起的光之巨像 Clawd（半透明玻璃 + 珍珠母贝 + 光丝）。"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha


OUT = str(ASSETS)
g = rng(10)
x, y = grid()
SS = 2
HZ = 735

# ---------- 天空与海 ----------
t = np.clip(y / HZ, 0, 1)
sky = np.array([0.035, 0.045, 0.07])[None, None] * (1 - t[..., None]) + np.array([0.20, 0.23, 0.27])[None, None] * t[..., None] ** 2.2
img = sky.astype(np.float32) * np.ones((H, W, 3), np.float32)
# 海面
sea_t = np.clip((y - HZ) / (H - HZ), 0, 1)
wave = fbm(H, W, 101, octaves=5, base=6)
# 横向拉伸的波纹
wv = np.asarray(Image.fromarray((fbm(H // 8, W // 2, 102, octaves=4, base=10) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
sea = np.array([0.10, 0.12, 0.15])[None, None] * (1 - sea_t[..., None] * 0.5) * (0.75 + 0.5 * wv[..., None])
img = np.where((y > HZ)[..., None], sea, img)
# 远处低矮岛屿剪影（左）
im = Image.new("L", (W * SS, H * SS), 0)
d = ImageDraw.Draw(im)
pts = [(0, HZ)]
for i in range(0, 61):
    xx = i * 11
    pts.append((xx, HZ - 18 * math.sin(i / 60 * math.pi) ** 0.8 - 4 * math.sin(i * 0.9)))
pts.append((660, HZ))
d.polygon([(px * SS, py * SS) for px, py in pts], fill=255)
d.polygon([(860 * SS, HZ * SS), (930 * SS, (HZ - 10) * SS), (1010 * SS, (HZ - 6) * SS), (1060 * SS, HZ * SS)], fill=255)
isl = to_arr(im.resize((W, H), Image.LANCZOS))
img = over(img, np.array([0.07, 0.08, 0.10]), isl * 0.9)

# ---------- 光之巨像 Clawd ----------
S = 40
PAD = 260
masks, (ox, oy), (pw, ph) = clawd_masks(S, PAD)
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
m = np.clip(masks["body"] + masks["arms"], 0, 1)
inner = edge_dist(m, 40)
rim = np.clip(m - blur(m, 6) * 1.0, 0, 1) * 2 + np.clip(blur(m, 3) - m, 0, 1) * 1.5
# 珍珠母贝的虹彩
nac = fbm(ph, pw, 111, octaves=4, base=5)
iri = hsv2rgb((0.55 + 0.5 * nac + gy * 0.02) % 1, np.full_like(nac, 0.28), np.ones_like(nac))
warm = np.array([1.0, 0.92, 0.80])
glass = warm[None, None] * (0.16 + 0.85 * (1 - inner) ** 3.0 + 0.14 * nac)[..., None]
glass = glass * (0.75 + 0.25 * iri)
# 内部流动的光丝
flow = np.sin(gx * 2.3 + np.sin(gy * 1.1 + nac * 4) * 2.5) * 0.5 + 0.5
glass = glass + blur(np.clip(flow - 0.9, 0, 1) * 5, 6)[..., None] * warm * 0.18
# 橙色的“灵魂”：中心的淡淡暖橙光，保证是 Clawd
glass = glass + (np.exp(-((gx - 8) / 5) ** 2 - ((gy - 5.5) / 3.5) ** 2) * m)[..., None] * np.array([0.85, 0.45, 0.25]) * 0.5
# 眼睛：更深、透出背景的空洞
eyes = masks["eyes"]
glass = glass * (1 - 0.75 * eyes[..., None])
lum = (glass * m[..., None]) + rim[..., None] * warm * 1.2

GX, GY = 1385, 600
light = np.zeros((H, W, 3), np.float32)
def place(arr, cx, cy):
    out = np.zeros((H, W, arr.shape[2]), np.float32)
    h, w = arr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    out[Y0:Y1, X0:X1] = arr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    return out
giant_lum = place(lum, GX, GY)
giant_a = place(m[..., None], GX, GY)[..., 0]
# 海面以下：淡出、被雾吞没
fade = np.clip((HZ + 40 - y) / 120, 0, 1)
giant_a *= fade
giant_lum *= fade[..., None]
# 半透明：背后的天空透过来
img = img * (1 - 0.15 * giant_a[..., None]) + giant_lum * 0.85

# 光的花瓣/羽翼：从巨像身后向上、向两侧展开的弧形光片
rip = Image.new("L", (W * SS, H * SS), 0)
rd = ImageDraw.Draw(rip)
for k in range(13):
    ang = math.radians(-170 + k * 160 / 12 + g.uniform(-4, 4))
    L = g.uniform(380, 560) * (1 - 0.25 * abs(math.cos(ang)))
    wdt = g.uniform(55, 95)
    base_x, base_y = GX + math.cos(ang) * 120, GY - 60 + math.sin(ang) * 80
    tip_x, tip_y = base_x + math.cos(ang) * L, base_y + math.sin(ang) * L
    nx, ny = -math.sin(ang), math.cos(ang)
    curl = g.uniform(-60, 60)
    for side in (-1, 1):
        pts = []
        for i in range(30):
            u = i / 29
            bulge = math.sin(math.pi * u) * wdt * side
            px = base_x + (tip_x - base_x) * u + nx * (bulge + curl * u * u)
            py = base_y + (tip_y - base_y) * u + ny * (bulge + curl * u * u)
            pts.append((px * SS, py * SS))
        rd.line(pts, fill=int(g.uniform(110, 190)), width=int(2.5 * SS), joint="curve")
ripples = to_arr(rip.resize((W, H), Image.LANCZOS)) * np.clip((HZ + 10 - y) / 80, 0, 1)
ripples = ripples * (1 - 0.6 * blur(np.zeros((H, W), np.float32), 1))
light += (blur(ripples, 2) * 0.45 + blur(ripples, 14) * 0.9)[..., None] * warm

# 光丝：从身体边缘向外、向上飘散
fil = Image.new("L", (W * SS, H * SS), 0)
fd = ImageDraw.Draw(fil)
bx0, by0 = GX - 8.5 * S, GY - 6.5 * S
for k in range(90):
    side = g.integers(4)
    if side == 0:
        sx, sy = bx0 + g.uniform(0, 17 * S), by0
    elif side == 1:
        sx, sy = bx0 - 1 * S, by0 + g.uniform(0, 10 * S)
    elif side == 2:
        sx, sy = bx0 + 17 * S, by0 + g.uniform(0, 10 * S)
    else:
        sx, sy = bx0 + g.uniform(0, 17 * S), by0 + 10 * S
    ang = math.atan2(sy - GY, sx - GX) + g.uniform(-0.4, 0.4)
    L = g.uniform(120, 520)
    pts = []
    ph_ = g.uniform(0, 6)
    for i in range(40):
        u = i / 39
        px = sx + math.cos(ang) * L * u + math.sin(u * 7 + ph_) * 18 * u
        py = sy + math.sin(ang) * L * u - 140 * u * u + math.cos(u * 6 + ph_) * 14 * u
        pts.append((px * SS, py * SS))
    fd.line(pts, fill=int(g.uniform(40, 150)), width=int(g.uniform(1, 3) * SS), joint="curve")
fils = to_arr(fil.resize((W, H), Image.LANCZOS)) * np.clip((HZ + 10 - y) / 60, 0, 1)
light += (fils * 0.55 + blur(fils, 5) * 0.9)[..., None] * warm
# 大范围辉光与体积光
halo = (blur(giant_a, 90) * 1.4 + blur(giant_a, 30) * 0.6) * (1 - 0.85 * giant_a)
light += halo[..., None] * np.array([1.0, 0.88, 0.72]) * 0.55
ang = np.arctan2(y - GY, x - GX)
dist = np.sqrt((x - GX) ** 2 + (y - GY) ** 2)
rays_n = np.asarray(Image.fromarray((rng(5).random((1, 720)) * 255).astype(np.uint8)).resize((720, 1), Image.BICUBIC)).astype(np.float32)[0] / 255
ray = rays_n[((ang + np.pi) / (2 * np.pi) * 719).astype(int)]
god = np.clip(ray - 0.45, 0, 1) * np.exp(-dist / 700) * np.clip((dist - 250) / 200, 0, 1)
light += blur(god, 6)[..., None] * np.array([0.9, 0.85, 0.75]) * 0.55
img = img + light * 0.85

# 海面上的倒影光柱（巨像下方）
refl_x = np.exp(-((x - GX) / 260) ** 2)
wv2 = np.asarray(Image.fromarray((fbm(H, W // 8, 103, octaves=3, base=110) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
sparkle = np.clip(wv2 * 2.2 - 1.25, 0, 1) ** 1.5 * 2
refl = refl_x * sparkle * (y > HZ) * np.exp(-(y - HZ) / 260)
img = img + refl[..., None] * warm * 0.9
img = img + (refl_x * (y > HZ) * np.exp(-(y - HZ) / 60))[..., None] * warm * 0.25

# ---------- 小 Clawd：陶瓷 + 钩针镂空长裙 ----------
s = 9
pad = 40
dress = [(1.5, 5.5), (14.5, 5.5), (16.6, 9.4), (16.4, 11.6), (0.0, 11.8), (-0.3, 9.4)]
check_proportions(extra={"dress": [dress]})
sm, (sox, soy), (spw, sph) = clawd_masks(s, pad, extra={"dress": [dress]})
sgy_, sgx_ = np.mgrid[0:sph, 0:spw].astype(np.float32)
sgx, sgy = (sgx_ - sox) / s, (sgy_ - soy) / s
sb = np.clip(sm["body"] + sm["arms"] + sm["legs"], 0, 1)
sin_ = edge_dist(sb, 6)
ceramic = np.array(CLAWD_ORANGE) / 255 * (0.55 + 0.5 * sin_)[..., None]
# 暖色轮廓光来自右侧巨像
rimR = np.clip(sb - np.roll(sb, -3, axis=1), 0, 1)
ceramic = ceramic + rimR[..., None] * np.array([1.0, 0.85, 0.65]) * 0.9
ceramic = ceramic * (1 - 0.85 * sm["eyes"][..., None])
spr = np.concatenate([ceramic, sb[..., None]], -1)
# 裙子上身部分（镂空格纹）
dm = sm["dress"]
holes = ((np.sin(sgx * 9) * np.sin(sgy * 9)) > 0.6).astype(np.float32)
dcol = np.array([0.93, 0.91, 0.87]) * (0.75 + 0.35 * np.clip((sgx + 1) / 18, 0, 1))[..., None]
da = dm * (1 - 0.85 * holes)
spr[..., :3] = over(spr[..., :3], dcol, da)
spr[..., 3] = np.maximum(spr[..., 3], da)

export_sprite("10a",spr,s,(sox,soy))
# ---------- 雾、粒子、电影调色 ----------
mist = np.exp(-((y - HZ) / 55) ** 2) * (0.6 + 0.4 * fbm(H, W, 120, octaves=3, base=5))
img = img + mist[..., None] * np.array([0.35, 0.37, 0.40]) * (0.5 + 0.8 * np.exp(-((x - GX) / 600) ** 2))[..., None]
dust = np.zeros((H, W), np.float32)
for _ in range(220):
    px, py = int(g.uniform(0, W)), int(g.uniform(0, HZ))
    dust[py, px] = g.uniform(0.4, 1)
img = img + blur(dust, 1.2)[..., None] * 8 * warm * 0.25
img = bloom(img, 0.75, 40, 0.5)
img = tonemap(img * 1.1)
# 冷暗部、暖高光
lum = img.mean(-1, keepdims=True)
img = img + (1 - lum) * np.array([-0.01, 0.0, 0.025]) + lum * np.array([0.02, 0.005, -0.02])
img = grain(img, 0.018, 12)
img = vignette(img, 0.35, 2.5)
export_bg("10a", img)

