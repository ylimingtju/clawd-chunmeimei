# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
"""10｜珠玉 · 电影写实（延续 D 图）
10-b：歌手 2025 舞台：白色光束从四面八方打来、在身后交汇，蓝灰烟雾；Clawd（磨砂陶瓷）穿白色垂坠礼服，
      肩部荷叶边、珍珠耳饰，安静地站在光束交汇处。
10-c：结尾：十颗珍珠串成一条链，每颗里映着一首歌的世界，环绕成专辑封面的万花筒（回到 A 图）。"""
import sys, math, os, glob
import numpy as np
from PIL import Image, ImageDraw
from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha

from pearl import pearl

OUT = str(ASSETS)
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()
ORANGE = np.array(CLAWD_ORANGE) / 255
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def mask_from(fn):
    im = Image.new("L", (W * SS, H * SS), 0)
    fn(ImageDraw.Draw(im), SS)
    return to_arr(im.resize((W, H), Image.LANCZOS))

# =====================================================================
# 10-b
# =====================================================================
img = np.zeros((H, W, 3), np.float32)
img[:] = np.array([0.06, 0.08, 0.11])
haze = fbm(H, W, 1001, octaves=5, base=4)
img = img + (haze - 0.3)[..., None] * np.array([0.10, 0.13, 0.17])
CXb, CYb = 960, 470   # 光束交汇点（在 Clawd 身后上方）
light = np.zeros((H, W), np.float32)
g = rng(1002)
for k in range(14):
    ang = math.radians(-170 + k * 160 / 13 + g.uniform(-5, 5))
    sx, sy = CXb + math.cos(ang) * 1400, CYb + math.sin(ang) * 900 - 200
    spread = g.uniform(0.04, 0.08)
    a0 = math.atan2(CYb - sy, CXb - sx)
    p1 = (sx, sy)
    p2 = (CXb + math.cos(a0 + math.pi / 2) * 40, CYb + math.sin(a0 + math.pi / 2) * 40)
    p3 = (CXb - math.cos(a0 + math.pi / 2) * 40, CYb - math.sin(a0 + math.pi / 2) * 40)
    width = 1400 * spread
    p0 = (sx + math.cos(a0 + math.pi / 2) * width, sy + math.sin(a0 + math.pi / 2) * width)
    p4 = (sx - math.cos(a0 + math.pi / 2) * width, sy - math.sin(a0 + math.pi / 2) * width)
    m = mask_from(lambda d, s: d.polygon([(p0[0] * s, p0[1] * s), (p2[0] * s, p2[1] * s), (p3[0] * s, p3[1] * s), (p4[0] * s, p4[1] * s)], fill=255))
    light += blur(m, 18) * g.uniform(0.35, 0.7)
light = light * (0.55 + 0.6 * haze)
img = img + light[..., None] * np.array([0.75, 0.82, 0.92])
core = np.exp(-(((x - CXb) / 260) ** 2 + ((y - CYb) / 200) ** 2))
img = img + core[..., None] * np.array([0.6, 0.66, 0.75])
# 地面
img = img * (1 - 0.45 * np.clip((y - 900) / 180, 0, 1))[..., None]

S = 30
gown = [(-0.4, 5.8), (16.4, 5.8), (16.0, 8.2), (16.8, 10.4), (18.2, 11.9), (8.0, 11.8), (-2.2, 11.9), (-0.8, 10.4), (0.0, 8.2)]
ruffle_l = [(-1.4, 5.4), (5.0, 5.6), (6.6, 6.4), (5.4, 7.4), (1.6, 8.0), (-1.4, 7.6), (-2.2, 6.4)]
ruffle_r = [(17.4, 5.4), (11.0, 5.6), (9.4, 6.4), (10.6, 7.4), (14.4, 8.0), (17.4, 7.6), (18.2, 6.4)]
arm_mic = rigid_arm("R", 40)
check_proportions(arms=[arm_mic], extra={"gown": [gown], "ruffle": [ruffle_l, ruffle_r]})
cm, (ox, oy), (pw, ph) = clawd_masks(S, 120, bbox=(-6, -4, 22, 12), arms=[(-1, 3, 0, 5), arm_mic],
                                     extra={"gown": [gown], "ruffle": [ruffle_l, ruffle_r]})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
inner = edge_dist(body, S * 0.7)
# 磨砂陶瓷：柔和体积、顶部受光、边缘被身后光束勾亮
col = ORANGE[None, None] * (0.6 + 0.5 * inner)[..., None] * (0.9 + 0.3 * np.clip(1 - gy / 10, 0, 1))[..., None]
col = col * (0.97 + 0.06 * fbm(ph, pw, 1003, octaves=3, base=40))[..., None]
col = over(col, np.array([0.04, 0.03, 0.04]), cm["eyes"])
spr = np.dstack([col, body])
satin = np.array([0.93, 0.93, 0.95])
fold = np.sin((gx - 8) * 1.1 + np.sin(gy * 0.6) * 0.8) * 0.5 + 0.5
gm = cm["gown"]
gcol = satin[None, None] * (0.8 + 0.16 * fold + 0.1 * np.clip(1 - (gy - 6) / 6, 0, 1) - 0.12 * np.clip((gy - 9) / 3, 0, 1))[..., None]
spr[..., :3] = over(spr[..., :3], gcol, gm)
spr[..., 3] = np.maximum(spr[..., 3], gm)
rm = cm["ruffle"]
rf = np.sin(np.hypot(gx - 8, gy - 3) * 2.2) * 0.5 + 0.5
rcol = satin[None, None] * (0.86 + 0.14 * rf)[..., None]
spr[..., :3] = over(spr[..., :3], rcol, rm)
spr[..., 3] = np.maximum(spr[..., 3], rm)
# 麦克风
tip = ((arm_mic[1][0] + arm_mic[2][0]) / 2, (arm_mic[1][1] + arm_mic[2][1]) / 2)
mic = Image.new("RGBA", (pw * 2, ph * 2), (0, 0, 0, 0))
md = ImageDraw.Draw(mic)
P = lambda a, b: ((ox + a * S) * 2, (oy + b * S) * 2)
end = (tip[0] - 1.0, tip[1] - 1.5)
md.line([P(*tip), P(*end)], fill=(26, 26, 30, 255), width=int(0.75 * S * 2))
ex, ey = P(*end)
md.ellipse([ex - S * 1.7, ey - S * 1.7, ex + S * 1.7, ey + S * 1.7], fill=(205, 208, 214, 255))
mm = to_arr(mic.resize((pw, ph), Image.LANCZOS))
spr[..., :3] = over(spr[..., :3], mm[..., :3], mm[..., 3])
spr[..., 3] = np.maximum(spr[..., 3], mm[..., 3])
# 珍珠耳饰：身体两侧上方
for side_x in (-0.6, 16.6):
    pe = pearl(int(S * 0.55), light=(-0.3, -0.7, 0.6))
    py0 = int(oy + 2.2 * S - pe.shape[0] / 2)
    px0 = int(ox + side_x * S - pe.shape[1] / 2)
    sub = spr[py0:py0 + pe.shape[0], px0:px0 + pe.shape[1]]
    a = pe[..., 3:4]
    sub[..., :3] = sub[..., :3] * (1 - a) + pe[..., :3] * a
    sub[..., 3:4] = np.maximum(sub[..., 3:4], a)
export_sprite("10b",spr,S,(ox,oy))
CX, FOOT = 960, 930
CY = FOOT - (oy + 12 * S - ph / 2)
af = sprite_alpha_layer(spr, CX, CY)
img = img * (1 - 0.35 * blur(af, 50))[..., None]
img = paste_rgba(img, spr, CX, CY)
# 身后光束在边缘的强烈勾边与少量溢光
img = img + blur(np.clip(af - np.clip((blur(af, 4) - 0.5) * 6 + 0.5, 0, 1), 0, 1), 1.5)[..., None] * np.array([0.8, 0.85, 0.95])
img = screen(img, blur(core * (1 - af), 30)[..., None] * 0.15)
# 脚下光池
img = img + np.exp(-(((x - CX) / 320) ** 2 + ((y - FOOT - 8) / 26) ** 2))[..., None] * np.array([0.35, 0.38, 0.45])
img = bloom(img, 0.75, 40, 0.5)
img = tonemap(img * 1.15)
lum = img.mean(-1, keepdims=True)
img = img + (1 - lum) * np.array([-0.01, 0.0, 0.025]) + lum * np.array([0.015, 0.005, -0.015])


img = grain(img, 0.018, 1004)
img = vignette(img, 0.35, 2.5)
export_bg("10b", img)


