"""10｜珠玉 · 电影写实（延续 D 图）
10-b：歌手 2025 舞台：白色光束从四面八方打来、在身后交汇，蓝灰烟雾；Clawd（磨砂陶瓷）穿白色垂坠礼服，
      肩部荷叶边、珍珠耳饰，安静地站在光束交汇处。
10-c：结尾：十颗珍珠串成一条链，每颗里映着一首歌的世界，环绕成专辑封面的万花筒（回到 A 图）。"""
import sys, math, os, glob
import numpy as np
from PIL import Image, ImageDraw
from common import *
from pearl import pearl

OUT = sys.argv[1]
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
img[:70] *= 0
img[-70:] *= 0
img = grain(img, 0.018, 1004)
img = vignette(img, 0.35, 2.5)
save(img, os.path.join(OUT, "10-b_珠玉_白色光束.png"))
print("10-b ok")

# =====================================================================
# 10-c 结尾：珍珠链回到万花筒
# =====================================================================
A = to_arr(Image.open(os.path.join(ROOT, "造型图", "第1批_代码绘制", "A_00_专辑整体_珍珠万花筒.png")).convert("RGB"))
worlds = ["关键帧/第1批/01-a_还有什么更好的_纸壳小屋.png", "造型图/第1批_代码绘制/B_02_纯妹妹_巨像花园.png",
          "关键帧/第1批/03-a_向日葵朝着夜_月下花田.png", "关键帧/第1批/04-b_假如我们还爱着_金色羽饰.png",
          "关键帧/第1批/05-a_我管不住我自己_晨光阳台.png", "关键帧/第2批/06-a_有趣_镜面长廊.png",
          "造型图/第1批_代码绘制/C_07_我表示理解_夜城蓝光.png", "关键帧/第2批/08-a_还你茉莉_泼墨鼓点.png",
          "关键帧/第2批/09-a_多少的光阴_潮间带.png", "造型图/第1批_代码绘制/D_10_珠玉_飞向巨像.png"]
worlds = [to_arr(Image.open(os.path.join(ROOT, p)).convert("RGB").resize((640, 360), Image.LANCZOS)) for p in worlds]
img = A * 0.62
CXc, CYc = 960, 500
RX, RY = 640, 430
# 链子：一条细金线
chain = mask_from(lambda d, s: d.ellipse([(CXc - RX) * s, (CYc - RY) * s, (CXc + RX) * s, (CYc + RY) * s], outline=255, width=int(3 * s)))
img = over(img, np.array([0.85, 0.72, 0.4]), chain)
# 小珍珠（链上的间隔珠）
for k in range(60):
    a = 2 * math.pi * k / 60 + math.radians(18)
    if k % 6 == 0:
        continue
    p = pearl(13, light=(-0.4, -0.7, 0.6), ambient=0.12)
    img = paste_rgba(img, p, CXc + RX * math.sin(a), CYc - RY * math.cos(a))
# 十颗大珍珠：按曲序顺时针，01 在正上方
for i in range(10):
    a = 2 * math.pi * i / 10 + math.radians(18)   # 错开 18°，珍珠落在万花筒的 Clawd 之间
    px, py = CXc + RX * math.sin(a), CYc - RY * math.cos(a)
    sh = np.zeros((H, W), np.float32)
    p = pearl(78, light=(-0.4, -0.7, 0.6), world=worlds[i], world_mix=0.85, ambient=0.15)
    img = img * (1 - 0.4 * blur(sprite_alpha_layer(p, px + 10, py + 14), 10))[..., None]
    img = paste_rgba(img, p, px, py)
# 中心的光：万花筒中心变亮，像一切收束于此
d = np.sqrt((x - CXc) ** 2 + (y - CYc) ** 2)
img = img + np.exp(-(d / 260) ** 2)[..., None] * np.array([0.35, 0.25, 0.18])
img = bloom(img, 0.85, 30, 0.3)
img = vignette(img, 0.4, 2.2)
img = grain(img, 0.012, 1005)
save(img, os.path.join(OUT, "10-c_珠玉_珍珠链回到万花筒.png"))
print("10-c ok")
