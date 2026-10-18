# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
"""04｜假如我们还爱着 · 黑白胶片多重曝光 → 金色舞台
04-a：灰蓝暗调；三个 Clawd（侧面 → 四分之三 → 正面特写）错位叠影，像双重曝光；黑色无袖衣装；只保留一点淡橙。
04-b：西安现场：暖金色舞台烟雾；Clawd 穿金色长裙，头顶黄色羽毛帽饰，麦克风垂在身侧。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha

from titles import film_title
place = no_actor
from clawd3d import clawd_boxes, sprite

OUT = str(ASSETS)
os.makedirs(OUT, exist_ok=True)
x, y = grid()
orange = np.array(CLAWD_ORANGE) / 255


def film_finish(img, seed, grain_amt=0.05, scratches=True):
    g = rng(seed)
    # 光晕（halation）：高光外溢一圈暖色
    hi = np.clip(img.mean(-1) - 0.55, 0, 1)
    img = img + blur(hi, 14)[..., None] * np.array([0.35, 0.18, 0.1]) * 0.6
    if scratches:
        for _ in range(7):
            sx = g.uniform(0, W)
            wv = g.uniform(0.4, 1.2)
            line = np.exp(-((x - sx - 3 * np.sin(y / g.uniform(80, 200))) / wv) ** 2) * g.uniform(0.04, 0.12)
            img = img + line[..., None] * (1 if g.random() < 0.5 else -1)
        dust = np.zeros((H, W), np.float32)
        for _ in range(60):
            dust[g.integers(0, H), g.integers(0, W)] = g.uniform(-1, 1)
        img = img + blur(dust, 1.3)[..., None] * 6
    img = grain(img, grain_amt, seed, size=1.4)
    img = grain(img, grain_amt * 0.4, seed + 1)
    return img


# =====================================================================
# 04-a
# =====================================================================
img = np.zeros((H, W, 3), np.float32)
# 灰蓝底：左侧窗光，带柔和的窗格光影
img += 0.10 + np.clip(1 - x / W, 0, 1)[..., None] ** 1.3 * np.array([0.30, 0.33, 0.38])
win = ((np.sin(x / 70) > 0.2) & (np.sin(y / 70) > -0.6)).astype(np.float32) * np.clip(1 - x / 900, 0, 1)
img += blur(win, 40)[..., None] * 0.08
img = blur(img, 20)

# 去色：只留一点橙
lum = img @ np.array([0.3, 0.55, 0.15], np.float32)
bw = np.stack([lum * 0.92, lum * 0.98, lum * 1.1], -1)
img = bw * 0.7 + img * 0.3 * np.array([1.05, 0.85, 0.75])
img = np.clip((img - 0.04) * 1.25, 0, None)
img = film_finish(img, 41, 0.055)
title = film_title("假如我们还爱着", size=104, seed=1)
place(img, title, 110, 850)
img = vignette(img, 0.5, 2.0)
export_bg("04a", img)


# =====================================================================
# 04-b 金色舞台
# =====================================================================
img = np.zeros((H, W, 3), np.float32)
img += np.array([0.20, 0.13, 0.05])
haze = fbm(H, W, 61, octaves=5, base=5)
img += (haze - 0.4)[..., None] * np.array([0.25, 0.17, 0.06])
# 顶部逆光
top_beam = np.exp(-((x - 960) / 420) ** 2) * np.clip(1 - y / 900, 0, 1)
img += top_beam[..., None] * np.array([0.55, 0.42, 0.18]) * (0.6 + 0.6 * haze[..., None])
# 金色光斑
g = rng(62)
bok = np.zeros((H, W), np.float32)
for _ in range(40):
    bx, by, r = g.uniform(0, W), g.uniform(0, 700), g.uniform(10, 40)
    X0, X1, Y0, Y1 = int(max(bx - r - 2, 0)), int(min(bx + r + 2, W)), int(max(by - r - 2, 0)), int(min(by + r + 2, H))
    dd = np.sqrt((x[Y0:Y1, X0:X1] - bx) ** 2 + (y[Y0:Y1, X0:X1] - by) ** 2)
    bok[Y0:Y1, X0:X1] += np.clip(r - dd, 0, 1) * g.uniform(0.2, 0.6)
img += blur(bok, 2)[..., None] * np.array([1.0, 0.8, 0.4])
# 地面
img = img * (1 - 0.35 * np.clip((y - 880) / 200, 0, 1))[..., None]

S = 26
gown = [(0.2, 5.4), (15.8, 5.4), (15.6, 8.2), (16.4, 10.4), (18.0, 11.8), (8.0, 12.0), (-2.0, 11.8), (-0.4, 10.4), (0.4, 8.2)]
sheer = [(0, 0), (0, 0), (0, 0)]
mic = [(-0.85, 4.6), (-0.15, 4.6), (-0.05, 7.6), (-0.95, 7.6)]   # 麦克风握在原手臂末端，垂在身侧
arm_down = (-1, 3, 0, 5)
check_proportions(extra={"gown": [gown]})
cm, (ox, oy), (pw, ph) = clawd_masks(S, 120, bbox=(-7, -9, 23, 14), arms=[arm_down, (16, 3, 17, 5)],
                                     extra={"gown": [gown], "sheer": [sheer], "mic": [mic]})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
body = np.clip(cm["body"] + cm["arms"], 0, 1)
col = orange[None, None] * (0.75 + 0.45 * np.clip(1 - gy / 9, 0, 1))[..., None] * np.array([1.05, 0.95, 0.8])
col = over(col, np.array([0.05, 0.03, 0.02]), cm["eyes"])
spr = np.dstack([col, body])
# 金色长裙：亮片 + 竖向流光
gm = cm["gown"]
gg = rng(63)
seq = gg.random((ph, pw)).astype(np.float32)
seq = (seq > 0.985).astype(np.float32)
seq = blur(seq, 0.8) * 4
flow = 0.5 + 0.5 * np.sin((gx - 8) * 2.2 + np.sin(gy * 0.8) * 0.6)
gold = np.array([0.86, 0.68, 0.30])[None, None] * (0.6 + 0.45 * flow + 0.2 * np.clip(1 - gy / 13, 0, 1))[..., None]
gold = gold + np.clip(seq, 0, 1)[..., None] * np.array([1.0, 0.95, 0.75]) * 0.8
spr[..., :3] = over(spr[..., :3], gold, gm)
spr[..., 3] = np.maximum(spr[..., 3], gm)
# 外层薄纱
sm_ = cm["sheer"] * (1 - gm) * 0.35
spr[..., :3] = over(spr[..., :3], np.array([0.95, 0.82, 0.5]), sm_)
spr[..., 3] = np.maximum(spr[..., 3], sm_)
# 裙摆流苏
fr = Image.new("L", (pw * 2, ph * 2), 0)
frd = ImageDraw.Draw(fr)
for i in range(120):
    u = i / 119
    fx_ = -1.8 + 19.6 * u
    fy_ = 11.7 + 0.1 * math.sin(math.pi * u)
    frd.line([((ox + fx_ * S) * 2, (oy + fy_ * S) * 2), ((ox + (fx_ + gg.uniform(-0.3, 0.3)) * S) * 2, (oy + min(fy_ + gg.uniform(0.2, 0.4), 12.0) * S) * 2)], fill=255, width=3)
frm = to_arr(fr.resize((pw, ph), Image.LANCZOS))
spr[..., :3] = over(spr[..., :3], np.array([0.95, 0.78, 0.38]), frm)
spr[..., 3] = np.maximum(spr[..., 3], frm)
# 麦克风
spr[..., :3] = over(spr[..., :3], np.array([0.15, 0.13, 0.12]), cm["mic"])
spr[..., 3] = np.maximum(spr[..., 3], cm["mic"])

# 羽毛帽饰：一根弯曲的羽轴 + 两侧细羽枝
fe = Image.new("RGBA", (pw * 2, ph * 2), (0, 0, 0, 0))
fd = ImageDraw.Draw(fe)
def P(a, b):
    return ((ox + a * S) * 2, (oy + b * S) * 2)
spine = []
for i in range(60):
    u = i / 59
    # 从头顶左侧出发，向上、向右弯，再垂下
    px = 3.0 + 14.5 * u
    py = -0.2 - 7.5 * math.sin(math.pi * u * 0.85) + 2.0 * u * u
    spine.append((px, py))
for i in range(len(spine) - 1):
    (ax, ay), (bx, by) = spine[i], spine[i + 1]
    tx, ty = bx - ax, by - ay
    n = math.hypot(tx, ty)
    nx, ny = -ty / n, tx / n
    u = i / (len(spine) - 1)
    L = (0.6 + 2.6 * math.sin(math.pi * min(u * 1.1, 1)))
    for side_ in (-1, 1):
        for k in range(3):
            o = k / 3
            sx_, sy_ = ax + tx * o, ay + ty * o
            ex, ey = sx_ + (nx * side_ * L + tx / n * 0.9) , sy_ + (ny * side_ * L + ty / n * 0.9)
            ex += 0.5 * math.sin(i * 0.7 + k)
            c = (236, 200, 90, 210) if side_ > 0 else (214, 176, 70, 210)
            fd.line([P(sx_, sy_), P(ex, ey)], fill=c, width=3)
fd.line([P(*p) for p in spine], fill=(250, 226, 140, 255), width=6)
feather = to_arr(fe.resize((pw, ph), Image.LANCZOS))
spr[..., :3] = over(spr[..., :3], feather[..., :3], feather[..., 3])
spr[..., 3] = np.maximum(spr[..., 3], feather[..., 3])

export_sprite("04b", spr, S, (ox,oy))
CX, CY = 960, 751
af = sprite_alpha_layer(spr, CX, CY)
img = img * (1 - 0.3 * blur(af, 40))[..., None]
img = paste_rgba(img, spr, CX, CY)
# 脚下的光池
pool = np.exp(-(((x - CX) / 330) ** 2 + ((y - 1000) / 36) ** 2))
img = screen(img, pool[..., None] * np.array([0.5, 0.38, 0.15]))
# 逆光勾边
rim = np.clip(af - shift(af, 4, 0), 0, 1)
img = img + blur(rim, 2)[..., None] * np.array([1.0, 0.85, 0.5]) * 0.8
# 烟雾盖在前面一点
img = screen(img, (haze * np.clip((y - 500) / 600, 0, 1) * 0.25)[..., None] * np.array([1.0, 0.8, 0.45]))
img = bloom(img, 0.7, 30, 0.5)
img = tonemap(img * 1.2) / tonemap(np.array(1.2))
img = film_finish(img, 64, 0.035, scratches=False)
img = vignette(img, 0.45, 2.0)
export_bg("04b", img)

