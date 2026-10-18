# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
"""C｜07 我表示理解 · 夜城蓝光
低保真 DV / 16mm 夜拍；毛绒 Clawd 穿过大黑色皮夹克、窄墨镜、握麦克风；冷蓝紫夜光 + 直闪；画面边缘白色手写涂鸦。"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha


OUT = str(ASSETS)
g = rng(7)
x, y = grid()
SS = 2

# ---------- 背景：高架桥下 ----------
HZ = 470  # 视平线
img = np.zeros((H, W, 3), np.float32)
navy = np.array([0.05, 0.05, 0.14])
img[:] = navy
# 远处天空/城市漫射光
img += (np.exp(-((y - HZ) / 160) ** 2))[..., None] * np.array([0.10, 0.10, 0.30])
# 桥底（上方深色斜面，带结构梁）
ceil = np.clip((HZ - 120 - y) / 300, 0, 1)
beams = (np.sin((x - W * 0.55) / (np.maximum(HZ - y, 30)) * 18) > 0.85) * ceil
img = img * (1 - 0.7 * ceil[..., None]) + beams[..., None] * np.array([0.05, 0.05, 0.12])
# 路面（透视）
road = np.clip((y - HZ) / (H - HZ), 0, 1)
img += road[..., None] * np.array([0.05, 0.06, 0.16])
# 车道线
im = Image.new("L", (W * SS, H * SS), 0)
d = ImageDraw.Draw(im)
vx = W * 0.62
for k, off in enumerate([-900, -300, 500, 1300]):
    d.line([(vx * SS, HZ * SS), ((vx + off * 1.4) * SS, H * SS)], fill=150 if k in (1, 2) else 90, width=6 * SS)
for i in range(12):  # 中间虚线
    t0, t1 = (i / 12) ** 2, ((i + 0.45) / 12) ** 2
    d.line([((vx + 120 * t0 * 1.4) * SS, (HZ + (H - HZ) * t0) * SS), ((vx + 120 * t1 * 1.4) * SS, (HZ + (H - HZ) * t1) * SS)], fill=170, width=int((2 + 10 * t1) * SS))
lines = blur(to_arr(im.resize((W, H), Image.LANCZOS)), 1.5)
img += lines[..., None] * np.array([0.25, 0.25, 0.45])
# 立柱
for (px, w_, sh) in [(140, 170, 0.0), (520, 90, 0.2), (1480, 120, 0.15), (1790, 210, 0.0)]:
    m = ((x > px) & (x < px + w_) & (y > 60)).astype(np.float32)
    m = blur(m, 2)
    col = np.array([0.03, 0.03, 0.08]) + np.clip((x - px) / w_, 0, 1)[..., None] * np.array([0.03, 0.03, 0.10])
    img = over(img, col, m * (1 - sh))
# 路灯光斑（失焦）
bok = np.zeros((H, W, 3), np.float32)
for i in range(46):
    bx, by = g.uniform(0, W), g.uniform(HZ - 260, HZ + 60)
    r = g.uniform(12, 46)
    c = [np.array([0.6, 0.7, 1.0]), np.array([0.9, 0.9, 1.0]), np.array([0.7, 0.5, 1.0]), np.array([1.0, 0.8, 0.6])][g.integers(4)]
    X0, X1, Y0, Y1 = int(max(bx - r - 3, 0)), int(min(bx + r + 3, W)), int(max(by - r - 3, 0)), int(min(by + r + 3, H))
    dd = np.sqrt((x[Y0:Y1, X0:X1] - bx) ** 2 + (y[Y0:Y1, X0:X1] - by) ** 2)
    disc = np.clip(r - dd, 0, 1) * (0.55 + 0.45 * np.clip(dd / r, 0, 1) ** 3)
    bok[Y0:Y1, X0:X1] += disc[..., None] * c * g.uniform(0.25, 0.7)
img += blur(bok, 1.5)
# 两侧的城市灯带
img += (np.exp(-((y - (HZ - 30)) / 14) ** 2) * (0.5 + 0.5 * np.sin(x / 23) ** 8))[..., None] * np.array([0.2, 0.2, 0.5]) * 0.6
img = blur(img, 3)  # 背景整体虚焦

# ---------- Clawd：毛绒 + 皮夹克 + 墨镜 + 麦克风 ----------
S = 42
PAD = 140
# 过大的皮夹克：从“肩部”垂下、前襟敞开，露出脸和眼睛
jacket = [[(-2.6, -0.4), (2.9, -0.4), (3.1, 2.2), (2.7, 6.5), (3.4, 11.6), (-2.9, 11.6), (-3.3, 4.0)],
          [(13.1, -0.4), (18.6, -0.4), (19.3, 4.0), (18.9, 11.6), (12.6, 11.6), (13.3, 6.5), (12.9, 2.2)],
          [(-2.9, 9.6), (18.9, 9.6), (18.9, 11.6), (-2.9, 11.6)]]
collar_l = [(2.9, -0.4), (4.4, -0.2), (3.6, 3.0), (3.1, 2.2)]
collar_r = [(13.1, -0.4), (11.6, -0.2), (12.4, 3.0), (12.9, 2.2)]
glasses = [(2.7, 2.8, 5.5, 5.2), (10.5, 2.8, 13.3, 5.2), (5.5, 3.3, 10.5, 3.65)]
masks, (ox, oy), (pw, ph) = clawd_masks(S, PAD, extra={"jacket": jacket, "collar": [collar_l, collar_r], "glasses": glasses})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S

# 毛绒：边缘细毛 + 绒面噪声
fur_n = rng(71).normal(0, 1, (ph, pw)).astype(np.float32)
fur = blur(np.clip(fur_n * 0.2 + 0.5, 0, 1), 0.7)
body = masks["body"] + masks["arms"] + masks["legs"]
body = np.clip(body, 0, 1)
fuzzy = np.clip(blur(body, 3.5) + (fur - 0.5) * 2.4 * (1 - np.abs(blur(body, 2.5) * 2 - 1)), 0, 1)
orange = np.array(CLAWD_ORANGE) / 255
inner = edge_dist(body, 22)
fur2 = blur(np.clip(rng(73).normal(0, 1, (ph, pw)).astype(np.float32) * 0.25 + 0.5, 0, 1), 1.6)
col = orange[None, None] * (0.6 + 0.5 * inner)[..., None] * (0.7 + 0.35 * fur + 0.35 * fur2)[..., None]
alpha = fuzzy.copy()
# 黑色皮夹克
jm = masks["jacket"]
leather_n = fbm(ph, pw, 72, octaves=4, base=6)
folds = np.sin(gx * 2.2 + leather_n * 1.5 + gy * 0.15) * 0.5 + 0.5
jcol = np.array([0.035, 0.035, 0.05])[None, None] * (0.8 + 0.4 * leather_n)[..., None]
# 皮革高光：沿褶皱的冷白反光 + 闪光灯热点
spec = blur(np.clip((folds - 0.86) / 0.14, 0, 1) ** 2, 4) * (0.35 + 0.4 * leather_n)
spec += 0.8 * np.exp(-((gx + 0.5) / 1.8) ** 2 - ((gy - 0.6) / 0.8) ** 2)
spec += 0.6 * np.exp(-((gx - 16.3) / 1.6) ** 2 - ((gy - 0.8) / 0.8) ** 2)
spec += 0.35 * np.exp(-((gx - 1.6) / 0.6) ** 2 - ((gy - 7.5) / 3.0) ** 2)
jcol = jcol + spec[..., None] * np.array([0.45, 0.5, 0.75])
col = over(col, jcol, jm)
alpha = np.maximum(alpha, jm)
cm = masks["collar"]
col = over(col, np.array([0.02, 0.02, 0.03]) + (np.clip(gy - 5, 0, 2) / 2)[..., None] * 0.05, cm * 0.9)
# 衣领边缘高光
cedge = np.clip(cm - blur(cm, 2), 0, 1) * 3
col = screen(col, np.clip(cedge, 0, 1)[..., None] * np.array([0.5, 0.55, 0.8]))
# 眼睛（墨镜下仍可见形状）
col = over(col, np.array([0.03, 0.03, 0.04]), masks["eyes"])
# 窄墨镜：黑色镜片 + 蓝色反光
gm = masks["glasses"]
gl = np.array([0.02, 0.02, 0.03])[None, None] * np.ones_like(col)
refl = np.exp(-((gy - 3.4 - (gx - 2) * 0.06) / 0.16) ** 2) * 0.9
gl = gl + refl[..., None] * np.array([0.35, 0.5, 1.0])
col = over(col, gl, gm)
alpha = np.maximum(alpha, gm)

sprite = np.concatenate([col, alpha[..., None]], -1)

export_sprite("07city", sprite, S, (ox,oy), extra={"jacket":jacket,"collar":[collar_l,collar_r],"glasses":glasses})

img = img * np.array([0.8, 0.85, 1.15]) + np.array([0.02, 0.01, 0.05])
CX, CY = 1040, 611
ROT = -4
# 闪光灯：主体前方硬光 -> 后方背景压暗
alpha_full = sprite_alpha_layer(sprite, CX, CY, ROT)
img = img * (1 - 0.35 * blur(alpha_full, 60))[..., None]
# 地面硬阴影（直闪造成）
shadow = blur(np.roll(np.roll(alpha_full, 22, 0), 26, 1), 6)
img = img * (1 - 0.6 * shadow)[..., None]
img = paste_rgba(img, sprite, CX, CY, ROT)
# 闪光衰减：中心亮
flash = np.exp(-(((x - CX) / 700) ** 2 + ((y - CY) / 520) ** 2))
img = img * (0.75 + 0.45 * flash)[..., None]

# ---------- 白色手写涂鸦（不可辨认） ----------
sc = Image.new("L", (W * SS, H * SS), 0)
sd = ImageDraw.Draw(sc)
gg = rng(77)
def scribble_word(x0, y0, length, hgt):
    pts = []
    xx = x0
    while xx < x0 + length:
        pts.append(((xx + gg.uniform(-3, 3)) * SS, (y0 + gg.uniform(-hgt, hgt) * (0.5 + 0.5 * math.sin(xx / 9))) * SS))
        xx += gg.uniform(5, 13)
    sd.line(pts, fill=255, width=3 * SS, joint="curve")
for row in range(7):
    yy = 140 + row * 58 + gg.uniform(-8, 8)
    xx = 44 + gg.uniform(0, 20)
    for w_ in range(gg.integers(2, 4)):
        L = gg.uniform(50, 120)
        scribble_word(xx, yy, L, 16)
        xx += L + gg.uniform(16, 30)
for row in range(3):
    yy = 860 + row * 52
    xx = 1560 + gg.uniform(0, 30)
    for w_ in range(2):
        L = gg.uniform(60, 120)
        scribble_word(xx, yy, L, 14)
        xx += L + 22
sca = blur(to_arr(sc.resize((W, H), Image.LANCZOS)), 0.8)
img = screen(img, sca[..., None] * np.array([0.92, 0.92, 0.98]) * 0.85)

# ---------- 后期：蓝紫调色、DV 质感 ----------
img = img * np.array([0.93, 0.95, 1.06])
lum = img.mean(-1, keepdims=True)
img = lum + (img - lum) * 0.9
# 轻微横向运动模糊
mb = img.copy()
for k in (-6, -3, 3, 6):
    mb += np.roll(img, k, axis=1)
img = img * 0.55 + (mb / 5) * 0.45
img = bloom(img, 0.6, 18, 0.5)
# 色差
img[..., 0] = np.roll(img[..., 0], 3, axis=1)
img[..., 2] = np.roll(img[..., 2], -3, axis=1)
# 扫描线 + 粗颗粒
img = img * (0.94 + 0.06 * (np.sin(y * np.pi / 2) ** 2))[..., None]
img = grain(img, 0.06, 9, size=1.6)
img = grain(img, 0.025, 10)
img = img * 0.92 + 0.045  # DV 发灰的黑位
img = vignette(img, 0.55, 2.2)
export_bg("07city", img)

