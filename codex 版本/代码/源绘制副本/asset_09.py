# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
"""09｜多少的光阴 · 沙画（灯箱背光：沙越厚越暗，手指抹开处透光）
09-a：黄昏的潮间带：左边夕阳落进海面、右边月亮和星升起（日月同框表现时间流逝）；潮水涌上来抹掉一串脚印；
      Clawd 坐在沙滩上（彩沙），沙上写着歌名，最后两个字正被潮水抹去。
09-b：2024 舞台（沙画表现）：深色沙面、红紫色星点灯从沙里透出、台口一排地灯；Clawd 穿银色亮片长礼服、V 领，握麦克风。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha

from titles import sand_title

OUT = str(ASSETS)
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()
ORANGE = np.array(CLAWD_ORANGE) / 255


def mask_from(fn):
    im = Image.new("L", (W * SS, H * SS), 0)
    fn(ImageDraw.Draw(im), SS)
    return to_arr(im.resize((W, H), Image.LANCZOS))


GRAIN = rng(900).random((H, W)).astype(np.float32)
GRAIN2 = blur(rng(901).random((H, W)).astype(np.float32), 0.8)


def light_table(D, backlight, k=3.4, tint=None):
    """沙的厚度 D → 背光透射。tint：彩沙区域的颜色（RGBA，A 为彩沙覆盖）。"""
    Dg = np.clip(D * (0.82 + 0.36 * GRAIN2) + (GRAIN > 0.985) * 0.25 * (D > 0.08), 0, None)
    T = np.exp(-k * Dg)[..., None]
    img = backlight * T
    if tint is not None:
        c, a = tint[..., :3], tint[..., 3:4]
        img = img * (1 - a) + (np.maximum(backlight, 0.75) * c * np.exp(-1.0 * Dg)[..., None] * 1.15) * a
    # 散落的沙粒（亮背景上的小黑点）
    loose = (GRAIN > 0.997) * (D < 0.3)
    img = img * (1 - 0.6 * loose[..., None])
    return img


def ridge_lines(lines, width, soft=1.5):
    return blur(mask_from(lambda d, s: [d.line([(p[0] * s, p[1] * s) for p in pl], fill=255, width=int(width * s), joint="curve") for pl in lines]), soft)


def clawd_mask_at(S, cx, foot, extra=None, arms=None):
    check_proportions(arms=arms, extra=extra)
    cm, (ox, oy), (pw, ph) = clawd_masks(S, int(S * 2), arms=arms, bbox=(-4, -3, 21, 12), extra=extra)
    cy = foot - (oy + 12 * S - ph / 2)
    out = {}
    x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
    for k, m in cm.items():
        a = np.zeros((H, W), np.float32)
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + pw, W), min(y0 + ph, H)
        a[Y0:Y1, X0:X1] = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        out[k] = a
    gp = lambda gx_, gy_: (x0 + ox + gx_ * S, y0 + oy + gy_ * S)
    return out, gp

# =====================================================================
# 09-a 潮间带
# =====================================================================
HZ = 470
# 背光：左侧琥珀（黄昏）→ 右侧靛蓝（入夜）
u = np.clip(x / W, 0, 1)[..., None]
amber = np.array([1.0, 0.72, 0.36])
indigo = np.array([0.36, 0.42, 0.78])
back = amber * (1 - u) ** 1.2 + indigo * (1 - (1 - u) ** 1.2)
back = back * (0.9 + 0.15 * np.clip(1 - y / H, 0, 1))[..., None]
# 天空：薄沙，向右逐渐变厚（天黑）；手抹的弧形云带
D = 0.12 + 0.35 * np.clip((x - 700) / 1200, 0, 1) ** 1.2
D = D + 0.05 * fbm(H, W, 910, octaves=3, base=6)
sweeps = ridge_lines([[(xx, 150 + 60 * math.sin(xx / 260) + k * 70) for xx in range(-50, W + 60, 30)] for k in range(4)], 26, 6)
D = D - 0.06 * sweeps * (y < HZ)
# 夕阳：左侧，一半沉入海面（抹开的圆，透出强光）+ 放射状抹痕
SX, SY, SR = 360, HZ, 120
sun = mask_from(lambda d, s: d.ellipse([(SX - SR) * s, (SY - SR) * s, (SX + SR) * s, (SY + SR) * s], fill=255)) * (y < HZ)
D = D * (1 - 0.95 * blur(sun, 2))
rays = ridge_lines([[(SX + math.cos(a) * (SR + 30), SY + math.sin(a) * (SR + 30)), (SX + math.cos(a) * (SR + 230), SY + math.sin(a) * (SR + 230))]
                    for a in np.linspace(math.pi * 1.05, math.pi * 1.95, 11)], 9, 2)
D = D - 0.08 * rays * (y < HZ)
# 月牙：右上（大圆减去偏移圆）+ 手指点出的星
MX, MY, MR = 1600, 170, 70
moon = mask_from(lambda d, s: d.ellipse([(MX - MR) * s, (MY - MR) * s, (MX + MR) * s, (MY + MR) * s], fill=255))
bite = mask_from(lambda d, s: d.ellipse([(MX - MR + 34) * s, (MY - MR - 14) * s, (MX + MR + 34) * s, (MY + MR - 14) * s], fill=255))
D = D * (1 - 0.95 * blur(np.clip(moon - bite, 0, 1), 1.5))
g = rng(912)
stars = mask_from(lambda d, s: [d.ellipse([(px - r) * s, (py - r) * s, (px + r) * s, (py + r) * s], fill=255)
                                for px, py, r in [(g.uniform(1100, W), g.uniform(30, HZ - 60), g.uniform(2, 6)) for _ in range(40)]])
D = D * (1 - 0.9 * stars)
# 海：较厚的沙，几道浪线（手指划出的细亮线）
sea = (y >= HZ) & (y < 700)
D = np.where(sea, 0.22 + 0.1 * np.clip((y - HZ) / 230, 0, 1) + 0.08 * fbm(H, W, 913, octaves=3, base=8), D)
waves = ridge_lines([[(xx, HZ + 30 + k * 42 + 10 * math.sin(xx / (80 + k * 20) + k)) for xx in range(-40, W + 40, 20)] for k in range(5)], 4, 1)
D = D - 0.2 * waves * sea
# 海面上夕阳的倒影光柱
refl = ridge_lines([[(SX - 110 + j * 6, HZ + 18 + j * 36), (SX + 110 - j * 12, HZ + 18 + j * 36)] for j in range(6)], 7, 2)
D = D - 0.22 * refl * sea
# 沙滩：更厚，潮水边缘一条起伏的湿沙线
def shore(xx):
    return 730 + 50 * math.sin(xx / 230) + 18 * math.sin(xx / 60)
shore_poly = [(xx, shore(xx)) for xx in range(-40, W + 41, 20)]
beach = mask_from(lambda d, s: d.polygon([(p[0] * s, p[1] * s) for p in shore_poly + [(W + 40, H + 10), (-40, H + 10)]], fill=255))
foam = (y >= 700) * (1 - beach)
D = np.where(foam > 0.5, 0.2 + 0.06 * fbm(H, W, 915, octaves=3, base=10), D)  # 涌上来的潮水（较薄、透亮）
foam_lines = ridge_lines([[(xx, shore(xx) - 8 - k * 12 + 5 * math.sin(xx / 25 + k)) for xx in range(-40, W + 41, 14)] for k in range(3)], 3, 1)
D = D - 0.25 * foam_lines * (y > 690)
D = np.where(beach > 0.5, 0.26 + 0.12 * np.clip((y - 760) / 320, 0, 1) + 0.1 * fbm(H, W, 914, octaves=3, base=10), D)
D = D + 0.22 * ridge_lines([shore_poly], 10, 4)  # 湿沙线
# 一串脚印：从左下走向 Clawd，靠近水的几只被潮水抹淡
prints = []
for i in range(9):
    px = 100 + i * 76
    py = 990 - i * 18 + (12 if i % 2 else -12)
    prints.append((px, py, 14, 9))
pm = mask_from(lambda d, s: [d.ellipse([(px - rx) * s, (py - ry) * s, (px + rx) * s, (py + ry) * s], fill=255) for px, py, rx, ry in prints])
erase = np.clip(1 - (y - 900) / 120, 0, 1) * np.clip((x - 400) / 300, 0, 1)
D = D - 0.55 * pm * (1 - 0.8 * erase)

D_hold = D.copy()
# Clawd：坐在沙滩上（彩沙：橙色），轮廓用手指勾出的亮线
cm, gp = clawd_mask_at(15, 880, 890)
body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
D = D * (1 - 0.75 * body)                    # 彩沙层比较薄
outline = np.clip(blur(body, 2.5) - body, 0, 1) * 3
D = D - 0.5 * np.clip(outline, 0, 1)
D = D + 1.2 * cm["eyes"]
tint = np.dstack([np.ones((H, W, 3)) * ORANGE, (body * (1 - cm["eyes"]) * 0.95)[..., None]])

actor = light_table(np.clip(D,0,None),back,tint=tint)
alpha = np.clip(body+outline,0,1)
export_sprite("09a",np.dstack([actor,alpha]),15,gp(0,0))
D = D_hold
tint = None
D = blur(D, 1.2)   # 沙层边界自然过渡
img = light_table(np.clip(D, 0, None), back, tint=tint)
img = img + blur(sun, 30)[..., None] * amber * 0.35
img = vignette(img, 0.35, 2.2)
export_bg("09a", img)


# =====================================================================
# 09-b 舞台
# =====================================================================
back = np.ones((H, W, 3), np.float32) * np.array([0.98, 0.86, 0.80])
D = 0.78 + 0.14 * fbm(H, W, 920, octaves=4, base=5)
# 手掌抹出的弧形沙纹
D = D - 0.12 * ridge_lines([[(960 + r * math.cos(a), 1150 + r * 0.6 * math.sin(a)) for a in np.linspace(math.pi * 1.02, math.pi * 1.98, 60)] for r in range(500, 1500, 70)], 14, 5)
# 舞台背后的雾光：从下往上抹薄
D = D - 0.55 * np.exp(-((y - 760) / 140) ** 2) * (0.7 + 0.3 * fbm(H, W, 921, octaves=3, base=4))
tintC = np.zeros((H, W, 4), np.float32)
# 红紫星点灯：手指点出的孔，透出彩光
g = rng(922)
cols = [np.array([1.0, 0.35, 0.45]), np.array([0.8, 0.4, 1.0]), np.array([1.0, 0.55, 0.3]), np.array([0.5, 0.6, 1.0])]
for _ in range(70):
    px, py, r = g.uniform(0, W), g.uniform(20, 640), g.uniform(5, 16)
    if abs(px - 960) < 260 and py > 250:
        continue
    m = mask_from(lambda d, s, px=px, py=py, r=r: d.ellipse([(px - r) * s, (py - r) * s, (px + r) * s, (py + r) * s], fill=255))
    m = blur(m, 2.5)
    D = D * (1 - 0.92 * m)
    c = cols[g.integers(len(cols))]
    tintC[..., :3] = tintC[..., :3] * (1 - m[..., None]) + c * m[..., None]
    tintC[..., 3] = np.maximum(tintC[..., 3], m * 0.9)
# 台口地灯：一排小孔
for k in range(30):
    px = 30 + k * 64
    m = mask_from(lambda d, s, px=px: d.ellipse([(px - 11) * s, (1020 - 9) * s, (px + 11) * s, (1020 + 9) * s], fill=255))
    D = D * (1 - 0.95 * blur(m, 2))
    tintC[..., :3] = tintC[..., :3] * (1 - m[..., None]) + np.array([1.0, 0.45, 0.35]) * m[..., None]
    tintC[..., 3] = np.maximum(tintC[..., 3], m)
# 舞台边沿：一道抹开的亮线
D = D - 0.5 * ridge_lines([[(0, 980), (W, 980)]], 6, 2)

D_hold = D.copy()
tint_hold = tintC.copy()
S = 26
gown = [(0.0, 5.4), (16.0, 5.4), (15.8, 8.6), (16.6, 11.0), (17.4, 12.0), (8.0, 11.8), (-1.4, 12.0), (-0.6, 11.0), (0.2, 8.6)]
vneck = [(4.6, 5.4), (11.4, 5.4), (8.0, 8.4)]
mic_arm = rigid_arm("R", 45)
cm, gp = clawd_mask_at(S, 960, 965, extra={"gown": [gown], "vneck": [vneck]}, arms=[(-1, 3, 0, 5), mic_arm])
body = np.clip(cm["body"] + cm["arms"], 0, 1)
gm = np.clip(cm["gown"] - cm["vneck"], 0, 1)
# 人物整体：沙抹薄（追光照在身上）
fig = np.clip(body + gm, 0, 1)
D = D * (1 - 0.8 * blur(fig, 1.5))
D = D - 0.4 * np.clip(blur(fig, 3) - fig, 0, 1) * 3          # 轮廓亮边
D = D + 1.3 * cm["eyes"]
# 银色亮片：裙面上竖向的亮片条 + 随机闪点（沙粒间透出的白光）
seq = blur(rng(924).random((H, W)).astype(np.float32), 1.6)
streak = np.asarray(Image.fromarray((rng(925).random((H // 10, W // 3)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
D = D + 0.45 * gm * np.clip((seq - 0.5) * 6 + 0.3 * (streak - 0.5) * 4, 0, 1)   # 亮片之间的细沙缝
spark = (rng(923).random((H, W)) > 0.975) * gm
D = D * (1 - 0.95 * np.clip(blur(spark.astype(np.float32), 1.0) * 2.5, 0, 1))
tintC[..., :3] = over(tintC[..., :3], ORANGE, body * (1 - cm["eyes"]))
tintC[..., 3] = np.maximum(tintC[..., 3], body * (1 - cm["eyes"]) * 0.95)
tintC[..., :3] = over(tintC[..., :3], np.array([0.86, 0.86, 0.94]), gm)
tintC[..., 3] = np.maximum(tintC[..., 3], gm * 0.85)
# 麦克风（右手）
mt = gp(17.6, 1.6)
mic = mask_from(lambda d, s: [d.line([(gp(16.8, 3.0)[0] * s, gp(16.8, 3.0)[1] * s), (mt[0] * s, mt[1] * s)], fill=255, width=int(18 * s)),
                              d.ellipse([(mt[0] - 26) * s, (mt[1] - 30) * s, (mt[0] + 26) * s, (mt[1] + 20) * s], fill=255)])
D = D + 1.0 * mic
actor = light_table(np.clip(D,0,None),back,tint=tintC)
alpha = np.clip(fig+mic,0,1)
export_sprite("09b",np.dstack([actor,alpha]),S,gp(0,0))
D = D_hold
tintC = tint_hold
fig = np.zeros((H,W),np.float32)
D = blur(D, 0.8)
img = light_table(np.clip(D, 0, None), back, tint=tintC)
img = img + blur(tintC[..., 3] * (1 - fig), 10)[..., None] * tintC[..., :3] * 0.35   # 彩灯的光晕
img = vignette(img, 0.3, 2.2)
export_bg("09b", img)

