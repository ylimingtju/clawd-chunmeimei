"""03｜向日葵朝着夜 · 油画厚涂
03-a：夜晚的向日葵田，花盘朝着夜空和月亮，几朵花苞未开；Clawd 独自坐在花田中抬头看月亮，身旁一个花苞转向它。
03-b：国乐无双舞台（蓝色背景 + 圆形月光光斑）；Clawd 穿驼色不对称上衣、古铜褶裙，戴紫色长手套，一只手握麦克风。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from titles import oil_title, place
from painterly import paint, impasto

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()


def layer():
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def done(im):
    return to_arr(im.resize((W, H), Image.LANCZOS))


def C(c, a=255):
    return tuple(int(v * 255) if v <= 1 else int(v) for v in c) + (a,)


def comp(dst, rgba):
    return over(dst, rgba[..., :3], rgba[..., 3])

# =====================================================================
# 03-a 底图
# =====================================================================
MX, MY, MR = 1390, 230, 92
t = y / H
sky = np.stack([0.04 + 0.12 * t, 0.07 + 0.2 * t, 0.22 + 0.32 * t], -1)
# 月光周围的大光晕与一圈圈的笔触光环（油画里常见的月晕）
d = np.sqrt((x - MX) ** 2 + (y - MY) ** 2)
sky = sky + (np.exp(-d / 360))[..., None] * np.array([0.18, 0.2, 0.22])
rings = (0.5 + 0.5 * np.cos(d / 26)) * np.exp(-d / 420) * (d > MR)
sky = sky + rings[..., None] * np.array([0.10, 0.12, 0.10])
img = sky.astype(np.float32)
moon = np.clip((MR - d) / 2, 0, 1)
img = over(img, np.array([0.99, 0.95, 0.78]), moon)
img = img + (np.exp(-((d - MR) / 18) ** 2))[..., None] * np.array([0.25, 0.22, 0.12])
# 星星
g = rng(31)
for _ in range(90):
    sx, sy = g.uniform(0, W), g.uniform(0, 560)
    if math.hypot(sx - MX, sy - MY) < 200:
        continue
    r = g.uniform(1.5, 4)
    dd = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    img = img + (np.clip(r - dd, 0, 1) * g.uniform(0.5, 1))[..., None] * np.array([1.0, 0.95, 0.8])
# 远山
im, dr = layer()
pts = [(0, 640)]
for i in range(0, W + 41, 40):
    pts.append((i, 610 - 40 * math.sin(i / 300 + 1) - 22 * math.sin(i / 97)))
pts += [(W, 700), (0, 700)]
dr.polygon([(a * SS, b * SS) for a, b in pts], fill=C((0.05, 0.08, 0.2)))
img = comp(img, done(im))
# 地面
ground = np.clip((y - 640) / 440, 0, 1)
gcol = np.stack([0.05 + 0.06 * ground, 0.11 + 0.1 * ground, 0.12 + 0.05 * ground], -1)
img = np.where((y > 640)[..., None], gcol, img)

# 向日葵（远小近大）
FACE = math.atan2(MY - 0, MX - 0)  # 花盘大致朝向右上方的月亮


def sunflower(dr, fx, fy, r, open_=1.0, tilt=0.0, stem_to=None, seed=0):
    gg = rng(seed)
    # 茎和叶
    if stem_to:
        dr.line([(fx * SS, fy * SS), (stem_to[0] * SS, stem_to[1] * SS)], fill=C((0.12, 0.26, 0.14)), width=max(2, int(r * 0.16)) * SS)
        for k in range(2):
            ly = fy + (stem_to[1] - fy) * (0.35 + 0.3 * k)
            lx = fx + (stem_to[0] - fx) * (0.35 + 0.3 * k)
            sgn = 1 if k == 0 else -1
            leaf = [(lx, ly), (lx + sgn * r * 0.9, ly - r * 0.35), (lx + sgn * r * 1.4, ly + r * 0.05), (lx + sgn * r * 0.7, ly + r * 0.25)]
            dr.polygon([(a * SS, b * SS) for a, b in leaf], fill=C((0.1, 0.24 + 0.04 * k, 0.15)))
    # 花盘：椭圆，朝右上倾斜（看到的是正面偏侧）
    sq = 0.55  # 压扁程度
    ca, sa = math.cos(tilt), math.sin(tilt)
    def P(u, v):
        return ((fx + u * ca - v * sa * sq) * SS, (fy + u * sa + v * ca * sq) * SS)
    if open_ > 0.3:
        n = 22
        for i in range(n):
            a = 2 * math.pi * i / n + gg.uniform(-0.05, 0.05)
            L = r * (1.85 + 0.15 * gg.random()) * open_
            w = r * 0.24
            tip = (math.cos(a) * L, math.sin(a) * L)
            l = (math.cos(a + 0.22) * r * 0.9, math.sin(a + 0.22) * r * 0.9)
            rr_ = (math.cos(a - 0.22) * r * 0.9, math.sin(a - 0.22) * r * 0.9)
            # 朝月亮一侧的花瓣更亮
            lit = 0.5 + 0.5 * math.cos(a - (-0.6))
            col = (0.92 + 0.08 * lit, 0.70 + 0.2 * lit, 0.16 + 0.14 * lit)
            dr.polygon([P(*l), P(*tip), P(*rr_)], fill=C(col))
        cr = r * 0.85
        for k in range(4):
            rr = cr * (1 - k * 0.2)
            col = (0.22 - k * 0.03, 0.12 - k * 0.02, 0.05)
            pts = [P(math.cos(a) * rr, math.sin(a) * rr) for a in np.linspace(0, 2 * math.pi, 30)]
            dr.polygon(pts, fill=C(col))
        # 花盘上的月光高光
        pts = [P(math.cos(a) * r * 0.45 + r * 0.3, math.sin(a) * r * 0.3 - r * 0.35) for a in np.linspace(0, 2 * math.pi, 20)]
        dr.polygon(pts, fill=C((0.42, 0.3, 0.14)))
    else:
        # 花苞：绿色萼片包着一点黄
        pts = [P(math.cos(a) * r * 0.75, math.sin(a) * r * 0.95) for a in np.linspace(0, 2 * math.pi, 24)]
        dr.polygon(pts, fill=C((0.16, 0.34, 0.16)))
        for k in range(5):
            a = -math.pi / 2 + (k - 2) * 0.35
            dr.polygon([P(math.cos(a) * r * 0.3, math.sin(a) * r * 0.3), P(math.cos(a) * r * 1.15, math.sin(a) * r * 1.25),
                        P(math.cos(a + 0.25) * r * 0.5, math.sin(a + 0.25) * r * 0.5)], fill=C((0.2, 0.42, 0.18)))
        pts = [P(math.cos(a) * r * 0.25, -r * 0.72 + math.sin(a) * r * 0.18) for a in np.linspace(0, 2 * math.pi, 16)]
        dr.polygon(pts, fill=C((0.9, 0.72, 0.2)))

# 按深度排序：远处（上方）先画
flowers = []
g = rng(41)
rows = 15
for rI in range(rows):
    depth = rI / (rows - 1)
    fy = 655 + depth ** 1.6 * 420
    r = 8 + depth ** 1.5 * 46
    n = int(30 - depth * 19)
    for k in range(n):
        fx = g.uniform(-60, W + 60)
        if 760 < fx < 1060 and 660 < fy < 770:
            continue  # 给 Clawd 留出空地
        bud = g.random() < 0.13
        tilt = -0.5 + g.normal(0, 0.15)
        flowers.append((fy, fx, r, bud, tilt))
flowers.sort()

# 远中景花
im, dr = layer()
for i, (fy, fx, r, bud, tilt) in enumerate(flowers):
    if fy > 800:
        continue
    sunflower(dr, fx, fy, r, 0.2 if bud else 1.0, tilt, (fx - r * 0.2, fy + r * 3.2), seed=i)
img = comp(img, done(im))

# ---------- Clawd：坐在花田中，抬头 ----------
S = 13
eyes_up = [(3.8, 1.6, 4.8, 3.6), (11.6, 1.6, 12.6, 3.6)]
cm, (ox, oy), (pw, ph) = clawd_masks(S, 20, extra={"eyes_up": eyes_up})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
orange = np.array(CLAWD_ORANGE) / 255
# 夜色里：整体偏冷偏暗，右上方被月光勾出亮边
col = orange[None, None] * 1.0 * np.array([1.0, 0.96, 0.98])
rim = np.clip(body - shift(body, 4, -4), 0, 1)
rim = blur(rim, 1.2)
col = col + rim[..., None] * np.array([0.75, 0.68, 0.45])
col = col * (1 + 0.25 * np.clip((gx - 4) / 12, 0, 1) * np.clip((6 - gy) / 6, 0, 1))[..., None]
col = over(col, np.array([0.03, 0.03, 0.05]), cm["eyes_up"])
spr = np.dstack([col, body])
CX, CY = 905, 735
img = img * (1 - 0.4 * blur(shift(sprite_alpha_layer(spr, CX, CY), 10, -12), 8))[..., None]
img = paste_rgba(img, spr, CX, CY)

# 转向 Clawd 的花苞（在 Clawd 右侧，花苞朝左）
im, dr = layer()
sunflower(dr, 1050, 700, 22, 0.25, 2.6, (1044, 812), seed=999)
img = comp(img, done(im))

# 近景花
im, dr = layer()
for i, (fy, fx, r, bud, tilt) in enumerate(flowers):
    if fy <= 800:
        continue
    sunflower(dr, fx, fy, r, 0.2 if bud else 1.0, tilt, (fx - r * 0.2, fy + r * 3.2), seed=i + 500)
img = comp(img, done(im))
# 月光：整体右上方提亮
img = img * (0.9 + 0.25 * np.exp(-(((x - MX) / 900) ** 2 + ((y - MY) / 700) ** 2)))[..., None]
base_a = np.clip(img, 0, 1)

# ---------- 转成油画 ----------
sky_angle = np.where(y < 640, np.arctan2(y - MY, x - MX) + np.pi / 2, 0.0)  # 天空：绕月亮打圈的笔触
scale_a = np.where(y < 630, 1.0, 0.42 + 0.35 * np.clip((y - 660) / 420, 0, 1))
painted, hmap = paint(base_a, [
    (16000, (30, 60), (10, 16), False),
    (60000, (14, 30), (5, 9), False),
    (60000, (6, 14), (2, 4), True),
], seed=3, default_angle=sky_angle, scale_map=scale_a)
# Clawd 与月亮保留更多细节：轻微混回底图
keep = np.clip(blur(sprite_alpha_layer(spr, CX, CY), 3) * 0.7 + blur(moon, 3) * 0.4, 0, 1)
painted = painted * (1 - keep[..., None]) + base_a * keep[..., None]
img = impasto(painted, hmap, 0.85, light=(1, -1))

title = oil_title("向日葵朝着夜", size=168, seed=5)
place(img, title, 90, 70)
img = vignette(img, 0.35, 2.2)
img = grain(img, 0.008, 3)
save(img, os.path.join(OUT, "03-a_向日葵朝着夜_月下花田.png"))
print("03-a ok")

# =====================================================================
# 03-b 国乐无双舞台
# =====================================================================
t = y / H
img = np.stack([0.22 + 0.1 * t, 0.36 + 0.08 * t, 0.70 - 0.12 * t], -1).astype(np.float32)
# 舞台上的圆形光斑（像月亮）
LX, LY = 520, 330
d = np.sqrt((x - LX) ** 2 + (y - LY) ** 2)
img = img + (np.exp(-d / 160))[..., None] * np.array([0.35, 0.33, 0.25])
img = over(img, np.array([0.98, 0.95, 0.82]), np.clip((70 - d) / 3, 0, 1))
# 舞台地面
floor = (y > 860).astype(np.float32)
img = over(img, np.array([0.08, 0.12, 0.26]) * 1.0, blur(floor, 2))
img = img + (np.exp(-(((x - 960) / 380) ** 2 + ((y - 900) / 50) ** 2)))[..., None] * np.array([0.25, 0.28, 0.38])
# 背景的竖向舞台幕布褶
img = img * (0.94 + 0.06 * np.sin(x / 38))[..., None] * (y < 860)[..., None] + img * (y >= 860)[..., None]

S = 30
top = [(-0.2, 5.0), (16.2, 5.0), (16.4, 9.2), (-0.4, 9.2)]
sash = [(16.4, 2.4), (17.6, 3.2), (6.0, 9.4), (2.2, 9.4)]
waist = [(-0.5, 8.6), (16.5, 8.6), (16.6, 9.6), (-0.6, 9.6)]
skirt = [(-0.6, 9.4), (16.6, 9.4), (17.6, 11.0), (18.4, 11.7), (15.6, 12.0), (12.6, 11.7), (9.6, 12.0), (6.4, 11.7), (3.4, 12.0), (0.4, 11.7), (-2.4, 11.7), (-1.6, 11.0)]
glove_l = [(-1.0, 3.0), (0.0, 3.0), (0.0, 5.0), (-1.0, 5.0)]          # 手套只覆盖原手臂块
glove_r = rigid_arm('R', 40)                                           # 右臂整块抬起握麦克风
mic = [(17.0, 2.6), (17.8, 2.2), (16.2, -0.6), (15.4, -0.2)]
check_proportions(arms=[glove_r], extra={"skirt": [skirt], "top": [top], "sash": [sash]})
cm, (ox, oy), (pw, ph) = clawd_masks(S, 40, bbox=(-4, -2, 21, 14), arms=[glove_l, glove_r],
                                     extra={"top": [top], "sash": [sash], "waist": [waist], "skirt": [skirt],
                                            "glove": [glove_l, glove_r], "mic": [mic]})
gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
body = np.clip(cm["body"], 0, 1)
col = orange[None, None] * (0.78 + 0.3 * np.clip(1 - gy / 10, 0, 1) + 0.12 * np.clip(1 - gx / 16, 0, 1))[..., None]
col = over(col, np.array([0.04, 0.03, 0.05]), cm["eyes"])
spr = np.dstack([col, body])

def lay(name, colr, shade):
    global spr
    m = cm[name]
    spr[..., :3] = over(spr[..., :3], colr * shade[..., None], m)
    spr[..., 3] = np.maximum(spr[..., 3], m)

camel = np.array([0.78, 0.66, 0.48])
lay("top", camel, 0.82 + 0.18 * np.sin(gx * 1.3 + gy * 0.6) + 0.12 * np.clip(1 - gx / 16, 0, 1))
lay("sash", camel * 1.08, 0.88 + 0.14 * np.sin((gx + gy) * 2.2))
lay("waist", np.array([0.32, 0.22, 0.62]), np.ones_like(gx))
pleat = 0.5 + 0.5 * np.sin((gx - 8) * 3.6 / (1 + np.clip(gy - 9.4, 0, 8) * 0.06))
bronze = np.array([0.55, 0.38, 0.22])
lay("skirt", bronze, 0.5 + 0.85 * pleat ** 3 + 0.2 * np.exp(-((gx - 5) / 3) ** 2) - 0.12 * np.clip((gy - 10) / 3, 0, 1))
glove = np.array([0.40, 0.24, 0.78])
lay("glove", glove, 0.75 + 0.45 * np.clip(np.sin(gx * 2.5 + gy * 1.5), 0, 1))
lay("mic", np.array([0.75, 0.76, 0.8]), 0.9 + 0.2 * np.sin(gx * 4))
# 眼睛重新压上
spr[..., :3] = over(spr[..., :3], np.array([0.04, 0.03, 0.05]), cm["eyes"] * (1 - cm["mic"]))
CX, CY = 1010, 720
img = img * (1 - 0.35 * blur(shift(sprite_alpha_layer(spr, CX, CY), 20, 30), 20))[..., None]
img = paste_rgba(img, spr, CX, CY)
# 顶光
img = img * (0.85 + 0.3 * np.exp(-(((x - 1010) / 520) ** 2 + ((y - 380) / 420) ** 2)))[..., None]
af = sprite_alpha_layer(spr, CX, CY)
rimb = np.clip(af - shift(af, 5, 5), 0, 1)
img = img + blur(rimb, 1.5)[..., None] * np.array([0.5, 0.55, 0.7])
haze = fbm(H, W, 77, octaves=4, base=5)
img = screen(img, (haze * np.clip((y - 400) / 500, 0, 1) * 0.18)[..., None] * np.array([0.7, 0.8, 1.0]))
mod1 = fbm(H, W, 91, octaves=3, base=4)
mod2 = fbm(H, W, 92, octaves=3, base=7)
img = img + ((mod1 - 0.5)[..., None] * np.array([-0.06, 0.04, 0.10]) + (mod2 - 0.5)[..., None] * np.array([0.10, 0.02, 0.08])) * (1 - af[..., None])
base_b = np.clip(img, 0, 1)
scale_b = np.where(af > 0.3, 0.75, 1.35)
painted, hmap = paint(base_b, [
    (14000, (30, 60), (10, 16), False),
    (40000, (12, 28), (5, 9), False),
    (40000, (5, 12), (2, 4), True),
], seed=7, default_angle=np.where(y < 860, np.pi / 2 + 0.5 * np.sin(x / 300), 0.0), jitter=0.16, scale_map=scale_b)
keep = np.clip(blur(sprite_alpha_layer(spr, CX, CY), 3) * 0.4, 0, 1)
painted = painted * (1 - keep[..., None]) + base_b * keep[..., None]
img = impasto(painted, hmap, 0.85, light=(-1, -1))
img = vignette(img, 0.35, 2.2)
img = grain(img, 0.008, 4)
save(img, os.path.join(OUT, "03-b_向日葵朝着夜_紫色长手套.png"))
print("03-b ok")
