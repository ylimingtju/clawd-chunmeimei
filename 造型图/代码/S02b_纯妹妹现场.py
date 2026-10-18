"""02-b｜纯妹妹 · 成都现场（羊毛毡，延续 B 图）
毡布舞台、背后挂着彩灯串；Clawd 穿银金色流苏亮片短裙、金色颈饰，一只手臂抬起握白色麦克风；
身边 4 只白色迷你 Clawd 穿白色毛毡花朵小裙，举手伴舞。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from felt import felt, layer, done, comp, C, SS, FINE

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
x, y = grid()
ORANGE = np.array(CLAWD_ORANGE) / 255

# ---------- 背景：深紫毡布幕 + 竖向褶 ----------
img = np.zeros((H, W, 3), np.float32)
img[:] = np.array([0.24, 0.17, 0.32])
img = img * (0.88 + 0.12 * np.sin(x / 46 + np.sin(y / 300))[..., None]) * (0.9 + 0.12 * FINE[..., None])
img = img * (0.75 + 0.35 * np.exp(-(((x - 960) / 700) ** 2)))[..., None]
# 彩灯串：三条下垂的线，挂着毛毡小灯泡
g = rng(201)
bulbs_cols = [(1.0, 0.55, 0.6), (1.0, 0.85, 0.4), (0.6, 0.85, 1.0), (0.8, 0.65, 1.0), (0.6, 0.95, 0.7)]
glow = np.zeros((H, W, 3), np.float32)
im, d = layer()
for k, (y0, sag) in enumerate([(70, 110), (190, 90), (300, 70)]):
    pts = [(xx, y0 + sag * (1 - ((xx - 960) / 1000) ** 2) + 18 * math.sin(xx / 180 + k)) for xx in range(-40, W + 41, 20)]
    d.line([(px * SS, py * SS) for px, py in pts], fill=C((0.18, 0.15, 0.16)), width=4 * SS)
    for i, (px, py) in enumerate(pts[::4]):
        c = bulbs_cols[(i + k * 2) % len(bulbs_cols)]
        d.ellipse([(px - 13) * SS, (py + 4) * SS, (px + 13) * SS, (py + 32) * SS], fill=C(c))
        if 20 < px < W - 20:
            glow[int(min(py + 18, H - 1)), int(px)] = c
img = comp(img, felt(done(im), 3, puff=4, fuzz=0.3, light=0.5, sh=3))
img = img + blur(glow, 26) * 900 * 0.05
# 毡布舞台地面
im, d = layer()
d.rectangle([0, 860 * SS, W * SS, H * SS], fill=C((0.52, 0.36, 0.42)))
img = comp(img, felt(done(im), 4, puff=18, fuzz=0.2, light=0.4))
img = img * (1 - 0.25 * np.exp(-((y - 862) / 10) ** 2))[..., None]
# 粉色追光圆
pool = np.exp(-(((x - 960) / 520) ** 2 + ((y - 930) / 70) ** 2))
img = screen(img, pool[..., None] * np.array([0.45, 0.32, 0.36]))


def felt_clawd(S, cx, foot, body_col, parts, arms=None, seed=0, eyes=True, mic=None):
    """毛毡 Clawd：parts=[(名称, 形状, 颜色)]；mic=(格坐标起点, 终点)。"""
    extra = {n: sh for n, sh, _ in parts}
    check_proportions(arms=arms, extra=extra)
    cm, (ox, oy), (pw, ph) = clawd_masks(S, int(S * 3), arms=arms, bbox=(-5, -5, 21, 12), extra=extra)
    spr = np.zeros((ph, pw, 4), np.float32)
    body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
    spr[..., :3] = body_col
    spr[..., 3] = body
    for n, _, c in parts:
        spr[..., :3] = over(spr[..., :3], np.array(c), cm[n])
        spr[..., 3] = np.maximum(spr[..., 3], cm[n])
    if mic:
        im = Image.new("RGBA", (pw * 2, ph * 2), (0, 0, 0, 0))
        dd = ImageDraw.Draw(im)
        (a0, b0), (a1, b1) = mic
        P = lambda a, b: ((ox + a * S) * 2, (oy + b * S) * 2)
        dd.line([P(a0, b0), P(a1, b1)], fill=(245, 245, 245, 255), width=int(0.9 * S * 2))
        hx, hy = P(a1, b1)
        dd.ellipse([hx - S * 1.5, hy - S * 1.5, hx + S * 1.5, hy + S * 1.5], fill=(235, 235, 238, 255))
        m = to_arr(im.resize((pw, ph), Image.LANCZOS))
        spr[..., :3] = over(spr[..., :3], m[..., :3], m[..., 3])
        spr[..., 3] = np.maximum(spr[..., 3], m[..., 3])
    if eyes:
        spr[..., :3] = over(spr[..., :3], np.array([0.08, 0.06, 0.07]), cm["eyes"])
    big = np.zeros((H, W, 4), np.float32)
    cy = foot - (oy + 12 * S - ph / 2)
    x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + pw, W), min(y0 + ph, H)
    big[Y0:Y1, X0:X1] = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    return big, (x0 + ox, y0 + oy)


def rosette_dress(n_rows=3):
    """白色毛毡花朵小裙：几排圆形“花朵”，下摆不低于 12 格。"""
    shapes = []
    for r in range(n_rows):
        yc = 6.6 + r * 1.75
        for i in range(6 + (r % 2)):
            xc = 0.6 + i * (14.8 / (5 + (r % 2)))
            rr = 1.15
            shapes.append([(xc + rr * math.cos(a), min(yc + rr * math.sin(a), 11.9)) for a in np.linspace(0, 2 * math.pi, 16)])
    return shapes

# ---------- 4 只白色迷你伴舞（先画，远一点、小一点）----------
WHITE = np.array([0.96, 0.95, 0.94])
for i, (cx, foot, ang_l, ang_r) in enumerate([(300, 905, 70, 20), (590, 890, 20, 70), (1330, 890, 70, 20), (1620, 905, 20, 70)]):
    big, _ = felt_clawd(11, cx, foot, WHITE * 0.97, [("dress", rosette_dress(), (1.0, 0.99, 0.98))],
                        arms=[rigid_arm("L", ang_l), rigid_arm("R", ang_r)], seed=i)
    sh = blur(shift(big[..., 3], 10, 12), 8)
    img = img * (1 - 0.3 * sh)[..., None]
    img = comp(img, felt(big, 10 + i, puff=8, fuzz=0.3, light=0.7, sh=5))

# ---------- 主角：银金流苏亮片短裙 ----------
S = 24
dress = [(-0.3, 5.6), (16.3, 5.6), (16.6, 9.0), (17.4, 11.4), (16.0, 11.9), (12.0, 11.6), (8.0, 11.9), (4.0, 11.6), (0.0, 11.9), (-1.4, 11.4), (-0.6, 9.0)]
collar = [(3.0, 5.0), (13.0, 5.0), (12.4, 6.2), (8.0, 6.8), (3.6, 6.2)]
gold_band = [(-0.4, 5.4), (16.4, 5.4), (16.4, 6.1), (-0.4, 6.1)]
ARM_R = rigid_arm("R", 60)
ARM_TIP = ((ARM_R[1][0] + ARM_R[2][0]) / 2, (ARM_R[1][1] + ARM_R[2][1]) / 2)
big, (gx0, gy0) = felt_clawd(S, 960, 905, ORANGE * 1.05,
                             [("dress", [dress], (0.80, 0.78, 0.76)), ("band", [gold_band], (0.90, 0.72, 0.30)), ("collar", [collar], (0.95, 0.78, 0.32))],
                             arms=[(-1, 3, 0, 5), ARM_R], mic=(ARM_TIP, (ARM_TIP[0] + 0.9, ARM_TIP[1] - 2.2)), seed=20)
dm = np.zeros((H, W), np.float32)
sh = blur(shift(big[..., 3], 16, 18), 12)
img = img * (1 - 0.35 * sh)[..., None]
clawd = felt(big, 21, puff=26, fuzz=0.22, light=1.0, sh=14)
img = comp(img, clawd)
# 裙面：金色与银色亮片拼块 + 流苏
im, d = layer()
gg = rng(205)
for k in range(260):
    gx_, gy_ = gg.uniform(-0.6, 16.8), gg.uniform(6.2, 11.4)
    px, py = gx0 + gx_ * S, gy0 + gy_ * S
    if gx_ < -0.2 + (gy_ - 9) * -0.5 or gx_ > 16.4 + (gy_ - 9) * 0.4:
        continue
    r = gg.uniform(2.5, 5.0)
    c = (0.95, 0.82, 0.42) if (gx_ + gy_ * 0.6) % 4 < 1.6 else (0.88, 0.88, 0.92)
    d.ellipse([(px - r) * SS, (py - r) * SS, (px + r) * SS, (py + r) * SS], fill=C(c))
for k in range(70):  # 下摆流苏
    gx_ = -1.0 + k * 0.26
    px, py = gx0 + gx_ * S, gy0 + 11.5 * S
    d.line([(px * SS, py * SS), ((px + gg.uniform(-4, 4)) * SS, (py + 0.45 * S) * SS)], fill=C((0.92, 0.86, 0.66)), width=3 * SS)
img = comp(img, felt(done(im), 22, puff=2, fuzz=0.15, light=0.6, sh=2))
# 亮片闪光
spark = np.zeros((H, W), np.float32)
for _ in range(45):
    px, py = int(gx0 + gg.uniform(0, 16) * S), int(gy0 + gg.uniform(6.2, 11.4) * S)
    if 0 <= py < H and 0 <= px < W:
        spark[py, px] = 1
img = img + np.clip(blur(spark, 2.2) * 30, 0, 1)[..., None] * 0.8
# 前景：几颗毛毡小星星彩带（虚焦）
im, d = layer()
for k in range(14):
    px, py = gg.uniform(0, W), gg.uniform(960, 1080)
    r = gg.uniform(10, 22)
    d.polygon([((px + r * (1 if i % 2 == 0 else 0.45) * math.cos(i * math.pi / 5 - math.pi / 2)) * SS,
                (py + r * (1 if i % 2 == 0 else 0.45) * math.sin(i * math.pi / 5 - math.pi / 2)) * SS) for i in range(10)],
              fill=C(bulbs_cols[k % 5]))
fg = felt(done(im), 23, puff=4, fuzz=0.3, light=0.5, sh=3)
fg = np.concatenate([blur(fg[..., :3], 3), blur(fg[..., 3], 3)[..., None]], -1)
img = comp(img, fg)
img = img * 0.97 + 0.03
img = vignette(img, 0.3, 2.4)
img = grain(img, 0.012, 7)
save(img, os.path.join(OUT, "02-b_纯妹妹_成都现场伴舞.png"))
print("02-b ok")
