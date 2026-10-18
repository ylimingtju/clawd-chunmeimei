"""07｜我表示理解 · 毛绒 + DV 夜拍（延续 C 图）
07-a：地下通道：一排白色荧光灯管伸向深处；Clawd 穿过大的黑色皮夹克，背对镜头朝通道深处走去。
07-b：武汉现场 · DV 饭拍：暖白舞台、烟雾；Clawd 穿白色抹胸垂坠长裙、白色长手套，一手握麦克风，一只钳子向前伸；
      前景是虚焦的观众剪影，画面带 REC 红点与时间码。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import *

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()
ORANGE = np.array(CLAWD_ORANGE) / 255


def mask_from(fn):
    im = Image.new("L", (W * SS, H * SS), 0)
    fn(ImageDraw.Draw(im), SS)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def plush_clawd(S, cx, foot, parts, arms=None, eyes=True, seed=0, angle=0.0):
    """毛绒 Clawd：边缘细毛 + 绒面噪声；parts=[(名称, 形状, 颜色, 材质)]，材质 'leather'/'satin'/'flat'。"""
    extra = {n: sh for n, sh, _, _ in parts}
    check_proportions(arms=arms, extra=extra)
    cm, (ox, oy), (pw, ph) = clawd_masks(S, int(S * 4), arms=arms, bbox=(-5, -5, 21, 12), extra=extra)
    gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
    gx, gy = (gx_ - ox) / S, (gy_ - oy) / S
    g = rng(seed)
    fur = blur(np.clip(g.normal(0, 1, (ph, pw)).astype(np.float32) * 0.2 + 0.5, 0, 1), 0.7)
    fur2 = blur(np.clip(g.normal(0, 1, (ph, pw)).astype(np.float32) * 0.25 + 0.5, 0, 1), 1.6)
    body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
    fuzzy = np.clip(blur(body, 3.0) + (fur - 0.5) * 2.2 * (1 - np.abs(blur(body, 3.0) * 2 - 1)), 0, 1)
    inner = edge_dist(body, S * 0.9)
    col = ORANGE[None, None] * (0.6 + 0.5 * inner)[..., None] * (0.72 + 0.32 * fur + 0.3 * fur2)[..., None]
    alpha = fuzzy
    for n, _, c, mat in parts:
        m = cm[n]
        if mat == "leather":
            ln = fbm(ph, pw, seed + 7, octaves=4, base=6)
            folds = np.sin(gx * 2.2 + ln * 1.5 + gy * 0.15) * 0.5 + 0.5
            cc = np.array(c)[None, None] * (0.8 + 0.4 * ln)[..., None]
            cc = cc + blur(np.clip((folds - 0.86) / 0.14, 0, 1) ** 2, 4)[..., None] * np.array([0.35, 0.4, 0.6]) * 0.6
        elif mat == "satin":
            folds = np.sin((gx - 8) * 0.7 + np.sin(gy * 0.5) * 0.6) * 0.5 + 0.5
            cc = np.array(c)[None, None] * (0.82 + 0.14 * folds + 0.08 * fur2 - 0.08 * np.clip((gy - 8) / 4, 0, 1))[..., None]
        else:
            cc = np.array(c)[None, None] * (0.85 + 0.2 * fur2)[..., None]
        col = over(col, cc, m)
        alpha = np.maximum(alpha, m)
    if eyes:
        col = over(col, np.array([0.03, 0.03, 0.04]), cm["eyes"])
    spr = np.dstack([col, alpha])
    cy = foot - (oy + 12 * S - ph / 2)
    return spr, (cx, cy), (cx - pw / 2 + ox, cy - ph / 2 + oy)


def dv_finish(img, seed, warm=False):
    if not warm:
        img = img * np.array([0.93, 0.95, 1.06])
    lum = img.mean(-1, keepdims=True)
    img = lum + (img - lum) * 0.9
    mb = img.copy()
    for k in (-5, -2, 2, 5):
        mb += np.roll(img, k, axis=1)
    img = img * 0.6 + (mb / 5) * 0.4
    img = bloom(img, 0.6, 18, 0.5)
    img[..., 0] = np.roll(img[..., 0], 3, axis=1)
    img[..., 2] = np.roll(img[..., 2], -3, axis=1)
    img = img * (0.94 + 0.06 * (np.sin(y * np.pi / 2) ** 2))[..., None]
    img = grain(img, 0.06, seed, size=1.6)
    img = grain(img, 0.025, seed + 1)
    img = img * 0.92 + 0.045
    return vignette(img, 0.5, 2.2)

# =====================================================================
# 07-a 地下通道（一点透视）
# =====================================================================
VX, VY = 1010, 455
NEAR = (-60, -60, W + 60, H + 60)
FAR = (VX - 110, VY - 70, VX + 110, VY + 70)
def at(t):
    """t=0 近处截面，t=1 远处截面。返回该深度的矩形。"""
    return tuple(NEAR[i] + (FAR[i] - NEAR[i]) * t for i in range(4))
img = np.zeros((H, W, 3), np.float32)
L0, T0, R0, B0 = NEAR
L1, T1, R1, B1 = FAR
wall_col = np.array([0.30, 0.36, 0.38])
ceil_col = np.array([0.22, 0.25, 0.28])
floor_col = np.array([0.12, 0.13, 0.15])
left = mask_from(lambda d, s: d.polygon([(L0 * s, T0 * s), (L1 * s, T1 * s), (L1 * s, B1 * s), (L0 * s, B0 * s)], fill=255))
right = mask_from(lambda d, s: d.polygon([(R0 * s, T0 * s), (R1 * s, T1 * s), (R1 * s, B1 * s), (R0 * s, B0 * s)], fill=255))
ceil = mask_from(lambda d, s: d.polygon([(L0 * s, T0 * s), (R0 * s, T0 * s), (R1 * s, T1 * s), (L1 * s, T1 * s)], fill=255))
floor = mask_from(lambda d, s: d.polygon([(L0 * s, B0 * s), (R0 * s, B0 * s), (R1 * s, B1 * s), (L1 * s, B1 * s)], fill=255))
img = over(img, wall_col, left)
img = over(img, wall_col * np.array([1.05, 0.92, 0.82]), right)  # 右墙偏暖（参照 MV 的橙色墙）
img = over(img, ceil_col, ceil)
img = over(img, floor_col, floor)
img = over(img, np.array([0.03, 0.03, 0.04]), mask_from(lambda d, s: d.rectangle([L1 * s, T1 * s, R1 * s, B1 * s], fill=255)))
# 深度雾：越远越暗
dist = np.sqrt(((x - VX) / (W / 2)) ** 2 + ((y - VY) / (H / 2)) ** 2)
img = img * np.clip(0.25 + 0.9 * dist, 0, 1.1)[..., None]
# 瓷砖缝：深度方向的横线 + 纵向线
lines = []
for k in range(1, 26):
    t = 1 - 1 / (1 + k * 0.28)
    l, tp, r, b = at(t)
    lines += [[(l, tp), (l, b)], [(r, tp), (r, b)], [(l, b), (r, b)]]
for v in np.linspace(0.1, 0.9, 7):
    lines.append([(L0, T0 + (B0 - T0) * v), (L1, T1 + (B1 - T1) * v)])
    lines.append([(R0, T0 + (B0 - T0) * v), (R1, T1 + (B1 - T1) * v)])
tiles = mask_from(lambda d, s: [d.line([(p[0] * s, p[1] * s) for p in pl], fill=255, width=int(2 * s)) for pl in lines])
img = img * (1 - 0.35 * tiles * (1 - floor))[..., None]
# 左墙扶手（铜色）
rail = mask_from(lambda d, s: d.line([(L0 * s, (T0 + (B0 - T0) * 0.62) * s), (L1 * s, (T1 + (B1 - T1) * 0.62) * s)], fill=255, width=int(12 * s)))
img = over(img, np.array([0.55, 0.38, 0.25]), rail)
# 荧光灯管：天花板中线上一串
glow = np.zeros((H, W), np.float32)
tubes = np.zeros((H, W), np.float32)
for k in range(14):
    t0 = 1 - 1 / (1 + k * 0.5 + 0.15)
    t1 = 1 - 1 / (1 + k * 0.5 + 0.38)
    l0, tp0, r0, _ = at(t0)
    l1, tp1, r1, _ = at(t1)
    cx0, cx1 = (l0 + r0) / 2, (l1 + r1) / 2
    w0, w1 = (r0 - l0) * 0.035, (r1 - l1) * 0.035
    yy0, yy1 = tp0 + (VY - tp0) * 0.04, tp1 + (VY - tp1) * 0.04
    m = mask_from(lambda d, s: d.polygon([((cx0 - w0) * s, yy0 * s), ((cx0 + w0) * s, yy0 * s), ((cx1 + w1) * s, yy1 * s), ((cx1 - w1) * s, yy1 * s)], fill=255))
    tubes = np.maximum(tubes, m)
glow = blur(tubes, 30) * 2.2 + blur(tubes, 8) * 0.8
img = img + glow[..., None] * np.array([0.75, 0.9, 1.0]) * 0.55
img = over(img, np.array([1.0, 1.0, 1.0]), tubes)
# 地面湿反光：灯管的倒影
refl = np.flipud(blur(tubes, 6))
shift_y = int(2 * VY - H)
refl = np.roll(refl, shift_y, axis=0) * floor
img = img + refl[..., None] * np.array([0.6, 0.75, 0.85]) * 0.5
img = img + blur(refl, 20)[..., None] * 0.15

# Clawd 背影：夹克从背后盖住身体上半到大半，露出橙色手臂与腿
S = 22
jacket_back = [(-1.6, -0.6), (17.6, -0.6), (18.4, 3.6), (18.0, 9.6), (-2.0, 9.6), (-2.4, 3.6)]
collar = [(2.0, -1.4), (14.0, -1.4), (15.0, -0.4), (1.0, -0.4)]
spr, (cx, cy), _ = plush_clawd(S, 1040, 930, [("jacket", [jacket_back], (0.04, 0.04, 0.05), "leather"),
                                              ("collar", [collar], (0.03, 0.03, 0.035), "leather")], eyes=False, seed=71)
af = sprite_alpha_layer(spr, cx, cy)
img = img * (1 - 0.5 * blur(shift(af, 14, 0), 14) * floor)[..., None]
img = paste_rgba(img, spr, cx, cy)
# 顶部灯光在夹克肩上的冷白高光
img = img + np.clip(af - shift(af, 4, 0), 0, 1)[..., None] * np.array([0.3, 0.36, 0.42])
img = dv_finish(img, 77)
save(img, os.path.join(OUT, "07-a_我表示理解_地下通道.png"))
print("07-a ok")

# =====================================================================
# 07-b 武汉现场 · DV 饭拍
# =====================================================================
img = np.zeros((H, W, 3), np.float32)
img[:] = np.array([0.16, 0.12, 0.08])
haze = fbm(H, W, 701, octaves=5, base=4)
img = img + (haze - 0.4)[..., None] * np.array([0.22, 0.17, 0.10])
# 顶光柱
beam = np.exp(-((x - 980 - (y - 0) * 0.08) / 260) ** 2) * np.clip(1 - y / 1000, 0, 1)
img = img + beam[..., None] * np.array([0.55, 0.48, 0.36]) * (0.6 + 0.6 * haze[..., None])
img = img * (1 - 0.3 * np.clip((y - 860) / 220, 0, 1))[..., None]

S = 32
gown = [(-0.4, 5.6), (16.4, 5.6), (16.6, 8.0), (17.6, 10.2), (19.0, 11.8), (8.0, 11.9), (-2.2, 11.8), (-1.0, 10.0), (-0.6, 8.0)]
drape = [(9.0, 5.6), (16.4, 5.6), (17.6, 10.2), (19.0, 11.8), (15.6, 11.8), (11.0, 7.8)]
arm_mic = rigid_arm("R", 55)
arm_reach = rigid_arm("L", -15)
tip = ((arm_mic[1][0] + arm_mic[2][0]) / 2, (arm_mic[1][1] + arm_mic[2][1]) / 2)
spr, (cx, cy), (gx0, gy0) = plush_clawd(S, 960, 900, [("gown", [gown], (0.94, 0.92, 0.88), "satin"),
                                                      ("drape", [drape], (0.98, 0.96, 0.92), "satin"),
                                                      ("glove", [arm_mic, arm_reach], (0.96, 0.95, 0.93), "flat")],
                                        arms=[arm_reach, arm_mic], seed=72)
af = sprite_alpha_layer(spr, cx, cy)
img = img * (1 - 0.25 * blur(af, 40))[..., None]
img = paste_rgba(img, spr, cx, cy)
# 麦克风（握在右手末端，朝向嘴边）
P = lambda a, b: (gx0 + a * S, gy0 + b * S)
mk = mask_from(lambda d, s: [d.line([(P(*tip)[0] * s, P(*tip)[1] * s), (P(tip[0] - 1.2, tip[1] - 1.6)[0] * s, P(tip[0] - 1.2, tip[1] - 1.6)[1] * s)], fill=255, width=int(0.8 * S * s)),
                             d.ellipse([(P(tip[0] - 1.2, tip[1] - 1.6)[0] - S * 0.9) * s, (P(tip[0] - 1.2, tip[1] - 1.6)[1] - S * 0.9) * s,
                                        (P(tip[0] - 1.2, tip[1] - 1.6)[0] + S * 0.9) * s, (P(tip[0] - 1.2, tip[1] - 1.6)[1] + S * 0.9) * s], fill=255)])
img = over(img, np.array([0.62, 0.62, 0.64]), mk)
# 逆光勾边
img = img + blur(np.clip(af - shift(af, 5, 0), 0, 1), 1.5)[..., None] * np.array([0.6, 0.52, 0.4])
# 前景：虚焦观众头部剪影 + 举起的手机屏
heads = mask_from(lambda d, s: [d.ellipse([(hx - hr) * s, (H - hr * 0.9) * s, (hx + hr) * s, (H + hr * 1.4) * s], fill=255)
                                for hx, hr in [(120, 170), (430, 150), (1560, 180), (1840, 150), (760, 120)]])
phone = mask_from(lambda d, s: d.rounded_rectangle([1260 * s, 780 * s, 1420 * s, 1060 * s], radius=18 * s, fill=255))
screen_m = mask_from(lambda d, s: d.rounded_rectangle([1272 * s, 792 * s, 1408 * s, 1048 * s], radius=10 * s, fill=255))
stage_crop = img[300:960, 560:1360].copy()
img = over(img, np.array([0.02, 0.02, 0.02]), blur(heads, 10))
img = over(img, np.array([0.03, 0.03, 0.03]), blur(phone, 6))
thumb = to_arr(to_img(np.clip(stage_crop, 0, 1)).resize((256, 212), Image.LANCZOS))
thumb = thumb[:, 60:196]   # 竖屏取景
pad_img = np.zeros((H, W, 3), np.float32)
pad_img[814:1026, 1272:1408] = thumb * 1.15
screen_m2 = screen_m * 0
screen_m2[814:1026, 1272:1408] = 1
screen_m2 = screen_m2 * screen_m
img = over(img, np.array([0.1, 0.1, 0.1]), screen_m)
img = over(img, pad_img, screen_m2)
img = img + blur(screen_m, 14)[..., None] * 0.05
img = dv_finish(img, 78, warm=True)
# DV 叠字：REC + 时间码 + 电池
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(ov)
f = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 44)
d.ellipse([80, 70, 116, 106], fill=(230, 40, 40, 255))
d.text((132, 62), "REC", font=f, fill=(245, 245, 245, 235))
d.text((80, H - 110), "APR 10 2026   21:37:52", font=f, fill=(245, 245, 245, 230))
d.rectangle([W - 190, 72, W - 96, 108], outline=(245, 245, 245, 230), width=4)
d.rectangle([W - 96, 82, W - 86, 98], fill=(245, 245, 245, 230))
d.rectangle([W - 182, 80, W - 130, 100], fill=(245, 245, 245, 230))
d.text((W - 330, H - 110), "SP  00:42", font=f, fill=(245, 245, 245, 230))
ova = to_arr(ov)
img = over(img, ova[..., :3], ova[..., 3])
save(img, os.path.join(OUT, "07-b_我表示理解_武汉DV饭拍.png"))
print("07-b ok")
