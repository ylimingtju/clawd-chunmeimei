"""08｜还你茉莉 · 水墨工笔 + 民族摇滚
08-a：宣纸上的工笔茉莉；鼓点落下墨迹炸开成泼墨、狂草横扫；工笔大鼓、手鼓、铃环绕；Clawd 穿素纱，双钳把茉莉抛向上方。
08-b：深色舞台：一笔浓墨弧代替光带，鼓点溅墨，朱红印章盖在四角；Clawd 穿黑色不对称垂坠长衣、红皮手套、金耳饰，一手握麦一手握拳高举。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from titles import ink_title, place

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()
INK = np.array([0.07, 0.06, 0.06])
CINNABAR = np.array([0.80, 0.15, 0.10])
ORANGE = np.array(CLAWD_ORANGE) / 255


def xuan(seed=800):
    """宣纸：暖白、纤维、轻微斑驳。"""
    fib = fbm(H, W, seed, octaves=3, base=200)
    blot = fbm(H, W, seed + 1, octaves=4, base=5)
    col = np.ones((H, W, 3), np.float32) * np.array([0.95, 0.92, 0.85])
    col = col * (0.95 + 0.04 * fib[..., None] + 0.03 * (blot[..., None] - 0.5))
    return col


def mask_from(draw_fn):
    im = Image.new("L", (W * SS, H * SS), 0)
    draw_fn(ImageDraw.Draw(im), SS)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def ink(img, m, density=0.9, bleed=1.2, color=INK, seed=0):
    """墨色：边缘晕开、内部浓淡不匀。"""
    n = fbm(H, W, 820 + seed % 30, octaves=3, base=12)
    mb = np.clip(blur(m, bleed) * 1.3 + (n - 0.5) * 0.25 * (blur(m, bleed * 3) > 0.05), 0, 1)
    d = np.clip(mb * density * (0.8 + 0.3 * n), 0, 1)[..., None]
    return img * (1 - d) + color * d


def splash(img, cx, cy, r, seed, n=40, density=0.95, color=INK, direction=None):
    """泼墨：一团主墨 + 放射状飞溅墨点与墨丝。"""
    g = rng(seed)
    def f(d, s):
        pts = []
        for i in range(48):
            a = 2 * math.pi * i / 48
            rr = r * (0.7 + 0.35 * g.random())
            pts.append(((cx + rr * math.cos(a)) * s, (cy + rr * math.sin(a)) * s))
        d.polygon(pts, fill=255)
        for _ in range(n):
            a = g.uniform(0, 2 * math.pi) if direction is None else direction + g.normal(0, 0.6)
            dist = r * g.uniform(1.0, 3.2)
            rr = max(1.5, r * 0.16 * g.uniform(0.2, 1) * (1.6 - dist / (r * 3.2)))
            px, py = cx + dist * math.cos(a), cy + dist * math.sin(a)
            d.ellipse([(px - rr) * s, (py - rr) * s, (px + rr) * s, (py + rr) * s], fill=255)
            if g.random() < 0.4:
                d.line([((cx + r * 0.8 * math.cos(a)) * s, (cy + r * 0.8 * math.sin(a)) * s), (px * s, py * s)],
                       fill=255, width=max(1, int(rr * 0.5)) * s)
    return ink(img, mask_from(f), density, 1.0, color, seed)


def dry_gaps(t, along, seed, start=0.05, end=0.65):
    """飞白：横截方向随机宽度的干笔缝；越靠近收笔越多；纵向也有断续。"""
    g = rng(seed + 900)
    band = g.random(4000).astype(np.float32)
    wob = fbm(H, W, 860 + seed % 20, octaves=3, base=10)
    idx = ((t + wob * 9) / g.uniform(2.0, 3.2)).astype(int) % 4000
    thr = start + (end - start) * along ** 1.3
    gap = (band[idx] < thr) & (wob > 0.3)
    return np.where(gap, 0.12, 1.0)


def brush_sweep(img, pts, width, seed, density=0.95):
    """一笔横扫：起笔粗、收笔细，带飞白。"""
    g = rng(seed)
    def f(d, s):
        n = len(pts)
        for i in range(n - 1):
            u = i / (n - 1)
            w = width * (0.5 + 0.8 * math.sin(math.pi * min(u * 1.3, 1)) ) * (1 - 0.55 * u)
            d.line([(pts[i][0] * s, pts[i][1] * s), (pts[i + 1][0] * s, pts[i + 1][1] * s)], fill=255, width=max(2, int(w * s)))
    m = mask_from(f)
    # 飞白：沿笔势方向的干笔缝
    ang = math.atan2(pts[-1][1] - pts[0][1], pts[-1][0] - pts[0][0])
    t = -x * math.sin(ang) + y * math.cos(ang)
    along = np.clip((x * math.cos(ang) + y * math.sin(ang) - (pts[0][0] * math.cos(ang) + pts[0][1] * math.sin(ang))) / 1000, 0, 1)
    m = m * dry_gaps(t, along, seed)
    return ink(img, m, density, 0.8, INK, seed)


def gongbi_jasmine(img, cx, cy, r, rot, seed, petal=(0.99, 0.98, 0.95)):
    """工笔茉莉：白瓣淡墨勾线 + 淡绿花萼。"""
    pts_all = []
    for k in range(6):
        a = rot + k * math.pi / 3
        pts = []
        for i in range(20):
            t = i / 19 * math.pi
            rr = r * math.sin(t) * 0.42
            L = r * (i / 19)
            pts.append((cx + L * math.cos(a) - rr * math.sin(a), cy + L * math.sin(a) + rr * math.cos(a)))
        pts_all.append(pts)
    fill = mask_from(lambda d, s: [d.polygon([(p[0] * s, p[1] * s) for p in pts], fill=255) for pts in pts_all])
    img = over(img, np.array(petal), fill * 0.95)
    img = img * (1 - (blur(fill, r * 0.25) * fill * 0.12)[..., None] * np.array([0.3, 0.2, 0.1]))  # 花心淡黄晕
    line = mask_from(lambda d, s: [d.line([(p[0] * s, p[1] * s) for p in pts + [pts[0]]], fill=255, width=int(1.6 * s)) for pts in pts_all])
    img = ink(img, line, 0.65, 0.4, INK, seed)
    core = mask_from(lambda d, s: d.ellipse([(cx - r * 0.12) * s, (cy - r * 0.12) * s, (cx + r * 0.12) * s, (cy + r * 0.12) * s], fill=255))
    return over(img, np.array([0.92, 0.82, 0.45]), core)


def leaf(img, cx, cy, L, ang, seed):
    pts = []
    for i in range(21):
        u = i / 20
        w = math.sin(math.pi * u) * L * 0.28
        pts.append((u, w))
    poly = [(cx + u * L * math.cos(ang) - w * math.sin(ang), cy + u * L * math.sin(ang) + w * math.cos(ang)) for u, w in pts]
    poly += [(cx + u * L * math.cos(ang) + w * math.sin(ang), cy + u * L * math.sin(ang) - w * math.cos(ang)) for u, w in pts[::-1]]
    m = mask_from(lambda d, s: d.polygon([(p[0] * s, p[1] * s) for p in poly], fill=255))
    img = over(img, np.array([0.55, 0.66, 0.5]), m * 0.85)
    vein = mask_from(lambda d, s: d.line([(cx * s, cy * s), ((cx + L * math.cos(ang)) * s, (cy + L * math.sin(ang)) * s)], fill=255, width=int(1.5 * s)))
    edge = np.clip(m - blur(m, 1.5) * 1.0, 0, 1) * 2
    return ink(img, np.clip(vein + edge, 0, 1), 0.55, 0.4, INK, seed)


def drum(img, cx, cy, rw, rh, seed, angle=0.0):
    """工笔大鼓 / 手鼓：鼓面淡赭、鼓身朱红、铆钉、淡墨勾线；alpha 由 angle 倾斜。"""
    ca, sa = math.cos(angle), math.sin(angle)
    def P(u, v):
        return (cx + u * ca - v * sa, cy + u * sa + v * ca)
    side = [P(-rw, 0), P(rw, 0), P(rw, rh), P(-rw, rh)]
    sm = mask_from(lambda d, s: d.polygon([(p[0] * s, p[1] * s) for p in side], fill=255))
    bottom = [P(rw * math.cos(t), rh + rw * 0.28 * math.sin(t)) for t in np.linspace(0, math.pi, 30)]
    bm = mask_from(lambda d, s: d.polygon([(p[0] * s, p[1] * s) for p in side + bottom[::-1]], fill=255))
    img = over(img, CINNABAR * 0.95, np.clip(sm + bm, 0, 1) * 0.92)
    top = [P(rw * math.cos(t), rw * 0.28 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 50)]
    tm = mask_from(lambda d, s: d.polygon([(p[0] * s, p[1] * s) for p in top], fill=255))
    img = over(img, np.array([0.88, 0.78, 0.6]), tm)
    nails = mask_from(lambda d, s: [d.ellipse([(px - 3) * s, (py - 3) * s, (px + 3) * s, (py + 3) * s], fill=255)
                                     for px, py in [P(rw * math.cos(t), rw * 0.28 * math.sin(t) + 12) for t in np.linspace(0.15, math.pi - 0.15, 9)]])
    img = over(img, np.array([0.85, 0.7, 0.3]), nails)
    lines = mask_from(lambda d, s: [d.line([(p[0] * s, p[1] * s) for p in top + [top[0]]], fill=255, width=int(2 * s)),
                                    d.line([(side[0][0] * s, side[0][1] * s), (side[3][0] * s, side[3][1] * s)], fill=255, width=int(2 * s)),
                                    d.line([(side[1][0] * s, side[1][1] * s), (side[2][0] * s, side[2][1] * s)], fill=255, width=int(2 * s)),
                                    d.line([(p[0] * s, p[1] * s) for p in bottom], fill=255, width=int(2 * s))])
    return ink(img, lines, 0.8, 0.4, INK, seed)


def bell(img, cx, cy, r, seed):
    pts = [(cx - r * 0.6, cy - r), (cx + r * 0.6, cy - r), (cx + r, cy + r * 0.7), (cx - r, cy + r * 0.7)]
    m = mask_from(lambda d, s: d.polygon([(p[0] * s, p[1] * s) for p in pts], fill=255))
    img = over(img, np.array([0.78, 0.62, 0.3]), m)
    ln = mask_from(lambda d, s: [d.line([(p[0] * s, p[1] * s) for p in pts + [pts[0]]], fill=255, width=int(2 * s)),
                                 d.line([(cx * s, (cy - r * 1.6) * s), (cx * s, (cy - r) * s)], fill=255, width=int(2 * s)),
                                 d.ellipse([(cx - r * 0.18) * s, (cy + r * 0.7) * s, (cx + r * 0.18) * s, (cy + r * 1.05) * s], fill=255)])
    return ink(img, ln, 0.8, 0.4, INK, seed)


def ink_clawd(img, S, cx, foot, parts, seed, arms=None, angle=0.0, opacity=0.92):
    """水墨 Clawd：淡橙渲染 + 浓墨勾边；parts=[(名称, 形状, 颜色, 不透明度)]。"""
    extra = {n: sh for n, sh, _, _ in parts}
    check_proportions(arms=arms, extra=extra)
    cm, (ox, oy), (pw, ph) = clawd_masks(S, int(S * 3), arms=arms, bbox=(-5, -4, 21, 12), extra=extra)
    cy = foot - (oy + 12 * S - ph / 2)

    def put(m):
        a = to_arr(to_img(m).rotate(angle, resample=Image.BICUBIC, expand=False))
        out = np.zeros((H, W), np.float32)
        x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + pw, W), min(y0 + ph, H)
        out[Y0:Y1, X0:X1] = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        return out
    body = put(np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1))
    # 淡橙晕染：中间淡、边缘积色（水墨渲染的感觉）
    wash = np.clip(blur(body, 2) * (0.8 + 0.2 * fbm(H, W, 850 + seed, octaves=3, base=10)), 0, 1)
    if opacity >= 1:
        wash = np.clip(blur(body, 1.2) * 1.15, 0, 1)
    img = img * (1 - wash[..., None] * opacity) + wash[..., None] * opacity * ORANGE * (1.05 + 0.12 * (1 - blur(body, 12)))[..., None]
    union = body.copy()
    for n, _, c, op in parts:
        m = put(cm[n])
        img = over(img, np.array(c), m * op)
        union = np.maximum(union, m * (op > 0.5))
    img = ink(img, put(cm["eyes"]), 1.0, 0.5, INK, seed + 1)
    edge = np.clip(union - np.clip((blur(union, 1.6) - 0.5) * 5 + 0.5, 0, 1) * (union > 0.5), 0, 1)
    img = ink(img, np.clip(edge * 1.6, 0, 1), 0.95, 0.6, INK, seed + 2)
    return img, (ox, oy, pw, ph, cx, cy)

# =====================================================================
# 08-a
# =====================================================================
img = xuan()
# 背景：大块淡墨晕（远山般的墨韵）
mist = fbm(H, W, 860, octaves=4, base=3)
img = img * (1 - (np.clip(mist - 0.45, 0, 1) * 0.35)[..., None] * (1 - np.array([0.3, 0.3, 0.32])))
# 鼓点：中心偏上一团泼墨炸开 + 狂草横扫
img = splash(img, 1300, 430, 115, seed=1, n=55, density=0.92)
img = splash(img, 1620, 300, 55, seed=2, n=26, density=0.85)
img = brush_sweep(img, [(200 + i * 34, 640 - 120 * math.sin(i / 44 * math.pi) + 14 * math.sin(i / 4)) for i in range(45)], 52, seed=3)
img = brush_sweep(img, [(1100 + i * 20, 760 - i * 9 + 10 * math.sin(i / 2)) for i in range(40)], 30, seed=4, density=0.8)
# 震动的同心冲击线（鼓点）
ring = mask_from(lambda d, s: [d.arc([(1300 - rr) * s, (430 - rr * 0.55) * s, (1300 + rr) * s, (430 + rr * 0.55) * s], 190, 350, fill=255, width=int(3 * s))
                               for rr in (230, 290, 360)])
img = ink(img, ring, 0.6, 0.5, INK, 5)
# 工笔大鼓、手鼓、铃
img = drum(img, 300, 820, 170, 150, seed=6, angle=-0.12)
img = drum(img, 1660, 840, 120, 100, seed=7, angle=0.15)
img = drum(img, 1500, 560, 70, 40, seed=8, angle=-0.35)
img = bell(img, 1560, 150, 38, seed=9)
img = bell(img, 1640, 215, 28, seed=10)
img = bell(img, 1760, 470, 34, seed=11)
img = ink(img, mask_from(lambda d, s: d.line([(1500 * s, 40 * s), (1700 * s, 70 * s)], fill=255, width=int(3 * s))), 0.8, 0.5, INK, 12)
# 茉莉枝（左上与右下），被鼓点震落的花瓣
for (lx, ly, L, a) in [(1760, 90, 120, 2.6), (1820, 150, 100, 2.0), (1720, 980, 130, -2.4), (1800, 930, 100, -1.8)]:
    img = leaf(img, lx, ly, L, a, seed=int(lx))
for i, (fx, fy, r) in enumerate([(1660, 120, 34), (1710, 200, 26), (1690, 900, 30), (1840, 990, 34)]):
    img = gongbi_jasmine(img, fx, fy, r, i * 0.4, seed=20 + i)

# Clawd：素纱 + 双钳上举（茉莉被抛向空中）
S = 22
gauze = [(-0.4, 4.6), (16.4, 4.6), (17.4, 8.6), (18.2, 11.6), (8.0, 11.9), (-2.2, 11.6), (-1.4, 8.6)]
arms = [rigid_arm("L", 70), rigid_arm("R", 70)]
img, (ox, oy, pw, ph, ccx, ccy) = ink_clawd(img, S, 960, 905, [("gauze", [gauze], (0.97, 0.97, 0.95), 0.55)], seed=30, arms=arms)
gl = mask_from(lambda d, s: [d.line([((960 - 8 * S + gxv * S) * s, (ccy - ph / 2 + oy + 4.8 * S) * s), ((960 - 8 * S + gxv * S * 1.08 - 0.6 * S) * s, (ccy - ph / 2 + oy + 11.6 * S) * s)], fill=255, width=int(1.4 * s))
                             for gxv in (2, 5, 8, 11, 14)])
img = ink(img, gl, 0.35, 0.5, INK, 31)
# 抛起的茉莉 + 花瓣
for i, (fx, fy, r) in enumerate([(960, 470, 42), (880, 400, 22), (1050, 420, 24), (930, 330, 16), (1010, 360, 14)]):
    img = gongbi_jasmine(img, fx, fy, r, 0.3 + i, seed=40 + i)
# 地面淡墨
img = ink(img, mask_from(lambda d, s: d.ellipse([700 * s, 890 * s, 1220 * s, 930 * s], fill=255)), 0.25, 8, INK, 50)

ink_t, seal = ink_title("还你茉莉", size=230, seed=2, seal_char="茉")
place(img, ink_t, 20, 0)
place(img, seal, ink_t.shape[1] - 60, 70)
img = img * (0.97 + 0.03 * fbm(H, W, 801, octaves=2, base=300)[..., None])
save(img, os.path.join(OUT, "08-a_还你茉莉_泼墨鼓点.png"))
print("08-a ok")

# =====================================================================
# 08-b 墨色舞台
# =====================================================================
img = xuan(seed=870)
# 大面积浓墨背景（留出右下一点纸白）
bgm = np.clip(1.05 - fbm(H, W, 871, octaves=3, base=2) * 0.25 - np.clip((x - 1500) / 600, 0, 1) * np.clip((y - 700) / 400, 0, 1) * 0.9, 0, 1)
img = ink(img, bgm, 0.93, 6, INK, 72)
# 浓墨弧（代替光带）——这里用“留白+淡墨”的反白弧：在墨底上刷出灰白弧
arc_m = mask_from(lambda d, s: d.arc([(-200) * s, 150 * s, (2120) * s, 1350 * s], 196, 344, fill=255, width=int(70 * s)))
rr_ = np.sqrt((x - 960) ** 2 + ((y - 750) * 1160 / 600) ** 2)
arc_m = arc_m * dry_gaps(rr_, np.clip(np.abs(x - 960) / 1100, 0, 1), 77, 0.05, 0.4)
img = img + arc_m[..., None] * (np.array([0.82, 0.8, 0.76]) - img) * 0.75
# 背后斜劈的两笔飞白（像扫弦）
for k, (p0, p1, wd) in enumerate([((300, 980), (1500, 380), 46), ((620, 1040), (1720, 560), 30)]):
    pts = [(p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u + 30 * math.sin(u * 3)) for u in np.linspace(0, 1, 40)]
    sm = mask_from(lambda d, s_, pts=pts, wd=wd: [d.line([(pts[i][0] * s_, pts[i][1] * s_), (pts[i + 1][0] * s_, pts[i + 1][1] * s_)], fill=255,
                                                         width=max(2, int(wd * (1.2 - 0.9 * i / 39) * s_))) for i in range(39)])
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    tt = -x * math.sin(ang) + y * math.cos(ang)
    al = np.clip(((x - p0[0]) * math.cos(ang) + (y - p0[1]) * math.sin(ang)) / 1300, 0, 1)
    sm = sm * dry_gaps(tt, al, 90 + k, 0.1, 0.7)
    img = img + sm[..., None] * (np.array([0.62, 0.6, 0.57]) - img) * 0.8
# 鼓点：灰白飞溅（像鼓槌砸下溅起的墨水），朱红溅点
PALE = np.array([0.78, 0.76, 0.72])
for i, (sx, sy, r, dirn) in enumerate([(360, 780, 34, -2.3), (1580, 720, 30, -0.8), (980, 300, 26, -1.57)]):
    img = splash(img, sx, sy, r, seed=60 + i, n=46, density=0.85, color=PALE, direction=dirn)
img = splash(img, 1720, 900, 40, seed=74, n=24, density=0.9, color=CINNABAR)
img = splash(img, 210, 300, 30, seed=75, n=18, density=0.9, color=CINNABAR)

S = 27
drape = [(-0.4, 5.8), (16.4, 5.8), (17.2, 7.6), (18.6, 9.6), (19.4, 11.8), (14.0, 11.9), (8.0, 11.6), (2.0, 11.9), (-0.6, 11.8), (-0.2, 8.4)]
cowl = [(-0.6, 5.5), (16.6, 5.5), (16.6, 6.6), (11.0, 7.6), (5.0, 6.6), (-0.6, 7.0)]
fist = rigid_arm("L", 80)
mic_arm = rigid_arm("R", 25)
img, (ox, oy, pw, ph, ccx, ccy) = ink_clawd(img, S, 960, 1000,
    [("drape", [drape], (0.08, 0.08, 0.09), 1.0), ("cowl", [cowl], (0.13, 0.13, 0.14), 1.0)], seed=80, arms=[fist, mic_arm], opacity=1.0)
# 垂坠褶：淡灰线
def gp(gx_, gy_):
    return (960 - pw / 2 + ox + gx_ * S, ccy - ph / 2 + oy + gy_ * S)
folds = mask_from(lambda d, s: [d.line([(gp(a, 6.4)[0] * s, gp(a, 6.4)[1] * s), (gp(b, 11.6)[0] * s, gp(b, 11.6)[1] * s)], fill=255, width=int(3 * s))
                                for a, b in [(3, 1.5), (7, 7.5), (11, 13), (14.5, 17.6)]])
img = img + folds[..., None] * (np.array([0.32, 0.32, 0.34]) - img) * 0.8
# 红皮手套（覆盖两只手臂）——全画面除印章外唯一的朱红
glove = mask_from(lambda d, s: [d.polygon([(gp(*p)[0] * s, gp(*p)[1] * s) for p in arm], fill=255) for arm in (fist, mic_arm)])
img = over(img, CINNABAR, glove)
img = img + np.clip(glove - shift(glove, 3, 3), 0, 1)[..., None] * 0.25
# 麦克风（右手）
mt = gp(17.9, 2.2)
mic = mask_from(lambda d, s: [d.line([(gp(17.2, 3.4)[0] * s, gp(17.2, 3.4)[1] * s), (mt[0] * s, mt[1] * s)], fill=255, width=int(16 * s)),
                              d.ellipse([(mt[0] - 22) * s, (mt[1] - 26) * s, (mt[0] + 22) * s, (mt[1] + 14) * s], fill=255)])
img = over(img, np.array([0.55, 0.55, 0.58]), mic)
# 金色耳饰：身体两侧上方
for gx_ in (-0.2, 16.2):
    ex, ey = gp(gx_, 1.8)
    m = mask_from(lambda d, s, ex=ex, ey=ey: [d.ellipse([(ex - 13) * s, (ey - 13) * s, (ex + 13) * s, (ey + 13) * s], fill=255),
                                              d.ellipse([(ex - 9) * s, (ey + 16) * s, (ex + 9) * s, (ey + 40) * s], fill=255)])
    img = over(img, np.array([0.88, 0.7, 0.3]), m)
# 握拳的冲击线（白）
punch = mask_from(lambda d, s: [d.line([((gp(-1.6, -1.6)[0] + dx) * s, (gp(-1.6, -1.6)[1] - 40 - k * 22) * s), ((gp(-1.6, -1.6)[0] + dx * 1.6) * s, (gp(-1.6, -1.6)[1] - 80 - k * 22) * s)], fill=255, width=int(4 * s))
                                for k, dx in enumerate((-50, 0, 50))])
img = img + punch[..., None] * (np.array([0.92, 0.9, 0.86]) - img)
# 四角朱红印章
_, seal = ink_title("还你茉莉", size=200, seed=5, seal_char="莉")
_, seal2 = ink_title("还你茉莉", size=170, seed=6, seal_char="茉")
place(img, seal, 60, 60)
place(img, seal2, W - 60 - seal2.shape[1], 70)
place(img, seal2, 80, H - 80 - seal2.shape[0])
place(img, seal, W - 70 - seal.shape[1], H - 90 - seal.shape[0])
# 低角度仰拍的透视感：上方略暗
img = img * (0.88 + 0.12 * np.clip(y / H, 0, 1))[..., None]
img = img * (0.97 + 0.03 * fbm(H, W, 879, octaves=2, base=300)[..., None])
save(img, os.path.join(OUT, "08-b_还你茉莉_墨色舞台.png"))
print("08-b ok")
