"""06｜有趣 · 赛璐璐扁平动画（粗描边、平涂、硬阴影）
06-a：镜面长廊，Clawd 的镜像排成一列、越远越小（参照单曲封面的多重镜像）；都穿珍珠链上衣 + 金黄长裙，
      各做不同小动作：闭眼歪头、挥手、歪身、背身转圈、眨眼，最后一个跑出画面。
06-b：歌手 2025 舞台：天花板垂下一排白色光管、侧逆光；Clawd 穿灰色不对称上衣配长裙，双臂张开，站在立式麦克风前。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from titles import cel_title, place

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()
INK = np.array([0.06, 0.05, 0.08])
ORANGE = np.array(CLAWD_ORANGE) / 255
ORANGE_SH = ORANGE * np.array([0.78, 0.66, 0.68])


def erode(m, r):
    return np.clip((blur(m, r) - 0.5) * 6 + 0.5, 0, 1) * (m > 0.5)


def cel_clawd(S, parts, light_split=11.0, eyes_closed=None, back=False, outline=0.32, arms=None):
    """parts: [(名称, 形状列表, 颜色, 阴影色)] 自下而上；返回 RGBA 精灵。
    eyes_closed: None / 'both' / 'left'；back=True 为背面（不画眼睛）。"""
    extra = {name: shapes for name, shapes, _, _ in parts}
    closed = [(3.2, 3.8, 5.0, 4.3), (11.0, 3.8, 12.8, 4.3)]
    extra["closed"] = closed if eyes_closed == "both" else closed[:1] if eyes_closed == "left" else [(0, 0, 0, 0)]
    cm, (ox, oy), (pw, ph) = clawd_masks(S, int(S * 2), arms=arms, bbox=(-4, -3, 21, 13), extra=extra)
    gy_, gx_ = np.mgrid[0:ph, 0:pw].astype(np.float32)
    gx = (gx_ - ox) / S
    body = np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1)
    union = body.copy()
    col = np.zeros((ph, pw, 3), np.float32)
    col[:] = ORANGE
    col = np.where((gx > light_split)[..., None] & (body > 0.5)[..., None], ORANGE_SH, col)
    edges = np.clip(body - erode(body, 1.2), 0, 1)
    for name, shapes, c, sh in parts:
        m = cm[name]
        cc = np.where((gx > light_split)[..., None], np.array(sh), np.array(c))
        col = over(col, cc, m)
        union = np.maximum(union, m)
        edges = np.maximum(edges, np.clip(m - erode(m, 1.0), 0, 1))
    if not back:
        # 睁眼：原眼睛；闭左眼：只保留右眼；双眼闭：不画原眼睛，只画闭眼线
        e = {None: cm["eyes"], "left": cm["eyes"] * (gx > 8), "both": 0 * cm["eyes"]}[eyes_closed]
        col = over(col, INK, e)
        col = over(col, INK, cm["closed"])
    # 外描边
    out = np.clip(blur(union, S * outline * 0.45) * 5, 0, 1)
    rgba = np.dstack([np.ones((ph, pw, 3)) * INK, out])
    rgba[..., :3] = over(rgba[..., :3], col, union)
    rgba[..., :3] = over(rgba[..., :3], INK, edges * 0.9)
    return rgba


def pearls_shapes(n_rows=4):
    """珍珠链（多段下垂的小圆，参照单曲封面）：返回多边形列表（格坐标）。"""
    shapes = []
    for k in range(n_rows):
        y0 = 5.4 + k * 0.55
        sag = 0.5 + k * 0.35
        for i in range(15):
            t = i / 14
            cx = 1.2 + 13.6 * t
            cy = y0 + sag * (1 - (2 * t - 1) ** 2)
            r = 0.28
            shapes.append([(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 10)])
    return shapes


SKIRT = [(-0.3, 6.6), (16.3, 6.6), (17.6, 9.0), (19.0, 11.6), (17.0, 12.0), (14.0, 11.7), (11.0, 12.0), (8.0, 11.7),
         (5.0, 12.0), (2.0, 11.7), (-1.0, 12.0), (-3.0, 11.6), (-1.6, 9.0)]
SKIRT_FOLDS = [[(sx, 7.0), (sx + 0.15, 7.2), (sx * 1.15 - 1.2 + 0.15, 11.8), (sx * 1.15 - 1.2, 11.8)] for sx in (3.0, 6.0, 10.0, 13.0)]
YELLOW, YELLOW_SH = (1.0, 0.80, 0.10), (0.92, 0.55, 0.06)
PEARL, PEARL_SH = (0.98, 0.96, 0.92), (0.82, 0.8, 0.82)
check_proportions(extra={"skirt": [SKIRT]})


def outfit_parts():
    return [("skirt", [SKIRT], YELLOW, YELLOW_SH), ("folds", SKIRT_FOLDS, YELLOW_SH, (0.8, 0.42, 0.05)),
            ("pearls", pearls_shapes(), PEARL, PEARL_SH)]

# =====================================================================
# 06-a 镜面长廊
# =====================================================================
img = np.zeros((H, W, 3), np.float32)
img[:] = np.array([0.05, 0.05, 0.1])
img += np.clip(1 - y / H, 0, 1)[..., None] * np.array([0.04, 0.05, 0.12])
# 地面：深色镜面（下半）
FLOOR_Y = 820
VX, VY = 1700, 560   # 消失点（右侧远处）
img = np.where((y > FLOOR_Y)[..., None], np.array([0.09, 0.08, 0.14]), img)
# 镜面竖缝：透视排列
seams = []
for k in range(9):
    t = (k / 8) ** 1.6
    sx = 80 + (VX - 80) * t
    seams.append(sx)
sm = np.zeros((H, W), np.float32)
im = Image.new("L", (W * SS, H * SS), 0)
d = ImageDraw.Draw(im)
for k, sx in enumerate(seams):
    top = 40 + (VY - 260 - 40) * (k / 8) ** 1.6
    bot = H + 10 - (H - VY - 60) * (k / 8) ** 1.6
    d.line([(sx * SS, top * SS), (sx * SS, bot * SS)], fill=255, width=int(max(2, 6 - k * 0.5) * SS))
seam_m = to_arr(im.resize((W, H), Image.LANCZOS))
img = over(img, np.array([0.55, 0.58, 0.7]), seam_m)
# 每块镜面里的平涂反光带（赛璐璐式斜条）
im = Image.new("L", (W * SS, H * SS), 0)
d = ImageDraw.Draw(im)
for k in range(len(seams) - 1):
    a_, b_ = seams[k], seams[k + 1]
    w_ = b_ - a_
    d.polygon([((a_ + w_ * 0.15) * SS, 0), ((a_ + w_ * 0.32) * SS, 0), ((a_ + w_ * 0.05) * SS, FLOOR_Y * SS), ((a_ - w_ * 0.12) * SS, FLOOR_Y * SS)], fill=255)
streaks = to_arr(im.resize((W, H), Image.LANCZOS))
img = img + streaks[..., None] * np.array([0.05, 0.06, 0.1])
# 缝顶上的星形闪光
def star4(img, sx, sy, r):
    pts = []
    for i in range(8):
        a = i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.22
        pts.append((sx + rr * math.cos(a), sy + rr * math.sin(a)))
    im = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(im).polygon([(px * SS, py * SS) for px, py in pts], fill=255)
    m = to_arr(im.resize((W, H), Image.LANCZOS))
    return over(img, np.array([1.0, 0.98, 0.9]), m)
for k, sx in enumerate(seams[:-1]):
    img = star4(img, sx, 70 + (VY - 300) * (k / 8) ** 1.6, 26 - k * 2)

# 6 个镜像 Clawd：(中心x, 脚底y, S, 角度, 动作)
crew = [
    (330, 905, 24, -7, dict(eyes_closed="both")),                              # 闭眼歪头（呼应封面）
    (730, 860, 18, 0, dict(arms=[(-1, 3, 0, 5), rigid_arm("R", 75)])),          # 挥手
    (1040, 820, 14, 10, {}),                                                    # 歪身
    (1280, 790, 11, 0, dict(back=True)),                                       # 背身转圈
    (1460, 770, 9, 0, dict(eyes_closed="left")),                               # 眨眼
    (1880, 748, 8, -16, {}),                                                    # 跑出画面
]
for (cx, foot, S, ang, kw) in crew[::-1]:
    spr = cel_clawd(S, outfit_parts(), **kw)
    ph = spr.shape[0]
    # sprite 中心对应格坐标 y=5（bbox -3..13 的中点），脚底 y=12 → 偏移 7 格
    cy = foot - 7 * S
    # 地面倒影
    refl = spr[::-1].copy()
    refl[..., 3] *= 0.22
    img = paste_rgba(img, refl, cx, foot + (foot - cy) - 0.2 * S, -ang)
    img = paste_rgba(img, spr, cx, cy, ang)
# 跑出画面的速度线
im = Image.new("L", (W * SS, H * SS), 0)
d = ImageDraw.Draw(im)
for k in range(5):
    yy = 640 + k * 22
    d.line([(1720 * SS, yy * SS), ((1800 - k * 8) * SS, yy * SS)], fill=255, width=4 * SS)
img = over(img, np.array([0.95, 0.95, 1.0]), to_arr(im.resize((W, H), Image.LANCZOS)) * 0.9)
# 挥手的动作弧线
im = Image.new("L", (W * SS, H * SS), 0)
d = ImageDraw.Draw(im)
d.arc([(790) * SS, (600) * SS, (930) * SS, (740) * SS], 200, 300, fill=255, width=4 * SS)
d.arc([(800) * SS, (575) * SS, (965) * SS, (740) * SS], 205, 290, fill=255, width=3 * SS)
img = over(img, np.array([0.95, 0.95, 1.0]), to_arr(im.resize((W, H), Image.LANCZOS)))

title = cel_title("有趣", size=230, seed=1)
place(img, title, 70, 40)
# 标题的镜像倒影（淡）
tr = title[::-1].copy()
cut = int(tr.shape[0] * 0.38)
tr = tr[:cut]
tr[..., 3] *= np.linspace(0.16, 0, cut)[:, None]
place(img, tr, 70, 40 + title.shape[0] - 60)
save(img, os.path.join(OUT, "06-a_有趣_镜面长廊.png"))
print("06-a ok")

# =====================================================================
# 06-b 光管舞台
# =====================================================================
img = np.zeros((H, W, 3), np.float32)
img[:] = np.array([0.10, 0.11, 0.16])
FLOOR_Y = 880
img = np.where((y > FLOOR_Y)[..., None], np.array([0.17, 0.17, 0.22]), img)
# 天花板垂下的光管：弧形排列，长短不一（平涂：白芯 + 浅青光边 + 描边）
g = rng(61)
im_glow = Image.new("L", (W * SS, H * SS), 0)
im_core = Image.new("L", (W * SS, H * SS), 0)
dg, dc = ImageDraw.Draw(im_glow), ImageDraw.Draw(im_core)
for k in range(46):
    tx = 40 + k * 41 + g.uniform(-6, 6)
    arc = 1 - ((tx - 960) / 1000) ** 2
    L = 140 + 300 * arc * g.uniform(0.7, 1.0)
    top = -10
    dg.rounded_rectangle([(tx - 9) * SS, top * SS, (tx + 9) * SS, (top + L + 8) * SS], radius=9 * SS, fill=255)
    dc.rounded_rectangle([(tx - 4) * SS, top * SS, (tx + 4) * SS, (top + L) * SS], radius=4 * SS, fill=255)
glow = to_arr(im_glow.resize((W, H), Image.LANCZOS))
core = to_arr(im_core.resize((W, H), Image.LANCZOS))
img = over(img, np.array([0.55, 0.82, 0.92]), glow)
img = over(img, np.array([1.0, 1.0, 1.0]), core)
# 侧逆光：从左侧打来的平涂光束
beam = Image.new("L", (W * SS, H * SS), 0)
ImageDraw.Draw(beam).polygon([(0, 420 * SS), (0, 640 * SS), (980 * SS, 960 * SS), (1080 * SS, 760 * SS)], fill=255)
bm = to_arr(beam.resize((W, H), Image.LANCZOS))
img = img + bm[..., None] * np.array([0.10, 0.11, 0.13])
# 光源本体（左侧的强光点）
img = star4(img, 30, 530, 70)
# 地面光斑
pool = ((x - 980) / 360) ** 2 + ((y - 925) / 42) ** 2 < 1
img = np.where(pool[..., None], np.array([0.34, 0.34, 0.4]), img)

S = 26
GREY, GREY_SH = (0.66, 0.66, 0.68), (0.46, 0.46, 0.5)
top_ = [(-0.2, 5.2), (16.2, 6.6), (16.2, 8.4), (-0.2, 8.4)]    # 斜领口：左高右低（不对称）
strap = [(-0.3, 5.2), (2.6, 5.4), (1.4, 10.6), (-1.6, 10.2)]    # 左侧垂片（不对称）
cutout = [(0, 0), (0, 0), (0, 0)]
skirt = [(-0.4, 8.2), (16.4, 8.2), (17.2, 11.8), (14.0, 12.0), (8.0, 11.8), (2.0, 12.0), (-1.2, 11.8)]
folds = [[(sx, 8.6), (sx + 0.12, 8.6), (sx + 0.4, 11.8), (sx + 0.28, 11.8)] for sx in (3, 6.5, 9.5, 13)]
check_proportions(arms=[rigid_arm("L", 28), rigid_arm("R", 28)], extra={"skirt": [skirt]})
parts = [("top", [top_], GREY, GREY_SH), ("strap", [strap], GREY, GREY_SH), ("skirt", [skirt], GREY, GREY_SH),
         ("folds", folds, GREY_SH, (0.36, 0.36, 0.4)), ("cutout", [cutout], tuple(ORANGE), tuple(ORANGE_SH))]
spr = cel_clawd(S, parts, light_split=12.0, arms=[rigid_arm("L", 28), rigid_arm("R", 28)])
CX, FOOT = 980, 925
CY = FOOT - 7 * S
img = paste_rgba(img, spr, CX, CY)
# 立式麦克风（在身前）：支架 + 麦头
im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
mx = CX + 30
top_y = CY - 5 * S + 6.4 * S          # 对应格坐标 y≈6.4（眼睛下方）
d.line([(mx * SS, (top_y + 30) * SS), (mx * SS, FOOT * SS)], fill=(16, 14, 20, 255), width=10 * SS)
d.line([((mx - 60) * SS, FOOT * SS), ((mx + 60) * SS, FOOT * SS)], fill=(16, 14, 20, 255), width=10 * SS)
d.rounded_rectangle([(mx - 18) * SS, (top_y - 22) * SS, (mx + 18) * SS, (top_y + 34) * SS], radius=16 * SS, fill=(16, 14, 20, 255))
d.rounded_rectangle([(mx - 12) * SS, (top_y - 16) * SS, (mx + 12) * SS, (top_y + 28) * SS], radius=11 * SS, fill=(150, 152, 160, 255))
d.line([((mx - 6) * SS, (top_y - 8) * SS), ((mx - 6) * SS, (top_y + 18) * SS)], fill=(225, 228, 235, 255), width=3 * SS)
mic = to_arr(im.resize((W, H), Image.LANCZOS))
img = over(img, mic[..., :3], mic[..., 3])
# 张开双臂的动作线
im = Image.new("L", (W * SS, H * SS), 0)
d = ImageDraw.Draw(im)
for sgn in (-1, 1):
    bx = CX + sgn * 10.6 * S
    for k in range(3):
        d.line([((bx + sgn * (20 + k * 18)) * SS, (CY - 3.2 * S - 30 + k * 26) * SS),
                ((bx + sgn * (50 + k * 18)) * SS, (CY - 3.2 * S - 46 + k * 26) * SS)], fill=255, width=4 * SS)
img = over(img, np.array([0.95, 0.96, 1.0]), to_arr(im.resize((W, H), Image.LANCZOS)))
save(img, os.path.join(OUT, "06-b_有趣_光管舞台.png"))
print("06-b ok")
