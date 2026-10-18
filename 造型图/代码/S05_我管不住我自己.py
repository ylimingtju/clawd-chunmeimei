"""05｜我管不住我自己 · 透明水彩
05-a：清晨小阳台，黑胶唱机在转，晾衣、绿植、咖啡；音符化成水彩小花飘出。Clawd 浇着花忍不住跟着摇摆。
05-b：南京现场（水彩表现）：深色背景、柔和追光、水彩光斑；Clawd 穿银色亮片层叠短裙，粉色立体玫瑰、荷叶边、亮片长靴，一只钳子按在胸前。"""
import sys, math, os
import numpy as np
from PIL import Image, ImageDraw
from common import *
from titles import watercolor_title
import watercolor as wc

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SS = 2
x, y = grid()


def poly_mask(polys):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([(px * SS, py * SS) for px, py in p], fill=255)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def ell_mask(items):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for (cx, cy, rx, ry) in items:
        d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS], fill=255)
    return to_arr(im.resize((W, H), Image.LANCZOS))


def line_mask(lines, width):
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for pl in lines:
        d.line([(px * SS, py * SS) for px, py in pl], fill=255, width=int(width * SS), joint="curve")
    return to_arr(im.resize((W, H), Image.LANCZOS))


def place_mask(m, cx, cy, angle=0.0):
    """把 Clawd 局部遮罩按中心位置与角度放到全画布。"""
    im = to_img(m)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    a = to_arr(im)
    out = np.zeros((H, W), np.float32)
    h, w = a.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    out[Y0:Y1, X0:X1] = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    return out


def flower5(cx, cy, r, rot=0.0):
    pts = []
    for i in range(60):
        a = rot + 2 * math.pi * i / 60
        rr = r * (0.55 + 0.45 * abs(math.cos(a * 2.5 - rot * 2.5)))
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


ORANGE_WC = (0.86, 0.52, 0.38)  # 水彩里的 Clawd 橙（叠两遍接近 #CA7C5E）

# =====================================================================
# 05-a 清晨阳台
# =====================================================================
img, TEX = wc.paper()
# 天空：上淡蓝、近地平暖黄（湿画法渐变）
skym = (y < 640).astype(np.float32)
img = wc.wash(img, skym * np.clip(1 - y / 640, 0, 1) ** 0.8, (0.62, 0.80, 0.95), 0.55, seed=1, bleed=10, edge=0.1, tex=TEX)
img = wc.wash(img, skym * np.clip((y - 250) / 390, 0, 1), (0.99, 0.86, 0.58), 0.45, seed=2, bleed=14, edge=0.05, tex=TEX)
# 太阳留白 + 光芒（留出纸白）
sun = np.exp(-(((x - 1560) ** 2 + (y - 170) ** 2) / (2 * 140 ** 2)))
img = img * (1 - sun[..., None] * 0.0) + sun[..., None] * (np.array([1, 0.99, 0.95]) - img) * 0.85
# 远处屋顶与树（很淡）
roofs = []
gg = rng(3)
xx = -40
while xx < W:
    w_ = gg.uniform(90, 200)
    h_ = gg.uniform(50, 130)
    roofs.append([(xx, 640), (xx, 640 - h_), (xx + w_ * 0.5, 640 - h_ - gg.uniform(20, 50)), (xx + w_, 640 - h_), (xx + w_, 640)])
    xx += w_ + gg.uniform(10, 40)
img = wc.wash(img, poly_mask(roofs), (0.78, 0.74, 0.82), 0.35, seed=4, bleed=3, tex=TEX)
trees = ell_mask([(gg.uniform(0, W), 620 - gg.uniform(0, 40), gg.uniform(30, 70), gg.uniform(25, 50)) for _ in range(14)])
img = wc.wash(img, trees, (0.66, 0.82, 0.62), 0.4, seed=5, bleed=4, tex=TEX)

# 晾衣绳 + 衣物
LDY = 75
line_y = lambda xv: 175 + LDY + 0.00012 * (xv - 1000) ** 2
rope = line_mask([[(xv, line_y(xv)) for xv in range(300, 1500, 20)]], 2)
img = img * (1 - rope[..., None] * 0.55)
shirt = [(560, line_y(560)), (720, line_y(720)), (760, 250), (730, 268), (720, 400), (560, 400), (550, 268), (520, 250)]
towel = [(860, line_y(860)), (1010, line_y(1010)), (1018, 350), (1000, 362), (980, 348), (960, 362), (940, 348), (920, 362), (900, 348), (880, 362), (852, 350)]
sock = [(1140, line_y(1140)), (1180, line_y(1180)), (1186, 260), (1210, 280), (1196, 300), (1150, 276)]
dyp = lambda P: [(a, b + (LDY if abs(b - line_y(a)) > 0.01 else 0)) for a, b in P]
shirt, towel, sock = dyp(shirt), dyp(towel), dyp(sock)
img = wc.wash(img, poly_mask([shirt]), (0.96, 0.66, 0.72), 0.65, seed=6, tex=TEX)
img = wc.wash(img, poly_mask([towel]), (0.62, 0.86, 0.78), 0.65, seed=7, tex=TEX)
img = wc.wash(img, poly_mask([towel]) * (np.sin(x / 9) > 0.5), (0.5, 0.78, 0.72), 0.4, seed=8, bleed=1.5, tex=TEX)
img = wc.wash(img, poly_mask([sock]), (0.98, 0.84, 0.5), 0.65, seed=9, tex=TEX)
pins = poly_mask([[(px - 5, line_y(px) - 12), (px + 5, line_y(px) - 12), (px + 4, line_y(px) + 14), (px - 4, line_y(px) + 14)]
                  for px in (575, 705, 875, 995, 1150, 1176)])
img = wc.wash(img, pins, (0.75, 0.6, 0.45), 0.8, seed=10, bleed=1, tex=TEX)

# 地砖（陶土色，透视格线）
floor = (y > 790).astype(np.float32)
img = wc.wash(img, floor, (0.93, 0.72, 0.58), 0.6, seed=11, bleed=3, tex=TEX)
vx = 960
tile_lines = []
for k in range(-14, 15):
    tile_lines.append([(vx + k * 40, 790), (vx + k * 220, H + 10)])
for yy in (830, 885, 960, 1060):
    tile_lines.append([(0, yy), (W, yy)])
img = img * (1 - line_mask(tile_lines, 2)[..., None] * 0.1)
# 阳台栏杆（铁艺，细线 + 卷花）
rail = [[(0, 600), (W, 600)], [(0, 780), (W, 780)]]
for k in range(0, W, 64):
    rail.append([(k, 600), (k, 780)])
    rail.append([(k + 32, 700), (k + 20, 688), (k + 32, 676), (k + 44, 688), (k + 32, 700)])
img = img * (1 - line_mask(rail, 4)[..., None] * np.array([0.55, 0.5, 0.5]))
# 左侧薄荷绿门框
door = poly_mask([[(0, 0), (230, 0), (230, H), (0, H)]]) - poly_mask([[(40, 60), (190, 60), (190, 1000), (40, 1000)]])
img = wc.wash(img, np.clip(door, 0, 1), (0.60, 0.84, 0.76), 0.7, seed=12, tex=TEX)
img = wc.wash(img, poly_mask([[(40, 60), (190, 60), (190, 1000), (40, 1000)]]), (0.85, 0.93, 0.96), 0.3, seed=13, tex=TEX)

# 龟背竹（左）+ 栏杆边的花盆
leaves = []
for (lx, ly, r, a) in [(330, 560, 110, -0.6), (440, 470, 120, -0.2), (250, 690, 100, -1.0), (520, 610, 90, 0.4), (360, 400, 90, -0.9)]:
    pts = []
    for i in range(40):
        t_ = 2 * math.pi * i / 40
        rr = r * (0.75 + 0.25 * math.cos(t_)) * (0.82 if math.sin(t_ * 6) > 0.6 else 1.0)  # 龟背竹的裂口
        pts.append((lx + rr * math.cos(t_ + a) * 1.0, ly + rr * math.sin(t_ + a) * 0.72))
    leaves.append(pts)
stems = line_mask([[(380, 880), (330, 560)], [(390, 880), (440, 470)], [(370, 880), (250, 690)], [(400, 880), (520, 610)], [(375, 880), (360, 400)]], 6)
img = wc.wash(img, stems, (0.5, 0.68, 0.45), 0.45, seed=16, bleed=1, tex=TEX)
img = wc.wash(img, poly_mask(leaves), (0.35, 0.66, 0.42), 0.72, seed=14, tex=TEX)
img = wc.wash(img, poly_mask(leaves[::2]), (0.3, 0.55, 0.36), 0.35, seed=15, bleed=6, tex=TEX)
# 叶脉
veins = []
for (lx, ly, r, a) in [(330, 560, 110, -0.6), (440, 470, 120, -0.2), (250, 690, 100, -1.0), (520, 610, 90, 0.4), (360, 400, 90, -0.9)]:
    veins.append([(lx - r * 0.8 * math.cos(a), ly - r * 0.5 * math.sin(a)), (lx + r * 0.8 * math.cos(a), ly + r * 0.5 * math.sin(a))])
pots = poly_mask([[(300, 860), (470, 860), (450, 990), (320, 990)], [(1240, 740), (1330, 740), (1320, 800), (1250, 800)],
                  [(640, 745), (710, 745), (702, 795), (648, 795)]])
img = wc.wash(img, pots, (0.86, 0.5, 0.38), 0.75, seed=17, tex=TEX)
blooms = [(1270, 715, 22), (1305, 700, 18), (1290, 690, 15), (660, 720, 16), (690, 712, 14)]
for i, (bx, by, br) in enumerate(blooms):
    img = wc.wet_bloom(img, bx, by, br, [(0.95, 0.5, 0.62), (0.98, 0.78, 0.4)][i % 2], seed=20 + i, density=0.8, tex=TEX)

# 小圆桌 + 黑胶唱机 + 咖啡
table = ell_mask([(1580, 770, 230, 46)])
img = wc.wash(img, table, (0.95, 0.92, 0.86), 0.5, seed=18, tex=TEX)
img = wc.wash(img, line_mask([[(1580, 790), (1580, 1000)], [(1500, 1000), (1660, 1000)]], 10), (0.55, 0.45, 0.4), 0.7, seed=19, tex=TEX)
player = poly_mask([[(1420, 690), (1690, 690), (1710, 760), (1400, 760)]])
img = wc.wash(img, player, (0.72, 0.5, 0.35), 0.8, seed=21, tex=TEX)
vinyl = ell_mask([(1545, 700, 105, 30)])
img = wc.wash(img, vinyl, (0.15, 0.13, 0.16), 0.9, seed=22, bleed=1.5, tex=TEX)
img = img + (ell_mask([(1520, 694, 40, 6)]) * 0.25)[..., None]               # 唱片反光（留白）
img = wc.wash(img, ell_mask([(1545, 700, 22, 7)]), (0.95, 0.45, 0.5), 0.9, seed=23, bleed=1, tex=TEX)
img = wc.wash(img, line_mask([[(1660, 690), (1640, 660), (1580, 700)]], 4), (0.5, 0.5, 0.52), 0.85, seed=24, bleed=0.8, tex=TEX)
cup = poly_mask([[(1730, 700), (1790, 700), (1784, 760), (1736, 760)]])
img = wc.wash(img, cup, (0.98, 0.98, 0.98), 0.4, seed=25, tex=TEX)
img = wc.wash(img, line_mask([[(1790, 712), (1808, 722), (1790, 742)]], 5), (0.85, 0.85, 0.88), 0.6, seed=26, bleed=1, tex=TEX)
steam = line_mask([[(1750 + 12 * math.sin(t_ / 9), 690 - t_) for t_ in range(0, 90, 3)],
                   [(1772 + 10 * math.sin(t_ / 8 + 1), 690 - t_) for t_ in range(0, 80, 3)]], 4)
img = wc.wash(img, steam, (0.82, 0.82, 0.88), 0.4, seed=27, bleed=3, tex=TEX)

# 音符化成水彩小花：从唱机飘向左上
gg = rng(30)
for i in range(16):
    u = i / 15
    fx = 1450 - u * 760 + 60 * math.sin(u * 7)
    fy = 640 - u * 330 - 50 * math.cos(u * 5)
    r = 14 + 14 * math.sin(math.pi * u)
    col = [(0.95, 0.5, 0.62), (0.98, 0.78, 0.36), (0.52, 0.8, 0.72), (0.72, 0.6, 0.9)][i % 4]
    img = wc.wash(img, poly_mask([flower5(fx, fy, r, gg.uniform(0, 1))]), col, 0.7, seed=40 + i, bleed=2.5, edge=0.7, tex=TEX)
    img = wc.wash(img, ell_mask([(fx, fy, r * 0.22, r * 0.22)]), (0.98, 0.8, 0.3), 0.8, seed=60 + i, bleed=1, tex=TEX)
# 几个铅笔画的音符
notes = []
for (nx, ny, s_) in [(1380, 560, 1.0), (1180, 470, 0.8), (980, 380, 0.9), (1290, 400, 0.7)]:
    notes.append([(nx, ny), (nx, ny - 50 * s_), (nx + 26 * s_, ny - 40 * s_)])
    notes.append([(nx - 10 * s_, ny + 4), (nx, ny + 8 * s_), (nx + 4, ny), (nx - 8 * s_, ny - 4 * s_), (nx - 10 * s_, ny + 4)])
img = wc.pencil(img, notes, seed=31, alpha=0.5, width=2)

# ---------- Clawd：摇摆着浇花 ----------
S = 17
ANG = -9
arm_can = (16, 3, 17, 5)                       # 原手臂，水壶挂在手臂末端
can = [(16.8, 4.4), (20.6, 4.4), (21.0, 8.0), (16.4, 8.0)]
spout = [(20.6, 5.2), (24.4, 3.0), (24.8, 3.6), (21.0, 6.4)]
handle = [(16.8, 4.4), (17.6, 2.8), (19.6, 2.8), (20.4, 4.4), (19.6, 4.4), (19.2, 3.6), (18.0, 3.6), (17.6, 4.4)]
cm, (ox, oy), (pw, ph) = clawd_masks(S, 40, bbox=(-2, -1, 27, 14), arms=[(-1, 3, 0, 5), arm_can],
                                     extra={"can": [can, spout, handle]})
CX, CY = 930, 657
bodyM = place_mask(np.clip(cm["body"] + cm["arms"] + cm["legs"], 0, 1), CX, CY, ANG)
eyeM = place_mask(cm["eyes"], CX, CY, ANG)
canM = place_mask(cm["can"], CX, CY, ANG)
# 投影
img = wc.wash(img, shift(bodyM, 18, 22) * (1 - bodyM), (0.75, 0.68, 0.72), 0.45, seed=70, bleed=8, tex=TEX)
img = wc.wash(img, bodyM, ORANGE_WC, 0.85, seed=71, bleed=1.2, edge=0.5, tex=TEX)
img = wc.wash(img, bodyM * np.clip((x - CX + 60) / 200, 0, 1), (0.8, 0.45, 0.35), 0.35, seed=72, bleed=4, tex=TEX)  # 背光侧加一层
img = wc.wash(img, eyeM, (0.12, 0.1, 0.12), 0.95, seed=73, bleed=0.6, edge=0.2, tex=TEX)
img = wc.wash(img, canM, (0.45, 0.78, 0.66), 0.85, seed=74, bleed=1, tex=TEX)
# 水滴弧线
gg = rng(75)
spout_tip = (CX + 290, CY - 60)
drops = []
for i in range(14):
    u = i / 13
    dx_ = spout_tip[0] + 30 + 120 * u
    dy_ = spout_tip[1] + 20 + 180 * u * u
    drops.append((dx_ + gg.uniform(-6, 6), dy_, 5, 7))
img = wc.wash(img, ell_mask(drops), (0.6, 0.8, 0.95), 0.75, seed=76, bleed=1, tex=TEX)
# 摇摆的动感线（铅笔）
img = wc.pencil(img, [[(CX - 200, CY - 120 + k * 30), (CX - 240, CY - 100 + k * 30)] for k in range(3)] +
                [[(CX + 10, CY + 160 + k * 12), (CX + 80, CY + 160 + k * 12)] for k in range(1)], seed=77, alpha=0.4)
# 铅笔底稿：主要物体的轮廓
outline = []
for poly in [shirt, towel, [(1420, 690), (1690, 690), (1710, 760), (1400, 760)]]:
    outline.append(poly + [poly[0]])
img = wc.pencil(img, outline, seed=78, alpha=0.3)

# 标题：水彩字
t = watercolor_title("我管不住我自己", size=150, seed=4, colors=((0.92, 0.42, 0.52), (0.95, 0.66, 0.2)))
th, tw = t.shape[:2]
t = t[20:]
th, tw = t.shape[:2]
tx, ty = 560, 0
tm = np.zeros((H, W), np.float32)
tc = np.zeros((H, W, 3), np.float32)
tm[ty:ty + th, tx:tx + tw] = t[..., 3]
tc[ty:ty + th, tx:tx + tw] = t[..., :3]
img = img * (1 - tm[..., None] * (1 - tc))
# 晨光：右上方暖光
img = img + (np.exp(-(((x - 1560) / 700) ** 2 + ((y - 170) / 500) ** 2)))[..., None] * np.array([0.05, 0.04, 0.0])
img = img * (0.97 + 0.03 * TEX[..., None])
save(img, os.path.join(OUT, "05-a_我管不住我自己_晨光阳台.png"))
print("05-a ok")

# =====================================================================
# 05-b 南京现场（水彩）
# =====================================================================
img, TEX = wc.paper(seed=520)
# 深色湿画背景：靛蓝 + 紫 + 暗玫瑰，多层叠
for k, (c, dns) in enumerate([((0.30, 0.30, 0.55), 0.9), ((0.42, 0.28, 0.5), 0.7), ((0.25, 0.25, 0.45), 0.8), ((0.5, 0.3, 0.45), 0.5)]):
    m = np.clip(fbm(H, W, 530 + k, octaves=3, base=3) * 1.4 - 0.1, 0, 1)
    img = wc.wash(img, m, c, dns, seed=531 + k, bleed=20, edge=0.3, tex=TEX)
# 追光：从上方落下的柔和锥形，提亮（擦洗出纸白）
cone = poly_mask([[(860, -20), (1060, -20), (1300, 1000), (620, 1000)]])
cone = blur(cone, 50) * np.clip(1 - y / 1300, 0.3, 1)
img = img + cone[..., None] * (np.array([0.98, 0.95, 0.96]) - img) * 0.55
# 地面光池
pool = np.exp(-(((x - 960) / 360) ** 2 + ((y - 980) / 55) ** 2))
img = img + pool[..., None] * (np.array([1, 0.97, 0.98]) - img) * 0.6
# 水彩光斑（湿渍）
gg = rng(540)
for i in range(22):
    bx, by = gg.uniform(0, W), gg.uniform(0, 760)
    if abs(bx - 960) < 380:
        continue
    img = wc.wet_bloom(img, bx, by, gg.uniform(18, 46), [(0.95, 0.75, 0.85), (0.8, 0.8, 1.0), (1.0, 0.9, 0.8)][i % 3], seed=541 + i, density=0.35, tex=TEX)
    img = img + (ell_mask([(bx, by, 6, 6)]) * 0.35)[..., None]

S = 26
ANG = 5


def ruffle(x0, x1, ytop, ybot, waves, amp=0.35, flare=0.0):
    top = [(x0 + flare * 0.3, ytop), (x1 - flare * 0.3, ytop)]
    bot = []
    for i in range(waves * 8 + 1):
        u = i / (waves * 8)
        xx = x1 + flare - (x1 - x0 + 2 * flare) * u
        bot.append((xx, ybot + amp * math.sin(u * waves * 2 * math.pi)))
    return top + bot

bodice = [(0.0, 5.6), (16.0, 5.6), (16.2, 8.4), (-0.2, 8.4)]
tiers = [ruffle(-0.4, 16.4, 7.8, 9.0, 6, 0.22, 0.6), ruffle(-0.8, 16.8, 8.8, 10.0, 7, 0.22, 1.2), ruffle(-1.2, 17.2, 9.8, 10.9, 8, 0.22, 1.8)]
dress = [bodice] + tiers
trail = [(16.6, 8.4), (18.8, 9.0), (20.8, 10.2), (22.2, 11.4), (21.4, 11.9), (20.6, 11.5), (19.8, 12.0), (19.0, 11.5), (18.2, 11.8), (17.4, 10.8)]
boots = None                                    # 腿保持原样，只把腿涂成银色亮片靴
arm_chest = rigid_arm('R', -35)                 # 右臂整块向内下压，贴近身体
check_proportions(arms=[arm_chest], extra={"dress": dress, "trail": [trail]})
cm, (ox, oy), (pw, ph) = clawd_masks(S, 40, bbox=(-4, -1, 24, 14), arms=[(-1, 3, 0, 5), arm_chest], legs=boots,
                                     extra={"dress": dress, "tiers": tiers[1:], "trail": [trail]})
CX, CY = 960, 801
# 地面
floor_m = np.clip((y - 930) / 30, 0, 1)
img = wc.wash(img, floor_m, (0.25, 0.22, 0.4), 0.75, seed=545, bleed=6, edge=0.1, tex=TEX)
img = img + (np.exp(-(((x - 960) / 330) ** 2 + ((y - 965) / 40) ** 2)))[..., None] * (np.array([1, 0.97, 0.98]) - img) * 0.6
B = place_mask(cm["body"], CX, CY, ANG)
A = place_mask(cm["arms"], CX, CY, ANG)
E = place_mask(cm["eyes"], CX, CY, ANG)
D = place_mask(cm["dress"], CX, CY, ANG)
T = place_mask(cm["tiers"], CX, CY, ANG)
TR = place_mask(cm["trail"], CX, CY, ANG)
L = place_mask(cm["legs"], CX, CY, ANG)
# 先把人物区域擦亮（水彩里留白），再上色
fig = np.clip(B + A + D + TR + L, 0, 1)
img = img + blur(fig, 2)[..., None] * (np.array([0.98, 0.97, 0.97]) - img) * 0.85
img = wc.wash(img, np.clip(B + A - D, 0, 1), ORANGE_WC, 0.85, seed=550, bleed=1.2, tex=TEX)
img = wc.wash(img, E, (0.12, 0.1, 0.12), 0.95, seed=551, bleed=0.6, edge=0.2, tex=TEX)
# 银色亮片裙：淡紫灰 + 层叠荷叶边更深
img = wc.wash(img, D, (0.80, 0.78, 0.88), 0.7, seed=552, bleed=1.5, edge=0.8, tex=TEX)
img = wc.wash(img, T, (0.74, 0.72, 0.86), 0.45, seed=553, bleed=1.2, edge=1.0, tex=TEX)
img = wc.wash(img, np.clip(T - shift(T, -10, 0), 0, 1), (0.6, 0.58, 0.76), 0.6, seed=5531, bleed=1.5, tex=TEX)
img = wc.wash(img, TR, (0.86, 0.84, 0.92), 0.55, seed=554, bleed=2, edge=0.9, tex=TEX)
img = wc.wash(img, L, (0.78, 0.78, 0.86), 0.75, seed=555, bleed=1.2, edge=0.8, tex=TEX)
# 亮片：白色水粉点 + 少量蓝紫点
img = wc.gouache_dots(img, np.clip(D + L, 0, 1), 520, seed=556, r=(1.2, 3.0))
img = wc.gouache_dots(img, np.clip(D + L, 0, 1), 120, seed=557, r=(1.0, 2.2), color=(0.72, 0.74, 0.95))
# 粉色立体玫瑰：左肩（左臂处）、右腰、裙摆
def gp(gxv, gyv):
    """格坐标 → 画布坐标（含旋转）。"""
    sx_, sy_ = ox + gxv * S - pw / 2, oy + gyv * S - ph / 2
    c, s_ = math.cos(math.radians(-ANG)), math.sin(math.radians(-ANG))
    return CX + sx_ * c - sy_ * s_, CY + sx_ * s_ + sy_ * c
for i, (rgx, rgy, rr) in enumerate([(-0.4, 5.8, 1.7), (15.2, 8.2, 1.5), (1.2, 10.0, 1.1), (20.4, 10.8, 1.1), (1.4, 5.0, 0.9), (16.6, 9.6, 0.9)]):
    px_, py_ = gp(rgx, rgy)
    r_ = rr * S
    img = wc.wet_bloom(img, px_, py_, r_, (0.92, 0.55, 0.66), seed=560 + i, density=0.85, tex=TEX)
    spiral = [[(px_ + r_ * 0.75 * (1 - t_ / 30) * math.cos(t_ * 0.55), py_ + r_ * 0.75 * (1 - t_ / 30) * math.sin(t_ * 0.55)) for t_ in range(30)]]
    img = wc.wash(img, line_mask(spiral, 3), (0.7, 0.3, 0.42), 0.7, seed=570 + i, bleed=0.6, tex=TEX)
    leaf = [(px_ - r_ * 1.2, py_ + r_ * 0.4), (px_ - r_ * 0.4, py_ + r_ * 0.2), (px_ - r_ * 0.6, py_ + r_ * 0.8)]
    img = wc.wash(img, poly_mask([leaf]), (0.55, 0.72, 0.6), 0.6, seed=580 + i, bleed=1, tex=TEX)
# 按在胸前的钳子
# 人物在追光里的柔光
img = img + blur(fig, 30)[..., None] * 0.04
img = img * (0.97 + 0.03 * TEX[..., None])
save(img, os.path.join(OUT, "05-b_我管不住我自己_银色亮片裙.png"))
print("05-b ok")
