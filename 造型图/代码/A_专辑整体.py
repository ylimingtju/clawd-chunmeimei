"""A｜00 专辑整体 · 珍珠万花筒
亮面珐琅潮玩质感；五只 Clawd 放射排列；珍珠链网上衣 + 橄榄金丝绸阔腿裤；虹彩液体漩涡、星芒。"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from common import *

OUT = sys.argv[1]
g = rng(1)
x, y = grid()
cx, cy = W / 2, 500

# ---------- 背景：六重镜像的虹彩液体 ----------
r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
th = np.arctan2(y - cy, x - cx)
seg = 2 * np.pi / 6
tk = np.abs(((th % seg) / seg) - 0.5) * seg  # 镜像折叠
# 在折叠后的坐标里采样噪声，形成万花筒
N = 1024
noise = fbm(N, N, 11, octaves=6, base=4)
noise2 = fbm(N, N, 12, octaves=4, base=3)
u = (r * np.cos(tk)) / 2.2 + 40
v = (r * np.sin(tk)) / 2.2 + 40
swirl = r / 260.0
u2 = u * np.cos(swirl) - v * np.sin(swirl) + 300
v2 = u * np.sin(swirl) + v * np.cos(swirl) + 300
n1 = map_coordinates(noise, [v2 % N, u2 % N], order=1)
n2 = map_coordinates(noise2, [(v2 * 0.6 + 200) % N, (u2 * 0.6) % N], order=1)
R = 720.0
hue = (0.03 + 0.5 * np.clip(r / R, 0, 1.3) + 0.6 * (n1 - 0.5) + 0.12 * (n2 - 0.5)) % 1.0
sat = np.clip(0.75 + 0.3 * n2, 0, 1)
val = np.clip(0.55 + 0.4 * n1, 0, 1)
PAL = np.array([[0.96, 0.10, 0.22], [1.0, 0.35, 0.55], [1.0, 0.62, 0.15], [1.0, 0.9, 0.25],
                [0.98, 0.35, 0.45], [0.15, 0.85, 0.85], [0.2, 0.55, 0.95], [0.55, 0.3, 0.95],
                [0.2, 0.9, 0.6], [0.96, 0.10, 0.22]], np.float32)
t = hue * (len(PAL) - 1)
i0 = np.floor(t).astype(int); fr = (t - i0)[..., None]
bg = PAL[i0] * (1 - fr) + PAL[np.minimum(i0 + 1, len(PAL) - 1)] * fr
bg = bg * (0.55 + 0.5 * val[..., None])
# 液体高光：噪声等值线处的白色流线
band = np.abs(np.sin(n1 * 38 + n2 * 9))
liquid = np.clip((band - 0.975) / 0.025, 0, 1) * np.clip(r / 250, 0, 1)
bg = screen(bg, blur(liquid, 1.2)[..., None] * 0.45)
# 中心粉红-金色发光圆
core = np.exp(-(r / 330) ** 2)
bg = screen(bg, core[..., None] * np.array([1.0, 0.2, 0.25]) * 0.5)
bg = screen(bg, np.exp(-(r / 160) ** 2)[..., None] * np.array([1.0, 0.62, 0.08]) * 0.6)
# 四角的饱和红块（参考封面）
for (px, py) in [(0, 0), (W, 0), (0, H), (W, H)]:
    d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    bg = over(bg, np.array([0.95, 0.12, 0.18]), np.clip(1 - d / 380, 0, 1) ** 1.5 * 0.6)
img = bg.astype(np.float32)

# ---------- Clawd：珐琅 + 珍珠 + 金裤 ----------
S = 17
PAD = 30
# 阔腿裤：两只裤腿各包住一对腿，向下展开
pants = [
    [(-0.1, 7.6), (7.8, 7.6), (7.3, 11.6), (-0.7, 11.6)],
    [(8.2, 7.6), (16.1, 7.6), (16.7, 11.6), (8.7, 11.6)],
]
masks, (ox, oy), (pw, ph) = clawd_masks(S, PAD, extra={"pants": pants})
px_, py_ = np.mgrid[0:ph, 0:pw].astype(np.float32)[::-1]
gx = (px_ - ox) / S  # 格坐标
gy = (py_ - oy) / S

def enamel_sprite(seed):
    base = np.array(CLAWD_ORANGE, np.float32) / 255
    m = masks["all"]
    inner = edge_dist(m, 4)
    shade = 0.82 + 0.22 * (1 - gy / 13) + 0.08 * inner
    col = base[None, None] * shade[..., None]
    # 镜面高光（左上圆角带）
    spec = np.exp(-((gx - 4.0) / 2.2) ** 2 - ((gy - 1.0) / 0.45) ** 2)
    spec += 0.6 * np.exp(-((gx - 13.5) / 0.9) ** 2 - ((gy - 1.6) / 0.5) ** 2)
    col = screen(col, (spec * masks["body"])[..., None] * 0.75)
    # 底部反射色光（青色）
    col = col * (1 - 0.25 * (np.clip((gy - 6) / 4, 0, 1) * masks["body"]))[..., None]
    # 眼睛：黑色珐琅 + 小高光
    eye = masks["eyes"]
    col = over(col, np.array([0.04, 0.03, 0.05]), eye)
    eh = np.zeros_like(eye)
    for ex in (3.6, 11.4):
        eh += np.exp(-((gx - ex - 0.3) / 0.18) ** 2 - ((gy - 3.4) / 0.25) ** 2)
    col = screen(col, (eh * eye)[..., None] * 0.9)
    alpha = m.copy()

    # 金色丝绸阔腿裤
    pm = masks["pants"]
    folds = 0.5 + 0.5 * np.sin(gx * 1.7 + np.sin(gy * 0.9) * 0.8)
    sheen = 0.55 + 0.35 * folds + 0.25 * np.exp(-((gx - 5) / 1.2) ** 2)
    gold = np.array([0.70, 0.56, 0.12])[None, None] * sheen[..., None]
    gold = screen(gold, (np.clip(folds - 0.8, 0, 1) * 4)[..., None] * np.array([1, 0.9, 0.5]) * 0.5)
    col = over(col, gold, pm)
    alpha = np.maximum(alpha, pm)

    # 珍珠链网：上身多条下垂的链 + 交叉网格
    im = Image.new("RGBA", (pw * 2, ph * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pr = S * 0.21 * 2
    def pearl(px, py, rr):
        for k in range(6, 0, -1):
            t = k / 6
            c = int(150 + 105 * (1 - t))
            d.ellipse([px - rr * t, py - rr * t, px + rr * t, py + rr * t], fill=(c, c - 6, c - 14, 255))
        d.ellipse([px - rr * 0.55, py - rr * 0.65, px - rr * 0.05, py - rr * 0.15], fill=(255, 255, 255, 255))
    def P(gxv, gyv):
        return ((ox + gxv * S) * 2, (oy + gyv * S) * 2)
    # 颈部紧贴的几圈
    for row in range(2):
        yy = 0.45 + row * 0.5
        for i in range(17):
            t = i / 16
            xx = 4.2 + 7.6 * t
            sag = 0.35 * (1 - (2 * t - 1) ** 2)
            pearl(*P(xx, yy + sag), pr)
    # 下垂的长链
    for k in range(4):
        y0 = 1.8 + k * 0.85
        sag = 1.4 + k * 0.55
        n = 24
        for i in range(n):
            t = i / (n - 1)
            xx = -0.6 + 17.2 * t
            yy = y0 + sag * (1 - (2 * t - 1) ** 2)
            if yy > 7.8 or 2.6 < yy - 0 and (3.2 < xx < 4.9 or 11.0 < xx < 12.8) and yy < 5.4:
                continue
            pearl(*P(xx, yy), pr * (0.9 if k % 2 else 1.0))
    pim = im.resize((pw, ph), Image.LANCZOS)
    pa = to_arr(pim)
    col = over(col, pa[..., :3], pa[..., 3])
    alpha = np.maximum(alpha, pa[..., 3])
    # 眼睛重新压在珍珠上方（保证识别度）
    col = over(col, np.array([0.04, 0.03, 0.05]), eye)
    col = screen(col, (eh * eye)[..., None] * 0.9)
    return np.concatenate([col, alpha[..., None]], -1)

sprite = enamel_sprite(0)

# 背后辉光
positions = []
Rr = 360
for k in range(5):
    phi = 90 + k * 72  # y 向下坐标中的角度；90° = 正下方
    if k == 0:
        rad = Rr + 20
    else:
        rad = Rr
    px = cx + rad * math.cos(math.radians(phi))
    py = cy + rad * math.sin(math.radians(phi))
    rot = -(phi - 90)
    positions.append((px, py, rot, k))
glow = np.zeros((H, W), np.float32)
for px, py, rot, k in positions:
    glow += sprite_alpha_layer(sprite, px, py, rot)
glow = blur(np.clip(glow, 0, 1), 40)
img = screen(img, glow[..., None] * np.array([1.0, 0.8, 0.5]) * 0.45)
# 阴影
shadow = np.zeros((H, W), np.float32)
for px, py, rot, k in positions:
    shadow += sprite_alpha_layer(sprite, px + 14, py + 18, rot)
img = img * (1 - blur(np.clip(shadow, 0, 1), 14) * 0.45)[..., None]
# 后面的 4 只先画，正中下方的主角最后画
for px, py, rot, k in sorted(positions, key=lambda p: -p[3]):
    img = paste_rgba(img, sprite, px, py, rot)

# ---------- 星芒、碎闪、色散 ----------
light = np.zeros((H, W, 3), np.float32)
def starburst(sx, sy, size, intensity, rays=4, rot=0.0, tint=(1, 1, 1)):
    ext = int(size * 3.5)
    X0, X1 = max(int(sx - ext), 0), min(int(sx + ext), W)
    Y0, Y1 = max(int(sy - ext), 0), min(int(sy + ext), H)
    if X1 <= X0 or Y1 <= Y0:
        return
    xs, ys = x[Y0:Y1, X0:X1], y[Y0:Y1, X0:X1]
    d = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2) + 1
    ang = np.arctan2(ys - sy, xs - sx)
    core = np.exp(-(d / (size * 0.08)) ** 2) * 2.0
    ray = np.zeros_like(d)
    for i in range(rays):
        a = rot + i * np.pi / rays
        diff = np.abs(np.sin(ang - a))
        ray += np.exp(-(diff * d / (size * 0.012 + 1)) ** 2) * np.exp(-d / size)
    halo = np.exp(-(d / (size * 0.35)) ** 2) * 0.35
    light[Y0:Y1, X0:X1] += (core + ray + halo)[..., None] * np.array(tint) * intensity

stars = [(470, 190, 380, 0.9, 4, 0.3), (1480, 260, 330, 0.8, 6, 0.1), (250, 760, 280, 0.7, 4, 0.7),
         (1690, 860, 300, 0.7, 4, 0.2), (1250, 760, 140, 0.6, 4, 0.9)]
for sx, sy, sz, it, rays, ro in stars:
    starburst(sx, sy, sz, it, rays, ro)
for i in range(30):
    sx, sy = g.uniform(0, W), g.uniform(0, H)
    starburst(sx, sy, g.uniform(20, 70), g.uniform(0.3, 0.9), 4, g.uniform(0, 1))
img = img + light * 0.45
# 棱镜色散：红蓝通道向外错位
img = bloom(img, 0.92, 30, 0.3)
img = np.clip(img, 0, 1)
img = img * img * (3 - 2 * img) * 0.35 + img * 0.65  # 轻 S 曲线
sh = 3
ca = img.copy()
ca[..., 0] = np.roll(img[..., 0], sh, axis=1)
ca[..., 2] = np.roll(img[..., 2], -sh, axis=1)
img = ca
img = grain(img, 0.02, 7)
save(img, OUT)
print("ok", OUT)
