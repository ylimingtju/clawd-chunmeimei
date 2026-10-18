"""00 开场 · 样张两张：
  00-1 三维 Clawd 背对镜头看展；画里是 Clawd 版专辑封面（万花筒），正下方的位置空着
  00-3 它被吸进画里、填上空位之后；展签「纯妹妹 × Clawd」露出来
画心是 2D 平面：放射对称的虹彩液体（分色成几档平涂）、星芒；四只（或五只）Clawd 头朝圆心、钳子举向中间，
穿珍珠网上衣和橄榄金丝绸裙。画框外的 Clawd 用三维光线投射，移轴相机与 v4_开场视角示意 相同。
用法：python S00_开场.py 输出目录"""
import math
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import clawd3d as c3
from kit import (K, PW, PH, X, Y, M, SM, circle, rot_pts, clawd, garment, g2p, rigid_arm, check_proportions,
                 gauss, fbm, value_noise, smooth, over, downsample, ishift, outline, dilate)
from common4 import W, H, to_px, rect_pts
import frame

G = np.random.default_rng(0)
CX, CY = 560, 315
ORANGE = np.array([202, 124, 94]) / 255
PAL = np.array([[0.96, 0.16, 0.30], [1.0, 0.45, 0.60], [1.0, 0.66, 0.20], [1.0, 0.90, 0.32], [0.30, 0.85, 0.80],
                [0.25, 0.55, 0.95], [0.58, 0.36, 0.95], [0.30, 0.88, 0.58], [0.96, 0.16, 0.30]], np.float32)


def kaleido_bg():
    x, y = X / K - CX, Y / K - CY
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    seg = 2 * np.pi / 10
    tk = np.abs(((th % seg) / seg) - 0.5) * seg
    N = 512
    n1 = fbm(N, N, 18, 1, 4)
    u, v = r * np.cos(tk) / 3 + 60, r * np.sin(tk) / 3 + 60
    sw = r / 300
    u2, v2 = u * np.cos(sw) - v * np.sin(sw) + 200, u * np.sin(sw) + v * np.cos(sw) + 200
    n = map_coordinates(n1, [v2 % N, u2 % N], order=1)
    hue = (0.08 + 0.55 * np.clip(r / 520, 0, 1.2) + 0.7 * (n - 0.5)) % 1.0
    q = hue * 10
    band = np.floor(q)                                  # 分色成平涂的色带，每条带里留一点渐变
    inner = q - band
    t = (band / 10) * (len(PAL) - 1)
    i0 = np.floor(t).astype(int)
    col = PAL[i0] * (0.88 + 0.16 * inner[..., None])
    col = gauss(col, 0.6 * K)
    # 色带交界处一道细细的白色液面反光（像真封面里那种液体高光）
    edge = np.zeros_like(hue)
    edge[:, 1:] = np.abs(band[:, 1:] - band[:, :-1]) > 0
    edge[1:, :] = np.maximum(edge[1:, :], np.abs(band[1:, :] - band[:-1, :]) > 0)
    edge = gauss(edge.astype(np.float32), 0.8 * K) * (fbm(PH, PW, 40, 7, 2) > 0.45)
    col = col + edge[..., None] * 0.28
    # 中心一圈暖光
    col = col * (0.85 + 0.25 * np.exp(-(r / 160) ** 2))[..., None] + np.exp(-(r / 70) ** 2)[..., None] * np.array([0.3, 0.25, 0.15])
    return col


def sparkle(img, x, y, s, rot=0.0):
    pts = []
    for i in range(8):
        a = rot + i * math.pi / 4
        rr = s if i % 2 == 0 else s * 0.14
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    return over(img, (1.0, 0.99, 0.95), M(pts))


def cover_clawd(img, theta_deg, S=10.5, Rf=262):
    """一只头朝圆心的 Clawd：theta 为从圆心指向它脚的方向。"""
    th = math.radians(theta_deg)
    fx, fy = CX + Rf * math.cos(th), CY + Rf * math.sin(th)
    ang = theta_deg - 90
    arms = [rigid_arm("L", 32), rigid_arm("R", 32)]
    check_proportions(arms=arms)
    cm, em = clawd(fx, fy, S, arms, ang)
    img = over(img, ORANGE * 1.02, cm)
    skirt = [(-0.6, 6.8), (16.6, 6.8), (17.6, 11.9), (-1.6, 11.9)]
    sm = garment(skirt, fx, fy, S, ang, arms=arms, name="skirt")
    sheen = 0.5 + 0.5 * np.sin(((X / K - fx) * math.cos(math.radians(ang)) + (Y / K - fy) * math.sin(math.radians(ang))) / S * 2.2)
    img = over(img, np.array([0.56, 0.50, 0.20])[None, None] * (0.8 + 0.35 * sheen)[..., None], sm)
    # 珍珠网上衣：身体上半部一格格的小珍珠
    pearls = np.zeros((PH, PW), np.float32)
    for j, gy in enumerate(np.arange(5.3, 7.2, 0.48)):
        for gx in np.arange(0.5, 16.0, 0.62):
            (px, py), = g2p([(gx + (0.31 if j % 2 else 0), gy)], fx, fy, S, ang)
            pearls = np.maximum(pearls, M(circle(px, py, 0.21 * S, 12)))
    img = over(img, (0.99, 0.97, 0.93), pearls)
    img = img * (1 - 0.18 * np.clip(pearls - ishift(pearls, -2, -2), 0, 1))[..., None]
    img = over(img, (0.06, 0.05, 0.06), em)
    return img


def painting(filled):
    img = kaleido_bg()
    for i in range(26):
        a, r = G.uniform(0, 2 * math.pi), G.uniform(60, 600)
        x, y = CX + r * math.cos(a), CY + r * math.sin(a) * 0.62
        img = sparkle(img, x, y, G.uniform(6, 22), G.uniform(0, 0.4))
    for th in (90 + 72 * k for k in range(1, 5)):
        img = cover_clawd(img, th)
    if filled:
        img = cover_clawd(img, 90)
    # 五只钳子举向中心：中间一颗大珍珠
    pr = np.hypot(X / K - CX + 6, Y / K - CY + 6) / 28
    img = img + np.exp(-(np.hypot(X / K - CX, Y / K - CY) / 60) ** 2)[..., None] * 0.35
    img = over(img, np.array([1.0, 0.98, 0.95])[None, None] * (1 - 0.2 * smooth(0, 1.4, pr))[..., None], M(circle(CX, CY, 26, 64)))
    return np.clip(downsample(img, K), 0, 1)


def viewer(full, yaw=0.0, x=8.0, z=0.0, lift=0.0):
    """画框外的三维 Clawd（kit3d：按面着色的矢量体块）。画里的彩光从身体边缘溢出来，勾出一圈彩色的边。"""
    import kit3d
    from scipy import ndimage as _nd
    boxes = kit3d.clawd_boxes()
    if lift:
        boxes = [((b0[0], b0[1] + lift, b0[2]), (b1[0], b1[1] + lift, b1[2]), n) for b0, b1, n in boxes]
    rgb, a = kit3d.draw(boxes, yaw, (x, z), 39.0, light=(-0.35, 0.75, 0.55), ambient=0.40)
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    ys = np.nonzero(a.max(1) > 0.5)[0]
    top_y = ys.min() if len(ys) else 0
    col = rgb * (1.0 - 0.25 * np.clip((yy - top_y) / 230, 0, 1))[..., None]
    core = _nd.grey_erosion(a, size=(7, 7))
    edge = np.clip(a - core, 0, 1) * a
    tint = np.clip(gauss(full, 30) * 1.4, 0, 1)
    col = col * (1 - 0.65 * edge[..., None]) + tint * 0.65 * edge[..., None]
    wide = gauss(np.clip(a - _nd.grey_erosion(a, size=(41, 41)), 0, 1), 6) * a
    col = col + (wide * 0.10)[..., None] * tint
    return over(full, col, a)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    wall = (0.20, 0.15, 0.22)
    p1 = painting(False)
    f1 = frame.compose(p1, wall, "纯妹妹 × Clawd")
    frame.save(viewer(f1), os.path.join(out_dir, "00-1_开场_看展.png"))
    p3 = painting(True)
    frame.save(frame.compose(p3, wall, "纯妹妹 × Clawd"), os.path.join(out_dir, "00-3_开场_封面凑齐.png"))
    frame.save(p3, os.path.join(out_dir, "00_画心.png"))
    # 结尾：Clawd 被放出来，转成侧身朝右走出镜头；画里正下方又空了出来
    frame.save(viewer(f1, yaw=90, x=17.0, z=-2.0), os.path.join(out_dir, "11_结尾_侧身走出.png"))
