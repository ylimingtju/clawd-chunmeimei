"""00 开场动画（0:00.00–0:08.00，帧 0–239）。
00-1 0.0–3.0s：三维 Clawd 背对镜头看展，歪了一下身子，抬起一只钳子像想摸画；画里的万花筒静止、星芒闪烁。
00-2 3.0–4.5s：万花筒开始旋转产生吸力，Clawd 被拉离地面，一边飞向画框一边转身、越来越小；
               飞到画框平面时正好转成正面、大小等于正下方的空位，填进去变成 2D；展签露出来。
00-3 4.5–8.0s：万花筒凑齐，五只 Clawd 一起举起钳子，整个万花筒旋转 30°。
画框外的相机与 kit3d / v4_开场视角示意 相同。"""
import math

import numpy as np
from scipy.ndimage import map_coordinates
from scipy import ndimage as nd

from kit import K, PW, PH, X, Y, M, circle, clawd, garment, g2p, rigid_arm, check_proportions, gauss, fbm, smooth, over, downsample, ishift
from common4 import W, H
import anim_core as A
import kit3d
import frame

FPS = 30
WALL = (0.20, 0.15, 0.22)
LABEL = "纯妹妹 × Clawd"
CX, CY = 560, 315
ORANGE = np.array([202, 124, 94]) / 255
PAL = np.array([[0.96, 0.16, 0.30], [1.0, 0.45, 0.60], [1.0, 0.66, 0.20], [1.0, 0.90, 0.32], [0.30, 0.85, 0.80],
                [0.25, 0.55, 0.95], [0.58, 0.36, 0.95], [0.30, 0.88, 0.58], [0.96, 0.16, 0.30]], np.float32)
N1 = fbm(512, 512, 18, 1, 4)
EDGE_N = fbm(PH, PW, 40, 7, 2) > 0.45
G = np.random.default_rng(0)
SPARKS = [(G.uniform(0, 2 * math.pi), G.uniform(60, 600), G.uniform(6, 22), G.uniform(0, 0.4), G.uniform(0, 2 * math.pi)) for _ in range(26)]

# 画框平面上的空位：画心 (560, 577) 处 S=10.5 的 Clawd。按相机反算它在三维里的位置
CAM_Z = 39.0
SLOT_S, SLOT_RF = 10.5, 262
Z_WALL = CAM_Z - kit3d.F / SLOT_S                      # 画框平面（墙）的 z
SLOT_FEET_Y = kit3d.EYE_Y + (kit3d.CYH - (frame.INNER[1] + CY + SLOT_RF)) * (CAM_Z - Z_WALL) / kit3d.F

T1, T2, T3 = 3.0, 4.5, 8.0


# ---------- 画心：万花筒 ----------
def kaleido(phi=0.0, swirl=0.0, cx=CX, cy=CY, k=1.0, flow=0.0):
    """十瓣万花筒。phi：整体转角（弧度）；swirl：额外的漩涡扭转（越靠中心越强）；k：内容缩放；flow：液体缓慢流动。"""
    x, y = (X / K - cx) / k, (Y / K - cy) / k
    r = np.hypot(x, y)
    th = np.arctan2(y, x) - phi - swirl * np.exp(-r / 380)
    seg = 2 * np.pi / 10
    tk = np.abs(((th % seg) / seg) - 0.5) * seg
    u, v = r * np.cos(tk) / 3 + 60, r * np.sin(tk) / 3 + 60
    sw = r / 300 + flow
    u2, v2 = u * np.cos(sw) - v * np.sin(sw) + 200, u * np.sin(sw) + v * np.cos(sw) + 200
    n = map_coordinates(N1, [v2 % 512, u2 % 512], order=1)
    hue = (0.08 + 0.55 * np.clip(r / 520, 0, 1.2) + 0.7 * (n - 0.5)) % 1.0
    q = hue * 10
    band = np.floor(q)
    inner = q - band
    i0 = np.floor((band / 10) * (len(PAL) - 1)).astype(int)
    col = PAL[i0] * (0.88 + 0.16 * inner[..., None])
    col = gauss(col, 0.6 * K)
    edge = np.zeros_like(hue)
    edge[:, 1:] = np.abs(band[:, 1:] - band[:, :-1]) > 0
    edge[1:, :] = np.maximum(edge[1:, :], np.abs(band[1:, :] - band[:-1, :]) > 0)
    edge = gauss(edge.astype(np.float32), 0.8 * K) * EDGE_N
    col = col + edge[..., None] * 0.28
    col = col * (0.85 + 0.25 * np.exp(-(r / 160) ** 2))[..., None] + np.exp(-(r / 70) ** 2)[..., None] * np.array([0.3, 0.25, 0.15])
    return col


def sparkle(img, x, y, s, rot=0.0):
    pts = []
    for i in range(8):
        a = rot + i * math.pi / 4
        rr = s if i % 2 == 0 else s * 0.14
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    return over(img, (1.0, 0.99, 0.95), M(pts))


def cover_clawd(img, theta_deg, arm=32.0, S=SLOT_S, Rf=SLOT_RF, cx=CX, cy=CY, k=1.0, mask=None):
    """一只头朝圆心的封面 Clawd（珍珠网上衣 + 橄榄金丝绸裙）。theta：从圆心指向它脚的方向；mask：只画在这块里。"""
    th = math.radians(theta_deg)
    fx, fy = cx + Rf * k * math.cos(th), cy + Rf * k * math.sin(th)
    S = S * k
    ang = theta_deg - 90
    arms = [rigid_arm("L", arm), rigid_arm("R", arm)]
    check_proportions(arms=arms)
    cm, em = clawd(fx, fy, S, arms, ang)
    mm = 1.0 if mask is None else mask
    img = over(img, ORANGE * 1.02, cm * mm)
    sm = garment([(-0.6, 6.8), (16.6, 6.8), (17.6, 11.9), (-1.6, 11.9)], fx, fy, S, ang, arms=arms, name="skirt")
    sheen = 0.5 + 0.5 * np.sin(((X / K - fx) * math.cos(math.radians(ang)) + (Y / K - fy) * math.sin(math.radians(ang))) / S * 2.2)
    img = over(img, np.array([0.56, 0.50, 0.20])[None, None] * (0.8 + 0.35 * sheen)[..., None], sm * mm)
    pearls = np.zeros((PH, PW), np.float32)
    for j, gy in enumerate(np.arange(5.3, 7.2, 0.48)):
        for gx in np.arange(0.5, 16.0, 0.62):
            (px, py), = g2p([(gx + (0.31 if j % 2 else 0), gy)], fx, fy, S, ang)
            pearls = np.maximum(pearls, M(circle(px, py, 0.21 * S, 12)))
    pearls = pearls * mm
    img = over(img, (0.99, 0.97, 0.93), pearls)
    img = img * (1 - 0.18 * np.clip(pearls - ishift(pearls, -2, -2), 0, 1))[..., None]
    img = over(img, (0.06, 0.05, 0.06), em * mm)
    return img


def painting(t, filled, phi, swirl, ring_rot, arm, flow=0.0):
    img = kaleido(phi, swirl, flow=flow)
    for (a, r, s, rot, ph) in SPARKS:                       # 星芒随整体转动，各自按自己的节奏闪
        a2 = a + math.radians(ring_rot)
        x, y = CX + r * math.cos(a2), CY + r * math.sin(a2) * 0.62
        tw = 0.65 + 0.35 * math.sin(t * 2.6 + ph)
        img = sparkle(img, x, y, s * tw, rot + math.radians(ring_rot))
    for th in (90 + 72 * k for k in range(1, 5)):
        img = cover_clawd(img, th + ring_rot, arm)
    if filled:
        img = cover_clawd(img, 90 + ring_rot, arm)
    pr = np.hypot(X / K - CX + 6, Y / K - CY + 6) / 28
    glow = 0.35 + 0.25 * smooth(4.9, 5.3, t) * (1 - smooth(5.6, 7.0, t))  # 举钳那一下中心珍珠亮一下
    img = img + np.exp(-(np.hypot(X / K - CX, Y / K - CY) / 60) ** 2)[..., None] * glow
    img = over(img, np.array([1.0, 0.98, 0.95])[None, None] * (1 - 0.2 * smooth(0, 1.4, pr))[..., None], M(circle(CX, CY, 26, 64)))
    return np.clip(downsample(img, K), 0, 1)


# ---------- 画框外：三维 Clawd ----------
def _corners(b0, b1):
    return np.array([(x, y, z) for x in (b0[0], b1[0]) for y in (b0[1], b1[1]) for z in (b0[2], b1[2])], np.float64)


def _rot_z(P, pivot, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    Q = P - np.asarray(pivot)
    return np.stack([Q[:, 0] * c - Q[:, 1] * s, Q[:, 0] * s + Q[:, 1] * c, Q[:, 2]], -1) + np.asarray(pivot)


def posed_boxes(lean=0.0, arm_r=0.0, arm_l=0.0, lift=0.0):
    """kit3d 的 Clawd 体块：手臂绕与身体相接处整块旋转（正值外端向上），整个身体绕脚底中心左右歪（正值向 +x 歪）。"""
    out = []
    for b0, b1, name in kit3d.clawd_boxes():
        P = _corners(b0, b1)
        if name == "body" and b0[0] >= 16 and b1[1] - b0[1] == 2:          # 右臂（x 16–17）
            P = _rot_z(P, (16, 8, 0), arm_r)
        elif name == "body" and b1[0] <= 0 and b1[1] - b0[1] == 2:         # 左臂（x -1–0）
            P = _rot_z(P, (0, 8, 0), -arm_l)
        P = _rot_z(P, (8, 0, 0), -lean)
        P[:, 1] += lift
        out.append((P, None, name))
    return out


def viewer(full, painting_img, boxes, yaw, x, z, flat=0.0):
    """flat：0..1，越接近画框平面越趋近画里 2D 的平涂橙色（到达时与空位里的 2D Clawd 一致，不跳色）。"""
    cols = {"body": kit3d.ORANGE * (1 - flat) + ORANGE * 1.02 * flat, "eye": np.array([0.05, 0.04, 0.05])}
    amb = 0.40 + 0.60 * flat                                     # 环境光拉满 = 不再有明暗面，等于平涂
    rgb, a = kit3d.draw(boxes, yaw, (x, z), CAM_Z, colors=cols, light=(-0.35, 0.75, 0.55), ambient=amb)
    if a.max() <= 0:
        return full
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    ys = np.nonzero(a.max(1) > 0.5)[0]
    top_y, bot_y = (ys.min(), ys.max()) if len(ys) else (0, H)
    span = max(60.0, (bot_y - top_y) * 0.5)
    col = rgb * (1.0 - 0.25 * (1 - flat) * np.clip((yy - top_y) / span, 0, 1))[..., None]
    er = max(2, int(round(7 * min(1.0, span / 230))))
    core = nd.grey_erosion(a, size=(er, er))
    edge = np.clip(a - core, 0, 1) * a
    tint = np.clip(gauss(full, 30) * 1.4, 0, 1)                 # 画里的彩光沿身体边缘溢出来
    e = 0.65 * (1 - flat)
    col = col * (1 - e * edge[..., None]) + tint * e * edge[..., None]
    return over(full, np.clip(col, 0, 1), a)


def frame_at(i):
    t = i / FPS
    # 画心参数
    if t < T1:
        phi, swirl, rot, arm, filled = 0.0, 0.0, 0.0, 0.0, False
    elif t < T2:
        s = A.seg(t, T1, T2)
        phi, swirl, rot, arm, filled = 0.9 * s ** 2, 1.6 * A.ease_io(s), 0.0, 0.0, False
    else:
        s = A.seg(t, T2, T3)
        relax = A.ease_out(A.seg(t, T2, T2 + 1.2))
        rot = 30 * A.ease_io(A.seg(t, T2 + 0.3, T3))
        phi = 0.9 + 0.6 * A.ease_out(A.seg(t, T2, T2 + 1.0)) + math.radians(rot)   # 先顺着余势再转一点，然后和圆环一起转
        swirl = 1.6 * (1 - relax)
        up = A.seg(t, 4.85, 5.25)
        arm = 32 * A.back_out(up, 2.2) if up > 0 else 0.0
        filled = True
    p = painting(t, filled, phi, swirl, rot, arm, flow=0.05 * max(0.0, t - T1))
    full = A.compose(p, WALL, LABEL)
    # 画框外的三维 Clawd
    if t < T1:
        lean = 5 * A.ease_io(A.seg(t, 0.6, 1.3)) - 2 * A.ease_io(A.seg(t, 1.3, 1.8))
        reach = A.ease_io(A.seg(t, 1.3, 2.2))
        arm_r = 80 * reach + 6 * math.sin(max(0.0, t - 2.2) * 9) * A.seg(t, 2.2, 2.4)
        boxes = posed_boxes(lean, arm_r, 0, 0.03 * math.sin(t * 2.2))
        return viewer(full, p, boxes, 0.0, 8.0, -0.6 * reach)
    if t < T2:
        s = A.seg(t, T1, T2)
        u = A.ease_in(A.seg(s, 0.08, 1.0), 2.0)
        hopz = 1.2 * math.sin(math.pi * A.seg(s, 0.0, 0.25))          # 先被拔离地面一下
        lift = hopz + (SLOT_FEET_Y) * u
        z = -0.6 + (Z_WALL - 4.0 + 0.6) * u                            # 正面（离相机最近的一面）到达画框平面
        yaw = 180 * A.ease_io(A.seg(s, 0.12, 0.92))
        boxes = posed_boxes(3 * (1 - A.ease_io(s)), 80 * (1 - A.ease_io(A.seg(s, 0, 0.6))), 25 * math.sin(math.pi * A.seg(s, 0.2, 0.9)), lift)
        return viewer(full, p, boxes, yaw, 8.0, z, flat=A.ease_in(A.seg(s, 0.55, 0.98), 1.5))
    return full


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a00_{i:04d}.png")
