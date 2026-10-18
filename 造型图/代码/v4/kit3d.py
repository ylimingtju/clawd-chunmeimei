"""画框外的三维 Clawd：长方体按面投影、背面剔除、按深度排序后用 skia 矢量绘制（边缘锐利，带面光照）。
相机与 v4_开场视角示意.py 相同：移轴透视，F=1484（像素），视平线在 y=1020，相机高 EYE_Y 格，朝 -Z 看。
Clawd 自身坐标沿用 clawd3d：X 右、Y 上、正面朝 -Z（脚底中心在 (8, 0, 0)）。"""
import math
import os
import sys

import numpy as np
import skia

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import clawd3d as c3

W, H = 1920, 1080
F, CYH, CXH, EYE_Y = 1484.0, 1020.0, 960.0, 8.0
ORANGE = np.array([202, 124, 94]) / 255
OVERLAY = {"eye", "glasses", "deco"}
FACES = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)), ((0, 4, 5, 1), (0, -1, 0)),
         ((2, 3, 7, 6), (0, 1, 0)), ((0, 2, 6, 4), (0, 0, -1)), ((1, 5, 7, 3), (0, 0, 1))]


def corners(bmin, bmax):
    return np.array([(x, y, z) for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])], np.float64)


def draw(boxes, yaw_deg, pos, cam_z, colors=None, light=(-0.4, 0.8, 0.45), ambient=0.42, ss=2):
    """boxes：clawd3d 格式；yaw 为 Clawd 绕自身竖轴转的角度（0=背对镜头、面朝墙；90=面朝画面右侧；180=面朝镜头）；
    pos=(x, z)：脚底中心在世界里的位置（相机在 x=8 正后方、z=cam_z）。返回 (rgb, alpha)，H×W。"""
    colors = colors or {"body": ORANGE, "eye": np.array([0.05, 0.04, 0.05])}
    a = math.radians(yaw_deg)
    ca, sa = math.cos(a), math.sin(a)
    L = np.array(light, np.float64)
    L /= np.linalg.norm(L)

    def xf(p):
        # clawd3d 的正面朝 -Z；相机在 +Z，所以 yaw=0 时就是背对镜头。yaw 为正时正面转向画面右侧
        x, y, z = p[..., 0] - 8, p[..., 1], p[..., 2]
        xr, zr = x * ca - z * sa, x * sa + z * ca
        return np.stack([xr + pos[0], y, zr + pos[1]], -1)

    cam = np.array([8.0, EYE_Y, cam_z])
    polys = []
    for bmin, bmax, name in boxes:
        # 盒子可以是轴对齐的 (min, max)，也可以直接给 8 个角点（bmax=None），用于整块旋转的手臂、倾斜的身体
        C = np.asarray(bmin, np.float64) if bmax is None else corners(bmin, bmax)
        P = xf(C)
        ctr = P.mean(0)
        for idx, _n in FACES:
            q = P[list(idx)]
            nw = np.cross(q[1] - q[0], q[3] - q[0])
            nn = np.linalg.norm(nw)
            if nn < 1e-9:
                continue
            nw = nw / nn
            c = q.mean(0)
            if np.dot(nw, c - ctr) < 0:
                nw = -nw
            if np.dot(nw, cam - c) <= 1e-6:
                continue
            d = np.linalg.norm(cam - c)
            shade = ambient + (1 - ambient) * max(0.0, float(np.dot(nw, L)))
            col = np.asarray(colors.get(name, ORANGE)) * shade
            scr = [(CXH + F * (v[0] - 8) / (cam_z - v[2]) * 1.0, CYH - F * (v[1] - EYE_Y) / (cam_z - v[2])) for v in q]
            polys.append((d - (6.0 if name in OVERLAY else 0.0), scr, col, name))   # 贴在表面上的薄块（眼睛、饰物）总画在上面
    polys.sort(key=lambda t: -t[0])
    surf = skia.Surface(W * ss, H * ss)
    cv = surf.getCanvas()
    cv.clear(skia.ColorTRANSPARENT)
    for d, scr, col, name in polys:
        p = skia.Paint(AntiAlias=True, Color=skia.Color4f(float(col[0]), float(col[1]), float(col[2]), 1.0))
        path = skia.Path()
        path.moveTo(scr[0][0] * ss, scr[0][1] * ss)
        for x, y in scr[1:]:
            path.lineTo(x * ss, y * ss)
        path.close()
        cv.drawPath(path, p)
        # 相邻面之间留一条极细的暗缝，让体块清楚
        e = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.2 * ss,
                       Color=skia.Color4f(float(col[0]) * 0.8, float(col[1]) * 0.8, float(col[2]) * 0.8, 0.6))
        cv.drawPath(path, e)
    arr = surf.makeImageSnapshot().toarray().astype(np.float32) / 255
    arr = arr.reshape(H, ss, W, ss, 4).mean(axis=(1, 3))
    alpha = arr[..., 3]
    rgb = np.where(alpha[..., None] > 1e-4, arr[..., :3] / np.maximum(alpha[..., None], 1e-4), 0)
    return rgb, alpha


def clawd_boxes(depth=8.0):
    return c3.clawd_boxes(depth=depth)
