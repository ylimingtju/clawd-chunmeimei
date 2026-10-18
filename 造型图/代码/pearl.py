"""珍珠 / 映着世界的珠子：球面法线 + 菲涅尔虹彩 + 高光；可把一张图按鱼眼映进珠子里。"""
import math
import numpy as np
from PIL import Image
from common import hsv2rgb, blur, to_arr


def pearl(r, light=(-0.45, -0.6, 0.66), base=(0.93, 0.9, 0.86), world=None, world_mix=0.0, ss=2, ambient=0.1):
    """返回 (2r+4)×(2r+4) 的 RGBA。world：HxWx3 图，按球面折射映入；world_mix：世界图可见程度。"""
    n = int(2 * r + 4) * ss
    c = n / 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    nx, ny = (xx - c) / (r * ss), (yy - c) / (r * ss)
    d2 = nx ** 2 + ny ** 2
    inside = d2 <= 1
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    fres = (1 - nz) ** 2.2
    # 珍珠层的虹彩：随视角变化的色相（粉、绿、蓝）
    hue = (0.85 + 0.55 * fres + 0.08 * nx - 0.05 * ny) % 1
    iri = hsv2rgb(hue, np.full_like(hue, 0.35), np.ones_like(hue))
    wrap = np.clip((nx * L[0] + ny * L[1] + nz * L[2] + 0.35) / 1.35, 0, 1)   # 包裹光，过渡更柔
    col = np.array(base)[None, None] * (ambient + 0.9 * wrap[..., None] ** 1.6) + iri * fres[..., None] * (0.25 + 0.4 * wrap[..., None])
    if world is not None and world_mix > 0:
        h, w = world.shape[:2]
        # 鱼眼：中心放大、边缘压缩，并左右/上下翻转（透过球看到的是倒像）
        rr = np.sqrt(d2) + 1e-6
        k = np.arcsin(np.clip(rr, 0, 1)) / (math.pi / 2) / rr
        u = np.clip(0.5 - nx * k * 0.5, 0, 0.999)
        v = np.clip(0.5 - ny * k * 0.5, 0, 0.999)
        wv = world[(v * h).astype(int), (u * w).astype(int)]
        col = col * (1 - world_mix * nz[..., None] ** 0.6) + wv * world_mix * nz[..., None] ** 0.6 * (ambient + 1.1 * wrap[..., None] ** 1.6)
        col = col + iri * fres[..., None] * 0.35
    # 高光：主高光 + 底部的小反光
    hx, hy = L[0] * 0.55, L[1] * 0.55
    spec = np.exp(-(((nx - hx) / 0.13) ** 2 + ((ny - hy) / 0.09) ** 2)) * 1.1 + np.exp(-(((nx - hx) / 0.35) ** 2 + ((ny - hy) / 0.28) ** 2)) * 0.18
    spec += np.exp(-(((nx - 0.3) / 0.28) ** 2 + ((ny - 0.74) / 0.07) ** 2)) * 0.15
    col = col + spec[..., None]
    # 轮廓暗边
    col = col * (1 - 0.35 * fres[..., None] ** 2)
    a = np.clip((1 - np.sqrt(d2)) * r * ss, 0, 1)
    rgba = np.dstack([col, a]).astype(np.float32)
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").resize((n // ss, n // ss), Image.LANCZOS)
    out = to_arr(im)
    # resize 后的 alpha 已经是 0..1；颜色允许略超 1（高光），这里把高光单独加回
    return out
