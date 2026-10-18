"""全片固定的展示层：纯色墙 + 金色雕花框（含投影）+ 展签。
画框由高度场程序生成（外圆珠线、卷草纹主带、珍珠串珠带、内斜面），按金属材质着色，结果缓存到 npz。"""
import math
import os

import numpy as np

from common4 import W, H, ishift, gauss, smooth, fbm, value_noise, label_layer, over, downsample

FRAME_OUT = (330, 130, 1590, 900)       # 外框（1260×770）
BORDER = 70                              # 框宽
INNER = (FRAME_OUT[0] + BORDER, FRAME_OUT[1] + BORDER, FRAME_OUT[2] - BORDER, FRAME_OUT[3] - BORDER)  # 画心 1120×630
PW, PH = INNER[2] - INNER[0], INNER[3] - INNER[1]
LABEL_CY = 960
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache_frame.npz")
SS = 3


def _height_field():
    fw, fh = (FRAME_OUT[2] - FRAME_OUT[0]) * SS, (FRAME_OUT[3] - FRAME_OUT[1]) * SS
    B = BORDER * SS
    Y, X = np.mgrid[0:fh, 0:fw].astype(np.float32)
    dl, dr, dt, db = X, fw - 1 - X, Y, fh - 1 - Y
    D = np.stack([dt, db, dl, dr])
    side = D.argmin(0)
    d = D.min(0)
    u = d / B                                            # 0 外沿 → 1 内沿
    s = np.where(side < 2, X, Y)                        # 沿边坐标
    L = np.where(side < 2, fw, fh).astype(np.float32)

    h = np.zeros_like(u)
    # 外沿平台与外圆珠线
    h += np.where(u < 0.05, 0.22 + 0.5 * u, 0)
    t = (u - 0.125) / 0.075
    h += np.where((u >= 0.05) & (u < 0.2), 0.25 + 0.55 * np.sqrt(np.clip(1 - t * t, 0, 1)), 0)
    h += np.where((u >= 0.2) & (u < 0.245), 0.28, 0)
    # 主带：拱起的弧面 + 卷草纹
    band = (u >= 0.245) & (u < 0.585)
    bu = (u - 0.245) / 0.34
    base = 0.45 + 0.28 * np.sin(np.pi * bu)
    period = 0.62 * B
    n_per = np.maximum(np.round(L / period), 1)
    a = (s / (L / n_per)) % 1.0                         # 每个纹样单元内 0..1
    # 单元关于中线镜像，保证对称、在角上收得住
    stem_b = 0.5 + 0.26 * np.sin(2 * np.pi * a)
    stem = np.exp(-((bu - stem_b) / 0.085) ** 2) * 0.36
    leaves = np.zeros_like(u)
    for k, (ac, up) in enumerate([(0.12, 1), (0.37, -1), (0.62, 1), (0.87, -1)]):
        da = (a - ac + 0.5) % 1.0 - 0.5
        bc = 0.5 + 0.26 * np.sin(2 * np.pi * ac) + up * 0.18
        ang = math.atan2(up * 0.6, 1.0)
        ca, sa = math.cos(ang), math.sin(ang)
        x1 = (da * ca + (bu - bc) * sa * 0.62)
        y1 = (-da * sa + (bu - bc) * ca * 0.62)
        r = (x1 / 0.19) ** 2 + (y1 / 0.095) ** 2
        lobe = np.clip(1 - r, 0, 1) ** 0.6
        vein = np.exp(-(y1 / 0.010) ** 2) * (np.abs(x1) < 0.17)
        lobes = lobe * (0.75 + 0.25 * np.cos(x1 * 60)) - vein * 0.25 * lobe
        leaves = np.maximum(leaves, lobes * 0.5)
    berry = np.zeros_like(u)
    for ac in (0.0, 0.5):
        da = (a - ac + 0.5) % 1.0 - 0.5
        bc = 0.5 + 0.26 * np.sin(2 * np.pi * ac)
        rr = (da / 0.07) ** 2 + ((bu - bc) / 0.16) ** 2
        berry = np.maximum(berry, np.sqrt(np.clip(1 - rr, 0, 1)) * 0.35)
    orn = np.maximum(np.maximum(stem, leaves), berry)
    h += np.where(band, base + orn, 0)
    # 槽 + 珍珠串珠带
    h += np.where((u >= 0.585) & (u < 0.62), 0.30, 0)
    pb = (u >= 0.62) & (u < 0.73)
    bead = 0.105 * B
    nb = np.maximum(np.round(L / bead), 1)
    pa = (s / (L / nb)) % 1.0 - 0.5
    pv = (u - 0.675) / 0.055
    rr = (pa / 0.46) ** 2 + pv ** 2
    h += np.where(pb, 0.30 + 0.42 * np.sqrt(np.clip(1 - rr, 0, 1)), 0)
    # 内斜面（凹弧）与内口小珠线
    sc = (u >= 0.73) & (u < 0.93)
    h += np.where(sc, 0.14 + 0.42 * np.clip((0.93 - u) / 0.2, 0, 1) ** 1.8, 0)
    t2 = (u - 0.965) / 0.035
    h += np.where((u >= 0.93) & (u <= 1.0), 0.10 + 0.14 * np.sqrt(np.clip(1 - t2 * t2, 0, 1)), 0)

    # 四角花饰：盖住主带的斜接缝
    rosette = np.zeros_like(u)
    for cx, cy in [(0, 0), (fw - 1, 0), (0, fh - 1), (fw - 1, fh - 1)]:
        px = np.abs(X - cx) - 0.415 * B
        py = np.abs(Y - cy) - 0.415 * B
        r = np.sqrt(px ** 2 + py ** 2) / (0.36 * B)
        th = np.arctan2(py, px)
        petals = (0.72 + 0.28 * np.cos(8 * th)) * np.sqrt(np.clip(1 - r ** 2, 0, 1))
        boss = np.sqrt(np.clip(1 - (r / 0.32) ** 2, 0, 1)) * 0.35
        rosette = np.maximum(rosette, np.where(r < 1, 0.62 + 0.4 * petals + boss, 0))
    inside_band = (u > 0.06) & (u < 0.75)
    h = np.where(inside_band & (rosette > 0), np.maximum(h, rosette), h)

    hole = u > 1.0
    h = np.where(hole, 0, h)
    return h.astype(np.float32), u, hole


def build_frame():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["rgb"], z["alpha"]
    hgt, u, hole = _height_field()
    hgt = gauss(hgt, 0.8 * SS / 3)
    amp = 7.0 * SS                                     # 浮雕高度（像素）
    gy, gx = np.gradient(hgt * amp)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Lg = np.array([-0.45, -0.65, 0.62]); Lg /= np.linalg.norm(Lg)
    V = np.array([0, 0, 1.0])
    Hh = Lg + V; Hh /= np.linalg.norm(Hh)
    ndl = np.clip((n * Lg).sum(-1), 0, 1)
    ndh = np.clip((n * Hh).sum(-1), 0, 1)
    # 反射向量的“上下”决定环境亮度（金属主要靠反射环境显形）
    R = 2 * n[..., 2:3] * n - V
    up = -R[..., 1]
    env = 0.16 + 0.95 * smooth(-0.35, 0.65, up) + 0.25 * np.exp(-((up + 0.1) / 0.12) ** 2)
    ao = np.clip(1 - (gauss(hgt, 6 * SS) - hgt) * 2.6, 0.35, 1)
    gold = np.array([1.0, 0.77, 0.36])
    deep = np.array([0.30, 0.19, 0.07])
    col = gold * (0.46 * env[..., None] + 0.52 * ndl[..., None]) * ao[..., None]
    col = col + np.array([1.0, 0.88, 0.62]) * (ndh ** 60 * 1.3 + ndh ** 10 * 0.22)[..., None] * ao[..., None]
    # 包浆：凹处偏暗褐，大面积缓慢起伏的金色差
    hgt_c = gauss(hgt, 4 * SS)
    crev = np.clip((hgt_c - hgt) * 3.0, 0, 1)
    col = col * (1 - 0.55 * crev[..., None]) + deep * 0.55 * crev[..., None]
    patina = value_noise(hgt.shape[0], hgt.shape[1], 90 * SS, 11)
    col *= (0.9 + 0.16 * patina)[..., None]
    # 高处被磨出的一点红色底泥（bole）
    rub = smooth(0.72, 0.95, hgt) * (fbm(hgt.shape[0], hgt.shape[1], 12 * SS, 21) > 0.68)
    col = col * (1 - 0.18 * rub[..., None]) + np.array([0.55, 0.22, 0.12]) * 0.18 * rub[..., None]
    alpha = (~hole).astype(np.float32)
    rgb = downsample(np.clip(col, 0, 1.4), SS).astype(np.float32)
    alpha = downsample(alpha, SS).astype(np.float32)
    np.savez_compressed(CACHE, rgb=rgb, alpha=alpha)
    return rgb, alpha


def compose(painting, wall, label, label_size=24, spill=0.0):
    """painting：PH×PW×3（0..1）；wall：RGB 0..1；label：展签文字；spill>0 时画心（灯箱）透出的彩光会落到画框周围的墙上。"""
    wall = np.asarray(wall, np.float32)
    img = np.ones((H, W, 3), np.float32) * wall
    if spill > 0:
        glow = np.zeros((H // 4, W // 4, 3), np.float32)
        from PIL import Image as _I
        small = np.asarray(_I.fromarray((np.clip(painting, 0, 1) * 255).astype(np.uint8)).resize((PW // 4, PH // 4), _I.BILINEAR)).astype(np.float32) / 255
        glow[INNER[1] // 4:INNER[1] // 4 + small.shape[0], INNER[0] // 4:INNER[0] // 4 + small.shape[1]] = small
        glow = gauss(glow, 24)
        glow = np.asarray(_I.fromarray((np.clip(glow, 0, 1) * 255).astype(np.uint8)).resize((W, H), _I.BILINEAR)).astype(np.float32) / 255
        img = img + glow * spill
    x0, y0, x1, y1 = FRAME_OUT
    # 画框在墙上的投影（光从左上来）：很浅、很柔
    sh = np.zeros((H, W), np.float32)
    sh[y0:y1, x0:x1] = 1
    sh = gauss(ishift(sh, 14, 9), 16) * 0.30 + gauss(ishift(sh, 4, 3), 4) * 0.18
    img *= (1 - np.clip(sh, 0, 0.42))[..., None]
    # 画心，以及画框压在画心上的一圈内阴影（左上更重）
    ix0, iy0, ix1, iy1 = INNER
    pt = painting.astype(np.float32).copy()
    yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
    inner_sh = np.exp(-yy / 9) * 0.38 + np.exp(-xx / 9) * 0.30 + np.exp(-(PH - 1 - yy) / 5) * 0.10 + np.exp(-(PW - 1 - xx) / 5) * 0.10
    pt *= (1 - np.clip(inner_sh, 0, 0.6))[..., None]
    img[iy0:iy1, ix0:ix1] = pt
    frgb, fa = build_frame()
    img[y0:y1, x0:x1] = over(img[y0:y1, x0:x1], frgb, fa)
    # 展签：比墙色深或浅几档
    lum = float(wall @ np.array([0.3, 0.59, 0.11]))
    ink = wall * 0.48 if lum > 0.42 else np.clip(wall + (1 - wall) * 0.55, 0, 1)
    if label:
        m = label_layer(label, label_size, W, H, W / 2, LABEL_CY)
        img = over(img, ink, m * 0.92)
    return np.clip(img, 0, 1)


def save(img, path):
    from PIL import Image
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


if __name__ == "__main__":
    import sys
    if os.path.exists(CACHE):
        os.remove(CACHE)
    test = np.ones((PH, PW, 3), np.float32) * np.array([0.92, 0.90, 0.86])
    out = compose(test, (0.86, 0.62, 0.48), "我管不住我自己")
    save(out, sys.argv[1])
    from PIL import Image
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).crop((300, 100, 760, 420)).resize((920, 640), Image.LANCZOS).save(sys.argv[1].replace(".png", "_角.png"))
