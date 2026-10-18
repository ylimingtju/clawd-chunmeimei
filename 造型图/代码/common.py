"""公共工具：Clawd 几何、噪声、混合与后期。所有场景脚本共用。"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
CLAWD_ORANGE = (202, 124, 94)  # 取自 素材/clawd 造型.png

# Clawd 几何（单位：格），按参考图像素测量。身体 16×10 格、手臂 1×2、腿 1.3×2。
BODY = (0, 0, 16, 10)
ARMS = [(-1, 3, 0, 5), (16, 3, 17, 5)]
EYES = [(3.6, 3, 4.6, 5), (11.4, 3, 12.4, 5)]
# 腿与身体相连（用户 2026-10-16 确认）：腿长仍为 2 格，整体高 12 格
LEGS = [(0, 10, 1.3, 12), (2.3, 10, 3.6, 12), (12.4, 10, 13.7, 12), (14.7, 10, 16, 12)]
BBOX = (-1, 0, 17, 12)


def rng(seed):
    return np.random.default_rng(seed)


# ---------- 基础数组工具 ----------
def to_arr(img):
    return np.asarray(img).astype(np.float32) / 255.0


def to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))


def blur(arr, r):
    """对 HxW 或 HxWxC 的 float 数组做高斯模糊。"""
    if arr.ndim == 2:
        return to_arr(to_img(arr).filter(ImageFilter.GaussianBlur(r)))
    chans = [to_arr(to_img(arr[..., i]).filter(ImageFilter.GaussianBlur(r))) for i in range(arr.shape[2])]
    return np.stack(chans, -1)


def blur_hdr(arr, r):
    """支持 >1 数值的模糊：先缩放到 0..1 再还原。"""
    m = max(float(arr.max()), 1e-6)
    return blur(arr / m, r) * m


def fbm(h, w, seed, octaves=5, base=8, persistence=0.5):
    """分形值噪声，返回 0..1。"""
    g = rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = base * (2 ** o)
        small = g.random((max(2, int(cells * h / max(h, w)) + 2), cells + 2)).astype(np.float32)
        layer = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255
        out += layer * amp
        total += amp
        amp *= persistence
    out /= total
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out


def grid(h=H, w=W):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x, y


def shift(a, dy, dx):
    """不回绕的平移（空出部分补 0）。"""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


def over(dst, src_rgb, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return dst * (1 - a) + src_rgb * a


def hsv2rgb(h, s, v):
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


def grain(arr, amount, seed, size=1.0):
    g = rng(seed)
    h, w = arr.shape[:2]
    n = g.normal(0, 1, (int(h / size), int(w / size))).astype(np.float32)
    if size != 1.0:
        n = np.asarray(Image.fromarray(((n * 0.25 + 0.5).clip(0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)).astype(np.float32) / 255 * 4 - 2
    return arr + n[..., None] * amount


def vignette(arr, strength=0.5, power=2.0):
    h, w = arr.shape[:2]
    x, y = grid(h, w)
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    return arr * (1 - strength * d ** power)[..., None]


def bloom(arr, threshold=0.7, radius=25, amount=0.6):
    bright = np.clip(arr - threshold, 0, None)
    return arr + blur_hdr(bright, radius) * amount


def tonemap(arr):
    """柔和的 filmic 压缩，避免高光硬裁切。"""
    return 1 - np.exp(-arr * 1.15)


# ---------- Clawd ----------
def clawd_masks(s, pad=0, ss=3, extra=None, arms=None, legs=None, bbox=None, body=None, eyes=None):
    """以格尺寸 s 渲染 Clawd 各部件遮罩。
    返回 (masks, origin, size)：masks 为各部件 float 数组；origin 为格坐标 (0,0) 在画布中的像素位置。
    extra: {名称: [矩形或多边形(格坐标)]}，用于服装等额外形状。
    arms/legs: 覆盖默认手臂/腿的形状（矩形或多边形），用于改变姿势；bbox 覆盖画布范围（格）。"""
    x0, y0, x1, y1 = bbox or BBOX
    pw = int((x1 - x0) * s + 2 * pad)
    ph = int((y1 - y0) * s + 2 * pad)
    ox, oy = -x0 * s + pad, -y0 * s + pad

    def render(shapes):
        im = Image.new("L", (pw * ss, ph * ss), 0)
        d = ImageDraw.Draw(im)
        for sh in shapes:
            if len(sh) == 4 and not isinstance(sh[0], (tuple, list)):
                a, b, c, e = sh
                if c <= a or e <= b:
                    continue          # 空矩形（占位）跳过
                d.rectangle([(ox + a * s) * ss, (oy + b * s) * ss, (ox + c * s) * ss - 1, (oy + e * s) * ss - 1], fill=255)
            else:
                d.polygon([((ox + px * s) * ss, (oy + py * s) * ss) for px, py in sh], fill=255)
        return to_arr(im.resize((pw, ph), Image.LANCZOS))

    masks = {
        "body": render(body if body is not None else [BODY]),
        "arms": render(arms if arms is not None else ARMS),
        "legs": render(legs if legs is not None else LEGS),
        "eyes": render(eyes if eyes is not None else EYES),
    }
    masks["all"] = np.clip(masks["body"] + masks["arms"] + masks["legs"], 0, 1)
    for k, shapes in (extra or {}).items():
        masks[k] = render(shapes)
    return masks, (ox, oy), (pw, ph)


def edge_dist(mask, r):
    """近似的“离边缘距离”：模糊后的遮罩，越靠内越接近 1。"""
    return np.clip(blur(mask, r) * mask, 0, 1)


def paste_rgba(canvas, rgba, cx, cy, angle=0.0):
    """把 RGBA float 精灵(中心对齐)以给定角度贴到 canvas(float RGB)。"""
    im = to_img(rgba)
    if angle:
        im = im.convert("RGBa").rotate(angle, resample=Image.BICUBIC, expand=True).convert("RGBA")
    arr = to_arr(im)
    h, w = arr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, canvas.shape[1]), min(y0 + h, canvas.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return canvas
    sub = arr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = sub[..., 3:4]
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - a) + sub[..., :3] * a
    return canvas


def sprite_alpha_layer(rgba, cx, cy, angle=0.0):
    """返回贴图位置处的全画布 alpha（用于阴影/辉光）。"""
    tmp = np.zeros((H, W, 3), np.float32)
    rg = rgba.copy()
    rg[..., :3] = 1
    tmp = paste_rgba(tmp, rg, cx, cy, angle)
    return tmp[..., 0]


def save(arr, path):
    to_img(arr).save(path, quality=95)


def rigid_arm(side, angle_deg):
    """整块旋转的手臂（保持 1×2 格原尺寸）。side='L'/'R'；angle_deg>0 表示向上抬起。
    以手臂与身体相接处的中点为轴：左臂 (0,4)，右臂 (16,4)。"""
    if side == "L":
        pivot, corners = (0.0, 4.0), [(0, -1), (-1, -1), (-1, 1), (0, 1)]
        a = math.radians(angle_deg)
    else:
        pivot, corners = (16.0, 4.0), [(0, -1), (1, -1), (1, 1), (0, 1)]
        a = math.radians(-angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [(pivot[0] + px * c - py * s, pivot[1] + px * s + py * c) for px, py in corners]


def check_proportions(arms=None, legs=None, extra=None, hem_limit=12.0):
    """检查姿势与服装是否改变了 Clawd 的比例：手臂块须为 1×2，腿须为默认腿，服装最低点 ≤ hem_limit。"""
    problems = []
    for a in (arms or []):
        pts = a if isinstance(a[0], (tuple, list)) else [(a[0], a[1]), (a[2], a[1]), (a[2], a[3]), (a[0], a[3])]
        d = sorted(math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
        if len(pts) != 4 or abs(d[0] - 1) > 0.05 or abs(d[-1] - 2) > 0.05:
            problems.append(f"手臂尺寸不是 1×2：{pts}")
    if legs is not None and [tuple(l) for l in legs] != [tuple(l) for l in LEGS]:
        problems.append("腿被修改")
    for name, shapes in (extra or {}).items():
        for sh in shapes:
            ys = [sh[1], sh[3]] if not isinstance(sh[0], (tuple, list)) else [p[1] for p in sh]
            if max(ys) > hem_limit + 1e-6:
                problems.append(f"{name} 最低点 {max(ys):.2f} > {hem_limit}")
    if problems:
        raise ValueError("Clawd 比例检查未通过：" + "；".join(problems))
