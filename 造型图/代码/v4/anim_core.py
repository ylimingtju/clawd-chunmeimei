"""v4 动画公共工具：时间与节拍、缓动、「一拍二」取帧、墙/画框/展签的缓存合成、并行渲染成 mp4。"""
import math
import os
import subprocess
import time
from multiprocessing import get_context

import numpy as np

import frame

FPS = 30
_PLATES = {}


# ---------- 时间与缓动 ----------
def clamp01(t):
    return min(max(t, 0.0), 1.0)


def seg(t, a, b):
    """t 在 [a, b] 内的归一化进度（两端截断）。"""
    return clamp01((t - a) / (b - a)) if b > a else float(t >= a)


def ease_io(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out(t, p=2.0):
    return 1 - (1 - clamp01(t)) ** p


def ease_in(t, p=2.0):
    return clamp01(t) ** p


def back_out(t, s=1.7):
    """带一点过冲的落定（按拍点「吸到位」）。"""
    t = clamp01(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def lerp(a, b, t):
    return a + (b - a) * t


def lerp2(p, q, t):
    return (lerp(p[0], q[0], t), lerp(p[1], q[1], t))


def hop(p0, p1, t, h):
    """从 p0 跳到 p1（画心坐标，y 向下），抛物线最高点比两端中较高者再高 h。"""
    t = clamp01(t)
    x = lerp(p0[0], p1[0], t)
    y = lerp(p0[1], p1[1], t) - 4 * h * t * (1 - t)
    return x, y


def twos(i):
    """一拍二：偶数帧的时间，奇数帧沿用上一帧的姿势。"""
    return (i - i % 2) / FPS


# ---------- 墙 + 画框 + 展签 ----------
def plate(wall, label):
    """缓存一块「墙 + 投影 + 金框 + 展签」，画心位置留空；返回 (底图, 画心处的框 alpha, 画心内阴影系数)。"""
    key = (tuple(np.round(wall, 4)), label)
    if key not in _PLATES:
        empty = np.zeros((frame.PH, frame.PW, 3), np.float32)
        base = frame.compose(empty, wall, label)
        frgb, fa = frame.build_frame()
        x0, y0 = frame.FRAME_OUT[:2]
        ix0, iy0, ix1, iy1 = frame.INNER
        fa_in = fa[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        yy, xx = np.mgrid[0:frame.PH, 0:frame.PW].astype(np.float32)
        PH, PW = frame.PH, frame.PW
        inner_sh = np.exp(-yy / 9) * 0.38 + np.exp(-xx / 9) * 0.30 + np.exp(-(PH - 1 - yy) / 5) * 0.10 + np.exp(-(PW - 1 - xx) / 5) * 0.10
        mul = ((1 - np.clip(inner_sh, 0, 0.6)) * (1 - fa_in))[..., None].astype(np.float32)
        _PLATES[key] = (base, mul)
    return _PLATES[key]


def compose(painting, wall, label, spill=0.0):
    """和 frame.compose 结果相同，但墙/框/展签只算一次。spill>0：灯箱画心透出的彩光落到墙上。"""
    base, mul = plate(wall, label)
    img = base.copy()
    if spill > 0:
        from PIL import Image as _I
        from common4 import gauss as _g
        W, H = img.shape[1], img.shape[0]
        small = np.asarray(_I.fromarray((np.clip(painting, 0, 1) * 255).astype(np.uint8)).resize((frame.PW // 4, frame.PH // 4), _I.BILINEAR)).astype(np.float32) / 255
        glow = np.zeros((H // 4, W // 4, 3), np.float32)
        glow[frame.INNER[1] // 4:frame.INNER[1] // 4 + small.shape[0], frame.INNER[0] // 4:frame.INNER[0] // 4 + small.shape[1]] = small
        glow = _g(glow, 24)
        glow = np.asarray(_I.fromarray((np.clip(glow, 0, 1) * 255).astype(np.uint8)).resize((W, H), _I.BILINEAR)).astype(np.float32) / 255
        ix0, iy0, ix1, iy1 = frame.INNER
        add = glow * spill
        add[iy0:iy1, ix0:ix1] = 0
        x0, y0, x1, y1 = frame.FRAME_OUT
        frgb, fa = frame.build_frame()
        add[y0:y1, x0:x1] *= (1 - fa[..., None])
        img = img + add
    ix0, iy0, ix1, iy1 = frame.INNER
    img[iy0:iy1, ix0:ix1] += painting.astype(np.float32) * mul
    return img


def zoom_painting(p, z, cx, cy):
    """画框里的推镜：以 (cx, cy) 为中心放大 z 倍（z ≥ 1），保持画心尺寸。"""
    if z <= 1.0001:
        return p
    from scipy.ndimage import map_coordinates
    H, W = p.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = cx + (xx - cx) / z
    sy = cy + (yy - cy) / z
    out = np.stack([map_coordinates(p[..., c], [sy, sx], order=1, mode="nearest") for c in range(3)], -1)
    return out


# ---------- 并行渲染 ----------
def _job(args):
    mod_name, fn_name, i = args
    import importlib
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return i, (np.clip(fn(i), 0, 1) * 255 + 0.5).astype(np.uint8)


def render(mod_name, fn_name, frames, out_path, stills=(), stills_dir=None, workers=None, scale=1.0, crf=16):
    """frames：要渲染的全局帧号列表；fn(i) 返回 1080×1920×3 的 0..1 浮点图。scale<1 时输出缩小的预览。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    W, H = int(1920 * scale) // 2 * 2, int(1080 * scale) // 2 * 2
    from imageio_ffmpeg import get_ffmpeg_exe
    cmd = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    from PIL import Image
    ctx = get_context("fork")
    with ctx.Pool(workers or max(1, os.cpu_count() - 1)) as pool:
        for n, (i, fr) in enumerate(pool.imap(_job, [(mod_name, fn_name, k) for k in frames], chunksize=1)):
            im = Image.fromarray(fr)
            if scale != 1.0:
                im = im.resize((W, H), Image.LANCZOS)
            proc.stdin.write(np.asarray(im).tobytes())
            if stills_dir and i in stills:
                os.makedirs(stills_dir, exist_ok=True)
                Image.fromarray(fr).save(os.path.join(stills_dir, f"帧_{i:04d}.png"))
            if n % 30 == 0:
                print(f"  帧 {n}/{len(frames)}  {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"完成：{out_path}  用时 {time.time() - t0:.0f}s", flush=True)
