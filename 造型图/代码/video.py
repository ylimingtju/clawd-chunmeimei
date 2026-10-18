"""把逐帧渲染函数并行跑完，直接管道送进 ffmpeg 编码成 mp4（不落地大量 PNG）。"""
import os, subprocess, time
import numpy as np
from multiprocessing import Pool

FPS = 30


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render_video(frame_fn, n_frames, out_path, W=1920, H=1080, workers=None, stills=(), stills_dir=None, crf=17):
    """frame_fn(i) -> HxWx3 float(0..1)。stills：要另存 PNG 的帧号。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(workers or max(1, os.cpu_count() - 1)) as pool:
        for i, fr in enumerate(pool.imap(_wrap, [(frame_fn, k) for k in range(n_frames)], chunksize=1)):
            proc.stdin.write(fr.tobytes())
            if stills_dir and i in stills:
                from PIL import Image
                os.makedirs(stills_dir, exist_ok=True)
                Image.fromarray(fr).save(os.path.join(stills_dir, f"帧_{i:04d}.png"))
            if i % 30 == 0:
                print(f"  帧 {i}/{n_frames}  {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"完成：{out_path}  用时 {time.time() - t0:.0f}s")


def _wrap(args):
    fn, k = args
    return (np.clip(fn(k), 0, 1) * 255 + 0.5).astype(np.uint8)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)
