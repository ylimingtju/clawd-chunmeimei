"""把各段 mp4 按顺序无损拼成整片（编码参数相同、每段从关键帧开始，用 concat 直接拷贝码流）。
用法：python concat_full.py"""
import os
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe

ROOT = "/Users/limingyang/Desktop/单依纯二专/视频/样片"
PARTS = [os.path.join(ROOT, "v4_试片_00开场-01还有什么更好的.mp4")] + [
    os.path.join(ROOT, "v4_分段", f) for f in ["02_纯妹妹.mp4", "03_向日葵朝着夜.mp4", "04_假如我们还爱着.mp4", "05_我管不住我自己.mp4",
                                              "06_有趣.mp4", "07_我表示理解.mp4", "08_还你茉莉.mp4", "09_多少的光阴.mp4", "10_珠玉_结尾.mp4"]]
OUT = "/Users/limingyang/Desktop/单依纯二专/视频/单依纯《纯妹妹》x Clawd_v4_整片.mp4"

if __name__ == "__main__":
    missing = [p for p in PARTS if not os.path.exists(p)]
    if missing:
        raise SystemExit("缺少分段：\n" + "\n".join(missing))
    lst = os.path.join(ROOT, "v4_分段", "_concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in PARTS:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    subprocess.run([get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", OUT], check=True)
    print("完成：", OUT)
