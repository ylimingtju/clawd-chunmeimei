"""按全片时间轴分派帧号并渲染成 mp4。
用法：python anim_run.py 曲目编号(如 02，或 all) [缩放]   或   python anim_run.py 起始帧 结束帧(不含) 输出.mp4 [缩放]"""
import importlib
import math
import os
import sys

import anim_core as A

OUT_DIR = "/Users/limingyang/Desktop/单依纯二专/视频/样片/v4_分段"
# (编号, 模块, 起始帧)；结束帧 = 下一段的起始帧
SONGS = [("00", "A00_开场", 0), ("01", "A01_还有什么更好的", 240), ("02", "A02_纯妹妹", None), ("03", "A03_向日葵朝着夜", None), ("04", "A04_假如我们还爱着", None), ("05", "A05_我管不住我自己", None), ("06", "A06_有趣", None), ("07", "A07_我表示理解", None), ("08", "A08_还你茉莉", None), ("09", "A09_多少的光阴", None), ("10", "A10_珠玉_结尾", None)]
_MODS = {}


def mod(name):
    if name not in _MODS:
        _MODS[name] = importlib.import_module(name)
    return _MODS[name]


def bounds():
    out = []
    for k, (code, name, f0) in enumerate(SONGS):
        if f0 is None:
            f0 = mod(name).FRAME0
        out.append([code, name, f0, None])
    for k in range(len(out) - 1):
        out[k][3] = out[k + 1][2]
    last = mod(out[-1][1])
    out[-1][3] = getattr(last, "FRAME1", None)
    return out


def frame_at(i):
    for code, name, f0, f1 in bounds():
        if f0 <= i < (f1 if f1 is not None else 10 ** 9):
            return mod(name).frame_at(i)
    raise ValueError(i)


def warm(names):
    for n in names:
        m = mod(n)
        if hasattr(m, "warm"):
            m.warm()
        if hasattr(m, "WALL"):
            A.plate(m.WALL, m.LABEL)


if __name__ == "__main__":
    B = bounds()
    if len(sys.argv) >= 4 and sys.argv[1].isdigit() and sys.argv[2].isdigit() and len(sys.argv[1]) > 2:
        i0, i1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        names = [n for c, n, a, b in B]
    else:
        code = sys.argv[1]
        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        row = [r for r in B if r[0] == code][0]
        i0, i1 = row[2], row[3]
        out = os.path.join(OUT_DIR, f"{code}_{row[1].split('_', 1)[1]}.mp4")
        names = [r[1] for r in B]
    warm(names)
    stills = set(range(i0, i1, 30))
    A.render("anim_run", "frame_at", list(range(i0, i1)), out, stills=stills,
             stills_dir=os.path.join(os.path.dirname(out), os.path.splitext(os.path.basename(out))[0] + "_静帧"), scale=scale)
