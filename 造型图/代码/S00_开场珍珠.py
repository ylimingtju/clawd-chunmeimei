"""00-1｜开场：黑场里一颗珍珠慢慢亮起，表面映出专辑封面的虹彩（镜头随后推进珍珠内部）。"""
import sys, os
import numpy as np
from PIL import Image
from common import *
from pearl import pearl

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
x, y = grid()
A = to_arr(Image.open(os.path.join(os.path.dirname(__file__), "..", "第1批_代码绘制", "A_00_专辑整体_珍珠万花筒.png")).convert("RGB"))

img = np.zeros((H, W, 3), np.float32) + 0.008
R = 210
p = pearl(R, light=(-0.72, -0.62, 0.32), world=A, world_mix=0.3, ambient=0.02)
img = paste_rgba(img, p, W // 2, H // 2)
# 珍珠周围极淡的虹彩光晕 + 底部反射
d = np.sqrt((x - W / 2) ** 2 + (y - H / 2) ** 2)
halo = np.exp(-np.clip(d - R, 0, None) / 90) * (d > R - 2)
ang = np.arctan2(y - H / 2, x - W / 2)
iri = hsv2rgb(((ang / (2 * np.pi)) + 0.5) % 1, np.full_like(ang, 0.55), np.ones_like(ang))
img = img + halo[..., None] * iri * 0.06
refl = np.exp(-(((x - W / 2) / 170) ** 2 + ((y - (H / 2 + R + 40)) / 18) ** 2))
img = img + refl[..., None] * np.array([0.10, 0.09, 0.09])
# 几粒漂浮的微光
g = rng(3)
dust = np.zeros((H, W), np.float32)
for _ in range(40):
    dust[g.integers(0, H), g.integers(0, W)] = g.uniform(0.3, 1)
img = img + blur(dust, 1.5)[..., None] * 6 * np.array([0.9, 0.85, 0.8])
img = bloom(img, 0.8, 40, 0.35)
img = grain(img, 0.01, 4)
save(img, os.path.join(OUT, "00-1_开场_珍珠亮起.png"))
print("00-1 ok")
