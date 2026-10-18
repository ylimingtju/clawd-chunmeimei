"""08 还你茉莉 · 敦煌壁画加摇滚 · 93 BPM（1:18.77–1:28.45）。画心动作「一拍二」。拍号 b 从 07 结束算起。
E  b0–2   舞台光拖出的慢门光轨在 Clawd 伸出的钳子前卷起来，凝固成一条飞天的飘带；飘带卷过的地方变成壁画，
          Clawd 换上素纱、背上反弹的琵琶。墙色和展签在它整个被卷进壁画的那一拍切换。
a  b2–9   石窟里两位飞天绕着 Clawd 飞，飘带拖在身后。重拍时 Clawd 把茉莉抛起来，茉莉散成花瓣雨。
          反弹琵琶的姿势就是摇滚吉他手的姿势；鼓点时画面推近一下并伴随震动。
M  b9–11  Clawd 一扫弦，火焰纹背光像冲击波一样往外扩；扫过的部分变成青金石蓝底的摇滚舞台，
          Clawd 换上黑色垂坠长衣和红手套。
b  b11–15 火焰纹背光变成舞台光，每个重拍亮一次；飞天们在台下甩飘带。低角度仰拍。"""
import math

import numpy as np

import S08_敦煌 as S8
from S08_敦煌 import (K, PW, PH, X, Y, Mural, M, circle, outline, ribbon, jasmine, apsara, flame, GROUND, PLASTER, FINE, GRAIN,
                     FLAKE_N, LAPIS, MALA, CINNA, EARTH, OCHRE, LEAD, INK, CLAWD)
from common4 import gauss, fbm, smooth, over, downsample, ishift, rigid_arm, stroke_mask
import anim_core as A
import frame
import A07_我表示理解 as A7

FPS = 30
T0 = A7.T0 + 19 * A7.BEAT
BEAT = 60 / 93
WALL7, LABEL7 = A7.WALL7, A7.LABEL7
WALL8, LABEL8 = (0.62, 0.22, 0.17), "还你茉莉"
FRAME0 = math.ceil(T0 * FPS)
FRAME1 = math.ceil((T0 + 15 * BEAT) * FPS)
MC = (560, 392)
CLAW = (560 + 8 * 16 + 14, 530 - 8 * 16)          # 07 结尾伸出的那只钳子附近
_CACHE = {}


def bt(t):
    return (t - T0) / BEAT


def capture(fn, patches):
    """跑一遍样张的 render 函数，在 Mural.age() 之前截住 2 倍分辨率的画面；patches 为临时替换的模块属性。"""
    snap = {}
    old = {k: getattr(S8, k) for k in patches}
    old_age = Mural.age
    Mural.age = lambda self: snap.__setitem__("img", self.img.copy())
    for k, v in patches.items():
        setattr(S8, k, v)
    S8.G = np.random.default_rng(8)
    try:
        fn()
    finally:
        Mural.age = old_age
        for k, v in old.items():
            setattr(S8, k, v)
    return snap["img"].astype(np.float32)


def plate_a():
    if "a" not in _CACHE:
        _CACHE["a"] = capture(S8.render, {"apsara": lambda *a, **k: None, "jasmine": lambda *a, **k: [(0, 0), (0.01, 0), (0, 0.01)]})
    return _CACHE["a"]


def plate_b():
    if "b" not in _CACHE:
        _CACHE["b"] = capture(S8.render_b, {"apsara": lambda *a, **k: None})
    return _CACHE["b"]


def age_maps():
    if "age" in _CACHE:
        return _CACHE["age"]
    n = fbm(PH, PW, 120, 30, 3)
    ridge = 1 - np.abs(2 * n - 1)
    crack = smooth(0.985, 0.996, ridge) * smooth(0.45, 0.7, fbm(PH, PW, 300, 31, 2))
    edge_w = smooth(150 * K, 20 * K, np.minimum.reduce([X, PW - X, Y, PH - Y]))
    flake = smooth(0.70, 0.73, FLAKE_N) * smooth(0.5, 0.7, fbm(PH, PW, 380, 33, 2)) * edge_w
    rough = PLASTER * (1 + 0.10 * GRAIN + 0.08 * FINE)[..., None]
    edge = np.clip(ishift(flake, 2 * K, 2 * K) - flake, 0, 1)
    salt = smooth(0.85, 0.97, fbm(PH, PW, 30, 34, 3)) * 0.12
    vign = np.clip(((X / PW - 0.5) ** 2 + (Y / PH - 0.5) ** 2) * 1.4, 0, 1)
    _CACHE["age"] = ((1 - 0.22 * crack).astype(np.float32), flake.astype(np.float32), rough.astype(np.float32),
                     (1 - 0.35 * gauss(edge, 0.8 * K)).astype(np.float32), salt.astype(np.float32), (1 - 0.22 * vign).astype(np.float32))
    return _CACHE["age"]


def age(img):
    mul, flake, rough, esh, salt, vig = age_maps()
    img = img * mul[..., None]
    img = img * (1 - flake[..., None]) + rough * flake[..., None]
    img = img * esh[..., None]
    img = over(img, (0.93, 0.90, 0.84), salt)
    return img * vig[..., None]


# ---------- 动态：飞天、茉莉 ----------
def orbit(theta, rx=330, ry=230, cy=MC[1] - 40):
    return MC[0] + rx * math.cos(theta), cy + ry * math.sin(theta)


def flying(Mu, b, k):
    """第 k 位飞天：沿椭圆绕着背光飞，飘带由它身后走过的轨迹生成。"""
    w = 0.55                                         # 每拍转的弧度
    th = -math.pi / 2 + k * math.pi + w * (b - 2.0)
    fx, fy = orbit(th)
    hist = [orbit(th - 0.12 * j) for j in range(1, 7)]
    ang = math.degrees(math.atan2(math.cos(th) * 230, -math.sin(th) * 330)) * 0.15
    flip = 1 if math.cos(th) < 0 else -1
    cols = [(LAPIS, MALA), (CINNA, OCHRE)][k]
    ribs = [([(fx + 4, fy - 30)] + [(x + 4, y - 30 + 8 * math.sin(b * 2 + j)) for j, (x, y) in enumerate(hist)], 12, cols[0], cols[1], 3.0, 0.3 * k + 0.4 * b)]
    S8.apsara(Mu, fx, fy, 6.5, ang + 10 * flip, [rigid_arm("L", 85 if k == 0 else -20), rigid_arm("R", -25 if k == 0 else 95)], ribs)


def petals_rain(Mu, b):
    """b2.5 茉莉被抛起，b3.4 在空中散开成花瓣雨。"""
    if b < 2.4:
        return
    if b < 3.4:
        u = A.seg(b, 2.4, 3.4)
        x, y = A.hop((470, 300), (560, 150), u, 120)
        Mu.paint(M(jasmine(x, y, 10, 4 * u)), LEAD, shade=0.1, line=INK, line_w=1, grain=0.5)
        Mu.paint(M(circle(x, y, 2.4, 12)), OCHRE, shade=0, wear=0.2)
        return
    g = np.random.default_rng(81)
    for i in range(40):
        a, sp = g.uniform(0, 2 * math.pi), g.uniform(60, 200)
        u = b - 3.4
        x = 560 + math.cos(a) * sp * min(1.0, u * 1.8) + 14 * math.sin(u * 2 + i)
        y = 150 + math.sin(a) * sp * 0.5 * min(1.0, u * 1.8) + 46 * u * u
        if y > 600:
            continue
        r = g.uniform(3.5, 6.5)
        Mu.paint(M(jasmine(x, y, r, g.uniform(0, 6) + u)), LEAD, shade=0.1, line=INK, line_w=1, grain=0.5)


def below_apsaras(Mu, b):
    sw = math.sin(math.pi * b * 0.5)
    S8.apsara(Mu, 150, 612, 7.0, -16, [rigid_arm("L", 30), rigid_arm("R", 30)],
              [([(176, 560), (230 + 30 * sw, 480), (210 - 20 * sw, 400), (262 + 40 * sw, 320), (240 - 30 * sw, 240)], 16, MALA, LAPIS, 3.2, 0.2 + 0.5 * b)])
    S8.apsara(Mu, 970, 612, 7.0, 16, [rigid_arm("L", 30), rigid_arm("R", 30)],
              [([(944, 560), (890 - 30 * sw, 480), (910 + 20 * sw, 400), (858 - 40 * sw, 320), (880 + 30 * sw, 240)], 16, CINNA, OCHRE, 3.2, 0.8 + 0.5 * b)])


def mural_on(img):
    Mu = Mural()
    Mu.img = img
    return Mu


# ---------- 一帧 ----------
def a_scene(b):
    S8.G = np.random.default_rng(int(b * 4) % 7 + 100)
    Mu = mural_on(plate_a().copy())
    for k in (0, 1):
        flying(Mu, b, k)
    petals_rain(Mu, b)
    return Mu.img


def b_scene(b):
    S8.G = np.random.default_rng(int(b * 4) % 7 + 200)
    Mu = mural_on(plate_b().copy())
    pulse = math.exp(-((b - 11.0) % 1.0) / 0.25) if b >= 11.0 else 0.0
    rr = np.hypot(X / K - 560, Y / K - 318)
    Mu.img = Mu.img * (1 + 0.20 * pulse * smooth(270, 180, rr))[..., None]       # 背光变成舞台光：每个重拍亮一下
    below_apsaras(Mu, b)
    return Mu.img


def painting(t, b):
    if b < 2.0:
        # E：从伸出的钳子前卷开一块壁画，边缘是一条翻卷的飘带
        con, cm = A7.concert(19.0, (560, 530, 16, 1.0))
        con = A7.S7.grade(con, cm)
        r = A.ease_in(A.seg(b, 0.1, 1.9), 1.6) * 1300
        cx, cy = CLAW
        ang = np.arctan2(Y / K - cy, X / K - cx)
        wav = 18 * np.sin(ang * 5 + b * 6)
        d = np.hypot(X / K - cx, Y / K - cy)
        inside = smooth(r + wav + 1, r + wav - 1, d) if r > 0 else np.zeros((PH, PW), np.float32)
        mur = age(a_scene(2.0))
        img = con * (1 - inside[..., None]) + mur * inside[..., None]
        # 卷边的飘带：沿着边界的一圈，正反两色交替翻卷
        if 0 < r < 1250:
            band = smooth(r + wav + 22, r + wav + 10, d) * smooth(r + wav - 2, r + wav + 8, d)
            tw = 0.5 + 0.5 * np.sin(ang * 3 + b * 4)
            col = np.array(MALA)[None, None] * tw[..., None] + np.array(CINNA)[None, None] * (1 - tw[..., None])
            img = over(img, col, band)
            img = over(img, INK, np.clip(outline(band, 1), 0, 1) * 0.8)
        # 光轨：钳子前一道卷起来的亮弧（越卷越短，凝固成飘带的起点）
        if b < 0.8:
            u = A.seg(b, 0.0, 0.8)
            arc = [(cx + (60 - 40 * u) * math.cos(a), cy + (40 - 20 * u) * math.sin(a)) for a in np.linspace(0, 2 * math.pi * (1.4 - u * 0.6), 50)]
            m = stroke_mask(PW, PH, [(x * K, y * K) for x, y in arc], 2.4 * K)
            img = img + (m * 0.9 + gauss(m, 4 * K) * 0.6)[..., None] * np.array([1.0, 0.85, 0.5])
        done = r > 520
        return img, done
    if b < 9.0:
        img = age(a_scene(b))
        return img, True
    # M：冲击波式外扩的背光
    r = A.ease_out(A.seg(b, 9.15, 10.6), 2.2) * 1300
    a_img = age(a_scene(b))
    b_img = age(b_scene(b))
    d = np.hypot(X / K - MC[0], Y / K - MC[1])
    inside = smooth(r + 1, r - 1, d)
    img = a_img * (1 - inside[..., None]) + b_img * inside[..., None]
    if 0 < r < 1250:
        ring = smooth(r - 26, r - 6, d) * smooth(r + 2, r - 4, d)
        ang = np.arctan2(Y / K - MC[1], X / K - MC[0])
        fl = 0.5 + 0.5 * np.cos(ang * 26)
        img = over(img, np.array(CINNA)[None, None] * (1 - fl[..., None]) + np.array(OCHRE)[None, None] * fl[..., None], ring * 0.95)
        img = over(img, INK, np.clip(outline(ring, 1), 0, 1) * 0.7)
    return img, True


def frame_at(i):
    t = i / FPS
    b = bt(A.twos(i))
    if b >= 11.0:
        p, sw = age(b_scene(b)), True
    else:
        p, sw = painting(t, b)
    p = np.clip(downsample(p, K), 0, 1)
    bb = bt(t)
    if 2.0 <= bb < 9.0:                                # 鼓点（每 2 拍）推近一下、震一下
        ph = (bb - 2.0) % 2.0
        kick = math.exp(-ph / 0.18)
        z = 1.02 + 0.035 * kick
        dx = 4 * kick * math.sin(t * 60)
        dy = 3 * kick * math.cos(t * 47)
        p = A.zoom_painting(p, z, 560 + dx, 380 + dy)
    elif bb >= 11.0:                                   # 低角度仰拍：画面略放大，镜头中心压低并慢慢上抬
        u = A.ease_io(A.seg(bb, 11.0, 15.0))
        p = A.zoom_painting(p, 1.06, 560, A.lerp(470, 430, u))
    return A.compose(p, WALL8 if sw else WALL7, LABEL8 if sw else LABEL7)


def warm():
    A7.warm()
    plate_a()
    plate_b()
    age_maps()
    for w, l in ((WALL7, LABEL7), (WALL8, LABEL8)):
        A.plate(w, l)


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    for i in [int(a) for a in sys.argv[2:]]:
        frame.save(frame_at(i), f"{out}/a08_{i:04d}.png")
