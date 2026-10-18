"""歌名标题：先渲染字形遮罩，再按各曲风格加工成 RGBA。"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import blur, to_arr, to_img, fbm, rng, shift

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "字体")
FONTS = {
    "huangyou": "ZCOOL_QingKe_HuangYou/ZCOOLQingKeHuangYou-Regular.ttf",
    "kuaile": "ZCOOL_KuaiLe/ZCOOLKuaiLe-Regular.ttf",
    "mashanzheng": "Ma_Shan_Zheng/MaShanZheng-Regular.ttf",
    "serif_light": "Noto_Serif_SC/NotoSerifSC-Light.ttf",
    "serif_xlight": "Noto_Serif_SC/NotoSerifSC-ExtraLight.ttf",
    "wenkai": "LXGW_WenKai/LXGWWenKai-Medium.ttf",
    "xiaowei": "ZCOOL_XiaoWei/ZCOOLXiaoWei-Regular.ttf",
    "longcang": "Long_Cang/LongCang-Regular.ttf",
    "maocao": "Liu_Jian_Mao_Cao/LiuJianMaoCao-Regular.ttf",
    "zhimang": "Zhi_Mang_Xing/ZhiMangXing-Regular.ttf",
}


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, FONTS[name]), size)


def text_mask(text, fname, size, pad=60, tracking=0.0, jitter=None, seed=0):
    """返回 (mask, 每个字的外框列表)。jitter=(dy像素, 旋转度) 时逐字随机偏移，做手作/跳动感。"""
    f = font(fname, size)
    g = rng(seed)
    adv = []
    for ch in text:
        bb = f.getbbox(ch)
        adv.append(max(bb[2], size * 0.35) if ch != " " else size * 0.4)
    w = int(sum(adv) + tracking * size * (len(text) - 1) + pad * 2)
    h = int(size * 1.5 + pad * 2)
    m = Image.new("L", (w, h), 0)
    boxes = []
    x = pad
    for ch, a in zip(text, adv):
        if jitter:
            dy, rot = g.uniform(-jitter[0], jitter[0]), g.uniform(-jitter[1], jitter[1])
            cim = Image.new("L", (int(size * 1.6), int(size * 1.8)), 0)
            ImageDraw.Draw(cim).text((size * 0.3, size * 0.2), ch, font=f, fill=255)
            cim = cim.rotate(rot, resample=Image.BICUBIC)
            m.paste(255, (int(x - size * 0.3), int(pad + dy - size * 0.2)), cim)
        else:
            ImageDraw.Draw(m).text((x, pad), ch, font=f, fill=255)
        boxes.append((x, pad, x + a, pad + size * 1.2))
        x += a + tracking * size
    arr = to_arr(m)
    ys, xs = np.nonzero(arr > 0.02)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
    return arr[y0:y1, x0:x1], [(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in boxes]


def dilate(mask, r):
    return np.clip(blur(mask, r) * 3.0, 0, 1)


def place(canvas, rgba, x, y):
    """把 RGBA 精灵以左上角 (x,y) 合成到 canvas。"""
    h, w = rgba.shape[:2]
    H, W = canvas.shape[:2]
    X0, Y0, X1, Y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return canvas
    sub = rgba[Y0 - y:Y1 - y, X0 - x:X1 - x]
    a = sub[..., 3:4]
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - a) + sub[..., :3] * a
    return canvas


def stack_rgba(*layers):
    """自下而上叠加同尺寸 RGBA。"""
    out = np.zeros_like(layers[0])
    for L in layers:
        a = L[..., 3:4]
        out[..., :3] = out[..., :3] * (1 - a) + L[..., :3] * a
        out[..., 3:4] = out[..., 3:4] + a * (1 - out[..., 3:4])
    return out


def shadow_layer(mask, dx, dy, r, strength):
    s = np.roll(np.roll(blur(mask, r), dy, 0), dx, 1) * strength
    out = np.zeros(mask.shape + (4,), np.float32)
    out[..., 3] = s
    return out


# ---------- 各曲风格 ----------

def paper_title(text, size=170, seed=0):
    """01：瓦楞纸剪字，粉色衬纸 + 牛皮纸面层 + 投影。"""
    m, _ = text_mask(text, "kuaile", size, pad=80, jitter=(6, 2.5), seed=seed)
    h, w = m.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    back = dilate(m, 7)
    back_rgba = np.dstack([np.full_like(m, 0.96), np.full_like(m, 0.74), np.full_like(m, 0.78), back])
    # 衬纸纸纹
    fib = fbm(h, w, seed + 3, octaves=3, base=60)
    back_rgba[..., :3] *= (0.94 + 0.06 * fib)[..., None]
    # 瓦楞纸面：竖向波纹 + 边缘露出的楞
    flute = 0.5 + 0.5 * np.sin(x / 3.2)
    kraft = np.array([0.78, 0.62, 0.44])
    face = kraft[None, None] * (0.88 + 0.1 * flute + 0.06 * fib)[..., None]
    edge = np.clip(m - np.clip(blur(m, 2) - 0.15, 0, 1), 0, 1)
    face = face * (1 - 0.35 * edge[..., None])
    face_rgba = np.dstack([face, m])
    face_rgba = np.roll(np.roll(face_rgba, -5, 0), -5, 1)
    hl = np.clip(m - np.roll(np.roll(m, 3, 0), 3, 1), 0, 1)
    face_rgba[..., :3] += np.roll(np.roll(hl, -5, 0), -5, 1)[..., None] * 0.25
    return stack_rgba(shadow_layer(back, 10, 14, 6, 0.45), back_rgba, shadow_layer(m, 3, 5, 2, 0.35), face_rgba)


def oil_title(text, size=180, seed=0, color=(0.98, 0.84, 0.36)):
    """03：刮刀厚涂的月光黄字，带笔触纹理与厚度高光。"""
    m, _ = text_mask(text, "mashanzheng", size, pad=70)
    h, w = m.shape
    g = rng(seed)
    n = g.normal(0, 1, (h, w)).astype(np.float32)
    # 横向拉长的笔触纹理
    streak = np.asarray(Image.fromarray(((np.clip(n * 0.25 + 0.5, 0, 1)) * 255).astype(np.uint8))
                        .resize((max(w // 14, 2), h), Image.BILINEAR).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255
    height = blur(m, 3) * (0.7 + 0.6 * streak)
    emb = height - np.roll(np.roll(height, 3, 0), 3, 1)
    base = np.array(color)
    col = base[None, None] * (0.82 + 0.3 * streak)[..., None]
    col = col + np.clip(emb, 0, 1)[..., None] * 0.6 - np.clip(-emb, 0, 1)[..., None] * 0.5
    # 笔画里混入一点白与橙
    mix = fbm(h, w, seed + 5, octaves=3, base=10)
    col = col * (1 - 0.25 * (mix > 0.65)[..., None]) + (mix > 0.65)[..., None] * np.array([1.0, 0.97, 0.85]) * 0.25
    a = np.clip(dilate(m, 1.2) * (0.85 + 0.3 * streak), 0, 1)
    return stack_rgba(shadow_layer(m, 4, 6, 4, 0.5), np.dstack([np.clip(col, 0, 1), a]))


def film_title(text, size=120, seed=0):
    """04：白色细宋体，两层错位叠影（双重曝光）。"""
    m, _ = text_mask(text, "serif_light", size, pad=60, tracking=0.25)
    glow = blur(m, 6) * 0.5
    a1 = np.clip(m * 0.95 + glow, 0, 1)
    a2 = np.roll(np.roll(np.clip(m * 0.4 + glow * 0.5, 0, 1), 9, 0), 16, 1)
    white = np.ones(m.shape + (3,), np.float32) * np.array([0.95, 0.96, 0.98])
    return stack_rgba(np.dstack([white, a2]), np.dstack([white, a1]))


def watercolor_title(text, size=170, seed=0, colors=((0.95, 0.55, 0.62), (0.98, 0.78, 0.35))):
    """05：水彩字：颜料晕染、边缘积色、颗粒；逐字轻微跳动。"""
    m, boxes = text_mask(text, "wenkai", size, pad=80, jitter=(14, 4), seed=seed)
    h, w = m.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    noise = fbm(h, w, seed + 11, octaves=4, base=14)
    bleed = np.clip(blur(m, 5) * 1.6 + (noise - 0.5) * 0.9, 0, 1)
    bleed = np.clip((bleed - 0.35) * 2.2, 0, 1)
    t = np.clip(x / w + (noise - 0.5) * 0.3, 0, 1)[..., None]
    c = np.array(colors[0]) * (1 - t) + np.array(colors[1]) * t
    edge = np.clip(bleed - blur(bleed, 4), 0, 1) * 2.5
    gran = fbm(h, w, seed + 12, octaves=2, base=160)
    dens = np.clip(0.55 + 0.35 * edge + 0.15 * (gran - 0.5), 0, 1) * bleed
    # 水彩为减色：颜色越深 alpha 越大，这里直接用 c 着色、dens 作 alpha
    col = c * (1 - 0.35 * edge[..., None])
    return np.dstack([np.clip(col, 0, 1), np.clip(dens, 0, 1)])


def cel_title(text, size=180, seed=0, fill=(1.0, 0.84, 0.12), shade=(0.96, 0.58, 0.08)):
    """06：赛璐璐：粗黑描边 + 亮黄填充 + 下半硬阴影 + 偏移投影。"""
    m, _ = text_mask(text, "xiaowei", size, pad=60, tracking=0.08)
    h, w = m.shape
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    bold = dilate(m, 2.2)
    outline = dilate(m, 7)
    face = np.array(fill)[None, None] * np.ones((h, w, 1))
    # 下 40% 做硬阴影色
    ys = np.nonzero(m.max(1) > 0.5)[0]
    cut = ys.min() + (ys.max() - ys.min()) * 0.62
    face = np.where((yy > cut)[..., None], np.array(shade)[None, None], face)
    # 左上的高光小条
    hl = np.clip(bold - shift(bold, 5, 5), 0, 1)
    face = face + hl[..., None] * 0.35
    black = np.zeros((h, w, 3), np.float32) + 0.05
    drop = np.dstack([black, shift(outline, 9, 9) * 0.9])
    return stack_rgba(drop, np.dstack([black, outline]), np.dstack([np.clip(face, 0, 1), bold]))


def ink_title(text, size=210, seed=0, thicken=4.2, seal_char="茉"):
    """08：狂草泼墨：刘建毛草加粗 + 飞白 + 飞溅墨点 + 朱红印章。返回 (墨色RGBA, 印章RGBA)。"""
    m, _ = text_mask(text, "maocao", size, pad=110, tracking=0.05)
    h, w = m.shape
    g = rng(seed)
    thick = np.clip(blur(m, thicken) * 2.6, 0, 1)
    # 飞白：沿横向拉长的干笔缝隙
    n = g.random((h // 5, max(w // 26, 2))).astype(np.float32)
    streak = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255
    n2 = fbm(h, w, seed + 7, octaves=3, base=6)
    dry = np.clip((streak * 0.6 + n2 * 0.6 - 0.32) * 5, 0, 1)
    dry = np.where(n2 > 0.45, 1.0, dry)                     # 只有部分区域出现飞白
    edge = np.clip(thick - blur(thick, 3), 0, 1)
    a = np.clip(thick * (0.55 + 0.45 * dry) + edge * 0.6, 0, 1)
    # 飞溅墨点
    sp = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(sp)
    ys, xs = np.nonzero(thick > 0.6)
    for _ in range(90):
        i = g.integers(len(xs))
        ang = g.uniform(0, 2 * np.pi)
        dist = g.uniform(20, 120)
        r = g.uniform(2, 9) * (1 - dist / 160)
        cx, cy = xs[i] + np.cos(ang) * dist, ys[i] + np.sin(ang) * dist
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    a = np.clip(a + to_arr(sp) * 0.95, 0, 1)
    ink = np.dstack([np.full((h, w, 3), 0.06, np.float32), a])
    # 印章：朱红方印，阴刻白字
    S_ = int(size * 0.62)
    seal = Image.new("L", (S_, S_), 255)
    sd = ImageDraw.Draw(seal)
    f = font("serif_light", int(S_ * 0.72))
    bb = f.getbbox(seal_char)
    sd.text(((S_ - (bb[2] - bb[0])) / 2 - bb[0], (S_ - (bb[3] - bb[1])) / 2 - bb[1]), seal_char, font=f, fill=0)
    sa = to_arr(seal)
    border = np.ones((S_, S_), np.float32)
    rough = blur(rng(seed + 1).random((S_, S_)).astype(np.float32), 1.2)
    sa = np.clip(sa * (0.75 + 0.5 * rough), 0, 1) * (rough > 0.32)
    seal_rgba = np.dstack([np.ones((S_, S_, 3), np.float32) * np.array([0.82, 0.14, 0.1]), sa.astype(np.float32)])
    return ink, seal_rgba


def sand_title(text, size=170, seed=0):
    """09：手指在沙上写字：返回 (挖开处 removal, 堆积边 pile) 两个遮罩。"""
    m, _ = text_mask(text, "zhimang", size, pad=60, tracking=0.12)
    groove = np.clip(blur(m, 1.5) * 1.4, 0, 1)
    pile = np.clip(dilate(m, 5) - groove, 0, 1)
    return groove, pile
