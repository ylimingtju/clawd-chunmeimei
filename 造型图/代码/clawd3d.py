"""Clawd 三维渲染：由长方体组成，numpy 光线投射（透视相机、主光+环境光、软阴影、地面接触阴影）。
单位：格（与二维几何一致）。坐标：X 向右、Y 向上、Z 朝向镜头外（正面朝 -Z 方向看向相机）。"""
import math
import numpy as np

ORANGE = np.array([202, 124, 94], np.float32) / 255


def clawd_boxes(depth=8.0, leg_gap=0.0, arm_depth=2.0, leg_depth=1.3, leg_rows=1, extra=None):
    """返回 [(min, max, 材质名)]。尺寸与二维一致：身体 16×10、手臂 1×2、腿 1.3×2（格）。
    腿从地面 Y=0 起，高 2 格；leg_gap 为身体与腿之间的空隙（参考图为 1，0 表示相连）。
    leg_rows=1：四条腿排成一排；=2：前后两排（每个位置前后各一条）。"""
    hz = depth / 2
    boxes = []
    for x0, x1 in [(0, 1.3), (2.3, 3.6), (12.4, 13.7), (14.7, 16)]:
        zs = [0.0] if leg_rows == 1 else [-hz + leg_depth, hz - leg_depth]
        for zc in zs:
            boxes.append(((x0, 0, zc - leg_depth / 2), (x1, 2, zc + leg_depth / 2), "body"))
    by0 = 2 + leg_gap
    boxes.append(((0, by0, -hz), (16, by0 + 10, hz), "body"))
    ay1 = by0 + 10 - 3                      # 手臂：距身体顶 3–5 格
    boxes.append(((-1, ay1 - 2, -arm_depth / 2), (0, ay1, arm_depth / 2), "body"))
    boxes.append(((16, ay1 - 2, -arm_depth / 2), (17, ay1, arm_depth / 2), "body"))
    for ex0, ex1 in [(3.6, 4.6), (11.4, 12.4)]:   # 眼睛：正面（-Z）略凸出的黑色薄块
        boxes.append(((ex0, ay1 - 2, -hz - 0.08), (ex1, ay1, -hz + 0.02), "eye"))
    return boxes + list(extra or [])


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.array(v, np.float64), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


def render(boxes, W, H, eye, target, fov=30.0, light_dir=(-0.5, 0.8, -0.6), ground=True,
           colors=None, ss=1, shadow_soft=0.6, shadow_samples=6):
    """返回 dict：rgb、alpha、depth、normal、mat_id、ground_shadow。"""
    colors = colors or {"body": ORANGE, "eye": np.array([0.05, 0.04, 0.05])}
    Wr, Hr = W * ss, H * ss
    cam, r, u, f = look_at(eye, target)
    aspect = Wr / Hr
    th = math.tan(math.radians(fov) / 2)
    px = (np.arange(Wr) + 0.5) / Wr * 2 - 1
    py = 1 - (np.arange(Hr) + 0.5) / Hr * 2
    PX, PY = np.meshgrid(px * th * aspect, py * th)
    D = f[None, None] + PX[..., None] * r + PY[..., None] * u
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    O = np.broadcast_to(cam, D.shape)

    names = sorted(set(b[2] for b in boxes))
    tbest = np.full((Hr, Wr), np.inf)
    nbest = np.zeros((Hr, Wr, 3))
    mbest = np.full((Hr, Wr), -1)

    def hit_boxes(O, D, tmax=None):
        """对所有盒子求最近交点。"""
        tb = np.full(D.shape[:-1], np.inf)
        nb = np.zeros(D.shape)
        mb = np.full(D.shape[:-1], -1)
        invD = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
        for i, (bmin, bmax, name) in enumerate(boxes):
            bmin, bmax = np.array(bmin), np.array(bmax)
            t1 = (bmin - O) * invD
            t2 = (bmax - O) * invD
            tmin = np.minimum(t1, t2)
            tmx = np.maximum(t1, t2)
            tn = tmin.max(-1)
            tf = tmx.min(-1)
            ok = (tf >= np.maximum(tn, 1e-4))
            if tmax is not None:
                ok &= tn < tmax
            closer = ok & (tn < tb)
            if closer.any():
                tb = np.where(closer, tn, tb)
                axis = tmin.argmax(-1)
                sign = -np.sign(np.take_along_axis(D, axis[..., None], -1)[..., 0])
                nn = np.zeros(D.shape)
                np.put_along_axis(nn, axis[..., None], sign[..., None], -1)
                nb = np.where(closer[..., None], nn, nb)
                mb = np.where(closer, names.index(name), mb)
        return tb, nb, mb

    tbest, nbest, mbest = hit_boxes(O, D)
    hit = np.isfinite(tbest)
    P = O + D * np.where(hit, tbest, 0)[..., None]
    L = np.array(light_dir, np.float64)
    L /= np.linalg.norm(L)

    # 软阴影：在光源方向附近抖动采样
    g = np.random.default_rng(0)
    def shadow_at(P, N):
        vis = np.zeros(P.shape[:-1])
        for k in range(shadow_samples):
            j = L + g.normal(0, 0.06 * shadow_soft, 3)
            j /= np.linalg.norm(j)
            Os = P + N * 1e-3 + j * 1e-3
            ts, _, _ = hit_boxes(Os, np.broadcast_to(j, P.shape))
            vis += ~np.isfinite(ts)
        return vis / shadow_samples

    rgb = np.zeros((Hr, Wr, 3))
    if hit.any():
        vis = np.ones((Hr, Wr))
        idx = np.nonzero(hit)
        vis_hit = shadow_at(P[idx][None], nbest[idx][None])[0]
        vis[idx] = vis_hit
        ndl = np.clip((nbest * L).sum(-1), 0, 1)
        sky = 0.5 + 0.5 * nbest[..., 1]         # 半球环境光
        for i, name in enumerate(names):
            m = (mbest == i)
            c = colors[name]
            rgb[m] = c * (0.32 * sky[m, None] + 0.12 + 0.75 * (ndl * vis)[m, None])
    # 地面接触阴影（Y=0 平面）
    gshadow = np.zeros((Hr, Wr))
    if ground:
        tg = -O[..., 1] / np.where(np.abs(D[..., 1]) < 1e-9, -1e-9, D[..., 1])
        gm = (~hit) & (tg > 0)
        Pg = O + D * np.where(gm, tg, 0)[..., None]
        idx = np.nonzero(gm)
        if len(idx[0]):
            Ng = np.zeros((len(idx[0]), 3)); Ng[:, 1] = 1
            v = shadow_at(Pg[idx][None], Ng[None])[0]
            gshadow[idx] = 1 - v
    out = {"rgb": rgb, "alpha": hit.astype(np.float32), "depth": np.where(hit, tbest, np.inf),
           "normal": nbest, "mat": mbest, "names": names, "ground_shadow": gshadow}
    if ss > 1:
        def ds(a):
            sh = (H, ss, W, ss) + a.shape[2:]
            return a.reshape(sh).mean(axis=(1, 3))
        for k in ("rgb", "alpha", "ground_shadow"):
            out[k] = ds(out[k])
    return out


def sprite(boxes, px_per_cell, az_deg=0.0, elev_deg=8.0, light_dir=(-0.6, 0.8, -0.5), colors=None,
           size_cells=(28, 22), target=(8, 6, 0), ss=1, shadow_samples=4, dist=400.0):
    """近似正交的相机渲染 Clawd，返回裁好边的 RGBA（float）。
    az_deg：相机绕 Y 轴的方位角，0=正面，-90=从 Clawd 右侧看（Clawd 朝向画面右方）。"""
    Wc, Hc = size_cells
    W, H = int(Wc * px_per_cell), int(Hc * px_per_cell)
    fov = math.degrees(2 * math.atan((Hc / 2) / dist))
    a, e = math.radians(az_deg), math.radians(elev_deg)
    tgt = np.array(target, np.float64)
    eye = tgt + dist * np.array([-math.sin(a) * math.cos(e), math.sin(e), -math.cos(a) * math.cos(e)])
    out = render(boxes, W, H, eye, tgt, fov=fov, light_dir=light_dir, ground=False, colors=colors,
                 ss=ss, shadow_samples=shadow_samples)
    rgba = np.dstack([out["rgb"], out["alpha"]]).astype(np.float32)
    ys, xs = np.nonzero(rgba[..., 3] > 0.01)
    p = int(px_per_cell)
    y0, y1, x0, x1 = max(ys.min() - p, 0), min(ys.max() + p, H), max(xs.min() - p, 0), min(xs.max() + p, W)
    return rgba[y0:y1, x0:x1]
