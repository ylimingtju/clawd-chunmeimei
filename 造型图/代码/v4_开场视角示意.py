"""开场视角示意：移轴（off-axis）透视相机，在 Clawd 背后、略低于它头顶，墙面与成像面平行，画框不变形。"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/Users/limingyang/Desktop/单依纯二专/造型图/代码")
import clawd3d as c3

W, H = 1920, 1080
F = 1484.0            # 焦距（像素），竖直视场约 40°
CY = 1020.0           # 视平线在画面里的 y（移轴：视平线压到画面底部附近）
EYE_Y = 8.0           # 相机高度（格）：低于 Clawd 头顶 12
Z_BACK = 35.0         # 相机到 Clawd 背面的距离（格）
D_WALL = 119.0        # 相机到墙的距离（格）
WALL = np.array([0.20, 0.14, 0.22])

def proj(x, y, z_dist, cx=960.0):
    return cx + F * x / z_dist, CY - F * (y - EYE_Y) / z_dist

def scene_bg():
    img = np.ones((H, W, 3)) * WALL
    # 画框：外框 1260×770，底边 900
    fx0, fx1, fy0, fy1 = 330, 1590, 130, 900
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([fx0 + 8, fy0 + 12, fx1 + 8, fy1 + 12], fill=(36, 24, 38))      # 投影
    d.rectangle([fx0, fy0, fx1, fy1], fill=(176, 138, 62))
    d.rectangle([fx0 + 22, fy0 + 22, fx1 - 22, fy1 - 22], fill=(214, 178, 96))
    d.rectangle([fx0 + 48, fy0 + 48, fx1 - 48, fy1 - 48], fill=(120, 90, 40))
    art = Image.open("/Users/limingyang/Desktop/单依纯二专/造型图/第1批_代码绘制/A_00_专辑整体_珍珠万花筒.png").convert("RGB")
    iw, ih = fx1 - fx0 - 140, fy1 - fy0 - 140
    a = art.resize((iw, int(art.height * iw / art.width)))
    a = a.crop((0, (a.height - ih) // 2, iw, (a.height - ih) // 2 + ih))
    im.paste(a, (fx0 + 70, fy0 + 70))
    return np.asarray(im).astype(np.float32) / 255

def clawd_layer(x_off, z_back, az_turn=0.0, lift=0.0):
    """Clawd 背对镜头（面朝墙 -Z），az_turn 为它自身绕 Y 轴转的角度。返回整幅 RGBA。"""
    boxes = c3.clawd_boxes(depth=8)
    a = math.radians(az_turn)
    def rot(p):
        x, y, z = p[0] - 8, p[1], p[2]
        return (x * math.cos(a) + z * math.sin(a) + 8, y, -x * math.sin(a) + z * math.cos(a))
    rb = []
    for bmin, bmax, m in boxes:
        cs = [rot((x, y, z)) for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])]
        cs = np.array(cs)
        rb.append((tuple(cs.min(0) + [x_off, lift, 0]), tuple(cs.max(0) + [x_off, lift, 0]), m))
    # 相机在 +Z：Clawd 背面 z=+4，相机 z = 4 + z_back
    zc = 4 + z_back
    sc = 2  # 半分辨率渲染再放大
    Hr = int(2 * CY / sc)
    fov = math.degrees(2 * math.atan((Hr / 2) / (F / sc)))
    out = c3.render(rb, W // sc, Hr, eye=(8, EYE_Y, zc), target=(8, EYE_Y, zc - 100), fov=fov,
                    light_dir=(-0.35, 0.85, 0.45), ground=False, shadow_samples=3)
    rgba = np.dstack([out["rgb"], out["alpha"]])[: H // sc]
    return np.asarray(Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255

bg = scene_bg()
frames = [("① 看展：背对镜头，头顶挡住画框底边一点", clawd_layer(0, Z_BACK)),
          ("② 被吸进去：边飞边转身，越来越小", clawd_layer(0, 70, az_turn=110, lift=14))]
outs = []
for title, L in frames:
    img = bg * (1 - L[..., 3:]) + L[..., :3] * L[..., 3:]
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    outs.append(im)
sheet = Image.new("RGB", (W, H * 2 + 20), "white")
for i, im in enumerate(outs):
    sheet.paste(im, (0, i * (H + 20)))
sheet.resize((W // 2, (H * 2 + 20) // 2)).save(sys.argv[1], quality=88)
