"""正片绘制副本的导出工具。源脚本保持只读；统一格画布，保存可复用绘制层。"""
from pathlib import Path
import sys, json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / '造型图/代码'
ROOT = SOURCE.parent.parent
ASSETS = HERE / '绘制层'
ASSETS.mkdir(exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
import common

BBOX = (-8, -10, 30, 14)
CELL = 32
ORIGIN = (-BBOX[0]*CELL, -BBOX[1]*CELL)
SIZE = ((BBOX[2]-BBOX[0])*CELL, (BBOX[3]-BBOX[1])*CELL)
LAST = {}


def checked_masks(s, pad=0, ss=3, extra=None, arms=None, legs=None, bbox=None, body=None, eyes=None):
    common.check_proportions(arms=arms or common.ARMS, legs=legs, extra=extra)
    if body is not None:
        assert body == [common.BODY], '身体必须是 16×10'
    result = common.clawd_masks(s, pad, ss, extra, arms, legs, bbox, body, eyes)
    LAST.update(s=s, origin=result[1], arms=arms or common.ARMS, extra=extra or {})
    return result


def rgba_over(a, b):
    aa, ba = a[..., 3:4], b[..., 3:4]
    alpha = ba + aa*(1-ba)
    rgb = (b[..., :3]*ba+a[..., :3]*aa*(1-ba))/np.maximum(alpha, 1e-8)
    return np.concatenate([rgb, alpha], -1)


def export_sprite(name, rgba, s=None, origin=None, arms=None, extra=None):
    s = s or LAST['s']; origin = origin or LAST['origin']
    arms = arms or LAST.get('arms', common.ARMS)
    extra = extra if extra is not None else LAST.get('extra', {})
    common.check_proportions(arms=arms, legs=common.LEGS, extra=extra)
    im = common.to_img(rgba).convert('RGBA')
    # Each output pixel samples the same grid coordinate in the original sprite.
    ratio = s/CELL
    coeff = (ratio, 0, origin[0]-ORIGIN[0]*ratio,
             0, ratio, origin[1]-ORIGIN[1]*ratio)
    norm = im.convert('RGBa').transform(SIZE, Image.Transform.AFFINE, coeff,
               resample=Image.Resampling.BICUBIC).convert('RGBA')
    norm.save(ASSETS/f'{name}_角色.png')
    (ASSETS/f'{name}_几何.json').write_text(json.dumps(dict(cell=CELL,bbox=BBOX,
        origin=ORIGIN,anchor=[8,12],body=common.BODY,legs=common.LEGS,arms=arms,extra=extra),
        ensure_ascii=False,indent=2))
    print('角色', name, flush=True)


def export_bg(name, arr):
    assert arr.shape == (1080,1920,3), (name,arr.shape)
    common.to_img(arr).save(ASSETS/f'{name}_背景.png')
    print('背景',name,flush=True)


def no_actor(canvas, *args, **kwargs):
    return canvas


def no_actor_alpha(*args, **kwargs):
    return np.zeros((1080,1920),np.float32)


def oil_sprite(rgba, seed):
    from painterly import paint, impasto
    rgb, h = paint(np.clip(rgba[...,:3],0,1),[(3000,(5,12),(2,4),False),
                       (6000,(3,7),(1,2),False)],seed=seed)
    rgb = impasto(rgb*.45+rgba[...,:3]*.55,h,.65)
    return np.dstack([rgb,rgba[...,3]])
