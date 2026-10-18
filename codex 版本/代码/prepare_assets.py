"""复制定稿绘制脚本，再在新副本中拆层。执行 python prepare_assets.py 00 01 ...。"""
from pathlib import Path
import sys, re, subprocess, os
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
SRC=HERE.parents[1]/'造型图/代码'
COPIES=HERE/'源绘制副本'
COPIES.mkdir(exist_ok=True)


def between(s, start, end, new=''):
    a=s.index(start); b=s.index(end,a)
    return s[:a]+new+s[b:]


def setup(name):
    s=(SRC/name).read_text()
    s=s.replace('OUT = sys.argv[1]', 'OUT = str(ASSETS)')
    s=s.replace('from common import *', '''from common import *
from asset_support import *
clawd_masks = checked_masks
paste_rgba = no_actor
sprite_alpha_layer = no_actor_alpha
''')
    s=re.sub(r'^print\(.*$', '', s, flags=re.M)
    return s


def save_as(s, tag):
    s=re.sub(r'^save\(img, .*$', f'export_bg("{tag}", img)', s, count=1, flags=re.M)
    return s


def build(tag):
    if tag=='00':
        s=setup('A_专辑整体.py')
        s=s.replace('sprite = enamel_sprite(0)', 'sprite = enamel_sprite(0)\nexport_sprite("00", sprite, S, (ox,oy), extra={"pants": pants})')
        s=save_as(s,'00')
    elif tag=='01':
        s=setup('S01_还有什么更好的.py')
        s=s.replace('from titles import paper_title, place','from titles import paper_title\nplace = no_actor')
        s=s.replace('CX, CY = 1010, 659','export_sprite("01a", spr, S, (ox,oy), arms=arms_up, extra={"veil":[veil]})\nCX, CY = 1010, 659')
        s=s.replace('CX, CY = 960, 776','export_sprite("01b", spr, S, (ox,oy))\nCX, CY = 960, 776')
        s=save_as(save_as(s,'01a'),'01b')
    elif tag=='02a':
        s=setup('B_纯妹妹.py').replace('rot = 3.0','rot = 0.0')
        s=s.replace('img = img * (1 - 0.35 * sh)[..., None]','')
        s=s.replace('img = comp(img, clawd)\nimg = comp(img, beads)',
            'export_sprite("02a", rgba_over(clawd, beads), S, (bx+ox, by+oy), extra={"gown":[gown], "flares":[flare_l,flare_r,hip_l,hip_r]})')
        s=between(s,'# 钉珠的小闪光','# 前景：花丛、松树、亭子')
        s=save_as(s,'02a')
    elif tag=='02b':
        s=setup('S02b_纯妹妹现场.py')
        s=s.replace('big, _ = felt_clawd(11','big, d_origin = felt_clawd(11')
        s=s.replace('    img = img * (1 - 0.3 * sh)[..., None]','')
        s=s.replace('    img = comp(img, felt(big, 10 + i, puff=8, fuzz=0.3, light=0.7, sh=5))',
            '    export_sprite(f"02d{i}", felt(big, 10+i, puff=8, fuzz=0.3, light=0.7, sh=5), 11, d_origin)')
        s=s.replace('img = img * (1 - 0.35 * sh)[..., None]','')
        s=s.replace('img = comp(img, clawd)','')
        s=s.replace('img = comp(img, felt(done(im), 22, puff=2, fuzz=0.15, light=0.6, sh=2))',
            'export_sprite("02b", rgba_over(clawd, felt(done(im),22,puff=2,fuzz=0.15,light=0.6,sh=2)), S, (gx0,gy0))')
        s=between(s,'# 亮片闪光','# 前景：几颗毛毡小星星彩带')
        s=save_as(s,'02b')
    elif tag=='03':
        s=setup('S03_向日葵朝着夜.py')
        s=s.replace('from titles import oil_title, place','from titles import oil_title\nplace = no_actor')
        s=s.replace('CX, CY = 905, 735','export_sprite("03a", oil_sprite(spr,31), S, (ox,oy))\nCX, CY = 905, 735')
        s=s.replace('CX, CY = 1010, 720','export_sprite("03b", oil_sprite(spr,37), S, (ox,oy))\nCX, CY = 1010, 720')
        s=save_as(save_as(s,'03a'),'03b')
    elif tag=='04':
        s=setup('S04_假如我们还爱着.py')
        s=s.replace('from titles import film_title, place','from titles import film_title\nplace = no_actor')
        s=between(s,'# 三维 Clawd：','# 去色：')
        s=s.replace('CX, CY = 960, 751','export_sprite("04b", spr, S, (ox,oy))\nCX, CY = 960, 751')
        s=save_as(save_as(s,'04a'),'04b')
    elif tag=='06':
        s=setup('S06_有趣.py')
        s=s.replace('from titles import cel_title, place','from titles import cel_title\nplace = no_actor')
        s=between(s,'# 6 个镜像 Clawd：','# =====================================================================\n# 06-b',
            'export_bg("06a", img)\nexport_sprite("06a", cel_clawd(32,outfit_parts()), 32, (192,160))\n\n')
        s=s.replace('CX, FOOT = 980, 925','export_sprite("06b", spr, S)\nCX, FOOT = 980, 925')
        s=between(s,'# 立式麦克风','save(img,')
        s=save_as(s,'06b')
    elif tag=='07':
        s=setup('S07_我表示理解.py')
        s=s.replace('af = sprite_alpha_layer(spr, cx, cy)',
            'export_sprite("07back", spr, S)\naf = sprite_alpha_layer(spr, cx, cy)',1)
        marker='# 麦克风（握在右手末端，朝向嘴边）'
        s=between(s,marker,'# 前景：虚焦观众头部剪影',
            'export_sprite("07b", spr, S)\n')
        s=between(s,'# 前景：虚焦观众头部剪影','save(img, os.path.join(OUT, "07-b',
            'img = dv_finish(img, 78, warm=True)\n')
        s=save_as(save_as(s,'07a'),'07b')
    elif tag=='07city':
        s=setup('C_我表示理解.py')
        s=between(s,'# 麦克风（右钳握住','img = img * np.array([0.8',
            'export_sprite("07city", sprite, S, (ox,oy), extra={"jacket":jacket,"collar":[collar_l,collar_r],"glasses":glasses})\n\n')
        s=save_as(s,'07city')
    elif tag=='08':
        s=setup('S08_还你茉莉.py')
        s=s.replace('from titles import ink_title, place','from titles import ink_title\nplace = no_actor')
        a=s.index('# Clawd：素纱'); b=s.index('# 抛起的茉莉',a)
        block=s[a:b]
        new='''bg_hold = img.copy()
img = np.zeros((H,W,3),np.float32)
'''+block+'''
black = img.copy()
img = np.ones((H,W,3),np.float32)
'''+block+'''
alpha = np.clip(1-(img-black).mean(-1),0,1)
rgba = np.dstack([black/np.maximum(alpha[...,None],1e-8),alpha])
export_sprite("08a",rgba,S,(ccx-pw/2+ox,ccy-ph/2+oy))
img = bg_hold
'''
        s=s[:a]+new+s[b:]
        s=between(s,'# 抛起的茉莉','# 地面淡墨')
        a=s.index('S = 27'); b=s.index('# 握拳的冲击线',a)
        block=s[a:b]
        new='bg_hold = img.copy()\nimg = np.zeros((H,W,3),np.float32)\n'+block+'''
black = img.copy()
img = np.ones((H,W,3),np.float32)
'''+block+'''
alpha = np.clip(1-(img-black).mean(-1),0,1)
rgba = np.dstack([black/np.maximum(alpha[...,None],1e-8),alpha])
export_sprite("08b",rgba,S,(960-pw/2+ox,ccy-ph/2+oy))
img = bg_hold
'''
        s=s[:a]+new+s[b:]
        s=between(s,'# 握拳的冲击线','# 低角度仰拍的透视感')
        s=save_as(save_as(s,'08a'),'08b')
        # Preserve a quiet first beat before the ink percussion enters.
        q=s[:s.index('# Clawd：素纱')]
        # The explicit background below is saved before the isolated actor block.
        s=s.replace('# 鼓点：中心偏上一团泼墨炸开 + 狂草横扫',
                    'quiet_paper = img.copy()\n# 鼓点：中心偏上一团泼墨炸开 + 狂草横扫')
        s=s.replace('bg_hold = img.copy()', '''quiet_paper=drum(quiet_paper,300,820,170,150,seed=6,angle=-.12)
quiet_paper=drum(quiet_paper,1660,840,120,100,seed=7,angle=.15)
quiet_paper=bell(quiet_paper,1560,150,38,seed=9)
quiet_paper=bell(quiet_paper,1640,215,28,seed=10)
export_bg("08quiet", quiet_paper)
bg_hold = img.copy()''',1)
    elif tag=='09':
        s=setup('S09_多少的光阴.py')
        s=s.replace('# Clawd：坐在沙滩上', 'D_hold = D.copy()\n# Clawd：坐在沙滩上')
        s=between(s,'# 身边不断流动的沙','# 歌名：', '''
actor = light_table(np.clip(D,0,None),back,tint=tint)
alpha = np.clip(body+outline,0,1)
export_sprite("09a",np.dstack([actor,alpha]),15,gp(0,0))
D = D_hold
tint = None
''')
        s=between(s,'# 歌名：','D = blur(D, 1.2)')
        s=s.replace('S = 26\ngown', 'D_hold = D.copy()\ntint_hold = tintC.copy()\nS = 26\ngown')
        s=s.replace('D = blur(D, 0.8)', '''actor = light_table(np.clip(D,0,None),back,tint=tintC)
alpha = np.clip(fig+mic,0,1)
export_sprite("09b",np.dstack([actor,alpha]),S,gp(0,0))
D = D_hold
tintC = tint_hold
fig = np.zeros((H,W),np.float32)
D = blur(D, 0.8)''')
        s=save_as(save_as(s,'09a'),'09b')
    elif tag=='10a':
        s=setup('D_珠玉.py')
        s=between(s,'SCX, SCY, SROT =','# ---------- 雾、粒子、电影调色',
            'export_sprite("10a",spr,s,(sox,soy))\n')
        s=between(s,'# 宽银幕遮幅','img = grain')
        s=save_as(s,'10a')
    elif tag=='10b':
        s=setup('S10_珠玉.py')
        s=s[:s.index('# =====================================================================\n# 10-c')]
        s=s.replace('CX, FOOT = 960, 930','export_sprite("10b",spr,S,(ox,oy))\nCX, FOOT = 960, 930')
        s=s.replace('img[:70] *= 0','').replace('img[-70:] *= 0','')
        s=save_as(s,'10b')
    else:
        raise ValueError(tag)
    pre='''# 定稿脚本的新副本：背景/角色拆层，原文件保持只读。
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_support import *
'''
    path=COPIES/f'asset_{tag}.py'
    path.write_text(pre+s)
    return path


if __name__=='__main__':
    tags=sys.argv[1:] or ['00','01','02a','02b','03','04','06','07','07city','08','09','10a','10b']
    for tag in tags:
        path=build(tag)
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,'-B',str(path)],check=True,env=env)
