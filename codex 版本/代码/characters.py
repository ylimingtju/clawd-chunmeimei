"""同一格画布的角色、刚性肢体动画与限定镜头的三维渲染。"""
import sys, math, json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'造型图/代码'))
from common import check_proportions, clawd_masks, ARMS, LEGS, BODY, rigid_arm, to_img
import clawd3d

ASSETS=HERE/'绘制层'
BBOX=(-8,-10,30,14)
CELL=32
ORIGIN=(256,320)
SIZE=(1216,768)


@lru_cache(maxsize=32)
def sprite(style):
    return Image.open(ASSETS/f'{style}_角色.png').convert('RGBA')


@lru_cache(maxsize=32)
def geometry(style):
    return json.loads((ASSETS/f'{style}_几何.json').read_text())


def initial_angle(shape, side):
    pts=shape if isinstance(shape[0],list) else [[shape[0],shape[1]],[shape[2],shape[1]],
                                                [shape[2],shape[3]],[shape[0],shape[3]]]
    cx=sum(p[0] for p in pts)/4;cy=sum(p[1] for p in pts)/4
    vx=cx-(0 if side=='L' else 16);vy=cy-4
    return math.degrees(math.atan2(-vy,abs(vx)))


@lru_cache(maxsize=32)
def limb_layers(style):
    g=geometry(style)
    cm,_,_=clawd_masks(CELL,0,bbox=BBOX,arms=g['arms'],extra=g['extra'])
    cloth=np.zeros((SIZE[1],SIZE[0]),np.float32)
    for k in g['extra']:
        if k not in ('closed','eyes_up','glove','mic','glasses','can'):
            cloth=np.maximum(cloth,cm[k])
    return cm,cloth


def posed_sprite(style, arm_delta=0, step=0, blink=False):
    base=sprite(style)
    g=geometry(style)
    # The stored outline is generated with the original check, then checked again here.
    check_proportions(arms=g['arms'],legs=LEGS,extra=g['extra'])
    if abs(arm_delta)<.05 and abs(step)<.01 and not blink:return base
    cm,cloth=limb_layers(style)
    arr=np.asarray(base).copy()
    if step:
        remove=cm['legs']*(1-cloth)
        arr[...,3]=(arr[...,3]*(1-remove)).astype(np.uint8)
        for k,(x0,y0,x1,y1) in enumerate(LEGS):
            lift=.55*max(0,math.sin(step*math.pi+(k%2)*math.pi))
            moved=(x0,y0-lift,x1,y1-lift)
            assert abs(moved[2]-moved[0]-1.3)<1e-6 and abs(moved[3]-moved[1]-2)<1e-6
            assert moved[1]<=10 and moved[3]>=10
            # Translate an entire original leg texture; clothing remains in front.
            x0p=round(ORIGIN[0]+x0*CELL);x1p=round(ORIGIN[0]+x1*CELL)
            y0p=ORIGIN[1]+10*CELL;y1p=ORIGIN[1]+12*CELL
            leg=np.asarray(base)[y0p:y1p,x0p:x1p].copy()
            yy=y0p-round(lift*CELL);hh=leg.shape[0]
            cover=cloth[yy:yy+hh,x0p:x1p]
            aa=leg[...,3:4]/255*(1-cover[...,None])
            sub=arr[yy:yy+hh,x0p:x1p]
            sub[...,:3]=(sub[...,:3]*(1-aa)+leg[...,:3]*aa).astype(np.uint8)
            sub[...,3]=np.maximum(sub[...,3],leg[...,3]*(1-cover)).astype(np.uint8)
    if abs(arm_delta)>=.05:
        remove=cm['arms']*(1-cm['body'])*(1-cloth)
        arr[...,3]=(arr[...,3]*(1-remove)).astype(np.uint8)
        out=Image.fromarray(arr)
        d=ImageDraw.Draw(out)
        new=[]
        for k,side in enumerate(('L','R')):
            a=initial_angle(g['arms'][k],side)+arm_delta
            sh=rigid_arm(side,a);new.append(sh)
            # Keep glove colour in the rotated block.
            color={'03b':(102,61,199,255),'07b':(242,239,235,255),
                   '08b':(204,38,26,255)}.get(style,(202,124,94,255))
            d.polygon([(ORIGIN[0]+px*CELL,ORIGIN[1]+py*CELL) for px,py in sh],fill=color)
        check_proportions(arms=new,legs=LEGS,extra=g['extra'])
        arr=np.asarray(out).copy()
    if blink and style not in ('02a',):
        for x0 in (3.6,11.4):
            x0p=round(ORIGIN[0]+x0*CELL); yp=ORIGIN[1]+3*CELL
            # Eyelids retain the rectangular language of the original character.
            patch=arr[yp:yp+64,x0p:x0p+32]
            patch[:]=(202,124,94,255)
            patch[28:38]=(20,17,20,255)
    return Image.fromarray(arr)


def paste_actor(canvas, style, p, arm_delta=0, step=0, blink=False, opacity=1, shadow=False, force3d=False):
    ax,ay,s,ang,yaw=p
    if force3d or style=='04a' or abs(yaw)>.05:
        im,anchor=three_d(style,round(yaw,1))
        scale=s/24
        im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
        anchor=(anchor[0]*scale,anchor[1]*scale)
        pos=(round(ax-anchor[0]),round(ay-anchor[1]))
        if opacity<1:im=im.copy();im.putalpha(im.getchannel('A').point(lambda q:round(q*opacity)))
        canvas.paste(im,pos,im)
        return canvas
    im=posed_sprite(style,arm_delta,step,blink)
    if s!=CELL:im=im.resize((round(38*s),round(24*s)),Image.Resampling.LANCZOS)
    # Common canvas centre = grid (11,2); anchor = grid (8,12).
    a=math.radians(ang);vx,vy=3*s,-10*s
    cx=ax+vx*math.cos(a)+vy*math.sin(a)
    cy=ay-vx*math.sin(a)+vy*math.cos(a)
    if ang:im=im.convert('RGBa').rotate(ang,Image.Resampling.BICUBIC,expand=True).convert('RGBA')
    if opacity<1:im=im.copy();im.putalpha(im.getchannel('A').point(lambda q:round(q*opacity)))
    if shadow:
        sh=Image.new('RGBA',canvas.size)
        ImageDraw.Draw(sh).ellipse((ax-8.4*s,ay-5,ax+8.4*s,ay+15),fill=(8,8,15,60))
        sh=sh.filter(ImageFilter.GaussianBlur(12));canvas.paste(sh,(0,0),sh)
    canvas.paste(im,(round(cx-im.width/2),round(cy-im.height/2)),im)
    return canvas


@lru_cache(maxsize=32)
def three_d(style,yaw):
    """Only called in the five turning shots named by v3. No 2-D width squeezing."""
    check_proportions(arms=ARMS,legs=LEGS)
    orange=np.array([202,124,94])/255
    colors={'body':orange,'eye':np.array([.035,.03,.04]),'cloth':np.array([.055,.055,.065]),
            'trim':np.array([.16,.17,.22]),'glove':np.array([.4,.24,.78])}
    extras=[]
    if style in ('04a',):extras=[((-.06,2,-4.06),(16.06,6,4.06),'cloth')]
    elif style in ('07a','07city','07back'):
        # Open front, continuous back; 8-cell body depth and a single central row of legs.
        extras=[((-2,1,-4.15),(2.9,12.4,4.15),'cloth'),((13.1,1,-4.15),(18,12.4,4.15),'cloth'),
                ((0,2,3.9),(16,12.4,4.15),'cloth'),((2.8,6.7,-4.25),(5.5,9.3,-4.1),'trim'),
                ((10.5,6.7,-4.25),(13.2,9.3,-4.1),'trim'),((5.5,8.4,-4.26),(10.5,8.7,-4.1),'trim')]
    elif style in ('03b','06b','04b'):
        colors['cloth']=np.array({'03b':[.72,.6,.42],'06b':[.64,.64,.67],'04b':[.86,.68,.30]}[style])
        extras=[((-.2,.15,-4.1),(16.2,6.4,4.1),'cloth')]
    boxes=clawd3d.clawd_boxes(depth=8,leg_rows=1,leg_depth=1.3,leg_gap=0,extra=extras)
    assert all(abs(b[0][2]+b[1][2])<1e-8 for b in boxes[:4])
    s=24;w,h=38*s,32*s;dist=400;a=math.radians(-yaw);e=math.radians(4*min(1,abs(yaw)/20))
    target=np.array([8.,6.,0.]);eye=target+dist*np.array([-math.sin(a)*math.cos(e),math.sin(e),-math.cos(a)*math.cos(e)])
    fov=math.degrees(2*math.atan(16/dist))
    light=(-.65*math.cos(a)+.6*math.sin(a),.8,-.65*math.sin(a)-.6*math.cos(a))
    result=clawd3d.render(boxes,w,h,eye,target,fov=fov,light_dir=light,ground=False,
                          colors=colors,ss=1,shadow_samples=2)
    rgb=result['rgb'];alpha=result['alpha']
    # Front surface carries the approved drawing's texture; sides are actual shaded volumes.
    texture_style={'07a':'07city','07back':'07city'}.get(style,style)
    if texture_style!='04a':
        front_view=math.cos(a)>0
        if not front_view:
            texture_style='07back' if style in ('07a','07city','07back') else texture_style
        tex=np.asarray(sprite(texture_style))/255
        cam,r,u,f=clawd3d.look_at(eye,target)
        xx=(np.arange(w)+.5)/w*2-1;yy=1-(np.arange(h)+.5)/h*2
        X,Y=np.meshgrid(xx*math.tan(math.radians(fov)/2)*w/h,yy*math.tan(math.radians(fov)/2))
        direction=f+X[...,None]*r+Y[...,None]*u
        direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        # Approved clothing is a textured front plane in the same three-dimensional model.
        # The depth test preserves real sides; the underlying arms and the one row of legs remain solid boxes.
        plane_z=-4.18 if front_view else 4.18
        distance=(plane_z-cam[2])/np.where(abs(direction[...,2])>1e-9,direction[...,2],1e-9)
        pts=cam+direction*distance[...,None]
        tx=np.clip(np.rint(ORIGIN[0]+pts[...,0]*CELL).astype(int),0,SIZE[0]-1)
        ty=np.clip(np.rint(ORIGIN[1]+(12-pts[...,1])*CELL).astype(int),0,SIZE[1]-1)
        cm,cloth=limb_layers(texture_style)
        ta=tex[...,3]*(1-np.maximum(cm['arms'],cm['legs'])*(1-cloth))
        good=(distance>0)&(distance<result['depth']+.03)
        pa=ta[ty,tx]*good
        material=tex[ty,tx,:3]
        if not front_view and texture_style=='07back':material=np.clip(material*1.15+.025,0,1)
        rgb=rgb*(1-pa[...,None])+material*pa[...,None]
        alpha=np.maximum(alpha,pa)
    if style=='04a':
        lum=rgb@np.array([.3,.55,.15]);rgb=1.28*(.75*lum[...,None]*np.array([.92,.98,1.1])+.25*rgb)
    rgba=to_img(np.dstack([rgb,alpha])).convert('RGBA')
    # Project the physical foot-centre (8,0,0), not the cropped image centre.
    _,r,u,f=clawd3d.look_at(eye,target);v=np.array([8.,0.,0.])-eye
    z=v@f;th=math.tan(math.radians(fov)/2)
    anchor=((v@r/z/th/(w/h)+1)*w/2,(1-v@u/z/th)*h/2)
    # clawd3d's front camera has screen-right = -world-X. Align it to the
    # common drawing grid without changing any box dimensions or faking a turn.
    return rgba.transpose(Image.Transpose.FLIP_LEFT_RIGHT),(w-anchor[0],anchor[1])


def self_check():
    for p in ASSETS.glob('*_几何.json'):
        g=json.loads(p.read_text());check_proportions(arms=g['arms'],legs=g['legs'],extra=g['extra'])
        assert g['body']==[0,0,16,10] and g['bbox']==list(BBOX)
    for side in ('L','R'):
        for angle in range(-90,91,5):check_proportions(arms=[rigid_arm(side,angle)])
    print('全部造型、统一格画布及刚性手臂检查通过。')


if __name__=='__main__':self_check()
