"""43 个子段的逐帧绘制。背景在世界坐标变换，角色独立按脚底锚点绘制。"""
import sys,math
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance
sys.dont_write_bytecode=True
from timeline import *
from characters import paste_actor,ASSETS
from film_titles import draw_title,title_image,seal
W,H=1920,1080
YY,XX=np.mgrid[:H,:W].astype(np.float32)


@lru_cache(maxsize=26)
def bg(style):return Image.open(ASSETS/f'{style}_背景.png').convert('RGB')


def camera(im,z=1,dx=0,dy=0,pivot=(960,540)):
    if z==1 and dx==0 and dy==0:return im.copy()
    cx,cy=pivot
    # Edge colour is only a safeguard; camera zooms provide an overscan margin.
    fill=tuple(np.asarray(im.resize((1,1)))[0,0])
    return im.transform((W,H),Image.Transform.AFFINE,(1/z,0,cx-(cx+dx)/z,0,1/z,cy-(cy+dy)/z),
                        Image.Resampling.BICUBIC,fillcolor=fill)


def circle_mask(cx,cy,r,rough=0):
    dist=np.hypot(XX-cx,YY-cy)
    if rough:
        a=np.arctan2(YY-cy,XX-cx);dist+=rough*(np.sin(a*19)+.45*np.sin(a*37+1))
    return Image.fromarray(np.uint8(np.clip((r-dist)/2+.5,0,1)*255))


def plane_mask(value):return Image.fromarray(np.uint8(np.clip(value/2+.5,0,1)*255))


def star(d,x,y,r,color=(255,245,210,230)):
    d.polygon([(x,y-r),(x+r*.18,y-r*.18),(x+r,y),(x+r*.18,y+r*.18),
               (x,y+r),(x-r*.18,y+r*.18),(x-r,y),(x-r*.18,y-r*.18)],fill=color)


def sparks(im,p,t,u,seed=0,count=45):
    d=ImageDraw.Draw(im,'RGBA');x,y,s,ang,_=p
    g=np.random.default_rng(seed)
    for k in range(count):
        gx,gy=g.uniform(-7.5,7.5),g.uniform(-6,-1)
        a=.5+.5*math.sin(t*2*math.pi*BPM.get(seed//10,80)/60+k*2.399)
        if a>.91:
            star(d,x+gx*s,y+gy*s,(3+4*a)*min(1,s/20),(255,252,240,int(220*(a-.9)*10)))


def flower(d,x,y,r,color=(250,227,214,240),angle=0):
    for k in range(5):
        a=angle+k*2*math.pi/5;cx=x+math.cos(a)*r*.55;cy=y+math.sin(a)*r*.55
        d.ellipse((cx-r*.5,cy-r*.5,cx+r*.5,cy+r*.5),fill=color)
    d.ellipse((x-r*.22,y-r*.22,x+r*.22,y+r*.22),fill=(245,193,67,255))


def microphone(im,p,standing=False,color=(220,223,222,255)):
    x,y,s,a,_=p;d=ImageDraw.Draw(im,'RGBA')
    if standing:
        mx=x+1*s;my=y-7*s
        d.line((mx,my,mx,y+1),fill=(32,33,42,255),width=6)
        d.ellipse((mx-48,y-8,mx+48,y+5),fill=(36,34,45,255))
    else:mx=x+9.1*s;my=y-8.2*s
    d.line((mx,my,mx-1.1*s,my+2*s),fill=color,width=max(3,round(s*.3)))
    d.ellipse((mx-.48*s,my-.48*s,mx+.48*s,my+.48*s),fill=(60,62,65,255),outline=color,width=2)


def dv(im,t,u,phones=False):
    d=ImageDraw.Draw(im,'RGBA')
    if phones:
        for k,(x,y) in enumerate([(130,895),(1610,855),(1740,960)]):
            d.rounded_rectangle((x,y,x+132,y+226),radius=14,fill=(12,13,20,255),outline=(132,135,143,230),width=3)
            thumb=im.resize((114,192));im.paste(thumb,(x+9,y+12))
            d=ImageDraw.Draw(im,'RGBA');d.ellipse((x+10,y+155,x+37,y+182),fill=(180,133,107,255))
    from titles import font
    if int(t*2)%2==0:d.ellipse((107,86,123,102),fill=(244,45,49,255))
    d.text((135,72),'REC',font=font('wenkai',32),fill=(245,245,238,245))
    d.text((1500,985),f'00:00:{int(t):02d}:{int(t*30)%30:02d}',font=font('wenkai',29),fill=(240,241,231,245))
    for x,y,sx,sy in [(72,63,1,1),(1848,63,-1,1),(72,1017,1,-1),(1848,1017,-1,-1)]:
        d.line((x+sx*44,y,x,y,x,y+sy*34),fill=(230,230,222,170),width=2)
    # DV scanline noise is deterministic and never modifies the silhouette.
    for k in range(12):
        yy=int((k*107+int(t*30)*17)%1080)
        d.line((0,yy,1920,yy),fill=(210,210,215,8),width=1)


def water_scene(style,u,t,p,step=0):
    import anim_05 as A
    from common import to_img,to_arr
    if style=='05a':
        at=2.25+2.25*u;world,tex=A.world_a(at);z,dx,c=A.cam_a(at)
        base=A.camera_fill(world,z,dx,0,*c)
        im,_=A.draw_a(base,*p[:4],A.walk_legs(step) if step else None,tex)
        im=A.water_drops(im,at,*p[:4],tex,on=True)
    else:
        bt=3*u;world,tex=A.world_b(bt);z,c=A.cam_b(bt)
        base=A.camera_fill(world,z,0,0,*c)
        im,_=A.draw_b(base,*p[:4],A.walk_legs(step) if step else None,tex,tw_seed=int(t*8)+900,lit=ease((bt-.1)/.75))
    return to_img(im).convert('RGB')


def world(style,u,t):
    env=math.sin(math.pi*u)**2
    if style=='00':return kaleidoscope(u,t,hero=False)
    if style=='07a':
        tunnel=bg('07a');city=bg('07city')
        q=ease((u-.16)/.53);z=1+.8*q
        a=camera(tunnel,z,0,0,(1010,490))
        if q:
            # Exit is an actual rectangular opening at the tunnel's vanishing point.
            halfw=90+1400*q**2;halfh=150+900*q**2
            mask=Image.new('L',(W,H));ImageDraw.Draw(mask).rectangle((1010-halfw,500-halfh,1010+halfw,500+halfh),fill=255)
            a=Image.composite(city,a,mask)
        return a
    z=1.;dx=dy=0
    if style=='01a':z=1.035;dx=20*(2*u-1)
    elif style=='01b':z=1.025;dy=14*(2*u-1)
    elif style=='02a':z=1.025;dx=18*(2*u-1)
    elif style in ('02b','03b','09b'):z=1+.025*u
    elif style=='03a':z=1.04;dy=25*(2*u-1)
    elif style=='04b':z=1.04-.04*u
    elif style=='06b':z=1.03;dy=18*(2*u-1)
    elif style=='07b':z=1.02;dx=5*math.sin(t*2.3)*env;dy=4*math.sin(t*3.7)*env
    elif style=='08a' and t>=60/93:z=1+.014*math.exp(-12*((t*93/60-1)%4))*env
    elif style=='08b':z=1+.018*u
    elif style=='10a':z=1+.015*u;dx=-12*u
    base=bg('08quiet') if style=='08a' and t<60/93 else bg(style)
    if style=='08a' and 60/93<=t<60/93+.28:
        r=2200*ease((t-60/93)/.28)
        base=Image.composite(bg('08a'),bg('08quiet'),circle_mask(1300,430,r,12))
    a=camera(base,z,dx,dy)
    d=ImageDraw.Draw(a,'RGBA')
    if style=='02a':
        for k,(x,y) in enumerate([(600,780),(1260,740),(1450,660),(400,710),(1150,580)]):
            r=21*ease(t*117/60-k*.8)
            if r:flower(d,x,y,r,(249,206,217,240))
        if t>.8:
            from titles import font
            for k in range(3):d.text((1100+42*k,440-35*k-12*math.sin(t*2)), 'Z',font=font('huangyou',32-k*5),fill=(242,240,250,200))
    elif style=='03a':
        # The nearby bud turns on its stem, without changing the Clawd geometry.
        x=1115+14*math.sin(math.pi*u);y=720-12*u
        d.line((1115,824,x,y),fill=(71,95,57,230),width=8)
        flower(d,x,y,27,(217,165,50,230),-.4+u*.7)
    elif style=='08a' and t>=60/93:
        beat=t*93/60;impact=math.exp(-5*((beat-1)%4))
        for k in range(45):
            aa=k*2.399;rr=(70+(k*71)%400)*(1+.2*(1-impact))
            cx=960+math.cos(aa)*rr*1.9;cy=640+math.sin(aa)*rr*.8
            r=(3+(k%8)*2)*impact
            if r>.5:d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(20,20,19,int(180*impact)))
    elif style=='10a':
        p=pose(style,u,t);x,y,s,*_=p
        for k in range(18):
            yy=y-3*s+k*2;tail=110+80*math.sin(t*2+k*.2)
            d.line((x-6*s,yy,x-6*s-tail,yy+16*math.sin(k+t)),fill=(185,221,255,45),width=1+k%2)
        star(d,x+230,y-115,16,(235,249,255,255))
    elif style=='09a':
        phase=math.sin(2*math.pi*u)
        lighting=Image.new('RGBA',(W,H));ld=ImageDraw.Draw(lighting)
        for x in range(0,W,12):
            shade=max(0,phase*(2*x/W-1))
            ld.rectangle((x,0,x+12,H),fill=(4,8,25,int(65*shade)))
        a.paste(lighting,(0,0),lighting)
    return a


def scene(style,u,t,p=None,step=0,arm=None,with_title=False,motion=None):
    p=list(p if p is not None else pose(style,u,t))
    song=int(style[:2]);beat=t*BPM.get(song,80)/60;env=math.sin(math.pi*u)**2
    if style.startswith('05'):im=water_scene(style,u,t,p,step)
    elif style=='00':return kaleidoscope(u,t,hero=True,p=p)
    else:
        im=world(style,u,t)
        if motion:im=camera(im,*motion)
        if style=='04a':
            q=ease((beat-1)/2)
            for k,(x,y,s,yaw) in enumerate([(1500,790,50,0),(1010,735,26,-38)]):
                ap=1-q
                if ap>.01:
                    gp=[mix(x,p[0],q),mix(y,p[1],q),mix(s,p[2],q),0,mix(yaw,p[4],q)]
                    paste_actor(im,style,gp,opacity=(.62 if k==0 else .45)*ap,force3d=True)
        if style=='06a':
            for k in range(5,0,-1):
                ss=24*(.72**k);xp=330+290*k;yp=905-82*k
                gp=[xp,yp,ss,env*5*math.sin(math.pi*beat/2-k),0]
                paste_actor(im,'06a',gp,arm_delta=env*18*math.sin(math.pi*beat/2-k),blink=(int(beat)%5==k))
                refl=Image.new('RGBA',(W,H));paste_actor(refl,'06a',gp)
                crop=refl.crop((round(xp-10*ss),round(yp-13*ss),round(xp+10*ss),round(yp)))
                crop=crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                crop.putalpha(crop.getchannel('A').point(lambda a:int(a*.22)))
                im.paste(crop,(round(xp-10*ss),round(yp+8)),crop)
        if style=='02b':
            for k,x in enumerate((350,620,1320,1580)):
                gp=[x,943,10,env*8*math.sin(math.pi*beat/2+k*math.pi),0]
                paste_actor(im,f'02d{k}',gp,arm_delta=env*22*math.sin(math.pi*beat/2),step=step)
        actual='07city' if style=='07a' and abs(p[4])<.05 else style
        if arm is None:
            arm=env*(17 if style in ('01a','06a','06b','07b','08b','02b') else 3)*math.sin(math.pi*beat/(2 if song==6 else 1))
        if style=='01a':arm+=42*math.sin(math.pi*u)**2
        if style=='08a':arm+=env*20*math.sin(math.pi*max(0,beat-1))
        paste_actor(im,actual,p,arm_delta=arm,step=step,blink=(env>.2 and beat%1<.08),force3d=style=='07a')
    if style in ('07city','07b') or (style=='07a' and abs(p[4])<45):microphone(im,p,color=(180,181,182,255))
    if style=='06b':microphone(im,p,standing=True)
    if style in ('02b','04b','09b'):sparks(im,p,t,u,song*10+2)
    if style=='08a':
        d=ImageDraw.Draw(im,'RGBA');q=max(0,min(1,(beat-1)/2))
        fy=p[1]-8*p[2]-190*math.sin(math.pi*q);fx=p[0]+9*p[2]+45*q
        flower(d,fx,fy,16,(251,250,234,250),t*.7)
    if style=='08b':
        d=ImageDraw.Draw(im,'RGBA')
        for k,(x,y) in enumerate([(160,175),(1640,190),(260,790),(1630,810)]):
            if beat>=k:
                st=seal('茉' if k%2==0 else '莉');im.paste(st,(x,y),st)
        for k in range(24):
            pulse=math.exp(-5*(beat%4));r=(4+k%8)*pulse;x=(k*367)%1920;y=(k*193)%1080
            d.ellipse((x-r,y-r,x+r,y+r),fill=(90,86,81,int(180*pulse)))
    if style in ('07a','07city','07b'):dv(im,t,u,phones=style=='07b')
    if with_title:draw_title(im,song,t,u)
    if style=='09a':sand_tide(im,u,t,p)
    return im


def sand_tide(im,u,t,p):
    d=ImageDraw.Draw(im,'RGBA');q=.5-.5*math.cos(2*math.pi*u)
    yline=1050-340*q
    wave=[(x,yline+22*math.sin(x/170+t)) for x in range(0,1921,16)]
    # Translucent water is a local scene prop, never a scene transition.
    d.polygon(wave+[(1920,1080),(0,1080)],fill=(168,196,197,75))
    d.line(wave,fill=(245,224,172,190),width=4)
    for k in range(6):
        x=600+k*45;y=950-k*13
        if y<yline:d.ellipse((x-5,y-2,x+6,y+3),fill=(91,66,41,120))
    if u>.65:
        top=650-650*ease((u-.65)/.35)
        if top>5:
            d.line([(x,top+15*math.sin(x/180+t)) for x in range(0,1921,10)],fill=(244,223,176,150),width=3)


def kaleidoscope(u,t,hero=True,p=None):
    angle=30*(1-ease(u))
    base=bg('00')
    if angle:
        a=math.radians(angle)
        z=math.cos(a)+W/H*abs(math.sin(a))+.01
        rw,rh=math.ceil(W*z),math.ceil(H*z)
        base=base.resize((rw,rh),Image.Resampling.BICUBIC).rotate(angle,Image.Resampling.BICUBIC)
        base=base.crop(((rw-W)//2,(rh-H)//2,(rw-W)//2+W,(rh-H)//2+H))
    else:base=base.copy()
    # Four surrounding figures and the independently anchored protagonist.
    for k,a0 in enumerate((-198,-126,-54,18)):
        a=math.radians(a0+angle);cx=960+590*math.cos(a);cy=515+350*math.sin(a)
        tilt=-a0-90-angle;tr=math.radians(tilt)
        p2=[cx+72*math.sin(tr),cy+72*math.cos(tr),12,tilt,0]
        paste_actor(base,'00',p2,arm_delta=35*math.sin(math.pi*u)**2)
    if hero:paste_actor(base,'00',p or pose('00',1),arm_delta=35*math.sin(math.pi*u)**2)
    return base


def opening(row,i):
    n=row['stop']-row['first'];u=i/(n-1);t=i/30
    if row['name']=='00-1':
        q=ease(u);p=[960,mix(630,982,q),mix(5,17,q),0,0]
        inside=kaleidoscope(0,0,False)
        r=190+210*u+1900*ease((u-.57)/.43)
        # Reuse the approved spherical normals, nacre and specular highlight.
        pearl=opening_pearl();scale=r/300
        layer=pearl.convert('RGBa').transform((W,H),Image.Transform.AFFINE,
            (1/scale,0,pearl.width/2-960/scale,0,1/scale,pearl.height/2-540/scale),Image.Resampling.BICUBIC).convert('RGBA')
        im=Image.new('RGB',(W,H));im.paste(layer,(0,0),layer)
        inner=r*ease((u-.35)/.65)
        im=Image.composite(inside,im,circle_mask(960,540,inner))
        paste_actor(im,'00',p)
        im=ImageEnhance.Brightness(im).enhance(.035+.965*ease(u*1.6))
        d=ImageDraw.Draw(im,'RGBA')
        if r<1400:
            d.ellipse((960-r,540-r,960+r,540+r),outline=(200,218,246,int(220*q)),width=4)
            d.arc((960-r*.84,540-r*.88,960+r*.84,540+r*.88),205,275,fill=(255,251,234,int(230*q)),width=9)
            star(d,960-r*.61,540-r*.69,14,(255,249,218,int(255*q)))
        return im
    im=kaleidoscope(u,t)
    title=title_image(0);d=ImageDraw.Draw(im,'RGBA')
    if t<2.5:
        s=.76+.24*ease(t/.5);tt=title.resize((round(title.width*s),round(title.height*s)),Image.Resampling.LANCZOS)
        tt=tt.rotate(8*(1-ease(t/.6)),Image.Resampling.BICUBIC,expand=True)
        im.paste(tt,((W-tt.width)//2,430),tt)
    else:
        for k in range(100):
            x=400+(k*97)%1120+(k%2*2-1)*(t-2.5)*250;y=490+(k*67)%160-(t-2.5)*110
            star(d,x,y,3,(255,236,186,int(255*(1-ease((t-2.5)/.47)))))
    return im


@lru_cache(maxsize=1)
def opening_pearl():
    from pearl import pearl
    from common import to_img
    return to_img(pearl(300,light=(-.72,-.62,.32),world=np.asarray(bg('00'))/255,
                        world_mix=.3,ambient=.02,ss=2)).convert('RGBA')


def transition(row,i):
    n=row['stop']-row['first'];u=i/(n-1);t=i/30
    old,new=sides(row);p=transition_pose(row,u,t);env=math.sin(math.pi*u)**2
    old_duration=3 if old=='00' else BY_NAME[f'{old[:2]}-{old[-1]}']['end']-BY_NAME[f'{old[:2]}-{old[-1]}']['start']
    if u==0:return scene(old,1,old_duration,p)
    if u==1:return scene(new,0,0,p)
    if row['name']=='05-M':return flower_transition(u,t,p)
    # Both worlds draw the identical screen-space anchor and cell size.
    # Their background camera move is independent of the actor's unscaled grid geometry.
    step=t*BPM[row['song']]/60*2*env
    movement=(1+.055*env,-42*env,-22*env)
    if row['name']=='04-E':movement=None
    a=scene('07city' if old=='07a' else old,1,old_duration,p,step=step,motion=movement)
    b=scene(new,0,0,p,step=step,motion=movement)
    name=row['name'];q=ease(u);meta=None
    if name=='01-E':
        cx=p[0]+(1-q)*80;cy=p[1]-6*p[2]
        r=140+1450*ease((u-.12)/.88)
        # The bubble's film reaches the hero before its membrane expands into the book world.
        cx+=560*(1-q)
        mask=circle_mask(cx,cy,r);meta=('bubble',cx,cy,r)
    elif name in ('01-M','03-E','06-E','08-M','10-M'):
        travel={'01-M':2400,'03-E':2550,'06-E':2400,'08-M':2500,'10-M':2450}[name]
        seam=p[0]+travel*(.5-q)
        curve=28*np.sin((YY-p[1])/210)*env if name=='01-M' else 0
        if name=='08-M':curve=45*np.sin(YY/120)+14*np.sin(YY/19)
        mask=plane_mask(XX-seam+curve)
        meta=('vertical',seam,name)
    elif name=='02-E':
        seam=p[1]-6*p[2]+1300*(.5-q)
        wave=25*np.sin(XX/120)+13*np.sin(XX/37)
        mask=plane_mask(YY-seam+wave);meta=('cloud',seam)
    elif name=='02-M':
        half=1450*q**1.25;cx=p[0]+180*(1-q)
        mask=plane_mask(half-np.abs(XX-cx)+10*np.sin(YY/80));meta=('curtain',cx,half)
    elif name=='03-M':
        seam=p[1]-6*p[2]+1400*(.5-q)
        mask=plane_mask(YY-seam-.13*(XX-960));meta=('moonpath',seam)
    elif name=='04-E':
        # Hero walks left out of the circular spotlight. The old circle recedes to the right.
        cx=p[0]+1400*q;cy=p[1]-6*p[2];r=1700*(1-q)+230
        mask=Image.fromarray(255-np.asarray(circle_mask(cx,cy,r)));meta=('spot',cx,cy,r)
    elif name in ('04-M','09-M'):
        seam=p[1]+300-1800*q
        mask=plane_mask(YY-seam);meta=('warm' if name=='04-M' else 'sand',seam)
    elif name=='05-E':
        # A real door first appears behind the complete 04-b stage, then the camera follows the exit.
        cx=1290-330*q;half=60+1500*ease((u-.16)/.84)
        top=640-1300*ease((u-.08)/.92);bottom=1050+500*q
        mask=Image.new('L',(W,H));ImageDraw.Draw(mask).rectangle((cx-half,top,cx+half,bottom),fill=255)
        meta=('door',cx-half,top,cx+half,bottom)
    elif name=='06-M':
        seam=p[0]+2450*(.5-q)
        mask=plane_mask(XX-seam);meta=('tube',seam)
    elif name=='07-E':
        seam=p[0]+2450*(.5-q);mask=plane_mask(XX-seam);meta=('dark',seam)
    elif name=='07-M':
        # Pull back the physical phone. The hero walks across its left bezel into the audience's world.
        left=-600+1450*q;right=2500-1050*q;top=-350+570*q;bottom=1500-510*q
        inner=Image.new('L',(W,H));ImageDraw.Draw(inner).rounded_rectangle((left,top,right,bottom),radius=22,fill=255)
        mask=Image.fromarray(255-np.asarray(inner));meta=('phone',left,top,right,bottom)
        if u>.82:
            # The phone is lowered out of the shot; the actor remains at the stage anchor.
            shift=1400*ease((u-.82)/.18)
            inner=Image.new('L',(W,H));ImageDraw.Draw(inner).rounded_rectangle((left,top+shift,right,bottom+shift),radius=22,fill=255)
            mask=Image.fromarray(255-np.asarray(inner));meta=('phone',left,top+shift,right,bottom+shift)
    elif name=='08-E':
        cx=p[0]+8*p[2];cy=p[1]-8*p[2];r=2400*q
        mask=circle_mask(cx,cy,r,rough=18*env);meta=('ink',cx,cy,r)
    elif name=='09-E':
        cx=p[0];cy=p[1];r=2300*ease((u-.1)/.9)
        mask=circle_mask(cx,cy,r,rough=20*env);meta=('sandcircle',cx,cy,r)
    elif name=='10-E':
        seam=p[1]-5*p[2]+1500*(.5-q)
        mask=plane_mask(YY-seam+.35*(XX-p[0]));meta=('stageedge',seam)
    else:raise ValueError(name)
    out=Image.composite(b,a,mask)
    boundary(out,meta,u,t,p)
    return out


def boundary(im,m,u,t,p):
    d=ImageDraw.Draw(im,'RGBA');kind=m[0];env=math.sin(math.pi*u)**2
    if kind=='bubble':
        _,x,y,r=m
        d.ellipse((x-r,y-r,x+r,y+r),outline=(235,240,255,235),width=4)
        d.arc((x-r*.98,y-r*.98,x+r*.98,y+r*.98),185,255,fill=(255,206,235,245),width=10)
    elif kind=='vertical':
        _,x,name=m
        col={'01-M':(245,222,204,255),'03-E':(246,220,138,255),'06-E':(191,221,245,255),
             '08-M':(20,19,17,255),'10-M':(213,239,250,210)}[name]
        w={'01-M':15,'03-E':4,'06-E':14,'08-M':26,'10-M':20}[name]
        pts=[(x-(28*math.sin((y-p[1])/210)*env if name=='01-M' else 0),y) for y in range(0,1081,12)]
        d.line(pts,fill=col,width=w)
        if name=='01-M':d.line([(x+20,y) for _,y in pts],fill=(117,92,78,100),width=3)
        if name=='06-E':d.line((x+26,0,x+26,1080),fill=(245,248,255,180),width=3)
        if name=='03-E':
            d.line((0,320,x,350,1920,250),fill=(244,211,145,200),width=3)
            r=18+55*ease(u);d.ellipse((x-r,350-r,x+r,350+r),fill=(249,223,150,240))
        if name=='08-M':
            for k in range(28):d.line((x-90,30+k*39,x+30,24+k*39),fill=(230,218,189,170),width=2+k%3)
    elif kind=='cloud':
        y=m[1]
        for x in range(-60,1980,90):
            yy=y-25*math.sin(x/120)-13*math.sin(x/37)
            d.ellipse((x-75,yy-30,x+75,yy+30),fill=(239,236,247,190))
    elif kind=='curtain':
        _,x,r=m
        for edge,sign in ((x-r,-1),(x+r,1)):
            d.line([(edge+8*math.sin(y/50),y) for y in range(0,1081,8)],fill=(190,130,174,240),width=18)
            for k in range(3):d.line((edge+sign*(25+k*17),0,edge+sign*(25+k*17),1080),fill=(180,115,154,70),width=3)
        flower(d,p[0]+6*p[2],p[1]-12*p[2]-30,45*(1-u),(243,195,208,220),t)
    elif kind=='moonpath':
        y=m[1];d.line([(x,y+.13*(x-960)) for x in range(0,1921,30)],fill=(241,217,140,210),width=7)
    elif kind=='spot':
        _,x,y,r=m;d.ellipse((x-r,y-r,x+r,y+r),outline=(223,199,145,180),width=4)
    elif kind in ('warm','sand'):
        y=m[1];d.line((0,y,1920,y),fill=(248,211,130,160),width=5 if kind=='warm' else 2)
        if kind=='sand':
            for k in range(150):
                x=(k*173)%1920;yy=y-((k*61+t*100)%170)
                star(d,x,yy,1+k%3,(248,222,164,180))
    elif kind=='door':
        _,left,top,right,bottom=m
        d.line((left,bottom,left,top,right,top,right,bottom),fill=(247,226,169,245),width=17)
        d.line((left+18,bottom,left+18,top+18,right-18,top+18,right-18,bottom),fill=(123,86,54,160),width=5)
    elif kind in ('tube','dark'):
        x=m[1]
        d.line((x,0,x,1080),fill=(234,241,250,230 if kind=='tube' else 110),width=9)
        if kind=='tube':d.line((x+12,0,x+12,1080),fill=(140,186,222,100),width=3)
    elif kind=='phone':
        _,left,top,right,bottom=m
        d.rounded_rectangle((left-15,top-20,right+15,bottom+20),radius=36,outline=(12,13,20,255),width=20)
        d.line((left+40,top-8,right-40,top-8),fill=(160,167,182,190),width=2)
    elif kind in ('ink','sandcircle'):
        _,x,y,r=m;g=np.random.default_rng(880)
        for k in range(100):
            a=g.uniform(0,2*math.pi);rr=r+g.uniform(-30,35);cx=x+rr*math.cos(a);cy=y+rr*math.sin(a);rad=g.uniform(1,7)
            d.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),fill=(24,22,18,200) if kind=='ink' else (224,191,130,220))
    elif kind=='stageedge':
        y=m[1];d.line([(x,y-.35*(x-p[0])) for x in range(0,1921,20)],fill=(181,170,192,230),width=7)
        star(d,p[0]+100+130*u,p[1]-100,12,(246,241,226,250))


def flower_transition(u,t,p):
    import anim_05 as A
    from common import to_img
    at=4.5+1.5*u;wa,ta=A.world_a(at);z,dx,c=A.cam_a(at)
    a=A.camera_fill(wa,z,dx,0,*c);fx,fy=A.to_screen(*A.FLOWER,z,dx,c)
    r=45*z;full=math.hypot(max(fx,W-fx),max(fy,H-fy))+50
    r+=full*ease((u-.81)/.19)
    wb,tb=A.world_b(0);b=A.camera_fill(wb,1+.4*(1-u),0,0,960,540)
    # The common actor flies to the flower before the camera carries it onto the stage.
    a0=pose('05a',1);b0=pose('05b',0)
    if u<.55:
        q=ease(u/.55);p=[mix(a0[0],fx,q),mix(a0[1],fy+75,q)-85*math.sin(math.pi*q),mix(a0[2],16,q),-12*math.sin(math.pi*q),0]
    else:
        q=ease((u-.55)/.45);p=[mix(fx,b0[0],q),mix(fy+75,b0[1],q),mix(16,b0[2],q),0,0]
    a,_=A.draw_a(a,*p[:4],None,ta,shadow=1-ease(u/.4))
    b,_=A.draw_b(b,*p[:4],None,tb,tw_seed=980,lit=0)
    out=Image.composite(to_img(b).convert('RGB'),to_img(a).convert('RGB'),circle_mask(fx,fy,r))
    d=ImageDraw.Draw(out,'RGBA');d.ellipse((fx-r,fy-r,fx+r,fy+r),outline=(239,183,114,150),width=max(2,min(30,round(z*2))))
    return out


@lru_cache(maxsize=10)
def pearl_world(n):
    # References inside the ten pearls come from our generated drawing layers.
    if n==5:
        im=scene('05a',.5,1.125)
    else:im=scene(f'{n:02d}a',.5,.8)
    im=im.resize((120,120),Image.Resampling.LANCZOS)
    return im


def ending(row,i):
    u=i/(row['stop']-row['first']-1);t=i/30
    if i==0:return scene('10b',1,2)
    p=[mix(960,960,ease(u)),mix(930,982,ease(u/.67)),mix(30,17,ease(u/.67)),0,0]
    a=scene('10b',1,2,p);b=kaleidoscope(1,3,True,p)
    q=ease((u-.19)/.48);radius=1500*q
    mask=circle_mask(960,530,radius)
    im=Image.composite(b,a,mask);d=ImageDraw.Draw(im,'RGBA')
    # Beams contract to a pearl, then the camera reveals the complete chain.
    gather=1-ease(u/.25)
    hand=(p[0]+9*p[2],p[1]-8*p[2])
    for x in (250,560,960,1360,1670):
        d.line((mix(hand[0],x,gather),mix(hand[1],0,gather),*hand),fill=(218,237,255,int(170*gather)),width=2)
    chain=ease((u-.12)/.44)
    for k in range(10):
        angle=-math.pi/2+2*math.pi*k/10
        x=mix(hand[0],960+600*math.cos(angle),chain);y=mix(hand[1],530+400*math.sin(angle),chain)
        r=mix(14,59,chain)
        if chain<.01 and k:continue
        thumb=pearl_world(k+1).resize((round(2*r),round(2*r)),Image.Resampling.LANCZOS)
        pm=Image.new('L',thumb.size);ImageDraw.Draw(pm).ellipse((0,0,thumb.width-1,thumb.height-1),fill=255)
        im.paste(thumb,(round(x-r),round(y-r)),pm);d=ImageDraw.Draw(im,'RGBA')
        d.ellipse((x-r,y-r,x+r,y+r),outline=(238,227,213,240),width=3)
        d.arc((x-r*.85,y-r*.85,x+r*.85,y+r*.85),195,275,fill=(255,253,234,220),width=4)
    if t>=8/3:
        im=ImageEnhance.Brightness(im).enhance(1-ease((t-8/3)/(119/30-8/3)))
    return im


def frame(name,i):
    row=BY_NAME[name];n=row['stop']-row['first'];assert 0<=i<n
    if row['song']==0:return opening(row,i)
    if row['part'] in ('E','M'):return transition(row,i)
    if row['part']=='c':return ending(row,i)
    u=i/(n-1);return scene(f"{row['song']:02d}{row['part']}",u,i/30,with_title=row['part']=='a')
