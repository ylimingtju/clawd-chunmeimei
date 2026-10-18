"""本地字体和原标题材质函数；标题动画以本曲拍点计时。"""
import sys, math
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'造型图/代码'))
import titles as T
from common import to_img, blur, fbm
from timeline import SONGS,BPM,ease


@lru_cache(maxsize=24)
def title_image(n):
    text=SONGS[n] if n else '纯妹妹 × Clawd'
    if n==1: a=T.paper_title(text,175,101)
    elif n==3:a=T.oil_title(text,178,103)
    elif n==4:a=T.film_title(text,153,104)
    elif n==5:a=T.watercolor_title(text,165,105)
    elif n==6:a=T.cel_title(text,210,106)
    elif n==8:a=T.ink_title(text,210,108,seal_char='茉')[0]
    elif n==9:
        m,p=T.sand_title(text,181,109);rgb=np.ones(m.shape+(3,))*[.91,.75,.46]
        rgb=rgb*(1-p[...,None]*.75);a=np.dstack([rgb,np.clip(m+p*.9,0,1)])
    else:
        fname={0:'huangyou',2:'huangyou',7:'longcang',10:'serif_xlight'}[n]
        m,_=T.text_mask(text,fname,205 if n in (0,2,7) else 220,pad=65)
        h,w=m.shape;y,x=np.mgrid[:h,:w]
        if n in (0,2):
            grain=fbm(h,w,120+n,octaves=3,base=70)
            col=np.ones((h,w,3))*([.96,.65,.74] if n==2 else [.94,.87,.63])
            col*= (.76+.24*grain)[...,None]
            edge=m-blur(m,3)
            col+=np.maximum(edge,0)[...,None]*.3
            if n==0:col+=.16*np.sin(x[...,None]/62+np.arange(3)*2.1)
            a=T.stack_rgba(T.shadow_layer(m,7,10,4,.6),np.dstack([np.clip(col,0,1),m]))
            if n==2:
                im=to_img(a).convert('RGBA');d=ImageDraw.Draw(im)
                inner=(m>.7)&(blur(m,2)<.85)
                for yy,xx in zip(*np.nonzero(inner & ((x+y)%20<2))):
                    d.line((int(xx),int(yy),int(xx)+2,int(yy)+3),fill=(255,226,230,210),width=1)
                return im
        else:
            alpha=np.clip(m+blur(m,6)*(.9 if n==10 else .2),0,1)
            a=np.dstack([np.ones(m.shape+(3,))*([.84,.95,1] if n==10 else [.98,.98,.96]),alpha])
    im=to_img(a).convert('RGBA')
    if im.width>1660:im=im.resize((1660,round(im.height*1660/im.width)),Image.Resampling.LANCZOS)
    return im


@lru_cache(maxsize=2)
def seal(char):
    assert char in ('茉','莉')
    return to_img(T.ink_title('茉',155,108,seal_char=char)[1]).convert('RGBA')


# Stroke centre-lines are entered locally in Chinese writing order. The actual
# visible outline remains the supplied Ma Shan Zheng font, including its paint texture.
STROKES={
 '向': [[(.5,.02),(.34,.14)],[(.13,.2),(.13,.95)],[(.15,.18),(.9,.18),(.9,.92),(.75,.95)],
        [(.35,.38),(.35,.75)],[(.35,.38),(.7,.38),(.7,.75)],[(.35,.75),(.7,.75)]],
 '日': [[(.18,.04),(.18,.96)],[(.18,.04),(.82,.04),(.82,.96)],[(.18,.5),(.82,.5)],[(.18,.96),(.82,.96)]],
 '葵': [[(.08,.14),(.92,.14)],[(.28,.02),(.28,.27)],[(.7,.02),(.7,.27)],
        [(.13,.34),(.38,.34),(.16,.64)],[(.16,.43),(.27,.53)],[(.65,.3),(.74,.43)],
        [(.88,.31),(.77,.47)],[(.72,.42),(.96,.61)],[(.3,.56),(.72,.56)],
        [(.2,.71),(.84,.71)],[(.49,.59),(.49,.79),(.16,.97)],[(.54,.78),(.89,.97)]],
 '朝': [[(.05,.2),(.5,.2)],[(.26,.02),(.26,.3)],[(.1,.32),(.1,.65)],
        [(.1,.32),(.43,.32),(.43,.65)],[(.1,.48),(.43,.48)],[(.1,.65),(.43,.65)],
        [(.03,.77),(.52,.77)],[(.26,.65),(.26,.98)],[(.62,.08),(.62,.7),(.52,.98)],
        [(.62,.08),(.9,.08),(.9,.95),(.78,.92)],[(.62,.35),(.9,.35)],[(.62,.61),(.9,.61)]],
 '着': [[(.28,.02),(.37,.13)],[(.72,.02),(.62,.13)],[(.14,.2),(.86,.2)],
        [(.21,.34),(.79,.34)],[(.05,.47),(.95,.47)],[(.5,.13),(.45,.53),(.1,.9)],
        [(.33,.56),(.33,.98)],[(.33,.56),(.8,.56),(.8,.98)],
        [(.33,.7),(.8,.7)],[(.33,.83),(.8,.83)],[(.33,.98),(.8,.98)]],
 '夜': [[(.5,.02),(.53,.12)],[(.1,.2),(.92,.2)],[(.39,.25),(.12,.59)],[(.25,.46),(.25,.98)],
        [(.64,.25),(.44,.62)],[(.57,.43),(.86,.43),(.66,.77),(.45,.98)],
        [(.58,.59),(.68,.7)],[(.46,.57),(.64,.8),(.94,.98)]]}


@lru_cache(maxsize=1)
def oil_stroke_arrival():
    mask,boxes=T.text_mask(SONGS[3],'mashanzheng',178,pad=70)
    h,w=mask.shape;arrival=np.full((h,w),99.,np.float32)
    for k,(ch,box) in enumerate(zip(SONGS[3],boxes)):
        left=max(0,int(box[0]-5));right=min(w,int(box[2]+6))
        ys,xs=np.nonzero(mask[:,left:right]>.05)
        if not len(xs):continue
        x0,x1=left+xs.min(),left+xs.max();y0,y1=ys.min(),ys.max()
        l=max(0,x0-13);r=min(w,x1+15);top=max(0,y0-12);bot=min(h,y1+17)
        yy,xx=np.mgrid[top:bot,l:r];X=(xx-x0)/max(1,x1-x0);Y=(yy-y0)/max(1,y1-y0)
        best=np.full(X.shape,np.inf);times=np.zeros(X.shape)
        strokes=STROKES[ch]
        for si,path in enumerate(strokes):
            lengths=[math.dist(a,b) for a,b in zip(path,path[1:])];total=sum(lengths);passed=0
            for (ax,ay),(bx,by),length in zip(path,path[1:],lengths):
                vx,vy=bx-ax,by-ay;f=np.clip(((X-ax)*vx+(Y-ay)*vy)/(vx*vx+vy*vy),0,1)
                dist=(X-ax-f*vx)**2+(Y-ay-f*vy)**2
                use=dist<best;best=np.minimum(best,dist)
                when=(si+(passed+length*f)/total)/len(strokes)
                times=np.where(use,when,times);passed+=length
        arrival[top:bot,l:r]=np.minimum(arrival[top:bot,l:r],(k+times)/4)
    return arrival


def draw_title(im,n,t,u):
    beat=t*BPM.get(n,80)/60
    base=title_image(n);shown=base.copy()
    if n==3:
        shown.putalpha(Image.fromarray((np.asarray(shown.getchannel('A'))*(oil_stroke_arrival()<=beat+.035)).astype(np.uint8)))
    if n in (5,7,9):
        count=len(SONGS[n]);q=min(count,1+beat*4)
        reveal=Image.new('L',base.size,0);rd=ImageDraw.Draw(reveal)
        # Local sweep in each glyph follows a brush path; font files contain no stroke-order data.
        cw=base.width/count
        for k in range(count):
            f=max(0,min(1,q-k))
            if f:rd.rectangle((k*cw,0,k*cw+cw*f,base.height),fill=255)
        shown.putalpha(Image.fromarray(np.minimum(np.asarray(shown.getchannel('A')),np.asarray(reveal))))
    scale=1.;dy=0
    if n in (1,2,6):
        scale=.9+.1*ease(beat*3)
        dy=-10*max(0,math.sin(beat*2*math.pi))*max(0,1-u)
        if n==1 and u>.90:scale*=1-.92*ease((u-.9)/.1)
    if n==8:dy=-13*math.exp(-12*(beat%1)) if beat>0 else 0
    if n==4:
        ghost=shown.copy();ghost.putalpha(ghost.getchannel('A').point(lambda a:int(a*.25*(1-u))))
        im.paste(ghost,(int((1920-ghost.width)/2+22*(1-u)),int(88+12*(1-u))),ghost)
    if n==10:
        d=ImageDraw.Draw(im,'RGBA')
        for k in range(34):
            x=680+(k*67)%570;yy=120+(k*31)%180
            p=max(0,1-beat/2)
            d.line((x-180*p,yy+100*p,x,yy),fill=(185,224,255,int(100*p)),width=1)
    if n==9 and u>.65:
        yy,xx=np.mgrid[:shown.height,:shown.width]
        front=650-650*ease((u-.65)/.35)
        keep=(yy+150<front+15*np.sin((xx+(1920-shown.width)/2)/180+t))
        shown.putalpha(Image.fromarray((np.asarray(shown.getchannel('A'))*keep).astype(np.uint8)))
    if scale!=1:shown=shown.resize((max(1,round(shown.width*scale)),max(1,round(shown.height*scale))),Image.Resampling.LANCZOS)
    x=(1920-shown.width)//2;y=int(70+dy)
    if n==6:x=1120;y=100
    if n==3:x=20;y=50
    if n==9:y=150
    if n==8:x=-25;y=0
    if n==10:x=230;y=170
    im.paste(shown,(x,y),shown)
    if n==6:
        ref=shown.transpose(Image.Transpose.FLIP_TOP_BOTTOM);ref.putalpha(ref.getchannel('A').point(lambda a:int(a*.18)))
        im.paste(ref,(x,y+shown.height+12),ref)
    if n==8 and beat>=1:
        s=seal('茉');im.paste(s,(min(1760,x+shown.width-70),y+shown.height-75),s)
    return im
