"""v3 帧边界与角色连续轨迹；所有段使用同一个绝对帧时间轴。"""
from pathlib import Path
import json, math
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]
SEGMENTS=json.loads((OUT/'时间轴.json').read_text())
BY_NAME={r['name']:r for r in SEGMENTS}
FPS=30
BPM={1:82,2:117,3:70,4:63,5:80,6:151,7:114,8:93,9:101,10:90}
SONGS=['','还有什么更好的','纯妹妹','向日葵朝着夜','假如我们还爱着','我管不住我自己',
       '有趣','我表示理解','还你茉莉','多少的光阴','珠玉']


def ease(v):
    v=max(0.,min(1.,v))
    return v*v*(3-2*v)


def mix(a,b,u):
    return a+(b-a)*u


# x / foot / pixels per grid cell / whole-body tilt / actual 3-D yaw
ENDPOINTS={
 '00':((960,982,17,0,0),(960,982,17,0,0)),
 '01a':((925,793,19,0,0),(895,793,19,0,0)),
 '01b':((960,908,24,0,0),(960,908,25,0,0)),
 '02a':((950,701,27,0,0),(905,701,27,3,0)),
 '02b':((960,905,24,0,0),(960,905,24.5,0,0)),
 '03a':((905,813,13,0,0),(905,813,13,0,0)),
 '03b':((965,885,29,0,0),(965,885,30,0,0)),
 '04a':((760,760,26,0,-75),(960,860,26,0,-40)),
 '04b':((960,998,28,0,0),(960,998,26,0,0)),
 '05a':((850.275,761.54,17.765,0,0),(887.75,762.6,17.85,0,0)),
 '05b':((908,944,26,0,0),(904.88,958.64,27.56,0,0)),
 '06a':((330,905,24,0,0),(330,905,24,0,0)),
 '06b':((980,925,26,0,0),(980,925,26,0,0)),
 '07a':((1040,930,22,0,180),(1040,930,35,0,0)),
 '07city':((1040,930,35,0,0),(1040,930,35,0,0)),
 '07b':((960,900,32,0,0),(960,900,32,0,0)),
 '08a':((960,905,22,0,0),(960,905,22,0,0)),
 '08b':((960,1000,27,0,0),(960,1000,27,0,0)),
 '09a':((880,890,15,0,0),(880,890,15,0,0)),
 '09b':((960,965,26,0,0),(960,965,26,0,0)),
 '10a':((560,575,9,14,0),(820,550,11,12,0)),
 '10b':((960,930,30,0,0),(960,930,30,0,0)),
}


def pose(style,u,t=0):
    a,b=ENDPOINTS[style]; e=ease(u)
    p=[mix(aa,bb,e) for aa,bb in zip(a,b)]
    env=math.sin(math.pi*u)**2
    song=int(style[:2]);beat=t*BPM.get(song,80)/60
    slow=beat/2 if song==6 else beat
    p[3]+=env*(3.5 if song in (1,2,5,6) else 1.5)*math.sin(math.pi*slow)
    p[1]-=env*(9 if song in (2,5,6) else 3)*abs(math.sin(math.pi*slow))
    if style=='04a':
        p[0]=mix(760,960,ease((u-.45)/.55));p[1]=mix(760,860,ease((u-.45)/.55))
        p[4]=mix(-75,-40,ease((u-.7)/.3))
    if style=='07a':
        if u<.5:
            q=ease(u/.5);p=[mix(1040,1010,q),mix(930,750,q),mix(22,17,q),0,180]
        else:
            q=ease((u-.5)/.5);p=[mix(1010,1040,q),mix(750,930,q),mix(17,35,q),0,mix(180,0,q)]
    return p


def sides(row):
    n=row['song'];part=row['part']
    if part=='E':
        return ('00' if n==1 else f'{n-1:02d}b'),f'{n:02d}a'
    if part=='M':return f'{n:02d}a',f'{n:02d}b'
    if part=='c':return '10b','00'
    raise ValueError(row)


def transition_pose(row,u,t):
    old,new=sides(row)
    a=pose(old,1);b=pose(new,0)
    if old=='07a': old='07city'
    q=ease(u);p=[mix(aa,bb,q) for aa,bb in zip(a,b)]
    env=math.sin(math.pi*u)**2
    name=row['name']; beat=t*BPM[row['song']]/60
    if name in ('01-E','02-E','02-M','06-M','10-E'):
        p[1]-=(180 if name!='10-E' else 130)*math.sin(math.pi*u)
        p[3]+=env*(-8 if name!='02-M' else 6)
    elif name in ('03-E',):
        p[0]+=230*math.sin(math.pi*u);p[1]-=330*math.sin(math.pi*u);p[3]-=env*10
    elif name=='04-E':
        p[0]-=100*env;p[4]=-75*ease(u)
    elif name=='04-M':p[4]=-40*(1-ease(u))
    elif name=='07-E':p[4]=180*ease(u)
    elif name=='07-M':
        p[2]+=7*env;p[0]-=220*env
    elif name=='08-E':p[3]+=5*env
    elif name=='08-M':p[3]-=9*math.sin(math.pi*u)
    elif name=='09-E':p[1]+=18*math.sin(math.pi*min(u*2,1))
    elif name=='10-M':p[1]-=70*env
    else:
        p[0]+=140*env;p[3]+=2.5*math.sin(math.pi*beat*2)*env
        p[1]-=5*abs(math.sin(math.pi*beat*2))*env
    return p


def self_check():
    assert SEGMENTS[0]['first']==0 and SEGMENTS[-1]['stop']==3196
    for a,b in zip(SEGMENTS,SEGMENTS[1:]):assert a['stop']==b['first']
    for row in SEGMENTS:
        if row['part'] in ('E','M'):
            old,new=sides(row)
            for end,style in [(0,old),(1,new)]:
                p=transition_pose(row,end,0)
                expected=pose(style,1-end)
                assert max(abs(x-y) for x,y in zip(p,expected))<1e-6,(row['name'],end,p,expected)
    print('时间轴与所有 E/M 的首尾锚点检查通过：3196 帧，43 子段。')


if __name__=='__main__':self_check()
