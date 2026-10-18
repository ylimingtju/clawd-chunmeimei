"""连接已检查的分段，解码复核并保留每个连接点前后各两帧。"""
import sys,json,time,subprocess,hashlib,re
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
sys.dont_write_bytecode=True
from timeline import *
from characters import self_check as character_check


def endpoint(r,end):
    if r['song']==0:return '00',pose('00',1)
    if r['part'] in ('E','M'):
        s=sides(r)[end];return ('07city' if s=='07a' and end==0 else s),transition_pose(r,end,0)
    if r['part']=='c':return ('10b',pose('10b',1)) if end==0 else ('00',pose('00',1))
    s=f"{r['song']:02d}{r['part']}";label='07city' if s=='07a' and end else s
    return label,pose(s,end)


def originals():
    before=json.loads((OUT/'检查/原文件SHA256.json').read_text())
    checked=[];changed=[]
    for name,want in before.items():
        p=ROOT/name
        actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        (checked if actual==want else changed).append(name)
    result={'checked':len(checked),'changed':changed,'all_unchanged':not changed}
    (OUT/'检查/原文件复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert not changed,changed
    return result


def main():
    self_check();character_check()
    exe=imageio_ffmpeg.get_ffmpeg_exe();dest=OUT/'纯妹妹xClawd_完整版_30fps.mp4'
    timings=[json.loads(s) for s in (OUT/'检查/渲染计时.jsonl').read_text().splitlines()]
    assert {r['segment'] for r in timings}=={r['name'] for r in SEGMENTS}
    joins=[]
    for a,b in zip(SEGMENTS,SEGMENTS[1:]):
        old,pa=endpoint(a,1);new,pb=endpoint(b,0)
        assert old==new,(a['name'],b['name'],old,new)
        err=max(abs(x-y) for x,y in zip(pa,pb));assert err<1e-6,(a['name'],b['name'],err)
        joins.append({'previous':a['name'],'next':b['name'],'frame':b['first'],'style':new,'pose':pa,'pose_error':err})
    concat=OUT/'检查/拼接清单.txt'
    concat.write_text(''.join("file '"+str(OUT/'分段'/f"{r['name']}.mp4").replace("'","'\\''")+"'\n" for r in SEGMENTS))
    start=time.perf_counter()
    if not dest.exists():
        with (OUT/'检查/拼接编码.log').open('wb') as err:
            subprocess.run([exe,'-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(concat),
                            '-map','0:v:0','-c','copy','-an','-movflags','+faststart','-n',str(dest)],stderr=err,check=True)
    frames,secs=imageio_ffmpeg.count_frames_and_secs(str(dest));assert frames==3196
    read=imageio_ffmpeg.read_frames(str(dest),pix_fmt='rgb24');meta=next(read)
    assert meta['size']==(1920,1080) and abs(meta['fps']-30)<1e-8,meta
    assert meta.get('codec')=='h264',meta
    diag=subprocess.run([exe,'-hide_banner','-i',str(dest)],capture_output=True,text=True).stderr
    assert 'yuv420p' in diag and 'Audio:' not in diag,diag
    (OUT/'检查/成片编码信息.txt').write_text(diag)
    wanted={j['frame']+offset for j in joins for offset in (-2,-1,0,1)}
    jd=OUT/'检查/拼接点';jd.mkdir(exist_ok=True);lookup={}
    for j in joins:
        for off in (-2,-1,0,1):lookup[j['frame']+off]=(j,off)
    luminance=[];large_changes=[];unexpected_black=[];bars=[];prior=None
    for i,raw in enumerate(read):
        pixels=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
        sample=pixels[::24,::24].astype(float)
        luminance.append(float(sample.mean()))
        if prior is not None:
            change=float(np.abs(sample-prior).mean())
            if change>22:large_changes.append({'frame':i,'sample_rgb_difference':change})
        prior=sample
        if 180<=i<3076:
            if pixels.max()<3:unexpected_black.append(i)
            if pixels[:12].max()<3 and pixels[-12:].max()<3:bars.append(i)
        if i in wanted:
            im=Image.frombytes('RGB',(1920,1080),raw);j,off=lookup[i]
            folder=jd/f"{j['previous']}_至_{j['next']}";folder.mkdir(exist_ok=True)
            im.save(folder/f'{i:04d}_{off:+d}.png')
        if i%600==0:print(f'成片解码检查 {i}/3196',flush=True)
    assert i+1==3196
    assert not unexpected_black and not bars,(unexpected_black,bars)
    for idx,j in enumerate(joins):
        folder=jd/f"{j['previous']}_至_{j['next']}";files=sorted(folder.glob('*.png'));assert len(files)==4
        sheet=Image.new('RGB',(1920,300),(23,25,29));d=ImageDraw.Draw(sheet)
        arrays=[]
        for k,p in enumerate(files):
            im=Image.open(p);arrays.append(np.asarray(im).astype(float));sheet.paste(im.resize((480,270)),(k*480,0))
            d.text((k*480+8,278),f"{j['previous']} -> {j['next']}   {p.stem}",fill='white')
        sheet.save(folder/'对比.jpg',quality=93)
        # Pixel change is diagnostic: titles appearing on a downbeat legitimately change pixels.
        pa=j['pose'];x,y,s,tilt,yaw=pa
        x0=max(0,round(x-11*s));x1=min(1920,round(x+12*s));y0=max(0,round(y-17*s));y1=min(1080,round(y+2*s))
        j['actor_crop_mean_abs_rgb_difference']=float(np.abs(arrays[1][y0:y1,x0:x1]-arrays[2][y0:y1,x0:x1]).mean())
        j['comparison']=str(folder/'对比.jpg')
    for page in range(6):
        part=joins[page*7:(page+1)*7]
        if not part:break
        sheet=Image.new('RGB',(1920,len(part)*300),(23,25,29))
        for k,j in enumerate(part):sheet.paste(Image.open(j['comparison']),(0,k*300))
        sheet.save(jd/f'拼接检查_{page+1}.jpg',quality=92)
    original_result=originals()
    render_start=min(datetime.fromisoformat(x['started_at']).timestamp() for x in timings)
    render_stop=max(datetime.fromisoformat(x['started_at']).timestamp()+x['render_seconds'] for x in timings)
    latest={x['segment']:x for x in timings}
    result={'path':str(dest),'frames':frames,'seconds':frames/30,'duration_timecode':'01:46.533',
            'width':1920,'height':1080,'fps':30,'codec':'h264','pixel_format':'yuv420p','audio_streams':0,
            'bytes':dest.stat().st_size,'segment_count':43,'preview_frames':301,'join_count':42,'join_check_frames':168,
            'segment_render_seconds':sum(x['render_seconds'] for x in timings),'render_wall_seconds':render_stop-render_start,
            'concat_and_decode_seconds':time.perf_counter()-start,
            'render_started_at':timings[0]['started_at'],'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'original_files':original_result,'joins':joins,'segments':[latest[r['name']] for r in SEGMENTS]}
    result['decode_scan']={'unexpected_black_frames':unexpected_black,'solid_black_top_bottom_bars':bars,
                           'large_adjacent_changes':large_changes,'mean_luminance_range':[min(luminance),max(luminance)]}
    (OUT/'检查/成片核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('joins','segments')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
