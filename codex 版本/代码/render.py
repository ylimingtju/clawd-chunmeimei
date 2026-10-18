"""先导出测试帧，检查后逐段编码；多进程仅从 __main__ 启动。"""
import os,sys,time,json,argparse,subprocess,multiprocessing as mp
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from PIL import Image,ImageDraw,ImageFont
from timeline import OUT,SEGMENTS,BY_NAME,FPS


def render_one(task):
    from film import frame
    name,i=task
    im=frame(name,i)
    assert im.mode=='RGB' and im.size==(1920,1080)
    return im.tobytes()


def previews(names,workers=3):
    tasks=[]
    for name in names:
        n=BY_NAME[name]['stop']-BY_NAME[name]['first']
        indices=sorted(set(round(q*(n-1)) for q in (0,.16,.33,.5,.67,.84,1)))
        dest=OUT/'抽帧'/name;dest.mkdir(parents=True,exist_ok=True)
        tasks.extend((name,i) for i in indices)
    start=time.perf_counter()
    with mp.get_context('spawn').Pool(workers,maxtasksperchild=35) as pool:
        for (name,i),raw in zip(tasks,pool.imap(render_one,tasks,chunksize=1)):
            im=Image.frombytes('RGB',(1920,1080),raw)
            im.save(OUT/'抽帧'/name/f'{name}_{i:04d}_全局{BY_NAME[name]["first"]+i:04d}.png')
            print(f'测试帧 {name} {i}',flush=True)
    for name in names:contact(name)
    print(f'测试帧耗时 {time.perf_counter()-start:.1f} 秒',flush=True)


def contact(name):
    paths=sorted((OUT/'抽帧'/name).glob('*.png'))
    im=Image.new('RGB',(4*480,2*300),(28,29,33));d=ImageDraw.Draw(im)
    for k,p in enumerate(paths):
        tile=Image.open(p).resize((480,270),Image.Resampling.LANCZOS)
        x=k%4*480;y=k//4*300;im.paste(tile,(x,y));d.text((x+8,y+276),p.stem,fill='white')
    dest=OUT/'检查';dest.mkdir(exist_ok=True)
    im.save(dest/f'{name}_测试帧.jpg',quality=92)


def encode(names,workers=4):
    import imageio_ffmpeg
    exe=imageio_ffmpeg.get_ffmpeg_exe();dest=OUT/'分段';dest.mkdir(exist_ok=True)
    log=OUT/'检查/渲染计时.jsonl'
    for name in names:
        row=BY_NAME[name];n=row['stop']-row['first'];path=dest/f'{name}.mp4'
        if path.exists():raise FileExistsError(f'已存在分段，避免误覆盖：{path}')
        approved=OUT/'检查'/f'{name}_检查通过.json'
        if not approved.exists():raise RuntimeError(f'{name} 尚未记录测试帧人工检查')
        start=time.perf_counter();stamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        args=[exe,'-hide_banner','-loglevel','warning','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
              '-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18',
              '-pix_fmt','yuv420p','-r','30','-video_track_timescale','15360','-movflags','+faststart','-threads','2','-n',str(path)]
        with (OUT/'检查'/f'{name}_ffmpeg.log').open('wb') as err:
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=err)
            try:
                with mp.get_context('spawn').Pool(workers,maxtasksperchild=48) as pool:
                    for k,raw in enumerate(pool.imap(render_one,[(name,i) for i in range(n)],chunksize=1)):
                        proc.stdin.write(raw)
                        if k%30==0:print(f'编码 {name} {k+1}/{n}，{time.perf_counter()-start:.1f}s',flush=True)
                proc.stdin.close();code=proc.wait()
                if code:raise RuntimeError(f'ffmpeg {code}')
            except BaseException:
                proc.kill();proc.wait();raise
        frames,secs=imageio_ffmpeg.count_frames_and_secs(str(path))
        assert frames==n,(name,frames,n)
        item=dict(segment=name,frames=frames,seconds=frames/30,bytes=path.stat().st_size,
                  render_seconds=round(time.perf_counter()-start,3),started_at=stamp,workers=workers)
        with log.open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        print(json.dumps(item,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preview','encode']);p.add_argument('names',nargs='+');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();names=[r['name'] for r in SEGMENTS] if a.names==['all'] else a.names
    (previews if a.mode=='preview' else encode)(names,a.workers)
