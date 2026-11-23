"""Overlay recorded EEF projections on actual Coffee dataset video frames (CPU)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'lerobot/src/lerobot').is_dir())
sys.path.insert(0,str(ROOT/'lerobot/skillVLA/libero/configs/train_skills/src'))
from train_skills_config import load_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=load_config(a.config)
    import av
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from lerobot.policies.skillVLA.focus_projection import project_eef
    from lerobot.policies.skill_expert.wrist_patch_target import WristCamera,project_into_wrist
    raw=ROOT/str(cfg['dataset_root'])/str(cfg.get('target_dataset','coffee_d0_full_full'))
    camera=json.loads((raw/'meta/mimicgen_camera.json').read_text());height,width=camera['height'],camera['width']
    project=np.asarray(camera['top_world_to_pixel']);flip=np.eye(4);flip[0,0]=-1;flip[0,2]=width-1
    project=flip@project
    episodes=pd.concat([pd.read_parquet(f) for f in sorted((raw/'meta/episodes').rglob('*.parquet'))],ignore_index=True).set_index('episode_index')
    frame=pd.concat([pd.read_parquet(f,columns=['episode_index','frame_index','timestamp','observation.state']) for f in sorted((raw/'data').rglob('*.parquet'))],ignore_index=True)
    info=json.loads((raw/'meta/info.json').read_text());samples=[]
    for ep, fraction in [(0,0.0),(0,.45),(125,.25),(350,.65)]:
        df=frame[frame.episode_index==ep].sort_values('frame_index');index=int(round(fraction*(len(df)-1)))
        samples.append((ep,index,df))
    images={};records=[]
    for ep,index,df in samples:
        state=np.stack(df['observation.state']);future=min(index+20,len(state)-1)
        for view,key in [('top','observation.images.image'),('wrist','observation.images.wrist_image')]:
            row=episodes.loc[ep];video=raw/info['video_path'].format(video_key=key,chunk_index=int(row[f'videos/{key}/chunk_index']),file_index=int(row[f'videos/{key}/file_index']))
            seconds=float(row[f'videos/{key}/from_timestamp'])+float(df.iloc[index]['timestamp'])
            with av.open(str(video)) as stream:
                vs=stream.streams.video[0];stream.seek(max(0,int((seconds-.25)/vs.time_base)),stream=vs)
                best=None
                for decoded in stream.decode(vs):
                    if decoded.time is None:continue
                    if best is None or abs(decoded.time-seconds)<abs(best[0]-seconds):best=(decoded.time,decoded.to_ndarray(format='rgb24'))
                    if decoded.time>=seconds:break
            if best is None or abs(best[0]-seconds)>1/(2*info['fps'])+.001:raise RuntimeError(f'Frame alignment failed {ep}:{index}')
            if view=='top':pixels=[project_eef(state[j,:3],project,height=height,width=width).raw_xy for j in [index,future]]
            else:
                cam=WristCamera(position=tuple(camera['wrist_camera_position_from_eef_site_in_eef_body']),
                    rotation=tuple(np.asarray(camera['wrist_camera_rotation_in_eef_body']).reshape(-1)),
                    fovy=camera['wrist_fovy_degrees'],mirror_x=False)
                uv,depth=project_into_wrist(torch.tensor(np.repeat(state[index:index+1,:3],2,axis=0)),
                    torch.tensor(np.repeat(state[index:index+1,3:6],2,axis=0)),torch.tensor(state[[index,future],:3]),cam,height=height,width=width)
                pixels=uv.numpy().tolist()
            scale=2;canvas=Image.fromarray(best[1]).resize((width*scale,height*scale));draw=ImageDraw.Draw(canvas)
            for (u,v),color in zip(pixels,['#39ff76','#00d5ff']):
                x,y=u*scale,v*scale
                if 0<=u<width and 0<=v<height:
                    draw.ellipse((x-9,y-9,x+9,y+9),outline='black',width=5)
                    draw.line((x-12,y,x+12,y),fill=color,width=3);draw.line((x,y-12,x,y+12),fill=color,width=3)
            label=f'Episode {ep} | frame {index} | {view}'
            draw.rectangle((0,0,width*scale,30),fill='#111827');draw.text((8,7),label,fill='white')
            records.append(dict(episode=ep,frame=index,view=view,video=str(video),timestamp=seconds,decoded_timestamp=best[0],eef_uv=pixels[0],future_frame=future,future_eef_uv=pixels[1]))
            images[(view,ep,index)]=canvas
    output=Path(__file__).resolve().parents[1]/'outputs/coffee_d0_uv_check';output.mkdir(parents=True,exist_ok=True)
    for view in ['top','wrist']:
        sheet=Image.new('RGB',(width*4,height*4+62),'#111827');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),'GREEN: current recorded EEF   |   CYAN: EEF after 20 frames (1 sec)',fill='white')
        draw.text((12,28),'No image flip or manual marker adjustment. Crosses outside the image are omitted.',fill='#b3bfd3')
        for i,(ep,index,_) in enumerate(samples):sheet.paste(images[(view,ep,index)],((i%2)*width*2,62+(i//2)*height*2))
        sheet.save(output/f'{view}_uv_overlay.png')
    (output/'projections.json').write_text(json.dumps(records,indent=2)+'\n')
    print(output)
if __name__=='__main__':main()
