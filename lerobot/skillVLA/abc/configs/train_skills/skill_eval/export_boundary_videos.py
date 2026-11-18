#!/usr/bin/env python3
"""Export three-camera ABC SBD presentations, retaining natural graph widths.

Existing HTML supplies all boundaries and SVG plots. White padding keeps video
and graph aspect ratios independent. Only CPU video composition is performed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import base64

from PIL import Image, ImageDraw


EXAMPLES = Path(__file__).resolve().parents[4]
SHARED = EXAMPLES/'libero/configs/train_skills/skill_eval/src/export_skill_segment_videos.py'
spec = importlib.util.spec_from_file_location('skill_video_shared', SHARED)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
DEFAULT_HTML = EXAMPLES/(
    'analysis/skill_boundary_detection/abc/abc_sim_all/'
    'abc_grab_clutter_obs60_true_h1_previous_action_unanchored_all_tasks_5ep_100k_nms45/'
    'interactive/index.html'
)


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(path)
    ]))['streams'][0]


def export_one(item, html_path, output, camera_width):
    match = re.search(r'task(\d+)\s*·\s*episode\s+(\d+)', item['caption'])
    if not match:
        raise ValueError(item['caption'])
    task, episode = map(int, match.groups())
    destination = output/f'task{task:02d}'/f'episode_{episode:07d}.mp4'
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.mp4')
    if partial.exists():
        raise FileExistsError(partial)
    source = (html_path.parent/unquote(item['source'])).resolve()
    source_info = probe(source)
    cameras = item['cameras']
    if len(cameras) != 3:
        raise ValueError('Expected left wrist, external, right wrist')
    other = next(c for c in cameras if 'data-source-height' in c)
    source_h = int(other['data-source-height'])
    primary_h = source_h-sum(int(c['data-row-end'])-int(c['data-row-start'])
                            for c in cameras if not c.get('primary'))
    # The ABC top view occupies the first band in the official stacked video.
    for c in cameras:
        if c.get('primary'):
            c['data-row-start'], c['data-row-end'] = '0', str(primary_h)
    if source_info['height'] != source_h:
        raise ValueError(f'Source camera layout mismatch: {source}')
    heights = [int(c['data-row-end'])-int(c['data-row-start']) for c in cameras]
    if len(set(heights)) != 1:
        raise ValueError('Camera bands must have equal heights')
    camera_h = round(camera_width*heights[0]/source_info['width'])
    camera_h += camera_h % 2
    video_w = camera_width*3
    svg = ET.fromstring(base64.b64decode(item['timeline_svg'].split(',', 1)[1]))
    # Native CSS size: SVG points -> 96-dpi pixels; do not stretch to video width.
    graph_w = round(float(svg.attrib['viewBox'].split()[2])*96/72)
    graph, box = shared.boundary_graph(item, graph_w)
    canvas_w = max(video_w, graph.width)
    canvas_w += canvas_w % 2
    video_x, graph_x = (canvas_w-video_w)//2, (canvas_w-graph.width)//2
    gap, bar_h, footer_h = 10, 34, 28
    bar_y = camera_h+gap
    graph_y = bar_y+bar_h+footer_h+gap
    canvas_h = graph_y+graph.height
    canvas_h += canvas_h % 2
    fps, total = float(item['data-fps']), int(item['data-n-frames'])
    skills = json.loads(item['data-skills'])
    if not skills or skills[0]['start'] != 0 or skills[-1]['end'] != total:
        raise ValueError('Incomplete skill coverage')
    if any(a['end'] != b['start'] for a, b in zip(skills, skills[1:])):
        raise ValueError('Non-contiguous skill coverage')
    font_bold = shared.font(16, True)
    font_path = shared.font(16).path
    bold_path = font_bold.path
    with tempfile.TemporaryDirectory(prefix='abc-boundary-video-') as tmp:
        tmp = Path(tmp)
        decoration = Image.new('RGBA', (canvas_w, canvas_h), 'white')
        decoration.paste(graph, (graph_x, graph_y))
        draw = ImageDraw.Draw(decoration)
        draw.rectangle((video_x, 0, video_x+video_w-1, camera_h-1), fill=(0,0,0,0))
        for i, skill in enumerate(skills):
            x0 = video_x+round(video_w*skill['start']/total)
            x1 = video_x+round(video_w*skill['end']/total)
            draw.rectangle((x0, bar_y, x1-1, bar_y+bar_h-1), fill=shared.PALETTE[i % len(shared.PALETTE)])
            draw.line((x1-1, bar_y, x1-1, bar_y+bar_h-1), fill='white')
            text = f'sk{i}'
            if draw.textlength(text, font=font_bold)+8 < x1-x0:
                draw.text(((x0+x1)/2, bar_y+bar_h/2), text, anchor='mm', font=font_bold,
                          fill='white', stroke_width=1, stroke_fill='#333333')
        decoration.save(tmp/'decoration.png')
        Image.new('RGBA', (3, bar_h), 'white').save(tmp/'bar_needle.png')
        plot_h = round(graph.height*(1-box['bottom']))-round(graph.height*box['top'])
        Image.new('RGBA', (2, plot_h), '#ef4444').save(tmp/'graph_needle.png')
        filters = ['[0:v]setpts=PTS-STARTPTS,split=3[a][b][c]']
        for i, (name, cam) in enumerate(zip(('a','b','c'), cameras)):
            row = int(cam['data-row-start'])
            filters.append(f'[{name}]crop=iw:{heights[i]}:0:{row},scale={camera_width}:{camera_h}:flags=lanczos[v{i}]')
        filters.append(f'[v0][v1][v2]hstack=inputs=3,pad={canvas_w}:{canvas_h}:{video_x}:0:white[base]')
        filters.append('[base][1:v]overlay=0:0:eof_action=repeat[decorated]')
        chain = []
        for i, cam in enumerate(cameras):
            x = video_x+i*camera_width+8
            chain.append(f"drawtext=fontfile='{bold_path}':text='{cam['label']}':fontsize=13:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=5:x={x}:y={camera_h-24}")
        badge_x = video_x+camera_width+10
        chain.append(f'drawbox=x={badge_x}:y=10:w=170:h=65:color=black@0.72:t=fill')
        for i, skill in enumerate(skills):
            start, end = int(skill['start']), int(skill['end'])
            enable = f"gte(n,{start})*lt(n,{end})"
            chain.append(f"drawtext=fontfile='{bold_path}':text='Skill {i}':fontsize=24:fontcolor=white:x={badge_x+10}:y=18:enable='{enable}'")
            chain.append(f"drawtext=fontfile='{font_path}':text='frames {start}–{end}':fontsize=14:fontcolor=white:x={badge_x+10}:y=48:enable='{enable}'")
            x0 = video_x+round(video_w*start/total)
            x1 = video_x+round(video_w*end/total)
            chain.append(f"drawbox=x={x0+1}:y={bar_y+1}:w={max(2,x1-x0-2)}:h={bar_h-2}:color=black:t=3:enable='{enable}'")
            if i:
                flash = f"gte(n,{start})*lt(n,{min(end,start+math.ceil(.26*fps))})"
                chain.append(f"drawbox=x={video_x}:y=0:w={video_w}:h={camera_h}:color=0x2563eb@0.30:t=fill:enable='{flash}'")
                chain.append(f"drawtext=fontfile='{bold_path}':text='SKILL CHANGE':fontsize=30:fontcolor=white:borderw=1:bordercolor=black:x={video_x}+({video_w}-text_w)/2:y=({camera_h}-text_h)/2:enable='{flash}'")
        chain.append(f"drawtext=fontfile='{font_path}':text='frame %{{n}} / {total-1}':fontsize=14:fontcolor=0x666666:x={video_x}:y={bar_y+bar_h+6}")
        filters.append('[decorated]'+','.join(chain)+'[labels]')
        bar_x = f'{video_x}+{video_w}*min(t*{fps},{total-1})/{total}'
        filters.append(f"[labels][2:v]overlay=x='{bar_x}':y={bar_y}:eof_action=repeat[bar]")
        play_x = f"{graph_x}+{graph.width}*({box['left']}+({box['right']}-{box['left']})*min(t*{fps},{total-1})/{box['frames']})"
        play_y = graph_y+round(graph.height*box['top'])
        filters.append(f"[bar][3:v]overlay=x='{play_x}':y={play_y}:eof_action=repeat,format=yuv420p[out]")
        # Generated filter text is an output artifact, not a modified repository config.
        (tmp/'filters.txt').write_text(';\n'.join(filters))
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-threads', '1',
                   '-ss', item['data-episode-start'], '-i', str(source)]
        for asset in ('decoration.png','bar_needle.png','graph_needle.png'):
            command += ['-i', str(tmp/asset)]
        command += ['-filter_complex_threads', '1', '-filter_complex_script', str(tmp/'filters.txt'),
                    '-map', '[out]', '-frames:v', str(total), '-an', '-c:v', 'libx264',
                    '-preset', 'veryfast', '-crf', '18', '-threads', '2',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
        subprocess.run(command, check=True)
    actual = probe(partial)
    if int(actual['nb_frames']) != total:
        raise RuntimeError(f'Frame-count mismatch: {destination}: {actual}')
    partial.rename(destination)
    print(f'{destination.relative_to(output)} ({total} frames; {canvas_w}x{canvas_h})', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, default=DEFAULT_HTML)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--camera-width', type=int, default=320)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    html_path = args.html.resolve()
    output = (args.output or html_path.parent.parent/'presentation_videos_with_boundary').resolve()
    parsed = shared.Players()
    parsed.feed(html_path.read_text())
    items = parsed.items[:args.limit] if args.limit else parsed.items
    if args.resume:
        items = [item for item in items if not (output/
                 ('task'+re.search(r'task(\d+)',item['caption'])[1])/
                 f"episode_{int(re.search(r'episode\s+(\d+)',item['caption'])[1]):07d}.mp4").exists()]
    if not parsed.items:
        raise ValueError('No episode players found')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda item: export_one(item,html_path,output,args.camera_width), items))
    print(f'Done: {len(items)} new episode presentations in {output}', flush=True)


if __name__ == '__main__':
    main()
