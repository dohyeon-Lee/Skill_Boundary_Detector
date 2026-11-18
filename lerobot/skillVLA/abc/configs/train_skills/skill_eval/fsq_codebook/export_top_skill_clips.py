#!/usr/bin/env python3
"""Export top-view skill clips from the most populated evaluated FSQ tokens.

Uses the completed report manifest, not full-training codebook counts. Each
output is exactly the existing HTML player's [frame_start, frame_end) interval.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import subprocess


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "outputs/abc_grab_clutter_absolute/checkpoints/epoch0800"
)


def choose_diverse(rows, count, seed):
    """Round-robin across tasks so early tasks do not monopolize the export."""
    by_task = defaultdict(list)
    for row in sorted(rows, key=lambda x: (x['task_id'], x['episode_id'], x['skill_index'])):
        by_task[row['task_id']].append(row)
    task_ids = sorted(by_task)
    random.Random(seed).shuffle(task_ids)
    queues = [deque(by_task[t]) for t in task_ids]
    chosen = []
    while len(chosen) < count:
        for queue in queues:
            if queue:
                chosen.append(queue.popleft())
                if len(chosen) == count:
                    break
        if not any(queues) and len(chosen) < count:
            raise ValueError(f'Only {len(rows)} occurrences; requested {count}')
    return chosen


def probe(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,r_frame_rate',
        '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)['streams'][0]


def export_one(job, checkpoint, output):
    row = job['occurrence']
    source = (checkpoint / row['video_path']).resolve()
    destination = output / job['clip_path']
    temporary = destination.with_suffix('.partial.mp4')
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    start, end = int(row['frame_start']), int(row['frame_end'])
    length = end-start
    assert length == row['length'] and length > 0, row
    fps = float(row['video_fps'])
    top, bottom = int(row['video_crop_row_start']), int(row['video_crop_row_end'])
    assert bottom > top >= 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-n', '-threads', '1',
        '-ss', f"{float(row.get('video_from_timestamp', 0.0))+start/fps:.9f}", '-i', str(source),
        '-map', '0:v:0', '-frames:v', str(length),
        '-vf', f'crop=iw:{bottom-top}:0:{top}', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-threads', '2', '-filter_threads', '1',
        '-movflags', '+faststart', str(temporary),
    ], check=True)
    info = probe(temporary)
    if int(info['nb_frames']) != length or int(info['height']) != bottom-top:
        raise RuntimeError(f'Incorrect exported interval/crop: {destination}: {info}')
    rate_num, rate_den = map(int, info['r_frame_rate'].split('/'))
    if abs(rate_num/rate_den-fps) > 1e-6:
        raise RuntimeError(f'Incorrect frame rate: {destination}: {info}')
    temporary.rename(destination)
    print(f"{job['clip_path']} ({length} frames)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path, help='Override the report manifest, e.g. a recovered legacy LIBERO report')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--per-token', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if min(args.top, args.per_token, args.workers) < 1:
        parser.error('top, per-token, and workers must be positive')
    checkpoint = args.checkpoint.resolve()
    manifest_path = (args.manifest or checkpoint/'metrics/manifest.json').resolve()
    manifest = json.loads(manifest_path.read_text())
    if not manifest['completed']:
        raise ValueError('Checkpoint report is not complete')
    records = list(manifest['records'].values())
    counts = Counter(row['token'] for row in records)
    ranked = sorted(counts, key=lambda t: (-counts[t], t))[:args.top]
    if len(ranked) != args.top:
        raise ValueError('Not enough used tokens')
    output = (args.output or checkpoint/f'skill_clips_top{args.top}_{args.per_token}each').resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    summary = {
        'epoch': manifest['epoch_tag'],
        'source_manifest': str(manifest_path),
        'ranking': 'evaluated occurrence count descending, then token ID ascending (not full-training counts)',
        'selection': 'seeded task round-robin; within-task episode/skill order',
        'seed': args.seed,
        'top_view_only': True,
        'tokens': [],
    }
    jobs = []
    for rank, token in enumerate(ranked, 1):
        rows = [row for row in records if row['token'] == token]
        selected = choose_diverse(rows, args.per_token, args.seed)
        levels = manifest['levels']
        base, coord = 1, []
        for level in levels:
            coord.append((token//base) % level)
            base *= level
        folder = f'{rank:02d}_token{token:04d}'
        token_jobs = []
        for index, row in enumerate(selected, 1):
            name = f"{index:02d}_task{row['task_id']:03d}_ep{row['episode_id']:05d}_skill{row['skill_index']:02d}_f{row['frame_start']:04d}-{row['frame_end']:04d}.mp4"
            job = {'clip_path': f'{folder}/{name}', 'occurrence': row}
            token_jobs.append(job)
            jobs.append(job)
        summary['tokens'].append({
            'rank': rank, 'token': token, 'coordinate': coord,
            'evaluated_occurrence_count': counts[token], 'exported': len(selected),
            'clips': token_jobs,
        })
    # Artifact metadata is generated alongside the exported videos for traceability.
    (output/'selection.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Top tokens:', [(t, counts[t]) for t in ranked], flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda job: export_one(job, checkpoint, output), jobs))
    print(f'Done: {len(jobs)} clips in {output}', flush=True)


if __name__ == '__main__':
    main()
