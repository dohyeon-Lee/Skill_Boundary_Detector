#!/usr/bin/env python3
"""Recover one legacy LIBERO HTML checkpoint and export its most used tokens.

Reads existing report-data JavaScript as JSON, never executes JavaScript, and
uses episode metadata to locate source frames in concatenated LeRobot videos.
No simulation, encoding-model inference, or Slurm submission is needed.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
EXAMPLES = WORKSPACE.parents[4]
REPO = EXAMPLES.parents[1]
ABC_EXPORTS = EXAMPLES/'abc/configs/train_skills/skill_eval/fsq_codebook'
DEFAULT_GROUP = WORKSPACE/'outputs/new_SBD_mode'


def read_checkpoint(compare, run_name, epoch):
    models = []
    for path in sorted(compare.glob('report-data-*.js')):
        for line in path.read_text().splitlines():
            if line.startswith('window.FSQ_GT_REPLAY_DATA.models.push('):
                models.append(json.loads(line.split('.push(', 1)[1][:-2]))
                continue
            match = re.fullmatch(r'window\.FSQ_GT_REPLAY_DATA\.models\[(\d+)\]\.checkpoints(?:\[(\d+)\]\.skills)?\.push\((.*)\);', line)
            if not match:
                continue
            model, index, payload = match.groups()
            model = models[int(model)]
            if model['run_name'] != run_name:
                continue
            data = json.loads(payload)
            if index is None:
                model['checkpoints'].append(data)
            else:
                model['checkpoints'][int(index)]['skills'].append(data)
    selected = [cp for model in models if model['run_name'] == run_name
                for cp in model['checkpoints'] if cp['epoch_tag'] == epoch]
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one {run_name}/{epoch}; found {len(selected)}')
    return selected[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, default=DEFAULT_GROUP)
    parser.add_argument('--run', default='FSQ333_contrastiveON_routeON_loss_new')
    parser.add_argument('--epoch', default='epoch2000')
    parser.add_argument('--dataset', type=Path, default=REPO/'dataset_filtered/libero_90_full_full')
    args = parser.parse_args()
    group, dataset = args.group.resolve(), args.dataset.resolve()
    cp = read_checkpoint(group/'compare', args.run, args.epoch)
    records = {row['uid']: dict(row) for skill in cp['skills'] for row in skill['occurrences']}
    assert len(records) == cp['occurrence_count']
    info = json.loads((dataset/'meta/info.json').read_text())
    key = 'observation.images.image'
    height = int(info['features'][key]['shape'][0])
    columns = ['episode_index', 'length', f'videos/{key}/chunk_index',
               f'videos/{key}/file_index', f'videos/{key}/from_timestamp']
    episodes = pd.concat([pd.read_parquet(p, columns=columns)
                          for p in sorted((dataset/'meta/episodes').glob('**/*.parquet'))],
                         ignore_index=True).set_index('episode_index')
    for row in records.values():
        ep = episodes.loc[row['episode_id']]
        if not 0 <= row['frame_start'] < row['frame_end'] <= ep['length']:
            raise ValueError(f'Invalid episode interval: {row}')
        path = dataset/info['video_path'].format(video_key=key,
                   chunk_index=int(ep[f'videos/{key}/chunk_index']),
                   file_index=int(ep[f'videos/{key}/file_index']))
        if not path.is_file():
            raise FileNotFoundError(path)
        row.update(video_path=str(path), video_fps=float(info['fps']),
                   video_from_timestamp=float(ep[f'videos/{key}/from_timestamp']),
                   video_crop_row_start=0, video_crop_row_end=height)
    checkpoint = group/args.run/'checkpoints'/args.epoch
    recovered_path = checkpoint/'metrics/clip_export_source.json'
    if recovered_path.exists():
        raise FileExistsError(recovered_path)
    recovered = {k: v for k, v in cp.items() if k != 'skills'}
    recovered.update(completed=True, records=records,
                     provenance='Recovered unchanged token assignments and cuts from legacy compare/report-data-*.js',
                     source_report=str(group/'compare/index.html'), dataset=str(dataset))
    recovered_path.parent.mkdir(parents=True, exist_ok=True)
    recovered_path.write_text(json.dumps(recovered, indent=2)+'\n')
    print('Recovered', len(records), 'occurrences; top 5:',
          Counter(row['token'] for row in records.values()).most_common(5), flush=True)
    subprocess.run([sys.executable, str(ABC_EXPORTS/'export_top_skill_clips.py'),
                    '--checkpoint', str(checkpoint), '--manifest', str(recovered_path)], check=True)
    clips = checkpoint/'skill_clips_top5_20each'
    subprocess.run([sys.executable, str(ABC_EXPORTS/'export_selected_cube_images.py'),
                    '--clips', str(clips)], check=True)


if __name__ == '__main__':
    main()
