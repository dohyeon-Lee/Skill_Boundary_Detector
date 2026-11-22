#!/usr/bin/env python3
"""Combine saved video reports by episode, without inference or re-rendering."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

import yaml

EVAL_DIR = Path(__file__).resolve().parents[1]


class Gallery(HTMLParser):
    """Slice original card markup without changing its inline plots."""
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.lines = [0]
        self.lines.extend(match.end() for match in re.finditer('\n', source))
        self.depth = 0
        self.card_start = None
        self.card_depth = 0
        self.cards = []
        self.players = []
        self.current_player = None
        self.feed(source)

    def source_position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if tag == 'div':
            self.depth += 1
            if 'card' in classes:
                if self.card_start is not None:
                    raise ValueError('Nested report cards are unsupported')
                self.card_start, self.card_depth = self.source_position(), self.depth
                self.current_player = None
            if 'episode-player' in classes and self.card_start is not None:
                self.current_player = attrs

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.card_start is not None and self.depth == self.card_depth:
                self.cards.append(self.source[self.card_start:self.source_position() + len('</div>')])
                self.players.append(self.current_player)
                self.card_start = None
            self.depth -= 1


def load_report(folder):
    source = (folder / 'index.html').read_text()
    metadata = json.loads((folder / 'src/boundary_results.json').read_text())
    settings = json.loads((folder / 'src/evaluation_settings.json').read_text())
    gallery = Gallery(source)
    records = metadata['episodes']
    if len(gallery.cards) != len(records) or not records or not all(gallery.players):
        raise ValueError(f'{folder.name}: video cards do not match saved episode records')
    cards = {}
    for record, markup, player in zip(records, gallery.cards, gallery.players, strict=True):
        key = (int(record['task_id']), int(record['episode_id']))
        if int(player['data-n-frames']) != int(record['n_frames']):
            raise ValueError(f'{folder.name}: episode length mismatch for {key}')
        caption = re.search(r'episode\s+(\d+)', markup)
        if not caption or int(caption.group(1)) != key[1]:
            raise ValueError(f'{folder.name}: episode order mismatch for {key}')
        cards.setdefault(key, []).append((record, markup, player))
    source_info = settings.get('source_manifest', {})
    dataset = source_info.get('dataset_marker_sha256') or source_info.get('dataset_dir')
    probe = settings['probe_variants'][0]
    cluster = settings['cluster_variants'][0]
    boundary = settings['boundary_variants'][0]
    label = (f"H={probe.get('descriptor_temporal_length', 'all')} · "
             f"replan={probe.get('replan_interval', '?')} · K={cluster.get('k', '?')} · "
             f"NMS={boundary.get('nms_frames', '?')} · SG={boundary.get('smooth_window', '?')}/{boundary.get('polyorder', '?')} · "
             f"power={boundary.get('score_power', 1):g}")
    return dict(folder=folder, cards=cards, dataset=dataset, label=label)


def build(config_path, *, dry_run=False, outputs_root=None):
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    folders = cfg.get('eval_folders')
    if not isinstance(folders, list) or len(folders) < 2 or len(set(folders)) != len(folders):
        raise ValueError('eval_folders must contain at least two different eval folder names')
    if any(not isinstance(name, str) or Path(name).name != name or name in {'.', '..', 'together'} for name in folders):
        raise ValueError('eval_folders entries must be folder names directly under outputs')
    name = str(cfg.get('output_name', 'comparison')).strip()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..'}:
        raise ValueError('output_name must be a folder name')
    outputs = Path(outputs_root) if outputs_root is not None else EVAL_DIR / 'outputs'
    destination = outputs / 'together' / name
    reports = [load_report(outputs / folder) for folder in folders]
    if any(report['dataset'] != reports[0]['dataset'] for report in reports[1:]):
        raise ValueError('Select reports from the same dataset to align episode IDs')
    common = sorted(set.intersection(*(set(report['cards']) for report in reports)))
    if not common:
        raise ValueError('Selected reports have no episodes in common')
    for key in common:
        lengths = {(int(record['n_frames']), float(player['data-fps']))
                   for report in reports for record, _, player in report['cards'][key]}
        if len(lengths) != 1:
            raise ValueError(f'Episode {key} has different frame counts or FPS across reports')
    print(f'Together: {len(reports)} reports, {len(common)} shared episodes -> {destination / "index.html"}')
    if dry_run:
        return destination / 'index.html'
    target = destination / 'index.html'
    if target.exists() and 'name="abc-sbd-together-report"' not in target.read_text()[:1000]:
        raise ValueError(f'Existing output is not a together report: {target}')
    results = []
    for index, report in enumerate(reports):
        relative = quote(os.path.relpath(report['folder'] / 'index.html', destination), safe='/')
        results.append({
            'id': str(index), 'short': report['label'], 'title': report['folder'].name,
            'meta': report['label'], 'src': relative,
            'episodeKeys': [f'{task}:{episode}' for task, episode in report['cards']],
        })
    def safe_json(value):
        return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')
    document = (Path(__file__).with_name('together_template.html').read_text()
                .replace('__SUMMARY__', f'{len(reports)}개 결과 · 공통 {len(common)} episodes · 스크롤 및 영상 연동')
                .replace('__RESULTS__', safe_json(results))
                .replace('__COMMON__', safe_json([f'{task}:{episode}' for task, episode in common])))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'src').mkdir(exist_ok=True)
    (destination / 'src/manifest.json').write_text(json.dumps({
        'kind': 'abc_sbd_together', 'eval_folders': folders, 'episodes': len(common),
        'sources': [{'folder': report['folder'].name, 'episodes': len(report['cards']), 'label': report['label']}
                    for report in reports],
        'episode_ids': [{'task_id': task, 'episode_id': episode} for task, episode in common],
        'inference': False, 'reencoded': False,
    }, ensure_ascii=False, indent=2) + '\n')
    temporary = destination / 'index.html.tmp'
    temporary.write_text(document)
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EVAL_DIR / 'outputs/together/together_config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    build(args.config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
