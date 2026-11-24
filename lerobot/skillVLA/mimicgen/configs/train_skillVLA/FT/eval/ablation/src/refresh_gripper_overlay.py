"""Recolor completed clips on CPU, optionally following an existing probe to completion."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'setup_env.sh').is_file())
sys.path[:0] = [str(ROOT / 'lerobot/src'),
               str(ROOT / 'lerobot/skillVLA/libero/configs/train_skillVLA/PT/eval/src')]

from probe_presentation import render_overlay, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--follow-status', type=Path)
    args = parser.parse_args()
    source = args.output / 'src'
    done = set()
    deadline = time.monotonic() + 3600
    while True:
        data = json.loads((source / 'results.json').read_text())
        for result in data['results']:
            if result['label'] in done:
                continue
            render_overlay(result, source, result['top_camera_transform'], require_plan=True)
            done.add(result['label'])
            reference = result['gripper_visualization_reference']
            (source / 'gripper_visualization_calibration.json').write_text(json.dumps(reference, indent=2))
            print(result['label'], reference, flush=True)
        # Re-read: the GPU probe may have completed another pair during encoding.
        latest = json.loads((source / 'results.json').read_text())
        if not {result['label'] for result in latest['results']} <= done:
            continue
        write_report(args.output, latest, latest['results'])
        terminal = not args.follow_status
        if args.follow_status:
            status = json.loads(args.follow_status.read_text())
            terminal = status['state'] in {'COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'}
        if terminal:
            print('Updated videos and HTML:', args.output / 'index.html', flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Probe did not finish within the overlay refresh window')
        time.sleep(5)


if __name__ == '__main__':
    main()
