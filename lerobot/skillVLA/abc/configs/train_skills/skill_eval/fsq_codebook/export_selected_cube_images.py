#!/usr/bin/env python3
"""Export selected FSQ cubes using the report's exact count-mode SVG geometry.

Requires cairosvg. Each image highlights one exported token while keeping the
same projection and count colors as the HTML, with a stronger selection halo.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg


DEFAULT_CLIPS = (
    Path(__file__).resolve().parent
    / 'outputs/abc_grab_clutter_absolute/checkpoints/epoch0800/skill_clips_top5_20each'
)


def coord(token):
    return [(token // (3**axis)) % 3 for axis in range(3)]


def project(point):
    x, y, z = [v-1 for v in point]
    yaw, pitch = -.63, .46
    xr = math.cos(yaw)*x-math.sin(yaw)*y
    yr = math.sin(yaw)*x+math.cos(yaw)*y
    return (300+xr*145,
            265+yr*145*math.sin(pitch)-z*145*math.cos(pitch),
            yr*math.cos(pitch)+z*math.sin(pitch))


def cube_svg(selected, counts):
    root = ET.Element('svg', xmlns='http://www.w3.org/2000/svg',
                      width='600', height='520', viewBox='0 0 600 520')
    ET.SubElement(root, 'rect', x='0', y='0', width='600', height='520', fill='white')
    for token in range(27):
        c = coord(token)
        for axis in range(3):
            if c[axis] == 2:
                continue
            neighbor = c.copy()
            neighbor[axis] += 1
            a, b = project(c), project(neighbor)
            ET.SubElement(root, 'line', x1=str(a[0]), y1=str(a[1]),
                          x2=str(b[0]), y2=str(b[1]), stroke='rgb(100,100,100)',
                          **{'stroke-opacity': '.42', 'stroke-width': '1.2'})
    used_counts = [count for count in counts.values() if count > 0]
    low, high = min(used_counts), max(used_counts)
    for token in sorted(range(27), key=lambda t: project(coord(t))[2]):
        if token == selected:
            continue  # Draw the selected token last so no lattice node hides it.
        x, y, _ = project(coord(token))
        used = counts[token] > 0
        fill = '#d7dde8'
        if used:
            norm = (counts[token]-low)/(high-low) if high > low else .6
            t = .1+.9*norm
            # Math.round in the browser uses floor(x+.5), not bankers rounding.
            rgb = [math.floor(a+(b-a)*t+.5) for a, b in zip((253,235,232), (136,8,8))]
            fill = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
        outlined = used and counts[token] > 10
        ET.SubElement(root, 'circle', cx=str(x), cy=str(y),
                      r='6' if used else '3.5', fill=fill,
                      stroke='#111' if outlined else 'none',
                      **{'stroke-width': '2' if outlined else '0'})
    x, y, _ = project(coord(selected))
    highlight = ET.SubElement(root, 'g', id=f'selected-token-{selected}')
    ET.SubElement(highlight, 'circle', cx=str(x), cy=str(y), r='24',
                  fill='#ff3b30', **{'fill-opacity': '.18'})
    ET.SubElement(highlight, 'circle', cx=str(x), cy=str(y), r='18',
                  fill='white', stroke='#ff3b30', **{'stroke-width': '3'})
    ET.SubElement(highlight, 'circle', cx=str(x), cy=str(y), r='12',
                  fill='#ff3b30', stroke='white', **{'stroke-width': '2'})
    return ET.tostring(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', type=Path, default=DEFAULT_CLIPS)
    parser.add_argument('--width', type=int, default=1800)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    selection = json.loads((args.clips/'selection.json').read_text())
    manifest = json.loads(Path(selection['source_manifest']).read_text())
    if manifest['levels'] != [3,3,3]:
        raise ValueError('This export matches the HTML 3x3x3 FSQ cube only')
    counts = Counter(row['token'] for row in manifest['records'].values())
    output = args.clips/'cube_images'
    output.mkdir(exist_ok=True)
    for token in selection['tokens']:
        ident = token['token']
        coordinates = '_'.join(map(str, coord(ident)))
        path = output/f"{token['rank']:02d}_token{ident:04d}_coord_{coordinates}.png"
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
        cairosvg.svg2png(bytestring=cube_svg(ident, counts), write_to=str(path),
                        output_width=args.width)
        print(path, flush=True)


if __name__ == '__main__':
    main()
