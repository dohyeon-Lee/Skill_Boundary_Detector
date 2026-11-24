"""Offline presentation of the probe's actual VSA readouts; raw clips stay intact."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from termination_chart import CHART_SCRIPT, chart_html


def render_overlay(result, source: Path, transform, *, require_plan=False, uv_enabled=True):
    import av
    from PIL import Image, ImageDraw
    from goal_overlay import (
        draw_fullskill, draw_goal, fullskill_from_trace, fullskill_gripper_from_trace,
        dataset_gripper_reference, marks_from_uv,
        marks_in_agent_view, values_from_trace,
    )
    from lerobot.utils.io_utils import write_video

    with av.open(str(source / result['video'])) as container:
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    height, combined_width = frames[0].shape[:2]
    width = combined_width // 2
    trace = result.get('skill_trace', [])
    initial, current = fullskill_from_trace(
        trace, batch_index=0, n_video_frames=len(frames), video_frame_stride=1,
    )
    if require_plan and initial is None:
        raise RuntimeError('VSA full-skill overlay requested but no actual plans were captured')
    settings = json.loads((source / 'settings.json').read_text())
    gripper_action = any('gripper_action' in point for record in trace
                         for point in record.get('fullskill_plans', []))
    reference = (None if gripper_action else
                 dataset_gripper_reference(settings['spec']['raw_dataset_dir'],
                                           settings['recipe']['episode_index']))
    result['fullskill_gripper_action'] = gripper_action
    result['gripper_visualization_reference'] = reference
    opening = fullskill_gripper_from_trace(
        trace, batch_index=0, n_video_frames=len(frames), video_frame_stride=1,
        gripper_reference=reference,
    )
    plans = []
    for plan in (initial, current):
        plans.append(None if plan is None else marks_in_agent_view(
            np.asarray(transform), plan.reshape(-1, 3), None, height=height, width=width,
        ).pixels.reshape(*plan.shape[:-1], 2))
    uvs = values_from_trace(trace, 'focus_uv', 2, batch_index=0,
                           n_video_frames=len(frames), video_frame_stride=1)
    pixels = None if uvs is None else marks_from_uv(uvs, height=height, width=width)
    annotated = []
    records = sorted(trace, key=lambda item: item.get('episode_timestep', 0))
    active = 0
    for index, frame in enumerate(frames):
        while active+1 < len(records) and records[active+1].get('episode_timestep', 0) <= index:
            active += 1
        top = Image.fromarray(frame[:, :width].copy())
        if plans[0] is not None:
            draw_fullskill(top, None, plans[1][index],
                           gripper_open_fraction=None if opening is None else opening[index])
        if pixels is not None and uv_enabled:
            draw_goal(top, pixels[index], color=(50, 220, 255))
        picture = Image.new('RGB', (combined_width, height + 48), '#10141c')
        picture.paste(top, (0, 0))
        picture.paste(Image.fromarray(frame[:, width:]), (width, 0))
        draw = ImageDraw.Draw(picture)
        skill_label = '' if not records else f" | token {records[active].get('codebook_token')}"
        if records and 'source_skill_index' in records[active]:
            skill_label += f" (skill {records[active]['source_skill_index']+1})"
        draw.text((8, height + 4), f"{result['label']} | step {min(index, result['steps'])}{skill_label}", fill='white')
        if result['kind'] == 'vsa':
            if opening is None:
                draw.text((8, height + 23), 'Latest plan: gripper not recorded', fill=(160, 170, 185))
            else:
                draw.text((8, height + 23), 'Open', fill=(45, 125, 255))
                draw.text((57, height + 23), 'Close command' if gripper_action else 'Grasp-width', fill=(245, 65, 65))
            draw.text((239, height + 23), 'GT UV input', fill=(50, 220, 255))
        else:
            draw.text((8, height + 23), 'GT action control (no VSA prediction)', fill='#b6c2d5')
        annotated.append(np.asarray(picture))
    name = result['label'] + '_annotated.mp4'
    temporary = source / (result['label'] + '_annotated.tmp.mp4')
    write_video(temporary, np.stack(annotated), 20)
    temporary.replace(source / name)
    result['annotated_video'] = name
    result['fullskill_plan_count'] = sum(len(s.get('fullskill_plans', [])) for s in trace)


def write_report(output: Path, info, results):
    cards = []
    for result_index, result in enumerate(results):
        video_id = f'probe-video-{result_index}'
        video = result.get('annotated_video', result['video'])
        error = result.get('boundary_xyz_error_m')
        boundary = (f'boundary XYZ error: {error:.6f} m' if error is not None else
                    f'actual rollout entry · replay state error: {result["boundary_repeat_max_error"]:.3g}')
        pending = ('<p class="muted">Full trajectory capture pending.</p>'
                   if result['kind'] == 'vsa' and not result.get('fullskill_plan_count') else '')
        transition = (f'<p>Final skill starts after {result["transition_step"]} actions '
                      f'({html.escape(str(result.get("transition_reason")))}). '
                      f'Final skill: {result.get("final_skill_steps", 0)} actions.</p>'
                      if 'transition_step' in result else '')
        if 'transitions' in result:
            transition = '<p>' + ' · '.join(
                f'Skill {t["from_skill_number"]}→{t["to_skill_number"]}: step {t["step"]} '
                f'({html.escape(t["reason"])})' for t in result['transitions']) + '</p>'
        cards.append(
            f'<section><h2>{html.escape(result["label"])} · success: {result["success"]}</h2>'
            f'<p>{result["steps"]} steps · {boundary}</p>'
            f'{transition}{pending}<video id="{video_id}" controls loop preload="metadata" src="src/{html.escape(video, quote=True)}"></video>'
            f'{chart_html(result, video_id)}'
            f'<p><a href="src/{html.escape(result["video"], quote=True)}">Original video</a></p></section>'
        )
    actual_entry = info.get('initialization') == 'rollout_entry'
    timed_transition = info.get('initialization') == 'transition_timing'
    protocol = (html.escape(info['protocol']) if actual_entry or timed_transition else
                f'GT prefix: frames [0,{info["frame_start"]}); videos begin at frame {info["frame_start"]}.<br>'
                'Original episode grounding retained. GT skill + GT UV fixed; no Predictor/Terminator.<br>'
                'The start is reconstructed, not a recorded intermediate full state.')
    prefixes = ''
    if actual_entry:
        prefixes = '<p>' + ' · '.join(
            f'<a href="src/{path.name}">{html.escape(path.stem)} · actual rollout prefix</a>'
            for path in sorted((output / 'src').glob('*_prefix.mp4'))
        ) + '</p>'
    content = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Coffee skill-start ablation</title><style>'
        'body{background:#10141c;color:#eee;font:16px/1.6 sans-serif;margin:0;text-align:center}'
        'main{max-width:1100px;margin:0 auto;padding:28px 18px}'
        'section{margin:32px auto 44px}h1,h2,p{overflow-wrap:anywhere}'
        'video{display:block;width:100%;max-width:1000px;height:auto;margin:16px auto;background:#000}'
        '.termination-panel{max-width:1000px;margin:0 auto}.termination-chart{display:block;width:100%;height:220px;cursor:crosshair}'
        'a{color:#87c6ff}.muted{color:#aab7cd}.open{color:#2d7dff}'
        '.closed{color:#f54141}.uv{color:#32dcff}</style><main>'
        f'<h1>Coffee {"transition A/B" if timed_transition else "rollout-entry reset" if actual_entry else "GT-start"} · skill {info["skill_index"]} / token {info["token"]}</h1>'
        + (f'<p>Evaluated skills (1-based): {" → ".join(map(str, info["skill_numbers"]))}</p>' if 'skill_numbers' in info else '')
        +
        f'<p>{protocol}</p>{prefixes}'
        '<p>Latest full-skill prediction only: <span class="open">Blue: open</span> · '
        '<span class="closed">Red: near demonstrated grasp width</span> · '
        '<span class="uv">Cyan: actual GT UV input</span><br>'
        'Color compares predicted finger spacing to stable GT open/grasp widths, '
        'with their midpoint as the threshold. An object keeps the fingers apart while grasped.<br>'
        'For action-based plans, blue/red indicate open/close commands; no finger-width calibration is used. '
        'For proprio plans, red indicates grasp-width state, not a close command. Neither verifies contact.<br>'
        'The latest plan is held between replans; gray means gripper data was not recorded.<br>'
        'Trajectory and UV are drawn on the top view. The wrist view stays unmodified.</p>'
        + ''.join(cards) + '</main>' + CHART_SCRIPT + '</html>'
    )
    temporary = output / 'index.tmp.html'
    temporary.write_text(content)
    temporary.replace(output / 'index.html')
