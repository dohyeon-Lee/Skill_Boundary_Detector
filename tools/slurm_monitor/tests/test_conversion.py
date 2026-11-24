from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversion import progress
from training import collect


def fixture(tmp_path):
    output = tmp_path / 'dataset' / 'droid'
    cache = output.with_name('.droid.building') / '.episodes'
    cache.mkdir(parents=True)
    launch = {'command': ['/repo/lerobot/skillVLA/droid/configs/generate_training_dataset/download_dataset/src/prepare_in_allocation.sh']}
    logs = {'out': f'Eligible selection: 4 episodes; excluded: 2\nOutput: {output}\nCPU workers: 3; action: cartesian_velocity\nEpisode 1/4 ready: 20 frames\n'}
    return output, cache, launch, logs


def test_out_of_order_markers_and_final_validation(tmp_path):
    output, cache, launch, logs = fixture(tmp_path)
    for rank in (0, 2, 3):
        folder = cache / f'{rank:06d}'
        folder.mkdir()
        (folder / 'done.json').write_text('{}')
    (cache / '000001').mkdir()  # In flight is not complete.
    current = progress({'terminal': False}, launch, logs)
    assert (current['completed'], current['total'], current['percent']) == (3, 4, 75)
    assert current['excluded'] == 2 and current['workers'] == 3
    (cache / '000001/done.json').write_text('{}')
    assert progress({'terminal': False}, launch, logs)['phase'] == 'finalizing'
    output.mkdir()
    (output / 'conversion_manifest.json').write_text('{"episodes":4}')
    logs['out'] += f'Converted 4 episodes / 80 frames -> {output}\n'
    assert progress({'terminal': True}, launch, logs)['phase'] == 'complete'


def test_stopped_attempt_does_not_borrow_new_markers(tmp_path):
    _, cache, launch, logs = fixture(tmp_path)
    for rank in range(4):
        folder = cache / f'{rank:06d}'
        folder.mkdir()
        (folder / 'done.json').write_text('{}')
    assert progress({'terminal': True}, launch, logs)['completed'] == 1
    assert progress({}, {'command': ['/repo/train.py']}, logs) is None
    assert progress({}, launch, {}) is None


def test_collected_card_contains_live_conversion(tmp_path):
    output, _, launch, logs = fixture(tmp_path)
    folder = tmp_path / '.cache/srun_sessions/convert'
    folder.mkdir(parents=True)
    log = folder / 'out'
    log.write_text(logs['out'])
    (folder / 'launch.json').write_text(json.dumps(launch))
    (folder / 'training_status.json').write_text(json.dumps({
        'job_id': '75000', 'run_id': 'example', 'started_at': '2026-11-24T07:42:00+00:00',
        'name': 'DROID_prepare_3workers', 'state': 'RUNNING', 'output': str(log)}))
    journal = tmp_path / '.cache/training_events'
    journal.mkdir()
    (journal / 'example.json').write_text((folder / 'training_status.json').read_text())
    card = collect(tmp_path, [{'job_id': '75000', 'state': 'RUNNING'}])['training:example']
    assert card['conversion_progress']['completed'] == 1
    assert card['outputs'] == [str(output)]
    assert card['training_progress'] == []
