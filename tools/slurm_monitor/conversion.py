"""Read DROID conversion markers without importing or interrupting the converter."""
from pathlib import Path
import re

if __package__:
    from .slurm import small_json
else:
    from slurm import small_json


def progress(row, launch, logs):
    command = launch.get('command', [])
    if not (isinstance(command, list) and command
            and Path(command[0]).name == 'prepare_in_allocation.sh'
            and 'droid' in Path(command[0]).parts):
        return None
    text = '\n'.join(logs.values())
    selection = re.search(r'^Eligible selection: (\d+) episodes; excluded: (\d+)', text, re.M)
    destination = re.search(r'^Output: (/.+)$', text, re.M)
    if not selection or not destination:
        return None
    total, excluded = map(int, selection.groups())
    if total <= 0:
        return None
    output = Path(destination[1].strip())
    ready = re.findall(r'^Episode (\d+)/\d+ ready:', text, re.M)
    completed = min(total, max(map(int, ready), default=0))
    phase = 'converting'
    # A stopped historical attempt must never borrow a later attempt's markers.
    # Live runs own the converter's locked, request-validated staging directory.
    if not row.get('terminal'):
        cache = output.with_name('.' + output.name + '.building') / '.episodes'
        count = sum(1 for path in cache.glob('*/done.json')
                    if re.fullmatch(r'\d{6}', path.parent.name))
        completed = max(completed, min(total, count))
    if completed == total:
        phase = 'finalizing'
    published = bool(re.search(r'^(?:Converted \d+ episodes / \d+ frames ->|Reuse verified completed dataset:)', text, re.M))
    if published:
        manifest = small_json(output / 'conversion_manifest.json')
        if manifest.get('episodes') == total:
            completed, phase = total, 'complete'
    workers = re.search(r'^CPU workers: (\d+)', text, re.M)
    return {'completed': completed, 'total': total, 'percent': 100 * completed / total,
            'phase': phase, 'excluded': excluded, 'workers': int(workers[1]) if workers else None,
            'output': str(output)}
