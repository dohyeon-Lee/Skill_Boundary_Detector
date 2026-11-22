"""Cancel only batch jobs recorded by one Yonsei launch, after controller verification."""
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def cancel(base, identifier):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', identifier):
        raise ValueError('Invalid launch identity')
    work = Path(base) / '.cache/yonsei_launches' / identifier
    manifest = json.loads((work / 'launch.json').read_text())
    if any(re.fullmatch(r'--[1-9][0-9]*', str(a)) for a in manifest.get('args', [])):
        raise ValueError('기존 할당 전체는 취소하지 않습니다. 이 버튼은 별도로 제출한 batch 잡에만 사용할 수 있습니다.')
    records = [json.loads(line) for line in (work / 'jobs.jsonl').read_text().splitlines() if line.strip()]
    ids = sorted({str(row['job_id']) for row in records})
    if not ids or not all(re.fullmatch(r'[1-9][0-9]*', value) for value in ids):
        raise ValueError('취소할 제출 잡 번호가 없습니다.')
    active = []
    # Validate every job before cancelling any; never accept job IDs from a browser.
    for job in ids:
        result = subprocess.run(['scontrol', '-o', 'show', 'job', job],
                                capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise ValueError('잡의 현재 소유권/실행 정보를 확인할 수 없습니다: ' + job)
        current = False
        for line in result.stdout.splitlines():
            fields = dict(re.findall(r'(?:^|\s)([A-Za-z]+)=([^\s]+)', line))
            if not fields:
                continue
            if fields.get('ArrayJobId', fields.get('JobId')) != job:
                raise ValueError('Slurm job identity mismatch')
            if not fields.get('UserId', '').endswith('(' + str(os.getuid()) + ')'):
                raise ValueError('잡 소유자가 현재 사용자와 일치하지 않습니다.')
            command = Path(fields.get('Command', ''))
            cwd = Path(fields.get('WorkDir', ''))
            try:
                command.resolve().relative_to(work.resolve())
                cwd.resolve().relative_to((work / 'project').resolve())
            except ValueError:
                raise ValueError('이 실행에서 제출한 잡과 일치하지 않습니다: ' + job) from None
            if fields.get('JobState') not in TERMINAL:
                current = True
        if current:
            active.append(job)
    if not active:
        raise ValueError('이미 모든 잡이 종료되었습니다.')
    result = subprocess.run(['scancel', '--user', pwd.getpwuid(os.getuid()).pw_name, *active],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError(result.stderr.strip()[-500:] or 'scancel failed')
    (work / 'cancel.json').write_text(json.dumps({'status': 'requested', 'job_ids': active}))
    return {'status': 'requested', 'job_ids': active}


if __name__ == '__main__':
    request = json.load(sys.stdin)
    try:
        print(json.dumps(cancel(request['base'], request['id'])))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
