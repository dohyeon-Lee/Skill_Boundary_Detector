"""CPU-only checks for trainer launch, bounded native pools and error cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


HELPER = (Path(__file__).resolve().parents[2]
          / "examples/libero/configs/src/stage1_runtime.sh")


@pytest.mark.parametrize("gpus", [1, 2])
def test_single_gpu_skips_accelerate_and_multi_gpu_keeps_it(tmp_path, gpus):
    trainer = tmp_path / "trainer with spaces"
    trainer.write_text('#!/bin/bash\nprintf "%s\\n" "$OMP_NUM_THREADS:$MKL_NUM_THREADS:$OPENBLAS_NUM_THREADS:$NUMEXPR_NUM_THREADS"\nexit 7\n')
    trainer.chmod(0o755)
    launcher = tmp_path / "accelerate"
    launcher.write_text('''#!/bin/bash
echo used > "$RECORD"
test "$1" = launch || exit 30
shift
test "$1" = --num_processes=2 || exit 31
shift
exec "$@"
''')
    launcher.chmod(0o755)
    record = tmp_path / "launcher_used"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               RECORD=str(record), STAGE1_CPU_THREADS="1", OPENBLAS_NUM_THREADS="128")
    result = subprocess.run(['bash', '-c', '''set -euo pipefail
source "$1"
stage1_set_cpu_threads
stage1_configure_launcher "$2" "$3"
"${TRAIN_LAUNCH[@]}"
''', '_', str(HELPER), str(gpus), str(trainer)], env=env, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    assert result.stdout.strip() == "1:1:1:1"
    assert record.exists() is (gpus > 1)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_failed_training_stops_own_proxy_and_preserves_exit_status(tmp_path, ignore_term):
    proxy = tmp_path / "wandb_ssh_proxy.py"
    proxy.write_text('''import os, signal, sys, time
from pathlib import Path
if sys.argv[2] == "yes": signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
''')
    ready = tmp_path / "ready"
    result = subprocess.run(['bash', '-c', '''set -euo pipefail
source "$1"
trap stage1_cleanup_runtime EXIT
"$2" "$3" "$4" "$5" &
export SBD_WANDB_TUNNEL_PID=$! SBD_WANDB_TUNNEL_OWNER=$BASHPID
for i in {1..100}; do [ ! -f "$4" ] || break; sleep .02; done
test -f "$4"
exit 7
''', '_', str(HELPER), sys.executable, str(proxy), str(ready),
        'yes' if ignore_term else 'no'], capture_output=True, text=True, timeout=12)
    assert result.returncode == 7, result.stderr
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()), 0)


def test_cleanup_does_not_stop_a_parent_proxy(tmp_path):
    proxy = tmp_path / "wandb_ssh_proxy.py"
    proxy.write_text('import time; time.sleep(60)\n')
    process = subprocess.Popen([sys.executable, str(proxy)])
    try:
        env = dict(os.environ, SBD_WANDB_TUNNEL_PID=str(process.pid))
        result = subprocess.run(['bash', '-c', '''set -euo pipefail
source "$1"
export SBD_WANDB_TUNNEL_OWNER=$BASHPID
trap stage1_cleanup_runtime EXIT
exit 9
''', '_', str(HELPER)], env=env, capture_output=True, text=True, timeout=3)
        assert result.returncode == 9
        assert process.poll() is None
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=3)
