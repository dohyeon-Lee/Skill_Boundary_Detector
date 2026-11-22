"""Job selectors preserve launcher args and never submit real Slurm work."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import pytest

from test_submit_job import CONFIGS, SUBMIT_JOB, _env, _job_script

PARSERS = [CONFIGS / "src/submit_args.sh", CONFIGS.parents[1] / "abc/configs/src/submit_args.sh"]


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("arguments,remaining,job_id", [
    ([], [], ""),
    (["--73428", "--config", "path with spaces.yaml", "--dry-run"],
     ["--config", "path with spaces.yaml", "--dry-run"], "73428"),
    (["Predictor", "--dry-run", "--73428"], ["Predictor", "--dry-run"], "73428"),
    (["--", "--73428"], ["--", "--73428"], ""),
])
def test_selector_preserves_options_and_positionals(parser, arguments, remaining, job_id):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBD_")}
    inspection = ('import json,os,sys; print(json.dumps([sys.argv[1:], '
                  '[os.environ.get(k, "") for k in '
                  '("SBD_SCHEDULER","SBD_SLURM_SUBMIT_MODE","SBD_SRUN_JOB_ID")]]))')
    program = ('set -euo pipefail; parser="$1"; shift; source "$parser"; '
               + shlex.quote(sys.executable) + ' -c ' + shlex.quote(inspection) + ' "$@"')
    result = subprocess.run(["bash", "-c", program, "_", str(parser), *arguments],
                            env=env, text=True, capture_output=True, check=True)
    args, settings = json.loads(result.stdout)
    assert args == remaining
    assert settings == (["slurm", "srun_existing", job_id] if job_id else ["", "", ""])


@pytest.mark.parametrize("arguments", [["--0"], ["--0123"], ["--73428oops"], ["--73428", "--73429"]])
def test_invalid_selector_stops_before_the_launcher(arguments):
    result = subprocess.run(["bash", "-c", 'set -euo pipefail; source "$1" "${@:2}"; echo launched',
                             "_", str(PARSERS[0]), *arguments], text=True, capture_output=True)
    assert result.returncode == 2
    assert "launched" not in result.stdout


def test_raw_batch_cannot_escape_the_selected_allocation(tmp_path):
    env = _env(tmp_path, "slurm")
    script = _job_script(tmp_path, "exit 0")
    command = ('set -euo pipefail; parser="$1"; helper="$2"; script="$3"; '
               'set -- --73428; source "$parser"; source "$helper"; '
               'submit_batch --parsable --array=0-2 "$script"')
    result = subprocess.run(["bash", "-c", command, "_", str(PARSERS[0]), str(SUBMIT_JOB), str(script)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "no new allocation" in result.stderr
    assert "sbatch " not in result.stdout


def test_every_launcher_rejects_invalid_selector_before_running():
    launchers = [p for root in (CONFIGS, CONFIGS.parents[1] / "abc/configs")
                 for p in root.rglob("submit*.sh")
                 if ".config_snapshots" not in p.parts and p.name not in {"submit_args.sh", "submit_job.sh"}]
    assert len(launchers) >= 57
    for launcher in launchers:
        result = subprocess.run(["bash", str(launcher), "--0"], text=True, capture_output=True, timeout=5)
        assert result.returncode == 2, (launcher, result.stderr)
        assert "Invalid job selector" in result.stderr, (launcher, result.stderr)
