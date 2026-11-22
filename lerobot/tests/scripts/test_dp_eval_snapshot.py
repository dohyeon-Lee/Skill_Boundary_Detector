"""An edit during validation must not change the checkpoint actually launched."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("domain", ["abc", "libero"])
@pytest.mark.parametrize("mode", [[], ["--local"]])
def test_snapshot_precedes_validation_and_is_used_for_execution(tmp_path, domain, mode):
    project = tmp_path / "project"
    configs = project / "lerobot/skillVLA" / domain / "configs"
    work = configs / "train_skills/DP/eval"
    shared = configs / "src"
    (work / "src").mkdir(parents=True)
    shared.mkdir()
    original = ROOT / "lerobot/skillVLA" / domain / "configs"
    shutil.copy2(original / "train_skills/DP/eval/submit_eval.sh", work / "submit_eval.sh")
    for name in ("snapshot_config.sh", "submit_args.sh"):
        shutil.copy2(original / "src" / name, shared / name)
    if domain == "abc":
        common = project / "lerobot/skillVLA/libero/configs/src"
        common.mkdir(parents=True)
        for name in ("snapshot_config.sh", "submit_args.sh"):
            shutil.copy2(ROOT / "lerobot/skillVLA/libero/configs/src" / name, common / name)
    (shared / "runtime_env.sh").write_text(":\n")
    (configs / "global_config.yaml").write_text("{}\n")
    # Execute the downstream CPU stub in place of dispatching to Slurm.
    (shared / "submit_job.sh").write_text(
        'submit_job() { "${PYTHON}" "${SCRIPT_DIR}/src/run_eval.py" --config "${CONFIG}"; }\n'
    )
    binary = project / ".venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(sys.executable)
    live = work / "eval_config.yaml"
    live.write_text(json.dumps({"dp_checkpoint": 10000}))
    exports = {
        "DP_ABLATION_OUTPUT_DIR": str(work / "outputs/result"), "DP_ABLATION_EPISODE_COUNT": "2",
        "DP_ABLATION_N_OBS_STEPS": "20", "DP_ABLATION_PREDICTION_HORIZON": "24",
        "DP_ABLATION_SBD_PREDICTION_HORIZON": "24", "DP_ABLATION_ARTIFACT_DIR": str(work / "partial_build/test"),
        "DP_ABLATION_CACHE_STATUS": "missing", "DP_ABLATION_CACHE_READY": "false",
        "DP_ABLATION_DEVICE": "cpu", "DP_ABLATION_CPUS": "1", "DP_ABLATION_PARTITION": "debug",
        "DP_ABLATION_QOS": "base", "DP_ABLATION_MEM": "1G", "DP_ABLATION_TIME": "00:05:00",
        "DP_ABLATION_NODELIST": "", "DP_ABLATION_EXCLUDE_NODES": "",
    }
    (work / "src/config.py").write_text(
        "import json,os,sys,shlex\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--config')+1])\n"
        "checkpoint=json.loads(p.read_text())['dp_checkpoint']\n"
        "Path(os.environ['LIVE_CONFIG']).write_text(json.dumps({'dp_checkpoint':100000}))\n"
        f"values={exports!r}\n"
        "values['VALIDATED_CHECKPOINT']=str(checkpoint)\n"
        "for key,value in values.items():print('export '+key+'='+shlex.quote(value))\n"
    )
    (work / "src/run_eval.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--config')+1])\n"
        "Path(os.environ['RESULT']).write_text(json.dumps({'config':str(p),"
        "'validated':int(os.environ['VALIDATED_CHECKPOINT']),"
        "'executed':json.loads(p.read_text())['dp_checkpoint']}))\n"
    )
    result = tmp_path / "result.json"
    env = dict(os.environ, SBD_EVAL_HF_READY="1", LIVE_CONFIG=str(live), RESULT=str(result))
    for key in ("SBD_GLOBAL_CONFIG", "ABC_DP_EVAL_CONFIG", "LIBERO_DP_EVAL_CONFIG"):
        env.pop(key, None)
    process = subprocess.run(["bash", str(work / "submit_eval.sh"), *mode], env=env,
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    actual = json.loads(result.read_text())
    assert json.loads(live.read_text())["dp_checkpoint"] == 100000
    assert actual["validated"] == actual["executed"] == 10000
    assert ".config_snapshots" in Path(actual["config"]).parts
