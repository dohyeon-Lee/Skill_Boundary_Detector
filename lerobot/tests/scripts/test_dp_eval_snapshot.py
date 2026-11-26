"""An edit during validation must not change the checkpoint actually launched."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("domain", ["abc", "libero"])
def test_initial_mask_changes_boundary_and_report_only(tmp_path, domain):
    path = ROOT / "lerobot/skillVLA/abc/configs/train_skills/DP/eval/src/defaults.py"
    spec = importlib.util.spec_from_file_location("mask_dp_defaults", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "n_obs_steps": 60, "horizon": 36, "history_encoder": "transformer",
        "down_dims": [128, 256, 512], "action_sequence_mode": "future_only",
    }))
    cfg = {"project_root": str(tmp_path), "policy_checkpoint": "model",
           "source_tasks": ["grab_clutter"], "output_name": "mask_comparison"}
    enabled, on = module.resolve_config(dict(cfg, initial_mask=True), domain=domain)
    disabled, off = module.resolve_config(dict(cfg, initial_mask=False), domain=domain)
    assert enabled["boundary_variants"][0]["initial_mask_frames"] == 60
    assert disabled["boundary_variants"][0]["initial_mask_frames"] == 0
    assert on["output_suffix"].endswith("_imask60")
    assert off["output_suffix"].endswith("_imask0")
    for key in ("probe_variants", "cluster_variants", "output_name"):
        assert enabled[key] == disabled[key]  # The same GPU/PCA/GMM caches remain usable.


def test_droid_terminal_mask_keeps_probe_caches_and_prior_report_separate(tmp_path):
    path = ROOT / "lerobot/skillVLA/abc/configs/train_skills/DP/eval/src/defaults.py"
    spec = importlib.util.spec_from_file_location("terminal_mask_dp_defaults", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "n_obs_steps": 30, "horizon": 18, "history_encoder": "transformer",
        "down_dims": [128, 256, 512], "action_sequence_mode": "future_only",
    }))
    cfg = {"project_root": str(tmp_path), "policy_checkpoint": "model",
           "output_name": "mask_comparison", "initial_mask": False}
    enabled, on = module.resolve_config(cfg, domain="droid")
    disabled, off = module.resolve_config(dict(cfg, terminal_mask=False), domain="droid")
    assert enabled["boundary_variants"][0]["terminal_mask_frames"] == 24
    assert disabled["boundary_variants"][0]["terminal_mask_frames"] == 0
    assert on["output_suffix"].endswith("_imask0")
    assert off["output_suffix"] == on["output_suffix"] + "_tmask0"
    for key in ("probe_variants", "cluster_variants", "experiments", "output_name"):
        assert enabled[key] == disabled[key]
    for invalid in ("false", 0, None):
        with pytest.raises(ValueError, match="terminal_mask must be true or false"):
            module.resolve_config(dict(cfg, terminal_mask=invalid), domain="droid")


def test_large_episode_settings_are_passed_by_file(tmp_path):
    path = ROOT / "lerobot/skillVLA/abc/configs/train_skills/DP/eval/src/config.py"
    spec = importlib.util.spec_from_file_location("large_dp_eval_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.dumps({"project_root": str(tmp_path), "episodes": "x" * 200_000})
    exports = module.shell_settings({"dp_ablation_settings_json": payload})
    assert "dp_ablation_settings_json" not in exports
    saved = Path(exports["dp_ablation_settings_file"])
    assert saved.read_text() == payload
    env = dict(os.environ, **{key.upper(): str(value) for key, value in exports.items()})
    subprocess.run([sys.executable, "-c", "pass"], env=env, check=True)


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
