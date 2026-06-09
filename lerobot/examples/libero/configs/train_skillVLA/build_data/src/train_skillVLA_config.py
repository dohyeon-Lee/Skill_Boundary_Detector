#!/usr/bin/env python3
"""Config helpers for configs/train_skillVLA (SkillVLA data generation).

Resolves paths + run tags for the pipeline that turns trained DP + FSQ models
into SkillVLA training data, and emits them as shell exports (--shell).

Root/yaml helpers are reused from train_skills_config.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# reuse the train_skills yaml-load + shell-emit helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "train_skillVLA_config.yaml"


def _levels(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    cleaned = str(value).replace("[", " ").replace("]", " ").replace(",", " ")
    return [int(v) for v in cleaned.split()]


def build_settings(cfg: dict, dataset: str | None = None) -> dict:
    root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = root / str(get_value(cfg, "dataset_root", "libero_dataset"))
    outputs_root = root / str(get_value(cfg, "outputs_root", "outputs"))
    # Fixed per-stage subdirs (match train_skills layout).
    dp_outputs_root = outputs_root / "DP"
    fsq_outputs_root = outputs_root / "FSQ"
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))

    source_dataset = dataset or str(get_value(cfg, "source_dataset", env="SOURCE_DATA"))

    # ── FSQ reference (declared like dp_policy_name: folder name + checkpoint) ──
    # The FSQ folder name encodes both the patch grid (dino<grid>) and the codebook
    # levels (fsq<levels>); parse both from it so they can't drift from the model you
    # reference. FSQ levels are single-digit (quantization bins, e.g. 5/8), so each
    # digit is one level; the build only needs num_embeddings = prod(levels).
    fsq_run_name = str(get_value(cfg, "fsq_run_name"))
    fsq_checkpoint = str(get_value(cfg, "fsq_checkpoint", "1000"))
    lv_match = re.search(r"fsq(\d+)", fsq_run_name)
    pg_match = re.search(r"_dino(\d+)", fsq_run_name)
    if not lv_match or not pg_match:
        raise ValueError(
            f"fsq_run_name must contain 'fsq<levels>' and 'dino<grid>' tags "
            f"(e.g. ..._fsq88_dino8), got: {fsq_run_name}"
        )
    fsq_digits = lv_match.group(1)
    fsq_levels = [int(d) for d in fsq_digits]
    patch_grid = int(pg_match.group(1))
    fsq_exp_suffix = fsq_run_name.split(f"_dino{patch_grid}", 1)[1]   # e.g. "_eqloss" or ""
    fsq_exp = fsq_exp_suffix.lstrip("_")

    # ── DINO (step 2) ──
    image_keys = as_list(get_value(cfg, "dino_image_keys", ["observation.images.image"]))
    dino_base_dataset = str(get_value(cfg, "dino_base_dataset", "libero_90"))
    base_dino_dir = dataset_root / f"{dino_base_dataset}_DINO" / f"pg{patch_grid}"

    # ── DP (step 3) ──
    dp_policy_name = str(get_value(cfg, "dp_policy_name"))
    dp_checkpoint = str(get_value(cfg, "dp_checkpoint", "100000"))
    dp_policy_path = dp_outputs_root / dp_policy_name / "checkpoints" / dp_checkpoint / "pretrained_model"

    # ── FSQ (step 4) — model path from the parsed run name + checkpoint ──
    fsq_model_dir = fsq_outputs_root / fsq_run_name
    if fsq_checkpoint in ("0", "best"):
        fsq_model_path = fsq_model_dir / "FSQ.pt"
        ckpt_tag = "best"
    else:
        fsq_model_path = fsq_model_dir / f"FSQ_epoch{int(fsq_checkpoint):04d}.pt"
        ckpt_tag = str(fsq_checkpoint)

    # ── output layout ──
    #   {skillvla_root}/{source_dataset}/{run_tag}/   ← final outputs (dino.npz, FSQ.pt, skillvla/)
    #   {skillvla_root}/{source_dataset}/_work/        ← intermediates, keyed by dependency:
    #       dino/pg{grid}/            (source+grid; shared across DP/FSQ)
    #       seg_{dp}_ck{ckpt}/        (DP-dependent: skillset + skill_tokens; shared across FSQ)
    run_tag = f"FSQ{fsq_digits}_dino{patch_grid}{fsq_exp_suffix}_{ckpt_tag}"
    source_out_dir = skillvla_root / source_dataset
    run_dir = source_out_dir / run_tag
    work_dir = source_out_dir / "_work"
    dino_per_episode_dir = work_dir / "dino" / f"pg{patch_grid}"
    # skillset + skill_tokens depend on the DP model (not FSQ), so key them by DP so a
    # different DP/checkpoint never reuses or clobbers another's segmentation.
    seg_dir = work_dir / f"seg_{dp_policy_name}_ck{dp_checkpoint}"
    skillset_dir = seg_dir / "skillset"

    def slurm(prefix: str, *, cpus: int, mem: str, time: str) -> dict:
        # partition/qos/nodelist/exclude are canonical (global_config.yaml train_*); output keys
        # keep the per-job prefix so submit scripts read the same $<PREFIX>_* vars.
        part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
        excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
        return {
            f"{prefix}_partition": part,
            f"{prefix}_qos": str(get_value(cfg, "train_qos", "base_qos")),
            f"{prefix}_gres": str(get_value(cfg, f"{prefix}_gres", "gpu:1")),
            f"{prefix}_cpus_per_task": int(get_value(cfg, f"{prefix}_cpus_per_task", cpus)),
            f"{prefix}_mem": str(get_value(cfg, f"{prefix}_mem", mem)),
            f"{prefix}_time": str(get_value(cfg, f"{prefix}_time", time)),
            f"{prefix}_nodelist": str(get_value(cfg, "train_nodelist", "")),
            f"{prefix}_exclude_nodes": excl,
        }

    settings: dict = {
        # roots
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "dataset_root": dataset_root,
        # source dataset
        "source_dataset": source_dataset,
        "raw_dataset_dir": dataset_root / source_dataset,
        # DINO (step 2)
        "dino_patch_grid": patch_grid,
        "dino_image_keys": ",".join(image_keys),
        "dino_image_key": image_keys[0] if image_keys else "observation.images.image",
        "dino_base_dataset": dino_base_dataset,
        "base_dino_dir": base_dino_dir,
        "dino_copy_mode": str(get_value(cfg, "dino_copy_mode", "symlink")),
        "dino_per_episode_dir": dino_per_episode_dir,
        # DP (step 3)
        "dp_policy_name": dp_policy_name,
        "dp_checkpoint": dp_checkpoint,
        "dp_policy_path": dp_policy_path,
        "skillset_dir": skillset_dir,
        # skill-level DINO tokens: FSQ-independent but DP-dependent → live in the DP-keyed
        # seg dir (shared across FSQ variants). FSQ vectors stay run-specific.
        "skill_tokens_path": seg_dir / "skill_tokens.npz",
        "skill_latents_path": run_dir / "skill_latents.npz",
        "skillset_dn_step": int(get_value(cfg, "skillset_dn_step", 7)),
        "skillset_n_gmm": int(get_value(cfg, "skillset_n_gmm", 5)),
        "skillset_smooth_window": int(get_value(cfg, "skillset_smooth_window", 7)),
        "skillset_savgol_polyorder": int(get_value(cfg, "skillset_savgol_polyorder", 4)),
        "skillset_replan_interval": int(get_value(cfg, "skillset_replan_interval", 3)),
        "skillset_nms_dist": int(get_value(cfg, "skillset_nms_dist", 25)),
        # parallelism: split tasks into shards of this size, one shard per Slurm array job (1 GPU each)
        "skillset_tasks_per_job": int(get_value(cfg, "skillset_tasks_per_job", 5)),
        "skillset_array_throttle": int(get_value(cfg, "skillset_array_throttle", 0)),
        # post-array verify: re-run tasks with missing episodes up to this many times
        "skillset_max_sweeps": int(get_value(cfg, "skillset_max_sweeps", 2)),
        # FSQ (step 4)
        "fsq_run_name": fsq_run_name,
        "fsq_exp": fsq_exp,
        "fsq_exp_suffix": fsq_exp_suffix,
        "fsq_model_dir": fsq_model_dir,
        "fsq_model_path": fsq_model_path,
        "fsq_checkpoint": fsq_checkpoint,
        "fsq_levels_str": " ".join(str(v) for v in fsq_levels),
        # SkillVLA build (step 5)
        "max_order": int(get_value(cfg, "max_order", 0)),
        "max_length": int(get_value(cfg, "max_length", 200)),
        "skill_pmax": int(get_value(cfg, "pmax", 10)),   # Stage-2 transition randomization 반폭 (ISS window)
        "skill_decoder_state_indices": str(get_value(cfg, "skill_decoder_state_indices", "[0,1,2,3,4,5,6,7]")),
        "cleanup_intermediate": str(get_value(cfg, "cleanup_intermediate", True)).lower(),
        # output layout
        "run_tag": run_tag,
        "skillvla_run_dir": run_dir,
        "skillvla_work_dir": work_dir,
        "skillvla_seg_dir": seg_dir,   # DP-keyed intermediates (skillset + skill_tokens)
        "iss_npz_path": run_dir / "skill_initial_state.npz",   # Stage-2 skill-initial-state (ISS)
        "dino_npz_path": run_dir / "dino.npz",                 # build_data_eval viz only (not used by training)
        "fsq_copy_path": run_dir / "FSQ.pt",
        "skillvla_dataset_dir": run_dir / "skillvla",
        # eval outputs (build_data_eval runs off: raw video + skillvla/ + dino.npz + FSQ.pt)
        "eval_dir": run_dir / "eval",
        "eval_dino_dir": run_dir / "eval" / "dino",
        "eval_skillset_dir": run_dir / "eval" / "skillset",
        "eval_fsq_patch_dir": run_dir / "eval" / "fsq_patch",
        "eval_fsq_recon_dir": run_dir / "eval" / "fsq_recon",
    }
    settings.update(slurm("skillvla", cpus=8, mem="64G", time="8:00:00"))
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--dataset", default=None, help="Override source_dataset")
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load_config(args.config), dataset=args.dataset)
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
