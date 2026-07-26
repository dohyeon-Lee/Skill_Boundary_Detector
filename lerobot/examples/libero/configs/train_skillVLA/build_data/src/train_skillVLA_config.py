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
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    get_value,
    load_config,
    print_shell,
    resolve_path,
    skillset_probe_settings,
)

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
    skillvla_data_mode = str(get_value(cfg, "skillvla_data_mode", "pt")).strip().lower()
    if skillvla_data_mode not in {"pt", "ft", "ft_own"}:
        raise ValueError(
            "skillvla_data_mode must be pt|ft|ft_own, "
            f"got {skillvla_data_mode!r}."
        )

    # ── FSQ reference (declared like dp_policy_name: folder name + checkpoint) ──
    # Parse codebook levels from the FSQ folder. The remaining suffix identifies
    # its live vision backbone / freeze mode / terminator architecture.
    fsq_run_name = str(get_value(cfg, "fsq_run_name"))
    fsq_checkpoint = str(get_value(cfg, "fsq_checkpoint", "1000"))
    lv_match = re.search(r"fsq(\d+)", fsq_run_name)
    if not lv_match:
        raise ValueError(
            f"fsq_run_name must contain an 'fsq<levels>' tag, got: {fsq_run_name}"
        )
    fsq_digits = lv_match.group(1)
    fsq_levels = [int(d) for d in fsq_digits]
    # Short run names end at the fsq<levels> tag, so the variant may be empty;
    # an empty variant is simply omitted from downstream folder names.
    fsq_variant = fsq_run_name[lv_match.end() :].strip("_")
    fsq_exp_suffix = f"_{fsq_variant}" if fsq_variant else ""
    fsq_exp = fsq_variant

    # ── DP (step 3) ──
    dp_policy_name = str(get_value(cfg, "dp_policy_name"))
    dp_checkpoint = str(get_value(cfg, "dp_checkpoint", "100000"))
    dp_policy_path = dp_outputs_root / dp_policy_name / "checkpoints" / dp_checkpoint / "pretrained_model"
    probe_settings = skillset_probe_settings(cfg)
    skillset_min_skills = int(get_value(cfg, "skillset_min_skills", 1))
    if skillset_min_skills < 1:
        raise ValueError(f"skillset_min_skills must be >= 1, got {skillset_min_skills}.")
    jitter_distribution = str(
        get_value(cfg, "transition_jitter_distribution", "half_normal")
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if jitter_distribution not in {"half_normal", "uniform"}:
        raise ValueError(
            "transition_jitter_distribution must be half_normal|uniform, "
            f"got {jitter_distribution!r}."
        )
    skill_pmax = int(get_value(cfg, "pmax", 10))
    if skill_pmax < 0:
        raise ValueError(f"pmax must be >= 0, got {skill_pmax}.")
    jitter_tag = "halfnormal" if jitter_distribution == "half_normal" else "uniform"
    data_identity_suffix = f"_ms{skillset_min_skills}_pmax{skill_pmax}_{jitter_tag}"

    # ── FSQ (step 4) — model path from the parsed run name + checkpoint ──
    fsq_model_dir = fsq_outputs_root / fsq_run_name
    if fsq_checkpoint in ("0", "best"):
        fsq_model_path = fsq_model_dir / "FSQ.pt"
        ckpt_tag = "best"
    else:
        fsq_model_path = fsq_model_dir / f"FSQ_epoch{int(fsq_checkpoint):04d}.pt"
        ckpt_tag = str(fsq_checkpoint)

    # ── output layout ──
    #   {skillvla_root}/{source_dataset}/{run_tag}/   ← final outputs (FSQ.pt, skillvla/)
    #   {skillvla_root}/{source_dataset}/_work/        ← intermediates, keyed by dependency:
    #       seg_{dp}_ck{ckpt}/        (DP-dependent: skillset + skill_tokens; shared across FSQ)
    # Segmentation mode changes both skill boundaries and the latent sequence. Keep it in the
    # final dataset identity as well as the intermediate seg_dir so a completed dataset built with
    # another mode can never short-circuit this build. It also makes FT snap references resolve only
    # against a PT vocabulary produced with the same segmentation mode.
    skillset_mode_suffix = probe_settings["skillset_probe_suffix"]
    base_run_tag = f"FSQ{fsq_digits}{fsq_exp_suffix}_{ckpt_tag}{skillset_mode_suffix}"
    run_tag = f"{base_run_tag}_{skillvla_data_mode}"
    # transfer 빌드(snap): 미지원 코드를 최근접 지원 코드로 snap한 빌드는 산출물(skill_latents/skillvla)이
    # 다르므로 폴더 분리 — run_tag에 _snap{min_freq} 부착 (downstream 파서들의 run_tag 정규식은
    # `FSQ\d+_dino\d+.*?` 꼴이라 그대로 통과). _work 중간물은 snap 무관(dino/segmentation)이라 공유 유지.
    fsq_snap = as_bool(get_value(cfg, "fsq_snap_to_supported", False))
    fsq_snap_reference = ""
    if fsq_snap:
        snap_suffix = f"_snap{int(get_value(cfg, 'fsq_snap_min_code_freq', 1))}"
        if skillvla_data_mode == "pt":
            # PT vocabulary pruning: the just-encoded raw distribution is the
            # reference, so no user-maintained path is needed.
            fsq_snap_reference = "self"
        else:
            # FT must use the PT vocabulary for this exact FSQ/checkpoint and
            # pruning threshold. Search source-dataset directories so the FT
            # config needs no duplicated PT dataset/path field.
            pt_run_tag = f"{base_run_tag}_pt{snap_suffix}{data_identity_suffix}"
            pt_refs = sorted(skillvla_root.glob(f"*/{pt_run_tag}/skill_latents.npz"))
            if len(pt_refs) != 1:
                found = "\n  ".join(str(p) for p in pt_refs) or "(none)"
                raise ValueError(
                    f"skillvla_data_mode={skillvla_data_mode} requires exactly one completed PT reference "
                    f"at */{pt_run_tag}/skill_latents.npz; found:\n  {found}\n"
                    "Build the matching PT data first (same FSQ checkpoint and "
                    "fsq_snap_min_code_freq)."
                )
            fsq_snap_reference = str(pt_refs[0])
        run_tag += snap_suffix
    # Final SkillVLA artifacts depend on episode filtering and jitter sampling. Keep those values in
    # the identity so changing either cannot short-circuit against an older completed dataset.
    run_tag += data_identity_suffix
    source_out_dir = skillvla_root / source_dataset
    run_dir = source_out_dir / run_tag
    work_dir = source_out_dir / "_work"
    # All SkillVLA builds use global boundaries. pt and ft_own reduce this
    # source's curves; ft reuses the one matching PT global threshold.
    seg_base = (
        f"seg_{dp_policy_name}_ck{dp_checkpoint}"
        f"{probe_settings['skillset_probe_suffix']}"
        f"_ms{skillset_min_skills}"
    )
    skillset_global_threshold_source = ""
    if skillvla_data_mode == "ft":
        pt_thresholds = sorted(
            skillvla_root.glob(f"*/_work/{seg_base}_globalmean/skillset/global_boundary_threshold.json")
        )
        if len(pt_thresholds) != 1:
            found = "\n  ".join(str(p) for p in pt_thresholds) or "(none)"
            raise ValueError(
                "skillvla_data_mode=ft requires exactly one completed PT global threshold "
                f"at */_work/{seg_base}_globalmean/skillset/global_boundary_threshold.json; "
                f"found:\n  {found}\nBuild the matching PT data first."
            )
        skillset_global_threshold_source = str(pt_thresholds[0])
    seg_suffix = "_globalref" if skillset_global_threshold_source else "_globalmean"
    seg_dir = work_dir / f"{seg_base}{seg_suffix}"
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
        "skillvla_data_mode": skillvla_data_mode,
        "raw_dataset_dir": dataset_root / source_dataset,
        # (DINO precompute emit 은퇴 — DINO는 어디서도 precompute 안 함. DP=state/raw-frames,
        #  FSQ 학습·terminator=ONLINE. dino_root/required/generate/base 등 소비자 0.)
        # DP (step 3)
        "dp_policy_name": dp_policy_name,
        "dp_checkpoint": dp_checkpoint,
        "dp_policy_path": dp_policy_path,
        "skillset_dir": skillset_dir,
        "skill_latents_path": run_dir / "skill_latents.npz",
        "skillset_dn_step": int(get_value(cfg, "skillset_dn_step", 7)),
        "skillset_n_gmm": int(get_value(cfg, "skillset_n_gmm", 5)),
        "skillset_smooth_window": int(get_value(cfg, "skillset_smooth_window", 7)),
        "skillset_savgol_polyorder": int(get_value(cfg, "skillset_savgol_polyorder", 4)),
        "skillset_replan_interval": int(get_value(cfg, "skillset_replan_interval", 3)),
        "skillset_nms_dist": int(get_value(cfg, "skillset_nms_dist", 25)),
        "skillset_min_skills": skillset_min_skills,
        **probe_settings,
        "skillset_boundary_threshold_mode": "global_mean",
        "skillset_global_threshold_source": skillset_global_threshold_source,
        "skillset_global_threshold_path": skillset_dir / "global_boundary_threshold.json",
        "skillset_dino_feature_dir": resolve_path(
            root, get_value(cfg, "skillset_dino_feature_dir", "")
        ),
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
        # transfer 안전망(B): 인코딩 시 미지원(학습때 안 쓰인) 코드 → 최근접 지원 코드로 snap.
        # (snap=true인데 reference가 없/틀리면 아래에서 이미 제출 전에 raise — 런타임 좀비 체인 방지)
        "fsq_snap_to_supported": fsq_snap,
        "fsq_snap_min_code_freq": int(get_value(cfg, "fsq_snap_min_code_freq", 1)),
        "fsq_snap_reference": fsq_snap_reference,
        "fsq_levels_str": " ".join(str(v) for v in fsq_levels),
        # SkillVLA build (step 5)
        "max_order": int(get_value(cfg, "max_order", 0)),
        "max_length": int(get_value(cfg, "max_length", 200)),
        "skill_pmax": skill_pmax,   # Stage-2 transition randomization 반폭 (ISS window)
        "skill_jitter_distribution": jitter_distribution,
        "skill_decoder_state_indices": str(get_value(cfg, "skill_decoder_state_indices", "[0,1,2,3,4,5,6,7]")),
        "cleanup_intermediate": str(get_value(cfg, "cleanup_intermediate", True)).lower(),
        # output layout
        "run_tag": run_tag,
        "skillvla_run_dir": run_dir,
        "skillvla_work_dir": work_dir,
        "skillvla_seg_dir": seg_dir,   # DP-keyed intermediates (skillset + skill_tokens)
        "iss_npz_path": run_dir / "skill_initial_state.npz",   # Stage-2 skill-initial-state (ISS)
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
