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
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

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
    fsq_name_grid = int(pg_match.group(1))   # grid tag in the FSQ NAME → run_tag label only
    fsq_exp_suffix = fsq_run_name.split(f"_dino{fsq_name_grid}", 1)[1]   # e.g. "_eqloss" or ""
    fsq_exp = fsq_exp_suffix.lstrip("_")

    # ── DINO (step 2) ── the build declares WHICH (grid × camera) DINO it needs (yaml), and
    # either VALIDATES a precomputed base has them or GENERATEs them on the source frames.
    # Two consumers, possibly different grids: DP segmentation (DP_patch_grid, 3rd-person) and
    # the FSQ terminator / skill DINO tokens / dino.npz (FSQ_patch_grid, dino_third/dino_wrist).
    THIRD_KEY, WRIST_KEY = "observation.images.image", "observation.images.wrist_image"
    dino_base_dataset = str(get_value(cfg, "dino_base_dataset", "libero_90")).strip()
    dino_generate = dino_base_dataset.lower() == "generate"
    dp_patch_grid = int(get_value(cfg, "DP_patch_grid", fsq_name_grid))
    fsq_patch_grid = int(get_value(cfg, "FSQ_patch_grid", fsq_name_grid))
    dino_third = as_bool(get_value(cfg, "dino_third", True))
    dino_wrist = as_bool(get_value(cfg, "dino_wrist", False))
    fsq_cameras = ([THIRD_KEY] if dino_third else []) + ([WRIST_KEY] if dino_wrist else [])
    if not fsq_cameras:
        raise ValueError("At least one of dino_third / dino_wrist must be true.")
    # Required (grid, camera) set: DP needs its grid+3rd; the FSQ side needs its grid×cameras.
    # de-dup while preserving order (DP and FSQ may share grid/camera).
    required = []
    for g, cam in [(dp_patch_grid, THIRD_KEY)] + [(fsq_patch_grid, c) for c in fsq_cameras]:
        if (g, cam) not in required:
            required.append((g, cam))
    # base DINO dir per grid (explicit base mode); generate writes into {source}_DINO instead.
    gen_base_name = f"{source_dataset}_DINO"            # generate target (source's own DINO)
    base_name = gen_base_name if dino_generate else f"{dino_base_dataset}_DINO"
    image_keys = fsq_cameras   # primary image_keys = FSQ cameras (extract/merge); DINO_IMAGE_KEY = 3rd

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
    #       dino/pg{grid}/            (per-grid; DP uses pg{DP_grid}, FSQ side uses pg{FSQ_grid})
    #       seg_{dp}_ck{ckpt}/        (DP-dependent: skillset + skill_tokens; shared across FSQ)
    run_tag = f"FSQ{fsq_digits}_dino{fsq_name_grid}{fsq_exp_suffix}_{ckpt_tag}"
    # transfer 빌드(snap): 미지원 코드를 최근접 지원 코드로 snap한 빌드는 산출물(skill_latents/skillvla)이
    # 다르므로 폴더 분리 — run_tag에 _snap{min_freq} 부착 (downstream 파서들의 run_tag 정규식은
    # `FSQ\d+_dino\d+.*?` 꼴이라 그대로 통과). _work 중간물은 snap 무관(dino/segmentation)이라 공유 유지.
    fsq_snap = as_bool(get_value(cfg, "fsq_snap_to_supported", False))
    fsq_snap_reference = str(get_value(cfg, "fsq_snap_reference", "") or "").strip()
    if fsq_snap:
        # 제출 시점 fail-fast: 예전엔 encode 잡 런타임에야 터져서 뒤의 build 잡이
        # DependencyNeverSatisfied 좀비로 남았음 — 여기서 막으면 체인 자체가 제출되지 않음.
        if not fsq_snap_reference:
            raise ValueError("fsq_snap_to_supported=true인데 fsq_snap_reference가 비어 있음 — 기준 빌드의 "
                             "skill_code_freq.npz 경로(전이 빌드) 또는 \"self\"(자기 데이터 어휘 정리)를 지정하세요.")
        if fsq_snap_reference.lower() == "self":
            # self-build 어휘 정리(pruning): 이 빌드가 방금 인코딩한 RAW 토큰 분포 자체를 기준표로 씀
            # (encode_FSQ_skills.py가 "self"를 그렇게 해석 — 외부 파일 불필요, 1-pass 자기완결).
            # 빈값(깜빡한 실수)과 구분하기 위해 명시적 "self"만 허용. 전이 빌드에 self를 쓰면 새 데이터의
            # 코드가 곧 '지원'이 되어 snap이 no-op이 되므로 의미 없음(그때는 원본 빌드 경로를 지정).
            fsq_snap_reference = "self"
        else:
            _ref = Path(fsq_snap_reference)
            if not _ref.is_absolute():
                _ref = root / fsq_snap_reference
            if not _ref.is_file():
                raise ValueError(f"fsq_snap_reference 파일이 존재하지 않음: {_ref}")
            fsq_snap_reference = str(_ref)
        run_tag += f"_snap{int(get_value(cfg, 'fsq_snap_min_code_freq', 1))}"
    source_out_dir = skillvla_root / source_dataset
    run_dir = source_out_dir / run_tag
    work_dir = source_out_dir / "_work"
    dino_root = work_dir / "dino"                         # per-grid subdirs: dino/pg{grid}/{camera}/
    dp_dino_dir = dino_root / f"pg{dp_patch_grid}"        # DP segmentation reads here (3rd)
    fsq_dino_dir = dino_root / f"pg{fsq_patch_grid}"      # extract/merge read here (3rd [+ wrist])
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
        # (DINO precompute emit 은퇴 — DINO는 어디서도 precompute 안 함. DP=state/raw-frames,
        #  FSQ 학습·terminator=ONLINE. dino_root/required/generate/base 등 소비자 0.)
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
        # transfer 안전망(B): 인코딩 시 미지원(학습때 안 쓰인) 코드 → 최근접 지원 코드로 snap.
        # (snap=true인데 reference가 없/틀리면 아래에서 이미 제출 전에 raise — 런타임 좀비 체인 방지)
        "fsq_snap_to_supported": fsq_snap,
        "fsq_snap_min_code_freq": int(get_value(cfg, "fsq_snap_min_code_freq", 1)),
        "fsq_snap_reference": fsq_snap_reference,
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
        # Stage-1 terminator wrist tokens: only when dino_wrist:true (FSQ trained terminator_use_wrist).
        # Same merge as dino.npz but the wrist camera → a second per-skill DINO token cache. Empty = none.
        "dino_wrist_npz_path": (run_dir / "dino_wrist.npz") if dino_wrist else "",
        "wrist_image_key": WRIST_KEY if dino_wrist else "",    # merge --image_key for dino_wrist.npz
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
