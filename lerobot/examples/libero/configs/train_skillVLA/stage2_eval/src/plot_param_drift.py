#!/usr/bin/env python3
"""Offline per-component parameter drift for ALREADY-TRAINED SkillVLA runs (Stage-2 or FT).

The live tracker (lerobot_train, track_param_drift) needs θ_init snapshotted at train start. For a
finished run we reconstruct θ_init from disk and diff each saved checkpoint against it → the same
‖θ_ckpt − θ_init‖ per component, at checkpoint (save_freq) granularity.

θ_init reconstruction (config.json is the source of truth):
  FT      : init = the Stage-2 checkpoint the run warm-started from (pretrained_path) — a single saved
            model whose keys already match, so it is diffed directly.
  Stage-2 : init is a MIX — the VLM (llm + vlm_vision) from pi05 (pretrained_path) and the
            expert/cond side from the Stage-1 checkpoint (stage1_checkpoint_path), rebuilt with the
            SAME remap the training uses (_remap_pi05_to_vlm / _remap_stage1_to_expert). SCRATCH
            Stage-2 (no Stage-1) has fresh cond/cond_vision → those are NOT on disk and are skipped.

Components (== modeling_skillVLA.SkillVLAPytorch.named_component_params, so numbers match the live graph):
  llm / vlm_vision / cond / cond_vision_encoder / action_expert.

Outputs, per model, under {FT_eval|stage2_eval}/outputs/update/{model_dir}/:
  param_drift_rel.png : ‖Δθ‖ / ‖θ_init‖ per component (normalized → 5 lines comparable on one axis)
  param_drift_abs.png : ‖Δθ‖ per component (absolute)
  param_drift.json    : raw numbers
--wandb also logs each checkpoint's drift to a wandb run "{model}_drift" under sections
  param_drift/* and param_drift_rel/* (a NEW section — overlays with the live train-time curves).

Usage:
  plot_param_drift.py --stage stage2 --all
  plot_param_drift.py --stage ft --model_dir <name> [--model_dir <name2> ...] --wandb
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

_HERE = Path(__file__).resolve()
_TSK = _HERE.parents[2]        # train_skillVLA
_CONFIGS = _HERE.parents[3]    # .../libero/configs
sys.path.insert(0, str(_CONFIGS / "train_skills" / "src"))
sys.path.insert(0, str(_CONFIGS.parents[2] / "src"))  # lerobot/src (configs→libero→examples→lerobot repo root)
from train_skills_config import get_value, load_config  # noqa: E402

GLOBAL_CFG = _CONFIGS / "global_config.yaml"

# component → the safetensors key PREFIXES that belong to it (matches named_component_params()).
GROUPS: dict[str, tuple[str, ...]] = {
    "llm": ("model.paligemma_with_expert.paligemma.model.language_model.",),
    "vlm_vision": ("model.paligemma_with_expert.paligemma.model.vision_tower.",),
    "cond": ("model.cond_encoder.", "model.image_proj."),
    "cond_vision_encoder": ("model.siglip.", "model.dino."),
    "action_expert": ("model.paligemma_with_expert.gemma_expert.model.",
                      "model.action_in_proj.", "model.action_out_proj.",
                      "model.time_mlp_in.", "model.time_mlp_out.",
                      "model.state_proj.", "model.skill_proj."),
}
COLORS = {"llm": "#d62728", "vlm_vision": "#ff7f0e", "cond": "#2ca02c",
          "cond_vision_encoder": "#1f77b4", "action_expert": "#9467bd"}


def _group_of(key: str) -> str | None:
    for g, prefs in GROUPS.items():
        if key.startswith(prefs):
            return g
    return None


def _safetensor_file(model_dir: Path) -> str:
    main = glob.glob(str(model_dir / "model.safetensors"))
    if not main:
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    return main[0]


def _read_all(model_dir: str) -> dict[str, torch.Tensor]:
    """Full raw state dict of a checkpoint's model.safetensors (for the training remap functions)."""
    out = {}
    with safe_open(_safetensor_file(Path(model_dir)), framework="pt") as z:
        for k in z.keys():
            out[k] = z.get_tensor(k)
    return out


def _tracked(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in d.items() if _group_of(k) is not None}


def build_init_tensors(stage: str, pol: dict) -> dict[str, torch.Tensor]:
    """θ_init as {stage2_key: tensor} over tracked keys. FT: the warm-start ckpt directly. Stage-2:
    VLM from pi05 + expert/cond from Stage-1, via the training remap (scratch → cond/cond_vision skipped)."""
    if stage == "ft":
        init_path = str(pol.get("pretrained_path") or "")
        if not init_path or not Path(init_path).is_dir():
            raise FileNotFoundError(f"FT init (pretrained_path) not found: {init_path!r}")
        with safe_open(_safetensor_file(Path(init_path)), framework="pt") as z:
            return {k: z.get_tensor(k) for k in z.keys() if _group_of(k) is not None}

    # stage2 — reuse the EXACT remap the model uses at warm-start
    from lerobot.policies.skillVLA.modeling_skillVLA import (  # noqa: PLC0415
        _remap_pi05_to_vlm, _remap_stage1_to_expert)
    pi05 = str(pol.get("pretrained_path") or "")
    s1 = str(pol.get("stage1_checkpoint_path") or "")
    if not pi05 or not Path(pi05).is_dir():
        raise FileNotFoundError(f"Stage-2 pi05 source (pretrained_path) not found: {pi05!r}")
    init: dict[str, torch.Tensor] = {}
    vlm = _remap_pi05_to_vlm(_read_all(pi05))
    init.update({k: v for k, v in vlm.items() if _group_of(k) in ("llm", "vlm_vision")})
    if s1 and Path(s1).is_dir():
        exp = _remap_stage1_to_expert(_read_all(s1))
        init.update({k: v for k, v in exp.items()
                     if _group_of(k) in ("cond", "cond_vision_encoder", "action_expert")})
    else:
        print("  [scratch] no Stage-1 → cond / cond_vision_encoder / action_expert fresh-init not on "
              "disk; those components are skipped for this run.", file=sys.stderr)
    return init


def process_model(stage: str, vla_root: Path, model_dir: str, out_root: Path,
                  wandb_project: str | None) -> None:
    run = vla_root / model_dir
    ckpt_root = run / "checkpoints"
    names = sorted((p.name for p in ckpt_root.glob("*") if p.name.isdigit()), key=int)
    steps = [int(n) for n in names]
    if not names:
        print(f"[skip] {model_dir}: no numeric checkpoints under {ckpt_root}", file=sys.stderr)
        return
    pol = json.loads((ckpt_root / names[-1] / "pretrained_model" / "config.json").read_text())

    print(f"[{model_dir}] ({stage})\n  steps: {steps}")
    init = build_init_tensors(stage, pol)
    init_norm_sq = {g: 0.0 for g in GROUPS}
    for k, t in init.items():
        init_norm_sq[_group_of(k)] += float(t.float().pow(2).sum())
    present = [g for g in GROUPS if init_norm_sq[g] > 0]   # skip components with no reconstructable init

    series = {g: [0.0] for g in present}
    series_rel = {g: [0.0] for g in present}
    xs = [0] + steps
    for name, st in zip(names, steps):
        acc = {g: 0.0 for g in present}
        with safe_open(_safetensor_file(ckpt_root / name / "pretrained_model"), framework="pt") as z:
            have = set(z.keys())
            for k, t0 in init.items():
                g = _group_of(k)
                if g not in acc or k not in have:
                    continue
                d = z.get_tensor(k).float() - t0.float()
                acc[g] += float(d.pow(2).sum())
        for g in present:
            absd = acc[g] ** 0.5
            series[g].append(absd)
            series_rel[g].append(absd / (init_norm_sq[g] ** 0.5))
        print(f"  @{st}: " + "  ".join(f"{g}={series_rel[g][-1]:.4f}" for g in present))

    out_dir = out_root / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "param_drift.json").write_text(json.dumps(
        {"model_dir": model_dir, "stage": stage, "steps": xs, "abs": series, "rel": series_rel,
         "init_norm": {g: init_norm_sq[g] ** 0.5 for g in present}}, indent=1))
    for kind, data, ylab in (("rel", series_rel, "relative drift  ‖Δθ‖ / ‖θ_init‖"),
                             ("abs", series, "absolute drift  ‖θ_ckpt − θ_init‖")):
        plt.figure(figsize=(8, 5))
        for g in present:
            plt.plot(xs, data[g], marker="o", label=g, color=COLORS[g])
        plt.xlabel(f"{stage} step"); plt.ylabel(ylab)
        plt.title(f"per-component update ({kind})\n{model_dir}", fontsize=8)
        plt.legend(fontsize=9); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(out_dir / f"param_drift_{kind}.png", dpi=130); plt.close()
    print(f"  wrote {out_dir}/param_drift_{{rel,abs}}.png")

    if wandb_project:
        import wandb  # noqa: PLC0415
        wr = wandb.init(project=wandb_project, name=f"{model_dir}_drift", reinit="finish_previous",
                        config={"stage": stage, "model_dir": model_dir})
        for i, st in enumerate(xs):
            wr.log({**{f"param_drift/{g}": series[g][i] for g in present},
                    **{f"param_drift_rel/{g}": series_rel[g][i] for g in present}}, step=st)
        wr.finish()
        print(f"  logged wandb run '{model_dir}_drift' → project '{wandb_project}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("ft", "stage2"), default="stage2")
    ap.add_argument("--model_dir", action="append", default=[], help="run folder name (repeatable)")
    ap.add_argument("--all", action="store_true", help="every run under the stage's output dir")
    ap.add_argument("--wandb", action="store_true", help="also log drift to a wandb run per model")
    ap.add_argument("--wandb_project", default="VLA_eval")
    ap.add_argument("--config", type=Path, default=GLOBAL_CFG)
    args = ap.parse_args()

    cfg = load_config(args.config)
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    sub = "skillVLA_FT" if args.stage == "ft" else "skillVLA_stage2"
    vla_root = outputs_root / sub
    eval_dir = "FT_eval" if args.stage == "ft" else "stage2_eval"
    out_root = _TSK / eval_dir / "outputs" / "update"

    models = list(args.model_dir)
    if args.all:
        models = sorted(p.name for p in vla_root.glob("*") if (p / "checkpoints").is_dir())
    if not models:
        ap.error("pass --model_dir <name> (repeatable) or --all")
    for m in models:
        process_model(args.stage, vla_root, m, out_root,
                      args.wandb_project if args.wandb else None)


if __name__ == "__main__":
    main()
