# SkillVLA backbone selection

Set `skillvla_backbone` in `PT/stage1_common_config.yaml` or
`FT/newtask_ft_common_config.yaml`:

```yaml
skillvla_backbone: pi05  # pi05 | groot_integrate | groot_seperate
```

`pi05` keeps the existing model, environment, checkpoint names, and launch path.
Both GR00T choices initialize the VLM **and action expert** from
`models/groot_n1_7_base` (NVIDIA GR00T N1.7). New SkillVLA readers, conditioning
projections, auxiliary heads and termination branch start fresh. Loading fails
if any pretrained weight is missing or incompatible.

The existing `PT/VSA`, `PT/Predictor`, `FT/VSA`, and `FT/Predictor` submit scripts
dispatch GR00T jobs to `.venv_groot_lerobot` while reusing the shared SkillVLA
dataset, trainer, checkpoint retention, and monitoring. No separate recipe
copies are necessary. Output names include `groot_integrate` or
`groot_seperate`; the latter spelling is intentional and is part of the selector.

## Model contract

- Native causal Cosmos/Qwen3-VL, 16 VLM layers; native 32-block DiT with its
  cross/self attention, AdaLN, state/action encoders, and action decoder.
- Native normalized current proprio remains a state token in the expert.
  It is never added to the VLM's skill/UV prompt.
- Two independent recurrent 100-token bottlenecks read all VLM layers.
  The first 8 cross+self pairs use the lower bank; the final 8 use the upper.
  Expert cross-attention only sees these banks.
- `groot_integrate`: all expert cross blocks read the whole 100-token bank.
- `groot_seperate`: 50 visual and 50 condition tokens, with shared self-attention
  between them; native text/image cross-attention alternation reads the
  corresponding half. Top/Wrist/XYZ auxiliary queries belong to the visual half.
- Skill broadcasts and UV-to-AdaLN conditioning act in the first half only;
  state tokens are excluded from the skill broadcast.
- Full-skill denoising shares the first half of the expert. Existing B-spline
  length, quantile normalization and optional pose6+gripper-action representation
  apply. It uses separate input/output heads.
- Detached per-layer termination readers have their own parameters and gradient
  clipping. Existing transition randomization, previous-skill code/UV, and
  post-end supervision are shared unchanged.
- Predictor keeps the existing mode2 skill and UV readers, losses and crop/UV
  transforms, replacing its underlying VLM with GR00T's pretrained VLM.

VSA camera preprocessing uses a native 256→230 crop (random during training,
center during inference), then a 224 input to its 16-pixel patch encoder.
UV conditions and patch labels follow that crop. The optional extra SkillVLA
Top crop also transforms current/past UV labels consistently. Native GR00T's
noise-at-zero flow convention and four inference updates are retained.

Both pi0.5 and GR00T SkillVLA defer auxiliary logging scalars to the shared
trainer's collection. The two banks reuse geometry, patch distributions and
normalized XYZ targets only within the current forward. This cache is discarded
on exit (including exceptions); every subsequent crop/UV gets fresh labels.
Only immutable quantiles, camera calibration tensors and patch-grid axes are
cached across forwards, keyed by their values, device and dtype. These caches
are transient and add no checkpoint keys or changes to the loss/reduction/RNG.

Saved policies use `type: skill_expert` / `skill_aux` to retain the common
dataset/processor contract. The existing semantic architecture label still
selects the dual-bottleneck data targets; `skillvla_backbone` selects the actual
model implementation. The GR00T adapter derives layer counts from GR00T rather
than treating the legacy `9_9` label as its layer count.

## FT source

GR00T FT resolves the matching component's PT run from the current PT recipe
and uses its 100k checkpoint. It rejects pi0.5 or the other GR00T variant.
For a different PT source, set the override in the same FT common YAML:

```yaml
warm_start:
  backend_sources:
    groot_integrate:
      vsa_checkpoint: {run: YOUR_GROOT_PT_VSA_RUN, checkpoint: "100000"}
      predictor_checkpoint: {run: YOUR_GROOT_PT_PREDICTOR_RUN, checkpoint: "100000"}
```

FT inherits architecture, state/action statistics, B-spline representation,
transition randomization and predictor geometry from its PT checkpoint.
The existing FT adaptation toggles and per-task episode selection apply.
`freeze_expert: true` freezes the entire native action head (including its VL
normalization/self-attention frontend), skill/UV projections and trajectory
input/output heads. Loss still passes through those frozen modules to the
trainable VLM and bottlenecks; visual auxiliary and termination heads train.

## Runtime assets and checks

Install the isolated environment with `bash setup_env.sh --groot-only`.
The GR00T base already contains the VLM weights; Cosmos tokenizer/image processor
sidecars are also required. To reuse the project HF cache, or download only those
small assets if absent, run from the repository root:

```bash
.venv_groot_lerobot/bin/python lerobot/skillVLA/libero/configs/train_skillVLA/PT/src/groot/prepare_assets.py
```

This stores them under `models/groot_n1_7_base/processor`, so remote launchers
transfer them with the model. Runtime falls back to the native Cosmos processor
repository/cache if that directory is absent. An incomplete tokenizer is rejected.

To inspect a resolved job without starting training:

```bash
.venv_groot_lerobot/bin/python lerobot/skillVLA/libero/configs/train_skillVLA/PT/src/groot/train.py \
  --config lerobot/skillVLA/libero/configs/train_skillVLA/PT/VSA/vsa_train_config.yaml \
  --phase pt --component VSA --print-only
```

Numerical CPU checks (native small Qwen3-VL and native small GR00T DiT):

```bash
PYTHONPATH=lerobot/src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv_groot_lerobot/bin/python -m pytest \
  --confcutdir=lerobot/tests/policies/skill_groot lerobot/tests/policies/skill_groot -q
```

The tests cover causal attention/camera masks, both expert memory layouts,
main/termination gradient separation, full-skill prefix gradients, FT freezing,
Predictor mode2, strict checkpoint round trips, and PT/FT configuration inheritance.
