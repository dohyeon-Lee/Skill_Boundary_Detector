# GR00T N1.7 — LeRobot PT / FT

`PT/` adapts the released **base** `nvidia/GR00T-N1.7-3B` model to LIBERO-90.
`FT/` adapts **our PT checkpoint** to LIBERO-10. No released LIBERO-tuned
checkpoint, SkillVLA skill inputs, XYZ goals or auxiliary heads are used.

## Installation and weights

The unmodified Hugging Face LeRobot source is pinned to
`146d8474932c8b566717eb2738ba459337ab71c4` in `third_party/lerobot_groot`.
The environment `.venv_groot_lerobot` is separate from pi05/SkillVLA and Molmo.
All 160 non-editable dependencies are pinned; Python 3.12 / CUDA 12.8 wheels.

From the project root:

```bash
bash setup_env.sh --groot-only           # existing server: only prepare GR00T
bash setup_env.sh --groot-only --check   # read-only imports/versions/source check

# Separate download; authenticate with hf auth login if NVIDIA requires it.
.venv_groot_lerobot/bin/hf download nvidia/GR00T-N1.7-3B --local-dir models/groot_n1_7_base
```

The root default `bash setup_env.sh` also installs GR00T, unless
`SBD_SETUP_GROOT=0`. Use `--groot-only` on an active server: the root default
installer recreates the main environment. Neither installer downloads weights
or datasets. Existing source edits and environments are preserved; a dirty
checkout at a different revision is rejected instead of overwritten.

CMake 3 is installed before LIBERO's legacy egl-probe build. Both OpenCV
distributions are pinned to the same version to prevent overlapping `cv2`
files from incompatible releases. Decord's upstream py3 wheel carries a stale
cp36 WHEEL tag: only that single known `uv pip check` diagnostic is accepted,
after a successful native-library import; all other failures remain fatal.

## Dataset and model semantics

- Same converted data as pi05: top + wrist images, 8D proprio (EEF XYZ3,
  axis-angle3, measured finger2), 7D delta EEF action (translation3, rotation3,
  gripper1). Actions are **already delta**; no second state subtraction occurs.
- `pt_proprio_grounding: true` reuses pi05's episode-start XYZ transform and
  global grounded state statistics. Actions and source dataset files are not
  rewritten. FT inherits the grounding convention from PT, with normalization
  rebuilt from the FT dataset. Resume restores the saved processor statistics.
- Our dataset quantiles/layout replace the raw base's robot-specific statistics.
  All eight state dimensions are retained. Processor checkpoints contain native
  LeRobot steps, camera order, layout, normalization and action masks.
- `dataset.action_horizon: 10` supervises 10x7 action entries. Native N1.7's
  40x132 tensor is padded and **masked**, including episode-end padding.
- Native GR00T image preprocessing owns the crop/resize; no Molmo-specific
  rotation or extra spatial crop is applied. Native LIBERO gripper decoding
  maps our zero-close dataset convention to the simulator's close +1/open -1.
- `training.tuning: default` uses the upstream default: frozen language/vision
  backbone, trainable projector + action head + VL norm/projector.
  `full` additionally trains language/vision. This is not Molmo's full-tuning
  default. Change `run.suffix` when comparing recipes.
- Upstream AdamW, flow-matching objective and precision are preserved: FP32
  model parameters and BF16 autocast. There is no Molmo Kahan optimizer and no
  claim that a given microbatch fits a 24/48GB GPU.
- `gradient_checkpointing: true` recomputes native action transformer blocks
  with RNG-preserving non-reentrant checkpointing. Full tuning also enables
  the backbone's own checkpointing. Architecture/state-dict keys are unchanged.

## Configure, preview and submit

Edit `PT/groot_PT_config.yaml` or `FT/groot_FT_config.yaml`. Slurm partition,
QoS and node exclusions inherit the existing global/server configuration.
The submit script chooses its own Python; manual venv activation is unnecessary.

```bash
bash lerobot/examples/libero/configs/train_groot/PT/submit_train.sh --dry-run
bash lerobot/examples/libero/configs/train_groot/FT/submit_train.sh --dry-run

.venv_groot_lerobot/bin/python lerobot/examples/libero/configs/train_groot/PT/src/train.py \
  --config lerobot/examples/libero/configs/train_groot/PT/groot_PT_config.yaml --check-data

cd lerobot/examples/libero/configs/train_groot/PT
./submit_train.sh --smoke   # 20 updates, batch 1, no W&B, separate *_smoke output
./submit_train.sh           # configured training
```

FT works identically after selecting our completed PT checkpoint. `--config`
selects another YAML. Each submission snapshots the configuration and pins
`warm_start.checkpoint: last` to a concrete checkpoint. A copied FT checkpoint
uses the local base model path; optionally set `model.checkpoint` in FT to
override the standard `models/groot_n1_7_base` location.

`batch_size` is global; `micro_batch_size` is per GPU. Accumulation is
`batch_size / (micro_batch_size * gpus)`. Steps, warmup, save/log intervals,
checkpoint folder names and W&B x-axis count **optimizer updates**. Native
training state/config retain micro-step counts for sample-exact upstream resume.

## Outputs, retention and resume

Default outputs:

```text
outputs_filtered/groot_PT/bs16_libero_90_full_full_base_n1_7_lerobot/
outputs_filtered/groot_FT/<PT run>_100000_libero_10_full_1_ft_bs16_n1_7_lerobot/
  checkpoints/010000/
    pretrained_model/   model/config/processors/stats + sbd_groot.json
    training_state/     optimizer, scheduler, RNG, micro-step metadata
  checkpoints/last -> 010000
```

`keep_checkpoints: 2` retains the newest two complete checkpoints plus
`protect_checkpoints` (PT: 30k/50k/100k; FT: 5k/10k). `-1` retains all.
Protection is retention, not an extra save schedule: protected steps must
coincide with a regular save or the final update. Pruning occurs after a
successful save; partial, symlinked and unrelated directories are not deleted.

Re-submit the same YAML to resume automatically. Optimizer, scheduler, RNG,
saved processors and data position are restored by upstream. An incomplete
latest save or recipe mismatch fails safely; use another suffix for a new run.

## Local verification

```bash
.venv_groot_lerobot/bin/python -m pytest --noconftest -q \
  lerobot/tests/scripts/test_groot_setup.py \
  lerobot/tests/scripts/test_groot_train_config.py \
  lerobot/tests/scripts/test_groot_checkpoint_retention.py \
  lerobot/tests/scripts/test_groot_training_adapter.py
```

`--check-data` requires no model weights or GPU. It is not a GPU training test:
run `--smoke` after downloading weights to validate CUDA capacity and the full
forward/backward path on the target GPU. This integration provides PT/FT
training; a project-specific simulation-eval launcher is not added here.
