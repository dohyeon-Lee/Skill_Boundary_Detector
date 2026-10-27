# MolmoAct2: HF LeRobot PT / FT

`PT/` adapts the released `models/molmoact2_base` weights to our LIBERO-90 data.
`FT/` adapts **our PT result** to LIBERO-10. Neither is foundation pretraining.

## Source and isolated environment

The policy, continuous/discrete objectives, model, BF16/Kahan AdamW optimizer,
training loop and checkpoint serialization are the **Hugging Face LeRobot port**:
https://github.com/huggingface/lerobot/tree/146d8474932c8b566717eb2738ba459337ab71c4

- Source: `third_party/lerobot_molmoact2`, pinned at the revision above (LeRobot 0.6.2).
- Environment: `.venv_molmoact2_lerobot` (Python 3.12 / torch 2.10 CUDA 12.8).
- `requirements.lock.txt` pins all non-editable packages. CMake 3 is installed
  first for LIBERO's legacy egl-probe build.
- The AllenAI original trainer and old `.venv_molmoact2` are no longer used.
  No recursive submodules, ai2-molmo or olmo dependency is needed.
- The main `.venv` and in-tree LeRobot used by pi05/SkillVLA are independent.
- Submission automatically selects the right Python; no manual activation.

From the project root:

```bash
bash setup_env.sh                             # new machine: main + Molmo environments
bash setup_env.sh --molmoact2-only             # existing machine: ONLY install/repair Molmo
bash setup_env.sh --molmoact2-only --check     # offline import/path/version checks, no GPU

# Weights/data remain separate. Existing models/molmoact2_base is reusable.
.venv_molmoact2_lerobot/bin/hf download allenai/MolmoAct2 --local-dir models/molmoact2_base
```

The installer is idempotent and preserves existing venvs. Dirty source at a
different revision is never overwritten. System git/gcc/g++/make and EGL/video
libraries must already be available. `SBD_SETUP_MOLMOACT2=0 bash setup_env.sh`
skips Molmo. Do not run the root default installer merely to migrate Molmo on
an active server: use `--molmoact2-only` to leave the main environment alone.

## Configuration and experimental settings

Use the existing `PT/molmoact2_PT_config.yaml` and `FT/molmoact2_FT_config.yaml`.
PT and FT now default to **continuous/8**, following the official robot
fine-tuning objective and training noise/time sampling count. `both` remains
available as an explicit joint discrete-action CE + flow-matching ablation;
it may download the official FAST tokenizer. Neither mode adds SkillVLA skill
tokens, XYZ goals or auxiliary heads.

Full tuning uses the official component learning rates: LLM `1e-5`, vision
encoder `5e-6`, image connector/projector `5e-6`, and action expert `5e-5`.
Both YAMLs list these rates explicitly. The optional legacy `base_lr` applies
one shared rate to unspecified components only when explicitly configured;
it is no longer present in the shipped recipes. Earlier experiments used PT
continuous/1 or FT both/8 with a shared `2.5e-5` rate and are not retroactively
changed. Use a new `run.suffix` when retraining with the updated recipe rather
than resuming an existing run with different settings. The batch, horizon,
step budget and warmup-constant schedule are otherwise unchanged.

Reference: https://huggingface.co/docs/lerobot/main/en/molmoact2

- Full tuning (`tuning: full`) selects upstream VLM FFT plus the action expert.
  `action_expert` freezes the VLM and requires continuous-only training.
- `policy.dtype=bfloat16` is fixed in the bridge: upstream uses mixed BF16/FP32
  parameter storage and its own Kahan-compensated AdamW. It does not guarantee
  every recipe fits a 48GB GPU.
  Local validation on A6000 (47.40 GiB), full tuning / continuous / one flow
  draw / microbatch 1 / checkpointing ON passed forward/backward but ran out of
  memory during the **first optimizer update**. BF16 alone is not sufficient
  to run this full-tuning recipe on a single 48GB card; use a larger-memory
  GPU or separately validate sharding. No LoRA/freeze/objective change is
  applied automatically to conceal a capacity failure.
- Both camera views, horizon 10, 8D state, 7D action and existing zero-close gripper
  convention are unchanged. Padded action dimensions use upstream masks.
- `pt_proprio_grounding: true` reuses pi05's episode-start XYZ transform and exact
  global grounded statistics. Raw parquet/stats are not rewritten. FT inherits
  the coordinate convention. FT normalization defaults to the **FT dataset**;
  `training.normalization_stats: pt` instead retains the saved PT state/action statistics.
- Warmup-constant LR and per-component LR overrides retain their existing meanings.
- Image augmentation uses the port's documented crop/rotation/color-jitter recipe.
- `num_flow_timesteps` counts training noise/time draws, not inference steps.

## Batch size and step units

`batch_size` is the **global batch**; `micro_batch_size` is per GPU.
Accumulation is `batch_size / (micro_batch_size * gpus)`.

The new upstream loop counts micro-batches. Our adapter maps the YAML's
`steps`, warmup, log/save intervals, checkpoint folder names and W&B x-axis to
**optimizer updates**, preserving the old interface. For example global 16 /
micro 1 / one GPU executes 16 micro-batches per update; YAML 100k still means
100k optimizer updates, **not 6,250**. Native saved training metadata/config
keep the micro-step count for correct sample-exact resume. Do not interpret
`training_step.json.step` as the user-facing checkpoint number.

## Preview, validation and launch

```bash
bash lerobot/examples/libero/configs/train_molmoact2/PT/submit_train.sh --dry-run
bash lerobot/examples/libero/configs/train_molmoact2/FT/submit_train.sh --dry-run

.venv_molmoact2_lerobot/bin/python \
  lerobot/examples/libero/configs/train_molmoact2/PT/src/train.py \
  --config lerobot/examples/libero/configs/train_molmoact2/PT/molmoact2_PT_config.yaml \
  --check-data

cd lerobot/examples/libero/configs/train_molmoact2/PT
./submit_train.sh --smoke   # 20 updates, batch 1, no W&B, separate *_smoke run
./submit_train.sh           # configured PT run
```

FT submission works identically in `FT/` after selecting our PT checkpoint.
`--config <path>` selects another YAML. Each submission snapshots its YAML and
pins `warm_start.checkpoint: last` to a concrete checkpoint.

## Output, checkpoints, retention and resume

PT/FT always pass upstream `wandb.disable_artifact=true`, including on resume.
W&B metrics remain enabled according to the YAML; model weights are not uploaded
to W&B. Local checkpoint saves, retention and resume are unchanged. This takes
effect at the next trainer launch, not in already-running processes. Previously
uploaded W&B artifacts are not removed automatically; deleting those cloud copies
does not delete server-side checkpoint files.

A backend suffix avoids mixing incompatible historical AllenAI runs:

- PT: `outputs_filtered/molmoact2_PT/bs16_libero_90_full_full_base_lerobot/`
- FT: `outputs_filtered/molmoact2_FT/<PT run>_100000_libero_10_full_1_ft_bs16_lerobot/`

Every saved checkpoint is a standard, transferable LeRobot checkpoint:

```text
checkpoints/010000/
  pretrained_model/   config.json, model.safetensors, train_config.json, processors/stats
                      sbd_molmoact2.json (grounding and step-clock provenance)
  training_state/     optimizer including Kahan state, scheduler, RNG, micro-step metadata
checkpoints/last -> 010000
```

`keep_checkpoints: 2` keeps the newest two **plus**
`protect_checkpoints: [30000, 50000, 100000]`. Protection does not itself schedule
extra saves; protected steps must coincide with save_every or the final step.
Pruning runs only after a completed native save and last-link update; partial
checkpoints, symlinks and unrelated folders are not deleted. `-1` retains all.

Re-submit the same YAML to resume the latest complete checkpoint automatically.
Model/optimizer/scheduler/RNG/data-order state are restored by upstream. Batch
shape, accumulation and objective/grounding changes require a new suffix.
An incomplete latest checkpoint or existing output without a resumable save
fails safely instead of silently overwriting results.

**Old AllenAI FSDP checkpoints cannot be resumed by the new backend.** They and
all existing results remain untouched; only the released HF-format base weights
are reused. Metadata `sbd_launch.json` records source revision, versions, recipe
and the distinction between optimizer-update and native micro-batch clocks.
## PT evaluation

`PT/eval/molmoact2_eval_config.yaml` and `PT/eval/submit_eval.sh` provide the same
LIBERO closed-loop evaluation workflow as pi05: checkpoint lists, task/GPU
packing, exact-demo or benchmark scenes, annotated videos, metrics and plots.
Run `./submit_eval.sh --dry-run` in `PT/eval` before submitting. The main venv
runs the existing simulator; the dedicated MolmoAct2 venv runs the unmodified
policy and saved processors. See `../src/baseline_eval/README.md` under configs
for the shared evaluation contract and grounding behavior.

## FT evaluation

FT reuses the same isolated native MolmoAct2 evaluation engine as PT. The FT
wrapper keeps logs and reports under `FT/eval/` and selects checkpoints from
`outputs_filtered/molmoact2_FT`. Its default YAML evaluates every LIBERO-10
task on the same exact held-out scenes used by the GR00T FT evaluation.

```bash
bash lerobot/examples/libero/configs/train_molmoact2/FT/eval/submit_eval.sh --dry-run
bash lerobot/examples/libero/configs/train_molmoact2/FT/eval/submit_eval.sh
```

## FT normalization statistics probe

Set `training.normalization_stats: ft | pt` in the FT YAML. Default `ft`
uses the grounded FT dataset statistics and preserves existing run names.
Selecting `pt` reads the warm-start checkpoint's saved state/action statistics
and adds `_ptstats` to the run name (before the backend suffix, when present).
Both input normalization and action unnormalization use the selected statistics,
saved in each checkpoint. Grounding is inherited independently from PT. Resume
retains its own FT checkpoint's processors and rejects changing modes; historical
FT checkpoints are treated as `ft`. Existing outputs are not converted.

## Optimizer state offload probe

In `FT/molmoact2_FT_config.yaml` select:

```yaml
training:
  optimizer:
    state_offload: none  # none | cpu; keep the existing component LR entries
```

`none` is the default and preserves existing runs. `cpu` stores Adam moments
and BF16 Kahan compensation on CPU, staging only one parameter's state on the
GPU during its update. The native update equations, component clipping, all
trainable parameters, learning rates and global batch are preserved. CPU/GPU
transfers add overhead. This probe currently supports one GPU per job.

CPU offload adds `_optcpu` before the run/backend suffix so its checkpoints do
not mix with GPU-state runs. Native optimizer checkpoints and resumed optimizer
configs retain the offload mode. Loading stages states directly on CPU instead
of first allocating the entire optimizer on CUDA. Changing modes within a
resumed run is rejected; use a separate run for the comparison.

For an L40S memory probe use `batch_size: 16`, `micro_batch_size: 1`, `gpus: 1`,
`gradient_checkpointing: true`, and `state_offload: cpu`. This reduces optimizer
state residency; full-model fit must still be measured on the target GPU.
Existing submitted YAML snapshots remain unchanged. Selecting this option does
not cancel or replace an existing job.
