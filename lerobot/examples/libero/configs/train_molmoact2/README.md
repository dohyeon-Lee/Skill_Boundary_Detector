# MolmoAct2: local LIBERO PT / FT

`PT/` adapts the public **allenai/MolmoAct2** checkpoint to our `libero_90_full_full`.
`FT/` adapts **our PT result** to `libero_10_full_1`. These are both downstream
fine-tuning stages in the paper's terminology, not re-runs of foundation pretraining.

The implementation adds settings, local dataset registration, and job submission.
The model, losses, normalization, trainer, optimizer, and checkpoint implementation
are executed from the official `third_party/molmoact2/experiments` checkout.
Our adapter applies pi05's existing episode-relative XYZ transform and selects
equivalent warmup-constant settings through the original scheduler's parameters.
Neither the main SkillVLA environment nor its LeRobot trainer is used.

## Source and environment

Official source: https://github.com/allenai/molmoact2

Integration target revision: `6070080a20321b4f498ab30f28e1d09ac465edb7`.
Original training instructions: `third_party/molmoact2/experiments/README.md`.

The **bundled `experiments/lerobot`** is the training dependency. The separate
root `molmoact2/lerobot` Git submodule is not installed in this environment.
Do not recursively initialize YAM: upstream's nested `YAM/molmoact2` gitlink has
no URL at this revision and is unrelated to LIBERO.

From the project root, environment setup on another machine is:

```bash
# Only for a new checkout; do not clone over the existing directory.
git clone --no-recurse-submodules https://github.com/allenai/molmoact2.git third_party/molmoact2
git -C third_party/molmoact2 checkout 6070080a20321b4f498ab30f28e1d09ac465edb7
uv venv --python 3.12 .venv_molmoact2
uv pip install --python .venv_molmoact2/bin/python 'cmake>=3.29,<4'
source .venv_molmoact2/bin/activate
hash -r
uv pip install --python .venv_molmoact2/bin/python --torch-backend cu128 \
  'cmake>=3.29,<4' 'torch==2.10.0' 'torchvision==0.25.0' \
  'torchcodec==0.10.0' 'transformers==5.3.0' \
  -e './third_party/molmoact2/experiments[train]' \
  -e './third_party/molmoact2/experiments/lerobot[async,libero]' debugpy
.venv_molmoact2/bin/hf download allenai/MolmoAct2 --local-dir models/molmoact2_base
```

The CMake cap avoids `egl-probe`'s CMake < 3.5 compatibility error with CMake 4.
These core versions are the environment used during integration, not an upstream
fully locked dependency release. Every training run records installed core
package versions and the actual source commit in `sbd_launch.json`.

## Preview and CPU validation

All commands below start in the project root unless otherwise stated.
Preview does not download models, submit a job, or load CUDA:

```bash
bash lerobot/examples/libero/configs/train_molmoact2/PT/submit_train.sh --dry-run
bash lerobot/examples/libero/configs/train_molmoact2/FT/submit_train.sh --dry-run
```

Check a real local sample, including both decoded camera frames and an H-step
action chunk. Model weights and GPUs are not required:

```bash
.venv_molmoact2/bin/python \
  lerobot/examples/libero/configs/train_molmoact2/PT/src/train.py \
  --config lerobot/examples/libero/configs/train_molmoact2/PT/molmoact2_PT_config.yaml \
  --check-data
```

Use the FT YAML for the corresponding FT dataset check. `--upstream-help`
instead checks original trainer imports and prints its CLI without CUDA startup.

## First GPU test

After the model download has finished:

```bash
cd lerobot/examples/libero/configs/train_molmoact2/PT
./submit_train.sh --smoke --dry-run  # inspect the test settings
./submit_train.sh --smoke           # submit 20 optimizer steps
```

The smoke mode uses global batch 1, microbatch 1, 1 GPU, 0 workers, log interval 1,
save interval 20, warmup 2, and no W&B. Its run name ends in `_smoke`.
The selected model, data, losses, tuning mode, and flow sample count are retained.
Passing this test requires checkpoint loading, forward/backward, optimizer steps,
and native checkpoint saving. No GPU training or VRAM requirement has been
established merely by passing the CPU checks.

## PT / FT training

Edit `PT/molmoact2_PT_config.yaml` and submit from `PT/` with `./submit_train.sh`.
Then set `warm_start.run_name` and `warm_start.checkpoint` in
`FT/molmoact2_FT_config.yaml` and submit from `FT/` the same way.
`--config /path/to/config.yaml` and `MOLMOACT2_TRAIN_CONFIG` select another YAML.

Defaults:

- Both cameras, the existing 8-dimensional state and 7-dimensional action;
  horizon 10, state discretization, q01/q99 normalization, existing zero-close
  gripper convention. The original model keeps its 32 action slots; the original
  dimension mask excludes the padded action dimensions.
- Full VLM/vision/connector/action-expert tuning, original `ft_embedding=lm_head`
  policy, no LoRA, no KV detach, activation checkpointing on.
- `action_format: both` uses discrete action CE plus continuous flow matching.
  The original `allenai/MolmoAct2-FAST-Tokenizer` is fetched on first use if needed.
  The original trainer also resolves the base tokenizer from `Qwen/Qwen3-4B`.
  `num_flow_timesteps: 8` is training noise/time replication, not inference steps.
- Unpacked, dynamic sequence lengths; global batch 16, per-GPU microbatch 1,
  hence 16 accumulation steps with one GPU. LRs are not scaled by GPU count.
- Original image augmentation; all four component learning rates default to
  `optimizer.base_lr: 2.5e-5`, matching pi05. Individual `llm_lr`, `vit_lr`,
  `connector_lr`, and `action_expert_lr` can override the common rate.
- `lr_mode: warmup_constant` matches pi05's linear warmup and constant LR after
  warmup. The original cosine implementation has `alpha_f=1` (floor equals peak)
  and zero warmup starting LR. Its native warmup is W+1 because Molmo calls the
  scheduler before optimizer step 1, whereas pi05's LambdaLR starts at step 0:
  the LR of optimizer step k is `base_lr * min(1, k/(W+1))` in both trainers.
  `cosine_decay` retains the original 0.1 final/peak ratio instead.
- PT matches pi05's 200k steps, 1000 warmup, 8 workers, log every 100, and save
  every 10k. FT matches its 100k steps, 500 warmup, 4 workers, and save every 5k.
  The per-device microbatch and accumulation remain Molmo-specific for memory.
- `training.tuning: action_expert` is an alternative original AE-only recipe and
  requires `action_format: continuous`; use a distinct `run.suffix` when comparing.

`PT/molmoact2_PT_config.yaml` exposes **`pt_proprio_grounding: true`**, as pi05
does. The adapter reuses `lerobot/src/lerobot/datasets/proprio_grounding.py` by
file path, without importing the main LeRobot package. It subtracts the episode's
first EEF XYZ before Molmo's normalization and state discretization. Other state
dimensions, actions, and images are unchanged. The same global-exact grounded
statistics and `episode_start_xyz_grounding_v2.npz` cache are used; parquet and
raw `stats.json` are not rewritten. The original stats collector receives grounded
state stats, and spawned workers apply the same transform when reopening data.

The grounding mode is saved inside each native checkpoint at
`model.robot_processor.metadata_by_tag.libero.proprio_grounding`, beside its
state normalization stats. **FT inherits this field from the selected PT step**;
there is no independent FT toggle. Older checkpoints without the field remain
ungrounded. Changing coordinates while resuming is rejected. Future rollout
evaluation must also subtract the rollout's initial XYZ when this metadata is on.

Placement, QoS, and excluded nodes follow `global_config.yaml` and `servers/`.
`submit_job.sh` handles Slurm or the local RunPod scheduler. Each submit freezes
the module/global/server YAMLs; FT `checkpoint: last` is pinned to a concrete PT
step at submission time. A CUDA guard runs with the MolmoAct2 interpreter.

## Outputs and resuming

```text
{outputs_root}/molmoact2_PT/bs16_libero_90_full_full_base/
{outputs_root}/molmoact2_FT/<PT-run>_step100000_libero_10_full_1_ft_bs16/
  sbd_launch.json                 # our paths, recipe, original argv, source commit, package versions
  config.yaml                     # original trainer config
  step10000/
    config.yaml
    model_and_optim/               # native distributed model + optimizer checkpoint
    train/rank0.pt                # native trainer/RNG state
  step100000-unsharded/            # final model-only checkpoint, additionally saved
```

Re-submit the **same YAML** to resume from the latest native `stepN` with optimizer
and training state. FT initialization from a PT checkpoint resets optimizer and
step counters through the original `initial_model_checkpoint` behavior.
No output folders are automatically removed. An incomplete latest checkpoint
fails validation instead of being overwritten. Checkpoints are retained by
default (`keep_checkpoints: -1`).

These checkpoints intentionally retain the upstream `step10000` layout; they
are not SkillVLA's `checkpoints/010000/pretrained_model` format. MolmoAct2 rollout
evaluation should use its upstream policy and normalization. Project-specific
PT/FT eval wrappers will be connected after the training smoke test, rather than
passing these weights through the pi05/SkillVLA evaluator.
