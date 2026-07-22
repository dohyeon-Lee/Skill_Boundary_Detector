# Original FSQ Closed-Loop Eval

`FSQ_eval` evaluates an original `outputs_filtered/FSQ/<run>/FSQ.pt` directly in
LIBERO. It does not consume a SkillVLA training parquet.

## Comparison Axis

All panels receive the same episode-exact reset state, GT FSQ skill sequence,
current state, and terminator inputs. The only panel difference is:

```text
broadcast_condition = broadcast_scale * skill_proj(z_q)
```

The scaled vector is added before expert self-attention in every expert layer
and at every flow-denoising step. The default YAML produces 100%, 60%, and 30%
panels. Scale 1.0 is numerically the historical behavior. Non-broadcast
checkpoints may only use 1.0 and fail fast for other values.

## Data

`submit_eval.sh` encodes the exact source skill NPZs once per checkpoint and
writes:

```text
outputs_filtered/FSQ/<run>/skill_latents_eval_<checkpoint>.npz
```

All scale panels share that cache. Exact reset states live at:

```text
dataset_filtered/FSQ_dataset/<source>/eval_init_states.npz
```

The submit script copies the legacy source-only cache when available or builds
it from the raw LeRobot dataset and original LIBERO HDF5s. Jobs do not read the
SkillVLA dataset. The raw LeRobot dataset is still required for episode
metadata and skill-HTML examples.

Historical `fsq_meta.json` files can lack `skillset_min_skills`. Set the exact
value per model in `fsq_eval_config.yaml`; the supplied broadcast runs use 1.
An explicit `skills_dir` may also be supplied per model.

## Run

```bash
cd lerobot/examples/libero/configs/train_skills/FSQ_eval
./submit_eval.sh
```

Each model can override `broadcast_scales` and `advance_mode`. `terminator`
uses the checkpoint terminator; `gt` advances by demonstration skill duration
while still recording terminator curves. Multiple models and scales become one
ordered panel list with per-panel success charts, a grouped comparison PNG,
skill HTML, and side-by-side videos.

## Validation

Validated on 2026-07-22 with the real FSQ333 broadcast checkpoint:

```text
checkpoint strict-load: broadcast, state=8, action=7, chunk=10, codebook=27
source cache:           9,620 skills
episode join:           3,921 episodes, 9,620 skills, 89 source scenes
resolved panels:        100%, 60%, 30%
```

The current source init-state cache lacks LIBERO task 51, so that task is
dropped with a warning if requested. A full GPU LIBERO rollout and Slurm fanout
remain to be launched from a compute allocation.
