# DROID FSQ evaluation

`eval_config.yaml` is the only public evaluation YAML. `models` names runs under
`outputs_DROID/FSQ`; `checkpoint` accepts `last`, one epoch, or a list of epochs.
Both joint-velocity and EEF delta-pose FSQ, including 333 and 3332 codebooks, use
the same launcher. Training contracts and checkpoints determine dimensions,
normalization, source dataset, skill boundaries and quantizer levels. Editing
the current FSQ training YAML never changes a saved model's evaluation.

```bash
cd lerobot/skillVLA/droid/configs/train_skills/FSQ/eval
./submit_eval.sh --dry-run          # read-only prerequisite/plan check
./submit_eval.sh --74999            # selected existing allocation + memory guard
./submit_eval.sh                    # server-profile scheduler
./submit_eval.sh --yonsei-plan      # preview portable dependencies
./submit_eval.sh --yonsei           # standard Yonsei launcher
```

Use `--local` only inside an allocation (or a private YAML with `device: cpu`
for a small CPU check). Submission pins `last` to numeric epochs in an immutable
snapshot; queued workers do not switch to a checkpoint saved later.

The global dataset and group apply. Evaluation uses the intersection with the
model's saved training episode selection and never renumbers episodes or
applies training-only idle filtering. All selected source skills are encoded
for full-corpus code usage; `task_ids`, `max_tasks` and `episodes_per_task` limit
the video gallery. `episodes_per_task` is an upper bound: rare tasks with fewer
demos use every available selected episode rather than being excluded.
`max_tasks: 0` includes all eligible tasks. Explicit task IDs must have at least
one selected skill episode. All compared models must select identical episodes.
Camera aliases are `exterior_1`,
`wrist` and `exterior_2`; they map to the three preserved converted streams.

One GPU job sequentially encodes the requested models in batches, optionally
performs the shared deterministic boundary-jitter sweep, then creates the same
interactive codebook, skill-position filters, start/end images and looping skill
video gallery as the other environments. It uses recorded DROID videos; no
simulator, policy rollout or video re-encoding is involved. EEF targets are read
with the training adapter, including first-state anchoring, SO(3) increments and
absolute gripper. Training datasets/checkpoints are never modified.

Results stay under `eval/outputs/<output_name>/`:

```text
<run>/checkpoints/epochNNNN/
  artifacts/skill_latents.npz
  artifacts/skill_boundary_jitter_sweep.npz   # optional
  evaluation_contract.json
  metrics/manifest.json
  summary.json
  evaluation_complete.json
  images/  videos/  index.html
<run>/index.html                            # checkpoint comparison
compare/index.html                          # model comparison
```

Video assets use hard links to the converted MP4s on the same filesystem, or
byte-for-byte copies across filesystems; no frames are re-encoded. This also
keeps collected Yonsei reports playable after transfer. `resume: true` reuses sealed encodings,
sweeps and rendered frames. Changed checkpoint content, source provenance,
episode/group selection or camera refuses the old output instead of mixing it.
Missing/incomplete source skillsets or checkpoints fail before submitting a job.
No evaluation is submitted just by editing the YAML.
