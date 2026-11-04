# pi05: native IID PT vs block-cyclic PT

Ported the task-group curriculum from `skillVLA_real`'s `train_pi05_cycle` onto the
**current** `train_pi05/PT` pipeline. Only IID and pure cyclic are supported;
there is no Reptile, old standalone update loop, extra objective or probe forward.

Both local YAMLs inherit `../../train_pi05/PT/pi05_PT_config.yaml`. Edit that
source to change common training/resource settings. Currently: LIBERO-90 full,
episode-start XYZ grounding ON, batch 16, workers 8, one GPU, CPU 16, RAM 128G,
100k steps, LR 2.5e-5 with 1k warmup then constant, checkpoints every 10k,
gradient checkpointing OFF and vision/language unfrozen. QoS/partitions/exclusions
come from the same global/server profile as ordinary pi05 PT; no local override.

```bash
# From this directory; only explicit non-dry-run commands submit a job.
TRAIN_CYCLIC_DRY_RUN=1 bash submit_train.sh iid
TRAIN_CYCLIC_DRY_RUN=1 bash submit_train.sh cyclic
bash submit_train.sh iid
bash submit_train.sh cyclic
```

- **IID** launches native `lerobot-train`, using the original PT batch script and
  every original CLI argument. Model, grounding statistics, optimizer/scheduler,
  shuffle/drop-last policy, seed defaults, save/resume and W&B are unchanged.
- **Cyclic** uses that same trainer with a single opt-in DataLoader factory.
  Tasks remain atomic and are greedily assigned to approximately frame-balanced
  groups. Each group supplies `phase_steps` full batches before switching; group
  order reshuffles each cycle. A group's shuffled frame cursor continues across
  revisits and reshuffles only when exhausted. No mixed-group batch. Fixed group
  budgets make frame exposure approximately, not exactly, equal to IID when
  atomic task sizes prevent perfectly balanced groups.
- Both modes are single-process for this comparison. Cyclic currently requires
  the full offline dataset; streaming/episode subsets/other samplers are rejected.
- Both retain native normalization/action unnormalization and save the usual
  pi05 checkpoints. Existing pi05 FT/eval can read them by explicit checkpoint
  path/output-root settings; their default checkpoint lookup is not changed.

Outputs are isolated under `{outputs_root}/pi05_cyclic_PT/` with `_iid` versus
`_cyclic_g8p250_s0` suffixes. W&B project stays inherited. Local launcher logs are
in `logs/`. Each submission freezes the local YAML **and its inherited PT YAML,
global config and server profiles** in `.config_snapshots/`.

Cyclic writes `groups.json` and `curriculum.json`. On resume, it reconstructs the
same index schedule up to the saved optimizer step (prefetch cannot advance the
saved cursor); changed grouping/batch/schedule is rejected. This guarantees sample
order, not bitwise equivalence of stochastic video/augmentation workers on resume.
The launcher refuses an existing output without a checkpoint; it never deletes
an incomplete run automatically. No existing pi05 outputs are reused or removed.
