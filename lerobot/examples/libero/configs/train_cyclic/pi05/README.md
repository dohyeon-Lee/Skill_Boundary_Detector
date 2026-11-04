# pi05: native IID PT vs block-cyclic PT

Ported the task-group curriculum from `skillVLA_real`'s `train_pi05_cycle` onto the
**current** `train_pi05/PT` pipeline. Only IID and pure cyclic are supported;
there is no Reptile, old standalone update loop or extra training objective.
IID and cyclic share read-only diagnostic probes in the native trainer.

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

- **IID** uses a thin entrypoint into native `lerobot-train`, with the original PT
  batch script and CLI arguments. Model, grounding statistics, optimizer/scheduler,
  global shuffle/drop-last policy, seed defaults and model checkpoints are unchanged.
  Only the isolated output, W&B project and shared diagnostic observer differ.
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
`_cyclic_g10p200_s0` suffixes. Both local modes use W&B project `cyclic_PT`; ordinary
`train_pi05` keeps its own project. Local launcher logs are
in `logs/`. Each submission freezes the local YAML **and its inherited PT YAML,
global config and server profiles** in `.config_snapshots/`.

Cyclic writes `groups.json` and `curriculum.json`. On resume, it reconstructs the
same index schedule up to the saved optimizer step (prefetch cannot advance the
saved cursor); changed grouping/batch/schedule is rejected. This guarantees sample
order, not bitwise equivalence of stochastic video/augmentation workers on resume.
The launcher refuses an existing output without a checkpoint; it never deletes
an incomplete run automatically. No existing pi05 outputs are reused or removed.

## Shared IID/cyclic logging

Both local YAMLs expose the same `logging` block. Default: `wandb_project: cyclic_PT`,
`probe_batches_per_group: 2`, `probe_seed: 12345`, `probe_grad_group: 0` (`-1` disables
the gradient diagnostic). Keep `cycle` and `logging` settings equal to overlay runs.
`cycle` in IID controls diagnostics only, never its global shuffled training loader.

- Native `train/*` metrics retain their 100-step cadence.
- `cycle/index`, `cycle/position`, `cycle/active_group`, `cycle/phase_step` and
  `epoch/g0..g9`, `epoch/global` are logged at native log steps and phase boundaries.
  Epochs count actual committed batch tasks, never prefetched sampler state.
  IID has active group `-1`; its cycle position is a common comparison clock.
- Every 200 steps (and initially/finally): `probe_loss/g0..g9`,
  `probe_forget/g0..g9`, plus legacy `probe/g*_loss` / `probe/g*_forget` aliases.
  Forgetting is a **ratio**, not percent. Cyclic references each group's own last
  phase-end loss; IID references the previous boundary (all groups are mixed).
  References update **after** logging, as in the original cyclic experiment.
- `grad_cos/g0` compares consecutive fixed-probe gradients. `grad_cos/interval_steps`
  records the actual comparison interval; legacy `probe/grad_cos_g0` is also logged.
  Default costs two extra probe gradient forwards/backwards per boundary and a
  CPU bf16 gradient snapshot (several GB for pi05), plus 20 loss-only probe batches.
  These are diagnostics, not additional optimizer steps or extra training losses.

Both modes select identical fixed raw frames and flow noise for a given group/seed.
Probe decoding/inference preserve Python, NumPy and torch RNG, module training flags
and parameter `.grad`; there is no optimizer/scheduler step during a probe.
Raw batches/config are cached once as `cyclic_probe_batches.pt` and
`cyclic_probe_config.json`. Metrics also append locally to `cyclic_metrics.jsonl`.
Each native checkpoint includes `cyclic_diagnostics.json` (references/exposure counts).
Resume restores these and the original raw probes; the gradient snapshot is rebuilt
at restored weights, so no multi-GB gradient file is added to every checkpoint.
For a mid-phase checkpoint, the first resumed cosine interval is shorter; normal
10k checkpoints fall exactly on 200-step boundaries. Missing diagnostic resume state
or a changed probe configuration is rejected rather than silently rebaselined.
