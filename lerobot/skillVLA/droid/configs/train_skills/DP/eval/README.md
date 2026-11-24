# DROID DP SBD evaluation

This evaluates segmentation on recorded DROID demonstrations. It is not a
closed-loop real-robot success evaluation. A completed FSQ/SBD skillset is not
required: use the completed joint-velocity dataset view and a saved DP checkpoint.

```bash
cd /data1/dohyeon/Skill_Boundary_Detector/lerobot/skillVLA/droid/configs/train_skills/DP/eval
./submit_eval.sh --dry-run
./submit_eval.sh --75000          # existing allocation; no extra GPU/Slurm step
./submit_eval.sh --report-only    # CPU regeneration from valid probe/PCA caches
```

`--config FILE` selects another recipe; `--local` executes in the current GPU
allocation. Missing dataset/checkpoint inputs fail before submission. The
launcher freezes local/global/server YAMLs before validation and execution.
Set `dp_checkpoint` to a saved step, such as `10000`, or `last`; its default is
`100000`. History and maximum prediction length are read from that checkpoint.
No evaluation is automatically chained after training.

The recipe follows LIBERO standard, explicitly retaining power **2** even if a
live LIBERO experiment YAML is changed to power 1:

| Setting | Value |
| --- | --- |
| Probes | 96 IID Gaussian, current GT action center, no mean sample, seed 42 |
| Denoising | DDIM, schedule index 7, one call, `pred_original_sample` |
| Descriptor | Full predicted horizon (18 currently), temporal mean |
| PCA | Variance 0.95, stride 3, standardized actions including gripper |
| GMM | Fixed K=5, diagonal, weighted, n_init=10, max_iter=300, no reference center |
| Boundary score | Covariance-gated cosine, power 2 |
| Threshold | One dataset-global mean over the selected evaluation episodes |
| Replan / SG / NMS / min skill | 8 frames / 7 anchors, order 4 / 23 frames / 10 frames |
| Masks | Initial checkpoint history; fixed final 24 frames |

The scoring and smoothing follow the maintained LIBERO evaluator. Replanning
uses 8 frames (~0.53 seconds), rounding half of chunk15 upward. NMS is
`ceil(1.5 * VLA chunk_size)`: DROID's chunk15 therefore uses 23 frames.
The controls remain **frame-based**;
DROID runs at 15Hz, so the elapsed times differ from LIBERO's 20Hz. The SG window
counts replan anchors. This uses a fixed terminal mask, not ABC's semantic
gripper-release mask. Probes stay in the stored 8D joint-velocity + absolute
gripper space. Joint positions are the proprioception; no EEF grounding,
delta-joint conversion or additional relative-action transform is applied.

Default selection is the first 10 dataset language-prompt IDs, up to 5 episodes
per ID, in ascending episode order. Some DROID prompts have fewer examples, so
the actual count is printed before submission. Edit `task_ids` to an explicit
ordered ID list, or use `task_ids: all` with `n_tasks: all`. These prompt IDs do
not identify fixed scenes or benchmark tasks. `pca_scope: dataset` fits a CPU
basis on the full converted dataset; `selected` uses only selected episodes.
The score threshold always uses the selected episodes. Each model/selection
has its own provenance-checked cache. Change `output_name` for a different
checkpoint, camera or selection.

Two episode workers share one GPU, with eight CPU threads total. `eval_workers`
and `eval_cpus_per_task` are editable; episodes are split by length without
overlap. Workers save raw and GMM caches only. Global threshold calibration and
report generation happen once after all workers finish. Resume reuses complete,
matching caches. Changing CPU-only scoring settings shares those probe caches.
Global training-only `idle_filter` switches do not remove frames from evaluation.

```text
partial_build/<model>_<output_name>/
  source/skillset_manifest.json
  pca/
  raw/
  clusters/
  workers/
outputs/<model>_<output_name>_<recipe>/
  index.html                  # synchronized video / skill bar / cosine graph
  src/
    summary.json
    threshold_statistics.json
    evaluation_complete.json
    boundary_results.json
    video_report_manifest.json
logs/
```

The report defaults to exterior_1 (`observation.images.image`); choose
`observation.images.wrist_image` or `observation.images.image2` in
`report_image_key` for wrist or exterior_2. Source MP4s are referenced without
re-encoding. Reports and partial evaluation caches stay in this `eval` folder;
the original converted data, training checkpoints and production skillsets are
never rewritten. The shared LIBERO/ABC evaluator provides cache validation,
interactive reports and CPU-only regeneration; DROID only adds its data contract.
