# ABC DP: partial SBD evaluation

Edit **`eval_config.yaml`**, then run **`./submit_eval.sh`**. Matching DP/PCA
caches run GMM clustering and HTML generation directly on CPU, including when
the cluster count or other GMM settings change. Missing or changed DP probe
caches submit one GPU job; no completed `train_skills/build_data` skillset is required.
It reads the ABC native dataset and a saved state-only DP checkpoint directly.

The main YAML fields are the normal SBD controls:

| Field | Meaning | Work needed after a change |
| --- | --- | --- |
| `sbd_prediction_horizon` | First N predicted actions, or `all` for the checkpoint's full prediction | DP descriptors + GMM |
| `gmm_components` | GMM component count | CPU GMM + report |
| `replan_interval` | Frames between SBD evaluations | DP descriptors + GMM |
| `gaussian_center` | `previous_action`, `current_gt`, or `pure_noise` | New probes unless cached |
| `pca_scope` | `dataset` = saved DP training episodes; `selected` = evaluation episodes | CPU PCA fit, then descriptors in that basis |
| `nms_frames` | Minimum spacing between boundary candidates, in original frames | CPU boundaries + report |
| `smooth_window` | Savitzky–Golay window, in replan samples | CPU boundaries + report |
| `polyorder` | Savitzky–Golay polynomial order | CPU boundaries + report |
| `score_power` | Positive finite score exponent: 1 for ABC standard, 2 for LIBERO standard | CPU boundaries + report |
| `peak_curvature_scale` | Minimum peak curvature = pooled mean × this non-negative scale; 0 disables | CPU boundaries + report |

Select a model with `dp_run_name` (the folder name inside DP) and a training
step with `dp_checkpoint`. The checkpoint path is assembled as
`<global outputs_root>/DP/<dp_run_name>/checkpoints/<dp_checkpoint>/pretrained_model`.
`outputs_root` is inherited from ABC's global YAML, including server overlays;
it is not repeated in this eval YAML. Numeric steps are padded to six digits
(`10000` becomes `010000`); `last` also works. Old `policy_checkpoint` paths
remain readable for frozen configurations.

DP observation history and maximum prediction length come from that checkpoint.
The launcher prints these lengths next to the SBD length. Asking for more than
the model predicts fails before any GPU submission. `smooth_window` accepts positive odd integers only (default 7); 1 disables
smoothing. Even lengths are rejected before submission, without rounding.
Build and eval share the same SG implementation and default polynomial order 4.
Probe/cluster/experiment names are connected automatically, so changing GMM K
does not require editing another reference. K=2 and K=5 use separate GMM caches.

Detailed defaults live in `src/defaults.py`; they preserve the existing probe,
PCA, threshold and report recipe. Optional `advanced: {eval_workers: 2}`
overrides hidden settings when needed. `gaussian_center: previous_action`
holds the last observed action across every predicted future slot, for both
full-length and first-step SBD. `current_gt` centers on the demonstration
future action sequence. `pure_noise` starts 96 fixed IID N(0,I) sequences,
runs one complete ten-step reverse trajectory, and uses schedule index 7
(the eighth UNet call) x0-hat for SBD. The HTML shows synchronized step-7 and
final-10 graphs with the same step-7 boundaries; final-10 is comparison only.
Both results are cached together, so later GMM/smoothing changes are CPU-only.
Long-model previous-action probe
names include `_hold_future` to separate them from older probes that left
later slots GT-centered. These are still
the existing temporal-mean descriptors, not a new trajectory representation.
Resolved recipes are saved in `outputs/<name>/src/evaluation_settings.json`.

The default is two episode workers sharing an eight-thread CPU budget (four
threads per worker). Single-worker runs and CPU report generation also apply
the CPU budget to PyTorch and loaded BLAS pools. Worker/thread changes preserve
completed DP/PCA caches and do not change the report name.
Old frozen YAMLs with explicit variant lists remain supported.

`output_name` is the base name. The actual checkpoint adds an architecture/size
prefix: `trans_small_`, `trans_base_`, `GRUiid_small_`, `GRUepi_base_`, etc.
`transformer_cls` uses `transCLS`, and flat conditioning uses `flat`.
The prefix is read from checkpoint config, not guessed from its folder name.
Report folders automatically append the eight
public values in this order: **prediction, replan, GMM K, NMS, window, polyorder, power, curvature scale**.
For example, `output_name: experiment` produces
`outputs/trans_small_experiment_pca_dataset_1_8_2_45_7_4_1_0_previous_action/index.html`
for a small Transformer with training-data PCA. `all` uses the checkpoint's resolved
numeric prediction length in the suffix. An identical combination updates its
own report; changing any of these values creates a different report folder.
New private caches stay under `partial_build/<model>_<size>_<output_name>` and are shared across
these reports, so CPU-only changes do not cause extra GPU inference. If SBD
uses fewer steps than the DP predicts, the private path adds `_sbdh<N>` to
separate horizon-dependent PCA/probe caches automatically. Compatible old
caches are reused in place. Worker
state uses a separate subfolder per suffix.
An older unprefixed partial build is reused in place only when its full source
manifest matches the selected checkpoint and episodes, avoiding repeat GPU work.
PCA scope is included before the numeric suffix (`_pca_dataset`,
`_pca_selected`, or the legacy `_pca_source_tasks`); probe centering is appended
after it (`_previous_action`, `_current_gt`, `_pure_noise`). For example:
`trans_small_experiment_pca_dataset_1_8_2_45_7_4_1_0_current_gt`.
Compatible private caches from before these name tags are reused in place;
different centers share the PCA but use distinct raw/GMM cache paths.
Existing report folders are not renamed. Checkpoint/episode changes within the
same architecture still require a distinct base `output_name`.

```bash
./submit_eval.sh --dry-run       # Validate selection and print paths; submit nothing.
./submit_eval.sh                 # Auto: cached CPU report, otherwise submit a GPU job.
./submit_eval.sh --73428         # Missing GPU caches: share existing allocation 73428.
./submit_eval.sh --local         # Run inside an existing GPU allocation.
./submit_eval.sh --report-only   # CPU: recut cached scores and regenerate the report.
```

`--JOB_ID` sends the frozen configuration to a worker in the original srun step.
It inherits that step's assigned GPU and does not create another Slurm job or
GPU step. Concurrent runs have separate `.run<id>` logs. New managed allocations
start this worker automatically; an older idle managed shell is initialized once.

Use `--config /path/to/file.yaml` to select another configuration. The default
selects `grab_clutter`, all 24 language tasks with five episodes each (120 total),
the obs60/future1 DP100k checkpoint, and NMS 45. `dp_run_name` and `dp_checkpoint`
select the model. `task_ids` selects language prompts
(59–82 for `grab_clutter`); `n_episodes` is the number **per prompt**.
The launcher freezes local/global YAML before checkpoint/cache validation.
Validation, launch records and execution use that same snapshot, including
local runs, so editing the live YAML during validation cannot switch models.

All ABC/LIBERO `submit*.sh` launchers accept `--JOB_ID` before or after their
usual options. It selects an existing Slurm allocation instead of requesting
another GPU and opens a separate tmux session. Existing ownership/resource
checks still apply. Array/dependency workflows that cannot reuse an allocation
stop with an error; they do not fall back to submitting new jobs. Cached CPU
reports, `--dry-run`, and explicit local modes retain their normal behavior.

All generated files stay under this directory:

```text
partial_build/<model>_<size>_<output_name>/
  source/skillset_manifest.json   # Selected episode IDs, checkpoint hashes, action contract
  pca/                           # PCA basis (see pca_scope)
  raw/                           # DP probe descriptors
  clusters/                      # GMM scores
  summary.json                   # Reusable report inputs, retained if outputs is deleted
  build_settings.json
  workers/                       # Resumable worker state and per-worker logs
outputs/<model>_<size>_<output_name>_<prediction>_<replan>_<K>_<NMS>_<window>_<polyorder>_<power>_<curvature_scale>/
  index.html                     # Video player, current skill, timeline and score graph
  src/                           # All other generated report files
    index.html                   # Statistical comparison and task pages
    evaluation_settings.json
    summary.json
    threshold_statistics.json
    evaluation_complete.json
    boundary_results.json
    video_references.json
    task_*/                      # Figures and frame assets
logs/                            # Scheduler stdout/stderr
```

`partial_build` contains evaluation intermediates, not a training FSQ skillset.
This command does not write to `dataset_ABC/FSQ_dataset` or `analysis`.
Source videos remain in the original dataset; the report generates only its own
figures/frame assets.
Video clips use system ffmpeg or the installed `imageio-ffmpeg` bundled binary,
so cached CPU reports can also run on login nodes without ffmpeg on PATH.

The checkpoint's saved normalization is retained. `pca_scope: dataset` reads
the checkpoint's `train_config.json` and applies its saved task filter, explicit
episode IDs, and random episode-selection manifest. It means the DP training
range, not every task in the dataset directory. For example, a five-task
1,000-episode-per-task DP uses those exact 5,000 episodes for PCA even when
evaluation selects only `grab_clutter`. Missing training provenance/selection
files fail before submission instead of silently fitting a different range.
The legacy `pca_scope: source_tasks` uses all episodes in the eval YAML's
`source_tasks`; `selected` uses only the requested evaluation episodes.
PCA fitting is CPU-only and does not require a complete SBD build. The shipped
`mode: std` includes all action dimensions, matching the saved Yonsei basis.
`pca_stride: 3` samples PCA anchors every three frames, while probe count is 96
and evaluation replan interval is 8. `threshold_scope: dataset` pools only the
evaluation selection, so its threshold can differ from a full-dataset build.
Terminal release/home-return masking uses the same detector as the normal ABC
builder. A standalone report has no previous training-segmentation baseline.

Both subset evaluation and full builds exclude `[0, checkpoint.n_obs_steps)`
before SG smoothing, threshold averaging, and peak detection. An obs60 policy
therefore masks frames 0–59 independently of the replan/NMS settings. Video
graphs shade this initial history mask blue. Cached DP descriptors remain
reusable; rerunning evaluation updates thresholds, cuts, and HTML on CPU.

Peak curvature is the negative local second derivative of the powered SG
score, measured using original frame timestamps (score/frame²). Pool all
eligible local peaks across the calibration episodes after initial/terminal
masking and the start guard, before height thresholding and NMS. A positive
`peak_curvature_scale` rejects peaks below their pooled arithmetic mean times
the scale. This is not a fixed rejection quota: 0 disables, 0.1 uses 10% of
the mean, and 1 uses the mean itself. The score-mean threshold is unchanged.
Blue crosses identify curvature rejections; red points are final boundaries.
The shared cutoff/counts are in `src/threshold_statistics.json`, and each peak's
frame/curvature and rejected frames are in `src/boundary_results.json`.
Semantic ABC terminal-release cuts remain separate boundary candidates.

The same YAML resumes cached work. Change `output_name` when changing the
checkpoint or episode selection. To compare NMS values on the same cached
scores, edit `nms_frames` and run the script normally (or use
`--report-only`); this writes the separately named report without DP inference.
To change GMM K=5 to K=2, set `gmm_components: 2` and run the same script:
valid DP descriptors are reused, and only GMM/thresholds/boundaries/reports are
recomputed on CPU. This also works with `--report-only`. CPU cache
runs hide CUDA devices and do not request or attach to a GPU allocation.
Deleting `outputs` is supported: keep `partial_build` to regenerate the report.

The main `index.html` is always the video gallery. Videos use the original ABC
source files with synchronized left-wrist, external, and right-wrist views;
report generation does not rerun DP or reencode source videos.

For a combined video comparison, edit `outputs/together/together_config.yaml`:
list two or more result folder names under `eval_folders`, in the desired column
order, and choose `output_name`. Run `./make_together.sh` from that same folder.
The result is `outputs/together/<output_name>/index.html`; auxiliary metadata
goes under its `src/`. Matching task/episode IDs are aligned across all inputs,
and only their intersection is shown. The Yonsei comparison layout opens the existing reports side by side in iframes,
with report selection, synchronized scrolling and playback. One to three selected
reports share the full viewport width equally; four or more keep three panels
visible and scroll horizontally. Panel widths follow window resizing automatically.
The scroll sync toggle also links graph horizontal scrolling for the same episode,
using relative scroll positions so different panel widths stay aligned.
Vertical scrolling follows the episode ID and the relative position inside its
card, so one-graph and step-7/final-10 two-graph reports remain aligned without
accumulating a height offset. Programmatic scroll updates do not echo back.
The small wrapper references existing reports without copying plots or videos, without GPU work,
DP inference or video encoding. Inputs must be saved video reports from the
same dataset with matching episode frame counts/FPS.
