# LIBERO DP SBD evaluation

Edit `eval_config.yaml`, then run `./submit_eval.sh`. This uses the same partial
build, CPU cache reuse, video report, and comparison player as ABC's `DP/eval`.
A completed `build_data`/FSQ skillset is not required.

```bash
./submit_eval.sh --dry-run       # Check checkpoint, episodes and cache readiness.
./submit_eval.sh                 # Cached descriptors: CPU. Missing descriptors: GPU job.
./submit_eval.sh --73428         # GPU work runs inside that existing srun allocation.
./submit_eval.sh --local         # Run in the current GPU allocation.
./submit_eval.sh --report-only   # Refit GMM/rebuild HTML from existing descriptors on CPU.
```

`--config FILE` selects another YAML. `--JOB_ID` never creates an extra GPU step.
When descriptors are cached, the command runs locally on CPU even with `--JOB_ID`.
Video clipping uses system ffmpeg when available, otherwise the executable
bundled with the installed `imageio-ffmpeg` package. Cached CPU reports therefore
also work on login nodes without an ffmpeg command on PATH.
The launcher freezes the YAML and inherited global settings before submission.
It freezes them before checkpoint/cache validation as well; validation, the
recorded launch settings, and execution all read the same snapshot even if the
live YAML is edited while the command is starting.
The existing RunPod input-transfer flow remains available with
`--prepare-runpod` / `--download-only`; RunPod automatically downloads verified
inputs before re-entering the same launcher.

| Public setting | Changing it requires |
| --- | --- |
| `sbd_prediction_horizon` (N or `all`) | New DP descriptors |
| `replan_interval` | New DP descriptors |
| `gaussian_center` (`current_gt`, `previous_action`, `pure_noise`) | New DP probes unless cached |
| `pca_scope` (`dataset` or `selected`) | CPU PCA fit/reuse; descriptors must match that basis |
| `gmm_components` | CPU GMM refit |
| `nms_frames` | CPU boundaries/report |
| `smooth_window` (positive odd integer) | CPU smoothing/report |
| `polyorder` | CPU smoothing/report |
| `score_power` | CPU scoring/report |
| `peak_curvature_scale` | CPU peak filtering/report; 0 disables, 1 uses mean peak curvature |

`dp_run_name` is the model folder under global `outputs_root/DP`.
`dp_checkpoint` selects a saved step (`10000` resolves to `010000`; `last` also
works). Observation history and maximum prediction length come from that
checkpoint. The shipped config uses the available obs20/future24 Transformer
10k checkpoint. Stateful GRU checkpoints consume one frame and hidden at a time,
including frames between replanning anchors; episodes reset hidden.
Result names automatically start with the checkpoint's encoder and U-Net size:
`trans_small_`, `trans_base_`, `GRUiid_small_`, `GRUepi_base_`, etc. `transformer_cls`
uses `transCLS`; flat conditioning uses `flat`. These are read from saved model
config, independently of its folder label. Keep only the experiment base name
in `output_name`.

The shared runtime defaults to two episode workers and eight CPU threads total
(four per worker). CPU thread limits also apply to single-worker and cached
report runs. Changing worker/thread counts reuses completed DP/PCA caches.

`task_ids` are LIBERO **suite** IDs by default. Scene-exact selection uses
`skillvla_dataset/<dataset>/eval_init_states.npz`, preserving the existing
LIBERO selection contract. `n_episodes` is per task; missing episodes fail before
submission. For dataset-internal IDs, set `advanced: {task_id_space: dataset}`.

`gaussian_center` selects `current_gt` (the shipped LIBERO setting, centered on
the GT action sequence as in the original PT/FT builds) or `previous_action`
(the previous action held across the predicted future). Both use 96 IID probes,
DDIM step 7, one denoising step, and temporal-mean PCA descriptors. Changing
the center requires new GPU probes if that center has no cache; PCA is shared.
Reports append `_current_gt` or `_previous_action` after the numeric suffix,
and raw/GMM cache paths include the center so the two conditions stay separate.
Older frozen YAMLs without this field retain the previous-action default.
`gaussian_center: pure_noise` instead starts 96 fixed IID N(0,I) action
sequences and runs the complete ten-step reverse trajectory, without a GT
action center. SBD uses the scheduler's x0-hat at schedule index 7 (the eighth
UNet call). The final ten-step actions are saved in the same raw cache.
Each episode's video timeline shows two synchronized graphs: step 7 for SBD,
and final 10 for comparison, with the same step-7 segmentation lines.
Both use the same PCA and descriptor horizon; final-10 GMM/mean calibration
is computed separately on CPU and never changes the SBD boundaries.
The report suffix is `_pure_noise`; raw caches include
`pure_noise_reverse10_x0step7_final10_v1`, and cached re-renders need no GPU.
This mode performs ten UNet calls per anchor rather than one.
LIBERO retains its covariance-gated cosine metric, 10-frame terminal mask,
power 2, replan 5, NMS 15, and SG 7/4 defaults. PCA includes the gripper dimension
as in the original LIBERO reference condition. Detailed defaults are documented
in `src/defaults.py`; optional `advanced` entries override hidden settings.

The initial mask follows the checkpoint's observation history automatically
(`obs20` masks frames 0–19). Evaluation and full builds exclude these anchors
before smoothing, episode/global threshold means, and boundary selection.
Video graphs shade the initial mask blue; existing DP caches can be reused.

The shipped `pca_scope: dataset` uses the **complete dataset** for PCA while
evaluation and the pooled score threshold still use only the selected episodes.
If the original full-build PCA for this checkpoint is available, its hash,
normalizer statistics, action dimensions, horizon, and PCA recipe are checked
before copying its exact numerical basis into the private evaluation cache.
Otherwise only PCA is fitted on CPU over the complete dataset; a full SBD build
is unnecessary. `pca_scope: selected` fits PCA only on the evaluation episodes.
Full-data PCA reports/caches add `_pca_dataset` to the base name to keep the two
scopes separate. The pooled threshold can differ from the old full-skillset
calibration over 3,921 episodes. Checkpoint
normalization and episode-start XYZ grounding remain those saved by training.

When SBD uses fewer steps than the DP predicts, its private partial-build
directory automatically adds `_sbdh<N>` (for example `_sbdh1`). This separates
horizon-dependent PCA provenance without requiring another `output_name`.
Full-horizon caches retain their existing path. Compatible older caches for
the requested horizon are reused in place. GMM and smoothing changes still
share the same partial build; report names keep their existing numeric suffix.

```text
partial_build/<model>_<size>_<output_name>/
  source/skillset_manifest.json   # Immutable selected episodes and checkpoint identity
  pca/                           # PCA bases
  raw/                           # DP probe descriptors
  clusters/                      # CPU GMM caches
  workers/                       # Worker state and logs
outputs/<model>_<size>_<output_name>_<prediction>_<replan>_<K>_<NMS>_<window>_<polyorder>_<power>_<curvature_scale>_<gaussian_center>/
  index.html                     # Video, skill timeline, synchronized graph
  src/                           # Other HTML, figures, JSON and video assets
outputs/together/
  together_config.yaml
  make_together.sh
logs/
```

Changing CPU controls creates a different output suffix while sharing the
partial build. Change `output_name` when changing checkpoint or episode selection.
An existing unprefixed build is reused in place only when its checkpoint and
episode manifest match exactly. Existing report folders keep their names.
You can delete generated report folders and regenerate them while keeping
`partial_build`. Evaluation writes neither training FSQ data nor `analysis`.

For two or more side-by-side results, edit
`outputs/together/together_config.yaml` and run its `./make_together.sh`.
Playback and graph scrolling are synchronized. Up to three columns fill the
screen; four or more use horizontal scrolling. It reuses existing reports on CPU.
Vertical sync follows the episode and relative card position, including when
comparing a single graph against the pure-noise step-7/final-10 double graph.

The shared SBD engine now lives in `src/evaluate.py` and `src/boundary_config.py`;
ABC references this location too. The cache/report orchestration and player are
shared with ABC to keep both workflows consistent. Historical explicit recipes
are archived under `legacy/`. Other `skill_eval` tools (FSQ/eef/galleries) remain
under their original components.
