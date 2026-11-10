# SkillVLA data build

`./submit_build_all.sh` builds the selected dataset.

## GPU execution

Keep the existing chunk/retry settings and choose the GPU execution with one key:

```yaml
skillset_tasks_per_job: 5
skillset_array_throttle: 0
skillset_max_sweeps: 2
skillset_single_gpu_workers: 2
```

- `skillset_single_gpu_workers: 0`: existing one-GPU-per-array-task submission;
  `skillset_array_throttle` limits concurrent array tasks (`0` is unlimited).
- `1`: complete the build in one GPU allocation, one chunk at a time.
- `2` or more: complete the build in one GPU allocation, processing that many
  chunks concurrently on its GPU. Each chunk contains up to
  `skillset_tasks_per_job` tasks. Array throttling is unused in this mode.

The single-GPU runner waits for all curve chunks before reducing one global
threshold, then runs cached segmentation, the existing verification/retry sweeps,
FSQ encoding, and final assembly. Complete skillsets and verified encoded outputs
are reused. It uses the same resolved environment snapshot and existing stage
scripts as the array path. Per-chunk stdout/stderr are under
`logs/single_gpu_<timestamp>_<id>/`.

Inside the original running srun shell, run `./submit_build_all.sh` to use its
visible GPU directly. No nested/overlapping srun or additional GPU is requested;
the runner requires exactly one visible CUDA device. From outside an allocation,
the same command requests one GPU using the server's `slurm_submit_mode`; local
RunPod uses the existing local job runner. `srun_existing` from outside the
original shell is rejected for this workflow. Existing allocations are never
cancelled/requeued by this build. Worker CUDA/OOM or storage-full errors stop
the build without automatic resubmission. Parallelism does not reserve a VRAM
budget per worker; choose the count for the GPU's available memory.

All stages share the allocation's remaining time. For a new single-GPU
allocation, `skillvla_time` covers the whole pipeline rather than each stage.

## FSQ segmentation contract

The selected FSQ run's `fsq_meta.json` and training `skillset_manifest.json` are
the source of truth for probe generation, Gaussian sampling/include-mean/seed,
denoising, PCA/action conventions, GMM, scoring, smoothing, terminal masking and
boundary rules. `pt`/`ft`/`ft_own`/`ft_epi` retain their documented threshold
selection. These settings are passed to the actual Python CLI in initial,
curve-only, cached-curve and missing-episode retry jobs; they are not just exports.

Before submitting or skipping stages, the build validates the inherited recipe
against existing segmentation artifacts. Strict SkillVLA curves carry a recipe
fingerprint, and reduction validates every curve before recording the complete
threshold provenance. FT's shared PT threshold must have matching provenance.
Encoding/final assembly record `segmentation_contract.json` beside `FSQ.pt`;
this proof survives optional intermediate cleanup and is required for reuse.

Historical builds affected by missing CLI arguments are **not** silently reused
or overwritten, even if all episode markers/final files exist. A mismatch or
missing proof stops with an error. Choose separate work/output locations for a
corrected dataset after deciding the rebuild scope; changing code alone does
not repair existing labels or trained checkpoints. No automatic deletion or
regeneration is performed. Slurm and RunPod/local share the same checks.
For example, selecting a new `skillvla_dataset_root` isolates both `_work` and
final outputs; changing only `skillvla_output_suffix` does not isolate `_work`.
Build the corrected PT data before selecting it as an FT threshold/snap reference.
The selected FSQ run's original `fsq_meta.json`, training skillset manifest and
encoder checkpoint must be available locally before an actual build.

## Optional camera UV targets

Set `focus_uv.enabled: false` to build without `eval_init_states.npz` or original
camera XML. Skill codes, boundaries, ISS, proprio/XYZ, and action/image data are
still built normally. XYZ/full-state Predictor targets and goal-XYZ Terminators
read canonical boundaries and endpoint states directly from the dataset parquet;
they do not require `skill_focus_uv.npz` or retained intermediate latents.

Keep `focus_uv.enabled: true` for UV prediction, foveated inputs, or Top-camera
patch supervision (including Both3 and Direct/Residual patch variants). These
objectives genuinely need camera calibration; disabling UV does **not** silently
disable their losses. Existing UV-enabled datasets retain their original target
lookup. Evaluation's episode-exact scene mapping is a separate requirement.

### Automatic episode-exact mapping

Run `./submit_build_all.sh` with `focus_uv.enabled: true`; no separate matching
command or YAML is needed. Immediately before generating missing/outdated UV
targets, the final build worker calls `src/ensure_episode_map.py`. It reuses an
existing map, or creates it from the source dataset and original demonstrations.
The default map is
`<dataset_root>/<skillvla_dataset_root>/<source_dataset>/eval_init_states.npz`;
`focus_uv.eval_init_states_path` can select another path. All inputs come from
the same resolved build snapshot as the other stages.

LIBERO matching compares action sequences with the original HDF5 demos without
rendering. The LangGap matcher is retained under `src/` and matches rendered
candidate initial states when applicable, using a private simulator configuration.
Both run inside the existing final build job, including local/RunPod execution.
With `focus_uv.enabled: false`, matching is skipped along with UV generation.

The shared map is locked while it is created and is published only after a
successful match. Matching failures stop UV generation; existing maps are never
overwritten. A source dataset with changed episode identities requires a new
map path rather than reuse of an old map. Existing complete dataset/ISS/FSQ
stages and current UV artifacts keep their normal skip behavior.
