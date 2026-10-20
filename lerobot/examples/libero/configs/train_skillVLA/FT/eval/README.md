# FT evaluation

`submit_eval.sh` + `ft_eval_config.yaml` run the existing simulator evaluation,
with results under `outputs/`.

## Offline Predictor evaluation

`submit_pred_eval.sh` + `ft_pred_eval_config.yaml` evaluate the Predictor alone,
without constructing a VSA, Terminator, LIBERO environment, or MuJoCo renderer.

```bash
./submit_pred_eval.sh --dry-run  # validate dataset/checkpoints; no model loading or submission
./submit_pred_eval.sh           # checkpoint-parallel GPU jobs; repository .venv selected automatically
```

Set `output_name` at the top of `ft_pred_eval_config.yaml` to choose the report
folder (a single name, not a path):

```yaml
output_name: predictor_fullvlm_eval
resume: true
```

This writes to `outputs_predictor/predictor_fullvlm_eval/`. Keep the same name to
resume the same experiment; use a different name for another experiment. Existing
reports are not renamed when this setting changes.

### Multiple GPUs

```yaml
eval_num_gpus: 5   # upper bound; one Predictor process per GPU
```

Model/checkpoint panels are split evenly across at most this many GPUs. Five
checkpoints with a limit of five request five one-GPU workers; with a limit of
two, the workers evaluate three and two checkpoints sequentially. Extra GPUs
beyond the number of panels are never requested. The default for older YAMLs
without this field is one GPU. `slurm.cpus`, memory and `cpu_threads` are per worker.

On Slurm this uses an array: each element can start when one GPU becomes available,
even on different nodes. Logs are `logs/FTpred_<array-job>_<worker>.out/.err`.
With `scheduler: local`, separate detached one-GPU jobs are launched through the
existing GPU allocator; there must be enough free GPUs (otherwise submission can
stop after launching some workers). `--local` is the manual, sequential one-GPU
path regardless of `eval_num_gpus`. `device: cpu` always uses one CPU worker.

All workers use exactly the same selected examples and checkpoint-owned
preprocessing. Per-checkpoint outputs and resume caches are unchanged. You may
resume with another GPU count without recomputing completed samples. Completed
panels are published with `completion.json`; the root `index.html` is atomically
rebuilt under a shared lock, so the last worker cannot discard another worker's
links. The overview shows completed and pending/running panels; refresh it while
the evaluation runs. No separate GPU/merge job is needed.

Results are separate:

```text
FT/eval/outputs_predictor/<output_name>/
  index.html
  selected_samples.json
  resolved_config.json
  <model-and-checkpoint>/
    index.html
    metrics.json
    predictions.csv
    predictions.jsonl
    images/
    records/                    # per-sample resume cache
```

Open `index.html`, then a checkpoint report. Each skill-start example shows both
recorded camera inputs, GT/predicted FSQ code and coordinates, correctness, GT XYZ,
XYZ conditioned on the predicted skill, and XYZ conditioned on the GT skill.
The top-camera overlay uses green for GT, red for predicted-skill XYZ, and orange
for GT-skill-conditioned XYZ. Task and wrong-skill filters are included.

The default data source is the **completed** `libero_10_full_full` SkillVLA build;
its oracle map alone does not contain the labels. **Training demos are included.**
`training_sources: [libero_10_full_1]` only identifies the original training demos
by `(scene_file, demo)`, not local episode number. The report separately summarizes
training and held-out demos and can filter either group. The same examples are
used for every checkpoint. Held-out means unseen **demo occurrences**, not
necessarily unseen FSQ code IDs. `episodes_per_task: 0` evaluates all episodes;
a positive limit samples deterministically and may omit the training example.

Only canonical skill starts are sampled; there is no transition jitter. The GT
endpoint follows `min(frame_end, episode_length - 1)`, matching the dataset builder.
Checkpoint-owned image/prompt/state normalization is retained; already-grounded
dataset states are not grounded a second time. XYZ targets and predictions remain
in the dataset's physical, episode-grounded coordinates (errors are saved in metres
and displayed in centimetres). Projection uses recorded camera calibration, not a
newly rendered scene; old datasets can use their original HDF5 camera XML plus the
action-aligned original episode-start EEF XYZ as a fallback (the same origin used
by the dataset builder, without MuJoCo initialization).

For an FT run, `predictor_checkpoint: "000000"` is a **PT zero-shot** panel: both
weights and preprocessing come from the original PT checkpoint recorded in the
FT config. Saved FT-zero statistics are not used. The PT source is relocated by
run name/checkpoint when evaluating on another server and must be available in
full; missing PT artifacts cause an error, not an FT-zero fallback. Later FT
checkpoints retain their own saved preprocessing. This does not change training
or simulator evaluation. `training_sources` still marks the **FT-reference**
training demos, including in the PT zero-shot panel; it is not PT training membership.

`mode1_preskill` / `mode2_preskill` use the GT previous skill and first-skill BOS.
This deliberately measures teacher-forced boundary prediction, **not** closed-loop
error accumulation or simulator success rate. `mode2` has no skill-conditioned XYZ
reader, so its two XYZ predictions should coincide.

`device: cpu` uses float32 and requests `slurm_cpu` rather than a GPU allocation.
It is supported, but a large VLM makes CPU inference slow and RAM-intensive; GPU
is recommended. `--local` runs directly instead of submitting: use a compute node,
not a shared login node. The launcher never silently falls back from CUDA to CPU.

`resume: true` skips completed samples with matching dataset, checkpoint, device
and visualization signatures. Changed evaluation inputs require another
`output_name`; results from incompatible settings are never silently mixed.
