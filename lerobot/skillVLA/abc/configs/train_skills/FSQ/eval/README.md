# ABC FSQ evaluation

Edit `eval_config.yaml`, then run from this directory:

```bash
./submit_eval.sh --dry-run
./submit_eval.sh --73428
```

`--73428` runs the complete pipeline inside that existing allocation: missing
FSQ latents are encoded on its GPU, then the video report runs on CPU. It does
not request a new allocation or Slurm step. Without a job selector, the shared
submitter handles the GPU prepass and dependent CPU replay jobs. `--config FILE`
selects a different YAML. Real submissions freeze the YAML and global settings.

The default configuration selects the absolute-action FSQ 3×3×3×2 model at epoch 1000.
Only `run_name` and `fsq_eval_checkpoint` need changing when selecting a model.
The HTML display name defaults to `run_name`, and a single run's output folder
defaults to `<run_name>/epochNNNN` (or `compare_epochNNNN_epochMMMM` for several
checkpoints). Explicit `model_name` and `output_name` remain optional overrides.
Its architecture and original training skillset come from the saved model and
`fsq_meta.json`; no DP architecture settings or new SBD build are required.
The evaluator selects five episodes for each of grab_clutter's 24 language
prompts and groups their skill videos by FSQ code.

The report is written to:

```text
outputs/FSQ3332_contrastiveON_routeON_loss_abc_grab_clutter_absolute_joint/epoch1000/index.html
```

The interactive codebook keeps three axes in each cube. From the fourth axis
onward, each combination of additional coordinates gets a separate cube, with
up to three cubes per row and a common color scale. FSQ 3×3×3×3 therefore shows
three cubes (axis 4 = −1, 0, +1), containing all 81 codes. Selecting a code also
outlines matching coordinates on axes 1–3 in the other cubes. Clicking a used
code shows inline top-view video players, each restricted to its skill interval.
`By task` groups them under task headings; `All skills · full width` keeps a
compact, clickable codebook above the videos, hides the statistics tables, and
fills the page width with a responsive video grid. The codebook stays visible
while scrolling so another code can be selected directly. Both views wrap vertically without horizontal scrolling. Returning to
`By task` restores the usual sidebar and statistics. Any player or `Play all` button
starts every video belonging to the selected code, including clips below the
viewport. Each loops independently, and `Pause all` pauses the entire group.
Switching views preserves playback. Videos are linked rather than copied or
transcoded. `resume: true` reuses completed checkpoints and their reports.

`src/run_eval.sbatch` is the allocation entry point; the evaluation engine is
shared with LIBERO. The old ABC `skill_eval/fsq_codebook` directory has moved
here. The additional analysis/export helpers also live under `src/`:

- `export_top_skill_clips.py`: export clips from an existing report.
- `analyze_arm_codebook.py` and `export_selected_cube_images.py`: legacy tools
  for three-dimensional FSQ333 codebooks, separate from the 4D HTML evaluator.
  Their legacy defaults still refer to an epoch-800 FSQ333 report; supply the
  appropriate report paths explicitly when using these optional tools.
