# Stage-1 wrist alignment and action maps

This standalone evaluator loads an `arch16_align`, `arch17_align`, or
`arch18_align` checkpoint and writes one panel per selected dataset frame:

- the wrist image and geometry-derived target patch;
- the final pooled alignment-head heatmap;
- the Top-N pre-pooling bottleneck-query maps.
- every action timestep's direct Action→bottleneck→vision attention map;
- every action timestep's action-output gradient saliency over vision patches.

Top-N is ranked by each query's **pooling-weighted spatial contribution**, not
by pooling weight alone. The saved NPZ contains all query logits and verifies
that their weighted sum exactly reconstructs the final map. These are learned
alignment-head contribution maps, not raw transformer MHA weights.

The action page bypasses the alignment head. Its top rows compose the actual
Action→bottleneck and bottleneck→vision MHA weights. Its bottom rows show
`∂||action_t||₂/∂vision_patch` at a deterministic flow probe (`x_t=0`,
`t=0.5`). Raw values, action outputs, bridge weights, and the probe time are
saved in each NPZ; only the displayed heatmaps are min-max normalized for
readability. Attention is also renormalized over patches for display, while its
pre-renormalization patch mass and excluded CLS mass remain in the NPZ and the
patch mass is printed above each map. The HTML has an
`Alignment head / Action maps` switch.

Each action panel begins with a paper-style whole-chunk summary: mean raw
attention across action slots (then patch-normalized for display) and RMS
gradient saliency across action slots. The following rows retain all per-slot
maps for diagnosis. Both summaries are also stored in the NPZ.

The YAML only asks for an architecture/checkpoint step and sample counts.
`global_config.yaml` supplies project/storage/Slurm placement, while
`stage1_common_config.yaml` supplies the shared SkillVLA dataset layout and
source. The checkpoint supplies its exact skillset, repo metadata, and DINO
model identity. Device (`cuda`), video backend (`pyav`), output path, and this
small evaluator's resource request are internal defaults.

`skills_per_episode: all` evaluates every valid skill in each selected episode;
`frames_per_skill` still controls how many evenly spaced frames are drawn from
each skill. An integer may be used when a smaller diagnostic subset is wanted.

Set `output_name` in the YAML to choose the result folder name under
`stage1/eval/outputs/`. Leave it empty to use the automatic
`<architecture>_<checkpoint-step>` name.

Edit `attention_eval_config.yaml`, then run `./submit.sh`. This folder does not
submit anything automatically.

Action maps are enabled by default without adding anything to the YAML. For an
advanced probe-time override or to reproduce the old alignment-only evaluator:

```yaml
action_maps:
  enabled: true
  probe_time: 0.5
```
