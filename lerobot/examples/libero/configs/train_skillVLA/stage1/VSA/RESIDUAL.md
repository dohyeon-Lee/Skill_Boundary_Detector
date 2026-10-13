# Residual_patch experiments

Select `architecture.name: residual_patch_1` through `residual_patch_6` in
`vsa_train_config.yaml`, or run `./submit_train.sh residual_patch_1` from this
directory. Each variant has its own run name and checkpoint config. All other
training settings and the existing main-action loss mask remain unchanged.
No dataset rebuild is needed.

The Notion specification is [Residual](https://app.notion.com/p/3f8ffbb8f49380acb08ec1bf50cb9098).
These six variants reuse Both3's top/wrist vision path and camera-dedicated
1+1+98 bottleneck queries. The `0.01` patch labels denote one query out of 100,
not a new loss coefficient; spatial loss retains the existing YAML weight.
Residual_lit is not selected by these six patch variants.

| Variant | Base direct input | Last-layer direct input (in addition to vision bottleneck) |
| --- | --- | --- |
| 1 | skill, current proprio, start/end XYZ | skill, current proprio, start/end XYZ |
| 2 | skill, current proprio, start/end XYZ | skill, start/end XYZ |
| 3 | skill, current proprio, start/end XYZ | skill, current proprio |
| 4 | skill, current proprio, start/end XYZ | skill |
| 5 | skill, current proprio, end XYZ | skill |
| 6 | skill, current proprio | skill |

Skill retains its layerwise broadcast. Proprio and XYZ use the existing MLP /
AdaRMS modules; last-layer conditions are constructed independently so removed
inputs are not added there a second time. Hidden states and the bottleneck can
still convey that information indirectly. Cond-Gemma retains current proprio,
skill and end XYZ for every variant.

## Flow and masking

The base sees all `4H` noisy action tokens: past `H`, current `H`, future `2H`.
The first 17 layers produce base velocities via the existing action head. Only
raw central hidden states `H:2H` reach the terminal Expert layer; its separate
linear residual head starts with zero weight and bias. The final current
velocity is `base[:, H:2H] + residual`. No detach is used.

Training uses one shared flow timestep and one `4H` noise/window state, not two
independent forwards. The base loss masks positions outside the selected
skill's original GT interval or its episode. Transition jitter selects which
skill/anchors condition the model; the base retains the canonical GT boundary
convention. An entirely out-of-skill jittered window contributes zero base loss.
The main action target and its existing padding/skill-end masking are unchanged.
Every token stays visible to attention, with the same fixed positions in training
and inference. The model receives neither a validity mask nor observed past actions.

Inference caches the vision/Cond path once, then jointly denoises `4H` each flow
step. Only `H:2H` is returned to the normal action queue. There is no phase,
duration head, or second denoising sequence. Default FT freezes the base including
its now-shared proprio projection, while adapting the residual/visual paths.
Stage-1 and FT evaluation restore the saved architecture automatically. Latent
Best-of-N, DSBC, and Joint routing are not supported by this first residual version.

## W&B

The top-level `residual/` section reports base/residual velocity RMS, their ratio,
base-only/final velocity MSE, MSE improvement, base valid fraction and residual
head gradient norm. MSE comparisons use the identical current-action targets,
real action dimensions and existing main-action mask. These are flow-training
diagnostics, not denoised action errors or rollout success rates. A zero residual
and zero terminal-layer gradients at the first update are expected with zero-init.
