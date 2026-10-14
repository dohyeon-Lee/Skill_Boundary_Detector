# Residual_patch experiments

Select `architecture.name: residual_patch_1` through `residual_patch_6` in
`vsa_train_config.yaml`, or run `./submit_train.sh residual_patch_1` from this
directory. Each variant has its own run name and checkpoint config. All other
training settings and the existing main-action loss mask remain unchanged.
No dataset rebuild is needed.

Append `_dGD` to any variant (`Residual_patch_1_dGD` through `Residual_patch_6_dGD`)
for detached-gradient training. Selection is case-insensitive; saved labels use
the lowercase family and retain `_dGD` (`residual_patch_1_dGD`). The action loss
does not update the 17-layer base, its shared conditioning projections, or the
shared final norm. The central hidden, base velocity and direct conditioning
are detached on the action path; norm parameters are read-only there while
gradients through its input still train the terminal layer/vision/residual head.
The masked 4H loss keeps its full base graph. No extra head/norm or YAML toggle
is introduced. Original attached names, losses, metrics, inference, checkpoint
shapes and default FT freeze rules are unchanged. Stage1/FT eval dispatch from
the suffixed checkpoint label automatically.

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

The existing debug schedule also logs the historical input-shuffle diagnostics
under `input_influence/` (including `state_shuffle/relative_output_delta`). They
compare the FINAL base-plus-residual central H velocity, keeping the noisy 4H
window and flow time fixed. Cached DINO tokens are reused; six no-gradient
Cond/Expert forwards run only at debug steps, not at every training update.
Auxiliary patch-loss caches and CPU/CUDA RNG state are restored before backward.
No loss term or parameter is added. Batch size 1 skips shuffle probes.
