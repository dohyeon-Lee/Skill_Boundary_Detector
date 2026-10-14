# Direct_patch experiments

Select `architecture.name: direct_patch_1` through `direct_patch_6` in
`vsa_train_config.yaml`, then run `./submit_train.sh`. Alternatively, leave the
YAML selection untouched and use `./submit_train.sh direct_patch_1`.
Names are case-insensitive at submission and saved as lowercase. Each variant
has its own run/checkpoint name. No dataset rebuild or extra YAML switch is needed.

Append `_dGD` to any variant (`Direct_patch_1_dGD` through `Direct_patch_6_dGD`)
to select detached-gradient training. Names retain the canonical `_dGD` suffix
(lowercase family: `direct_patch_1_dGD`). The 4H base is still trained by its
own masked loss. The action path reads detached 17-layer hidden/conditioning
and the SAME norm/head weights with parameter gradients disabled only in that
path. Input gradients through norm/head remain live, so the terminal Expert and
vision path learn normally. No decoder is copied, frozen globally, or reinitialized.
Inference, state_dict shapes, losses, W&B namespaces, and default FT freeze rules
are unchanged; Stage1/FT eval recognize the saved suffixed labels automatically.

The Notion specification is [Direct vs Residual](https://app.notion.com/p/3f8ffbb8f49380acb08ec1bf50cb9098).

## Architecture

Direct reuses the Residual_patch 4H window, input contracts and Both3 visual path
(top + wrist, dedicated 1+1+98 queries, original spatial-loss weight).
`0.01` in the patch labels means one query out of 100, not a loss weight.

| Variant | 17-layer direct input | Last-layer direct input, plus vision bottleneck |
| --- | --- | --- |
| 1 | skill, current proprio, start/end XYZ | skill, current proprio, start/end XYZ |
| 2 | skill, current proprio, start/end XYZ | skill, start/end XYZ |
| 3 | skill, current proprio, start/end XYZ | skill, current proprio |
| 4 | skill, current proprio, start/end XYZ | skill |
| 5 | skill, current proprio, end XYZ | skill |
| 6 | skill, current proprio | skill |

Skill uses the existing layerwise broadcast; proprio/XYZ use MLP -> AdaRMS.
These are direct-input ablations: hidden states can convey information indirectly,
and Cond-Gemma always receives skill, current proprio and end XYZ.

The 17-layer hidden states have length 4H: past H + current H + future 2H.
The original action head reads all 4H hidden states as base velocities. Raw hidden
states H:2H (before final normalization/output projection) continue through the
last Expert layer, conditioned by vision. The SAME final norm and SAME original
action head decode H direct velocities. No extra head is created or initialized.
The central velocity is the direct output, NOT base + direct output. Transformer
internal residual connections remain unchanged.

## Training and inference

One flow timestep and noisy 4H window are shared by both readouts. The base loss
masks positions outside the selected skill's canonical interval or episode;
the original H action targets and action-loss mask are unchanged. Both losses
update the shared head and 17-layer prefix, without detach. The last layer is
trained through the action loss. All 4H attention tokens remain visible in both
training and inference; GT validity masks are used only for loss.

One 4H flow rollout is used: H:2H updates with direct velocity, the other positions
with base velocity. Only H:2H actions are returned. No second denoising sequence,
phase predictor or duration head is needed. Stage-1 and FT evaluation dispatch
from the saved label/revision automatically. The shared dataset target retains
its existing internal `residual_window` name for checkpoint compatibility.

Default FT keeps the shared base/head frozen and adapts the terminal/visual path.
`unfreeze_action_head` / full unfreeze retain their existing meanings. DSBC,
latent Best-of-N and Joint routing are not supported by these window variants.

## W&B

Diagnostics use a separate top-level `direct/` section: base/direct velocity RMS,
direct-minus-base RMS, base/final velocity MSE, MSE improvement, base valid
fraction and shared-head gradient norm. MSE uses the original current-action
targets and mask; improvement may be negative. There are no residual-head
metrics or assumed zero velocity/terminal gradient at initialization.

At the existing `training.vsa_debug` steps, input-shuffle probes retain the
historical `input_influence/{state_shuffle,skill_shuffle,end_pose_shuffle,
top_image_shuffle,wrist_image_shuffle,both_images_shuffle}/relative_output_delta`
and `output_delta_rms` names. They compare the deployed central H velocity using
the SAME noisy 4H window and flow time, with cached DINO tokens (one DINO pass).
Six extra Cond/Expert forwards run without gradients at debug steps only; no
extra loss is added. Patch-loss caches and CPU/CUDA RNG state are restored before
backward. Batch size 1 cannot be shuffled and skips these probes.
