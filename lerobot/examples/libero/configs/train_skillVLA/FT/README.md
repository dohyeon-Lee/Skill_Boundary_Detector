# SkillVLA FT

Adapts a trained Stage-1 system to a new-task dataset. It mirrors `PT/`:
three independent component jobs plus one Joint job share [`newtask_ft_common_config.yaml`](newtask_ft_common_config.yaml)
(new dataset + the three source checkpoints). This is the current fine-tuning
pipeline. The code folder was renamed from `NewTask_FT` to `FT`, and its output
group is now `skillVLA_FT`. Existing checkpoints, run names, eval results, logs,
and snapshots are preserved. Historical `skillVLA_NewTask_FT` paths remain
readable. Internal `newtask_ft_*` fields, YAML filenames, and W&B project names
stay unchanged so saved configs and W&B resumes remain compatible. The obsolete
Stage-2/DSBC FT implementation was removed before this rename.

Resume launchers override the saved `output_dir` with the current local run
directory, so old checkpoint JSON does not recreate the former output group.
On other servers, move the real `skillVLA_NewTask_FT` checkpoint directory to
`skillVLA_FT` separately before resuming: ignored output data is not moved by
Git. Existing Hugging Face remote folders are not renamed by this local change.

| Component | Config | Launcher | Output |
| --- | --- | --- | --- |
| VSA | `VSA/vsa_ft_config.yaml` | `VSA/submit_train.sh` | `outputs_root/skillVLA_FT/VSA/<run>` |
| Predictor | `Predictor/predictor_ft_config.yaml` | `Predictor/submit_train.sh` | `outputs_root/skillVLA_FT/Predictor/<run>` |
| Terminator | `Terminator/terminator_ft_config.yaml` | `Terminator/submit_train.sh` | `outputs_root/skillVLA_FT/Terminator/<run>` |
| Joint | `Joint/joint_ft_config.yaml` | `Joint/submit_train.sh` | `outputs_root/skillVLA_FT/Joint/<run>` |

VSA training conditions on GT skills and GT spatial targets, so the three jobs
can run in parallel. `NEWTASK_FT_DRY_RUN=1 VSA/submit_train.sh` resolves and
prints the VSA job without submitting it.

VSA, Predictor, Terminator and Joint use the shared trainer, which never uploads
checkpoint weights or adapters to W&B, including when resuming old configs with
`wandb.disable_artifact=false`. W&B metrics/media, local checkpoint saves and
server-to-server archival are unchanged. This applies at the next trainer launch;
running processes and previously uploaded artifacts are untouched.

## Dataset requirements

* Source checkpoints are named as `{run, checkpoint}` in the shared YAML and located automatically.
* Labeled by the same FSQ model as the source checkpoints. This is verified
  from `fsq_source.json` on both sides; the checkpoints' `skill_code_space_id`
  is then kept even though the new run tag differs.
* Same FSQ levels, state/action dimensions, and `proprio_grounding`.
* Arch5--Arch13 and foveated checkpoints need `skill_focus_uv.npz`
  (`focus_uv.enabled=true` at build time).

## VSA (Arch4--Arch13 only)

The whole model contract is inherited from `warm_start.vsa_checkpoint` through
`--policy.path`; the YAML cannot restate it.

Direct/Residual checkpoints also retain their `18-N` base + `N` terminal layer
split. Default FT freezes those `18-N` base layers and trains the `N` terminal
layers; a `_layerN` run tag from Stage-1 is retained in the FT run lineage.

`policy.newtask_ft_enabled=true` freezes every parameter on the skill-only
trajectory route: Expert motion-core layers `1..(18 - visual_bridge_last_n_layers)`,
the final Expert norm, action in/out projections, the time MLP, the skill
broadcast, and the Expert-side end-pose AdaRMS of Arch9--Arch11. Cond-Gemma and
its condition inputs, the recurrent bottleneck, the visual bridge, the terminal
bridge Expert layers, and the UV/XYZ/termination heads train. DINO is always
frozen and standalone VSA FT does not expose full-unfreeze or DSBC variants.

Consequences:

* `adaptation.skill_flow_loss: false` skips skill auxiliary supervision and
  its dataset targets. Setting it to `true` enables full-skill action flow
  loss for Vis/canonical models, or normalized proprio spline MSE for Bspline
  models (never an extra canonical flow pass for Bspline).
  With `adaptation.freeze_expert: true`, **all Expert layers, final norm,
  action I/O and Bspline plan generator/head remain frozen**. Gradients still
  pass through them into trainable skill-side visual bridges, Cond and
  bottleneck tokens. The run gains `_frozenexpert_skillflow` or
  `_frozenexpert_skillbspline`, keeping it separate from loss-off checkpoints.
  With `freeze_expert: false`, enabling the loss preserves the previous
  behavior: all Expert layers, final norm/action head (plus the Bspline plan
  generator/head when applicable) train. DINO and route input projections
  remain frozen. Models without trainable visual inputs to their skill route
  do not gain upstream learning signal just by computing a frozen loss.
  Visual skill trajectories are not guaranteed to stay identical to PT even
  with frozen Expert weights, because the visual inputs/bridges can adapt.
* State/action normalization is inherited from the source checkpoint instead
  of being recomputed from the new dataset, because the frozen core and action
  head live in those coordinates. Transition jitter follows the new dataset.
* Arch0--Arch3 and latent Best-of-N checkpoints are rejected.

### VSA checkpoint offload

VSA FT supports the same `training.checkpoint_archive` / `keep_checkpoints`
settings as Predictor FT. The current YAML uses `scratch2` / `1` for training
on an external server: every completed checkpoint (including optimizer/RNG
resume state) is archived to Yonsei, and only remotely SHA256-verified older
local checkpoints are pruned. The destination retains **all** saved steps;
`keep_checkpoints` limits local copies only. Failed or unfinished transfers
remain local and retry, so more than one local save may temporarily exist.

With `steps: 10000` and `save_every: 1000`, a fresh completed run archives
`001000` through `010000` (10 checkpoints). The latest local checkpoint and
`checkpoints/last` remain available for resume. Normal training completion
waits for outstanding transfers; archive settings also apply when resuming.
If the options are absent from an older YAML, offload/pruning stay disabled.

When training **on Yonsei itself**, set `checkpoint_archive: ""` and
`keep_checkpoints: -1`; the worker rejects offloading to the same storage.
See [server-sync archive details](../../../../../../tools/server_sync/README.md#학습-중-체크포인트-자동-보관-predictor--vsa-ft)
for SSH requirements, logs, and retry commands. No UI process is needed.

## Predictor / Terminator

Thin wrappers around the unified auxiliary trainer (`../PT/Terminator/src/`) in
`mode: ft`. Contract and batch size come from the component checkpoint, the
dataset source is appended to the run lineage, and `newtask_ft: true` selects
the component's own checkpoint from the shared YAML and the
`skillVLA_FT` output group.

Predictor FT also inherits the warm-start checkpoint's normalization statistics,
including proprio quantiles used for the VLM state prompt. New-task statistics
do not replace those bounds, even when the VLM is trainable. They remain fixed
and are saved with every checkpoint, so evaluation uses the same input scale.
Resuming uses the **resumed checkpoint's own** statistics, not the original PT
source or freshly computed target statistics; old FT runs are not silently
converted. Scratch Predictor PT and Terminator-only training are unchanged.
The current Predictor YAML uses `run.suffix: base_ptstats` to start a fresh run
from PT instead of auto-resuming the earlier `base` run. Existing outputs are
not renamed or deleted. Choose another fresh suffix for another restart.

### Predictor VLM adaptation

`Predictor/predictor_ft_config.yaml` selects the VLM training scope with
`predictor_ft.unfreeze` (reader/skill/XYZ heads continue training in every case):

| `unfreeze` | Trainable VLM base parameters |
| --- | --- |
| `none` | None |
| `all` | Vision tower, image-language projector, and language model |
| `vision_encoder` | SigLIP vision tower **and** image-language projector; language-model base frozen |

`predictor_ft.lora` is unchanged. LoRA can accompany `none` or `vision_encoder`;
it trains language-model adapters without unfreezing language-model base weights.
`all` requires LoRA disabled. Vision-only training still backpropagates **through**
the frozen language model to the image stack; it does not detach that computation.
Vision/projector parameters use the existing `predictor_vlm_lr_scale` optimizer
group. `gradient_checkpointing: true` checkpoints both the frozen language-model
backward path and the trainable vision tower in this mode.

Run names distinguish `_fullvlm`, `_frozenvlm`, `_visionencoder`, and optional
`_lora`. The default remains `none`, equivalent to the previous `freeze_vlm: true`
with LoRA disabled. Omitted adaptation fields inherit the warm-start contract.
Old snapshots containing `freeze_vlm` remain readable; do not combine the old
switch with `unfreeze`. Saved checkpoints retain the training scope for resume
and evaluation, including the fine-tuned vision/projector weights.

Mode3 UV warm starts also work in standalone Predictor FT: spatial targets are
inherited, UV labels follow the same randomized image crop, and inference
restores the original-frame UV for a UV-based VSA. UV needs the new-task
dataset's existing Top-camera `skill_focus_uv.npz`; no raw start XYZ is needed
for the UV branch. Predictor normalization inheritance is unchanged.

## Joint

Joint warm-starts VSA, Predictor, and (by default) Terminator together. Predictor
outputs condition VSA differentiably according to the configured Joint routes.
Terminator receives a separate current-skill start view and detached Predictor
skill/XYZ outputs; its loss, optimizer group, and gradient clipping are isolated
from Predictor/VSA. Set `terminator.enabled: false` for the two-model ablation.

The VSA warm start may be an existing canonical model (Both3/LIT/etc.) or
`direct_patch_1..6` / `residual_patch_1..6`, including `_dGD` and non-default terminal
layer counts such as 9+9. Canonical models retain their original full-skill route.
Window models use their actual current-proprio-conditioned 4H motion prefix for
both local-code candidate ranking and differentiable STE route supervision;
the 4H padding mask applies to loss only. The prefix/norm/shared head stay frozen,
but gradients can still reach Predictor skill/XYZ inputs. `_dGD` still blocks the
action-to-base path; it does not block the separate Predictor route objective.
There is no extra ordinary VSA skill-flow update in Joint.

DINO stays frozen and Joint has no DSBC support or user-facing VSA adaptation
switches. The same dataset and common warm-start selectors are used; no dataset
rebuild or additional architecture selector is required.

Newterm7 retains its start-proprio/goal inputs and its disabled current-proprio input.
Its k-frame post-end positives use detached Predictor skill/XYZ from each past
occurrence's own start view and state prompt. Conflicting same predicted codes are
excluded. Frozen DINO features are reused for the current and past conditions.
Episode endpoints get k extra current-condition loss weights rather than duplicated
Joint samples, so VSA/Predictor sampling is unchanged. This matches the intended
terminal-positive emphasis, not identical minibatch ordering to standalone Newterm.
Terminator history/masking variants remain unsupported in Joint.

## Evaluation

`eval/` evaluates the adapted system in closed loop. It is the Stage-1 eval engine
(`../PT/eval/src`) run with [`eval/ft_eval_config.yaml`](eval/ft_eval_config.yaml); `logs/`
and `outputs/` are written under `eval/`.

```bash
eval/submit_eval.sh                        # submit
STAGE1_EVAL_DRY_RUN=1 eval/submit_eval.sh  # resolve + print the plan only
```

* Every `models[]` entry selects `vsa_dir`, `predictor_dir`, and `terminator_dir`, each with its own
  checkpoint. Run names are searched in the separate FT folders, the Joint folder, and Stage-1
  compatibility locations. Multiple entries are evaluated side by side.
* Set `predictor_dir: joint` and `predictor_checkpoint: ""` to use the co-trained Predictor
  already inside each selected VSA checkpoint. A `vsa_checkpoint` list then sweeps VSA and
  Predictor together without a separate Predictor overlay. `terminator_dir` may still select
  an external Terminator at a fixed checkpoint. Non-Joint checkpoints reject this selector.
  To deliberately mix checkpoints/components, keep naming the real Predictor/Terminator runs
  and their separate checkpoint steps as before.
* `predictor_dir: gt` selects GT skill and XYZ, while `terminator_dir: gt` selects GT transitions;
  leave that role's checkpoint blank.
* A `vsa_checkpoint` list automatically adds checkpoint-vs-success-rate line charts:
  `checkpoint_success_rates.png` overlays every model's overall SR, and
  `checkpoint_task_success_rates/*.png` contains one separate chart per model with one line
  per task. The x-axis is the numeric training step (including 0k); the y-axis is SR (%).
  No extra YAML switch is needed, for separate or Joint FT runs. Evaluation behavior and
  existing bar charts are unchanged.
* `vsa_checkpoint: ["000000", "001000", "005000"]` includes a 0k VSA baseline:
  `000000` resolves the warm-start VSA recorded in the FT checkpoint configs.
  The original checkpoint must also be available locally. Predictor/Terminator
  selectors are kept as specified; this is zero-shot VSA, not necessarily a
  zero-shot three-component system. Without `gt_dataset`, the 0k panel inherits
  the FT run's new-task dataset. Ordinary Stage-1 step-zero checkpoints still load normally.
  With `predictor_dir: joint`, virtual 0k also resolves the recorded pre-FT Predictor
  (`skill_predictor_checkpoint_path`); both original warm starts must be downloaded.
  A real Joint `000000` checkpoint, if present, loads its own saved VSA/Predictor normally.
* `gt_dataset` names the new-task SkillVLA run so every panel — including an original Stage-1
  checkpoint trained on another suite — uses the same scenes and GT sequences.
* `libero_10_full_1` has one demo per task. With `oracle.repeat_episodes: true`, that exact scene can
  be replayed for every requested episode; set it to `false` when only one rollout is desired.
