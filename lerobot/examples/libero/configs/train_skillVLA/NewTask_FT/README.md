# NewTask FT

Adapts a trained Stage-1 system to a new-task dataset. It mirrors `stage1/`:
three independent component jobs plus one Joint job share [`newtask_ft_common_config.yaml`](newtask_ft_common_config.yaml)
(new dataset + the three source checkpoints). The older `FT/` and `FT_eval/`
folders are the Stage-2/DSBC pipeline and are untouched.

| Component | Config | Launcher | Output |
| --- | --- | --- | --- |
| VSA | `VSA/vsa_ft_config.yaml` | `VSA/submit_train.sh` | `outputs_root/skillVLA_NewTask_FT/VSA/<run>` |
| Predictor | `Predictor/predictor_ft_config.yaml` | `Predictor/submit_train.sh` | `outputs_root/skillVLA_NewTask_FT/Predictor/<run>` |
| Terminator | `Terminator/terminator_ft_config.yaml` | `Terminator/submit_train.sh` | `outputs_root/skillVLA_NewTask_FT/Terminator/<run>` |
| Joint | `Joint/joint_ft_config.yaml` | `Joint/submit_train.sh` | `outputs_root/skillVLA_NewTask_FT/Joint/<run>` |

VSA training conditions on GT skills and GT spatial targets, so the three jobs
can run in parallel. `NEWTASK_FT_DRY_RUN=1 VSA/submit_train.sh` resolves and
prints the VSA job without submitting it.

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

* The skill-flow (full trajectory) loss has no trainable parameter, so it is
  not computed and its dataset targets are not loaded. The skill-only rollout
  of the FT checkpoint is identical to the source checkpoint.
  `adaptation.skill_flow_loss: true` is the sole optional adaptation choice: it
  enables that loss and automatically unfreezes all Expert transformer layers,
  the final Expert norm, and action head. DINO and the route input projections
  remain frozen.
* State/action normalization is inherited from the source checkpoint instead
  of being recomputed from the new dataset, because the frozen core and action
  head live in those coordinates. Transition jitter follows the new dataset.
* Arch0--Arch3 and latent Best-of-N checkpoints are rejected.

## Predictor / Terminator

Thin wrappers around the unified auxiliary trainer (`terminator/`) in
`mode: ft`. Contract and batch size come from the component checkpoint, the
dataset source is appended to the run lineage, and `newtask_ft: true` selects
the component's own checkpoint from the shared YAML and the
`skillVLA_NewTask_FT` output group.

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
(`../stage1_eval/src`) run with [`eval/ft_eval_config.yaml`](eval/ft_eval_config.yaml); `logs/`
and `outputs/` are written under `eval/`.

```bash
eval/submit_eval.sh                        # submit
STAGE1_EVAL_DRY_RUN=1 eval/submit_eval.sh  # resolve + print the plan only
```

* Every `models[]` entry selects `vsa_dir`, `predictor_dir`, and `terminator_dir`, each with its own
  checkpoint. Run names are searched in the separate FT folders, the Joint folder, and Stage-1
  compatibility locations. Multiple entries are evaluated side by side.
* For a Joint system, put the same Joint run name in all three `*_dir` fields. For separately trained
  components, put each component's run name in the corresponding field. Checkpoint steps may differ.
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
* `gt_dataset` names the new-task SkillVLA run so every panel — including an original Stage-1
  checkpoint trained on another suite — uses the same scenes and GT sequences.
* `libero_10_full_1` has one demo per task. With `oracle.repeat_episodes: true`, that exact scene can
  be replayed for every requested episode; set it to `false` when only one rollout is desired.
