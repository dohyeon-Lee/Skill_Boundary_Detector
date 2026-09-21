# NewTask FT

Adapts a trained Stage-1 system to a new-task dataset. It mirrors `stage1/`:
three independent jobs that share [`newtask_ft_common_config.yaml`](newtask_ft_common_config.yaml)
(new dataset + the three source checkpoints). The older `FT/` and `FT_eval/`
folders are the Stage-2/DSBC pipeline and are untouched.

| Component | Config | Launcher | Output |
| --- | --- | --- | --- |
| VSA | `VSA/vsa_ft_config.yaml` | `VSA/submit_train.sh` | `outputs_root/skillVLA_NewTask_FT/VSA/<run>` |
| Predictor | `Predictor/predictor_ft_config.yaml` | `Predictor/submit_train.sh` | `outputs_root/skillVLA_NewTask_FT/Predictor/<run>` |
| Terminator | `Terminator/terminator_ft_config.yaml` | `Terminator/submit_train.sh` | `outputs_root/skillVLA_NewTask_FT/Terminator/<run>` |

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

`policy.newtask_ft_enabled=true` freezes every parameter on the skill-only
trajectory route: Expert motion-core layers `1..(18 - visual_bridge_last_n_layers)`,
the final Expert norm, action in/out projections, the time MLP, the skill
broadcast, and the Expert-side end-pose AdaRMS of Arch9--Arch11. Cond-Gemma and
its condition inputs, the recurrent bottleneck, the visual bridge, the terminal
bridge Expert layers, and the UV/XYZ/termination heads train. DINO follows
`adaptation.train_dino`.

Consequences:

* The skill-flow (full trajectory) loss has no trainable parameter, so it is
  not computed and its dataset targets are not loaded. The skill-only rollout
  of the FT checkpoint is identical to the source checkpoint.
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

## Evaluation

`eval/` evaluates the adapted system in closed loop. It is the Stage-1 eval engine
(`../stage1_eval/src`) run with [`eval/ft_eval_config.yaml`](eval/ft_eval_config.yaml); `logs/`
and `outputs/` are written under `eval/`.

```bash
eval/submit_eval.sh                        # submit
STAGE1_EVAL_DRY_RUN=1 eval/submit_eval.sh  # resolve + print the plan only
```

* `model_dir`, `external_predictor_model`, and `external_terminator_model` are run names. They are
  searched in `skillVLA_NewTask_FT/<component>`, then the Stage-1 locations, so one grid can mix
  adapted and original components (FT system, zero-shot baseline, per-component ablations).
* `gt_dataset` names the new-task SkillVLA run so every panel — including an original Stage-1
  checkpoint trained on another suite — uses the same scenes and GT sequences.
* `libero_10_full_1` has one demo per task, so GT panels need `n_episodes: 1`.
