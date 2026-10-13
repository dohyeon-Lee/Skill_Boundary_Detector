# Predictor or terminator auxiliary training

`auxiliary_train_config.yaml` is the only user-facing training config in this
directory. It trains either the skill predictor or the FSQ terminator; enabling
both in one job is rejected because their sampling units differ.

Predictor training samples every real skill occurrence once per DataLoader
epoch. Each access draws one transition jitter and applies that same draw to
the start image, start proprio, GT skill code, and optional endpoint-focus UV.
It therefore does not repeat a single predictor target for every frame inside
the skill. Terminator training retains ordinary frame-level sampling.

## Newterm post-end experiments

Select `Newterm1`, `Newterm2`, `Newterm3`, or `Newterm4` in the Stage-1
`stage1/Terminator/terminator_train_config.yaml`:

```yaml
fsq_terminator: {architecture: Newterm1}
termination_loss: {target_sigma: 1.0, positive_weight: 2.0, post_end_steps: 10}
```

All four reuse the existing fusion Terminator, frozen top-camera DINO, and
direct raw-skill skip. Newterm1 has no proprio input; Newterm2 uses current
proprio; Newterm3 reuses the existing 20-frame proprio-history Transformer.
Newterm4 uses current proprio and a second token for the active skill's start
proprio, reusing the existing start-state token path. For its post-end branch,
the start token comes from the corresponding past skill in the same ISS store.
They have no start randomization, goal-XYZ input, or visual auxiliary loss.
Their specification is the Newterm rows in [the Notion comparison table](https://app.notion.com/p/3f0ffbb8f493811a9e90d4cb8e44654d).

Interior sampling is unchanged. A frame within `post_end_steps` frames of a
past skill's inclusive endpoint also receives termination target 1 under
that skill's code, using the same current image/proprio/history. Both branches
reuse the encoded image tokens. Loss is `(sum(current BCE) + sum(post-end BCE))
/ original batch size`, with positive weight 2 applied in both branches.
If several short skills end within k frames, all past conditions are supervised.
Past conditions with the same code as the current skill are excluded to avoid
contradictory targets on identical inputs. Other episodes are never used. For each episode,
its last frame is included `post_end_steps` extra times in the shuffled sampler;
those copies train only the final skill, with target 1.

`post_end_steps: 0` disables the extra conditions and endpoint repeats. Existing
term profiles retain their sampling, and old checkpoints default this field to
zero. No dataset rebuild is needed. FT inherits this setting from its component
checkpoint. Output names end in `terminator_Newterm1` (or 2/3/4) before any suffix.
Nondefault k values append `_k{k}` to keep different supervision runs separate.

From `stage1/Terminator`, select a variant in YAML or submit with an override:

```bash
./submit_train.sh Newterm1
./submit_train.sh Newterm2
./submit_train.sh Newterm3
./submit_train.sh Newterm4
```

The predictor can optionally add a second endpoint-focus UV branch. It shares
the predictor VLM forward, has its own reader/head, and is conditioned on the
jitter-aligned GT FSQ skill during training. The selected SkillVLA run must
contain `skill_focus_uv.npz`. Set `skill_predictor.focus_uv.enabled: true`.

`skill_predictor.freeze_vlm: true` retains the frozen-VLM/optional-LoRA setup.
Setting it to `false` co-trains the complete predictor VLM and automatically
disables predictor LoRA. Use `training.optimizer.predictor_vlm_lr_scale` to
control the VLM learning rate independently.

## PT and FT

- `mode: pt` initializes the predictor from pi0.5. The terminator is warm-started
  from the selected `dataset.source/run/FSQ.pt` only when its full context/architecture/
  backbone/freeze contract matches `fsq_terminator`; otherwise the requested
  terminator is initialized fresh.
- `mode: ft` infers its single train target from either
  `warm_start.predictor_checkpoint` or `warm_start.terminator_checkpoint`.
  Exactly one path must be non-empty. Its PT config owns the component contract;
  the PT-only YAML model sections are ignored.
- FT inherits its batch size from that component PT checkpoint.

The supported terminator is the same default state/image query terminator used
by current FSQ training. Historical image-only, wrist-only, state-only, and
state-RNN variants are intentionally not exposed by this trainer config.

Submit from this directory with:

```bash
./submit_train.sh
```

Output names omit a separate `pt`/`ft` token and preserve the dataset lineage:

```text
bs{PT batch}_{FSQ run}_{PT source}[_{FT source}...]_{enabled targets}
```

Each FT stage appends its current dataset source before the enabled target name.
PT suffixes are inherited separately from the dataset lineage and are always
re-attached after the current target name. A new FT suffix is appended after
the inherited suffixes.
