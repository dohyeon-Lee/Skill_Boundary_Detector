# Predictor or terminator auxiliary training

`auxiliary_train_config.yaml` is the only user-facing training config in this
directory. It trains either the skill predictor or the FSQ terminator; enabling
both in one job is rejected because their sampling units differ.

Predictor training samples every real skill occurrence once per DataLoader
epoch. Each access draws one transition jitter and applies that same draw to
the start image, start proprio, GT skill code, and optional endpoint-focus UV.
It therefore does not repeat a single predictor target for every frame inside
the skill. Terminator training retains ordinary frame-level sampling.

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
