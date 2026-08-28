# Predictor + terminator auxiliary training

`auxiliary_train_config.yaml` is the only user-facing training config in this
directory. It can train the skill predictor, the FSQ terminator, or both.

## PT and FT

- `mode: pt` initializes the predictor from pi0.5. The terminator is warm-started
  from the selected `dataset.source/run/FSQ.pt` only when its full context/architecture/
  backbone/freeze contract matches `fsq_terminator`; otherwise the requested
  terminator is initialized fresh.
- `mode: ft` infers its train targets from `warm_start.predictor_checkpoint` and
  `warm_start.terminator_checkpoint`. The paths may refer to different PT
  `skill_aux` checkpoints. Each non-empty path enables that component and its
  PT config owns the component contract; the PT-only YAML model sections are
  ignored.
- When both FT paths are present, their FSQ code-space identity must also match
  the target dataset. Equal codebook dimensions alone are not accepted as proof
  that integer codes have the same meaning.
- FT also inherits its batch size from the component PT checkpoint. Two FT
  component sources with different PT batch sizes are rejected.

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
