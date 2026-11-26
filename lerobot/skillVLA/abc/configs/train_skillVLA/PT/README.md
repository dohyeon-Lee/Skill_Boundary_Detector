# ABC SkillVLA PT

This directory mirrors the LIBERO Stage-1 VSA interface while using the ABC
native dataset in place. The shared trainer is intentionally reused so fixes to
the action-flow and dual-bottleneck implementation stay common.

Default ABC differences are explicit in `VSA/vsa_train_config.yaml`: three
ordered cameras, 14-D absolute-joint state/action/full-skill targets, 30-frame
chunks with 15 executed actions at 30 Hz, bimanual UV/XYZ auxiliaries, and
step-based transition jitter (`pmax_step: 23`) without gripper gating.

Run from `PT/VSA`:

```bash
./submit_train.sh
```

`Predictor/` provides a YAML and `submit_train.sh` with the
same `--JOB_ID` interface (for example `./submit_train.sh --73428`). Dataset and
transition settings come from `stage1_common_config.yaml`; submitted configs are
snapshotted. Existing ABC SkillVLA artifacts are reused without rebuilding.

The Predictor defaults to `mode2`: Top + left/right wrist images, 14D proprio
in the language prefix, and the previous skill (learned BOS at episode reset).
It predicts the four-axis FSQ skill and both Top-image endpoints in
`[uL,vL,uR,vR]` order. Both UV targets use the same sampled Top crop; invalid
projections are masked separately for each arm. Its occurrence-balanced start
sampling inherits the common +/-23-frame half-normal transition randomization.

Predictor uses the common backend in LIBERO `PT/src/auxiliary/` and writes to
`outputs_ABC/skillVLA_PT/Predictor/`. Standalone Terminator training entries were
removed. VSA's integrated termination reader and evaluation of saved external
Terminator checkpoints remain available.
