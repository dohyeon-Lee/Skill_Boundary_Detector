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
