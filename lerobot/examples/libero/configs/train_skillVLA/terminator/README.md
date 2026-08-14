# Auxiliary-only training

This directory trains the Stage-1 FSQ terminator, image-only terminator,
wrist-only terminator, and skill predictor without constructing or optimizing
an action/VSA model.

Select the training target in `terminator_train_config.yaml`:

`terminator.train`, `image_only_terminator.train`,
`wrist_only_terminator.train`, and `skill_predictor.train` are independent
booleans. Any non-empty combination is valid; setting all four to `false` is a
configuration error.

The normal state+image terminator supports both FSQ implementations. A joint
FSQ-v3 checkpoint warm-starts its saved `terminator.*` tensors. An
`FSQOriginalConfig` checkpoint contains no terminator tensors, so its FSQ
levels, hidden size, attention shape, and encoder state bounds are used to
construct a fresh terminator instead. The raw `fsq_initial` evaluation remains
v3-only because an FSQ-original checkpoint has no pretrained terminator to
evaluate.

The image-only model consumes the current skill code plus top/wrist images. It
uses the same progress and termination targets as the normal terminator, does
not consume transition-randomized predictor inputs, and loads no model tensor
from `FSQ.pt`. DINO comes from its original pretrained path and stays frozen;
the image projection, query decoder, and heads start from scratch. The FSQ file
is read only for the matching architecture and code contract.

The wrist-only model consumes the current skill code and current wrist-camera
image only. It has no state or top-camera input and, like the image-only model,
starts every terminator-specific tensor from scratch while keeping the original
DINO encoder frozen. Its checkpoint prefix is `model.fsq_wrist_term_train.*`.

Submit with:

```bash
./submit_train.sh
```

Single-GPU runs invoke `lerobot-train` directly so the `accelerate` launcher
does not perform a redundant cold Python/Torch import. Multi-GPU runs retain
`accelerate launch` with one process per configured GPU.

Runs use the separate `VLA_terminator` W&B project by default. Metrics are
reported under `train_terminator/*`, `train_image_terminator/*`,
`train_wrist_terminator/*`, and `train_skill_predictor/*`; action/VSA loss and
optimizer panels are not produced. Checkpoints use `model.fsq_term_train.*`,
`model.fsq_image_term_train.*`, `model.fsq_wrist_term_train.*`, and
`model.skill_predictor.*` prefixes. Wrist-only training is wired here; its
Stage-1 evaluator variant is intentionally a separate integration step.

For image-only Stage-1 evaluation, point `external_skill_model` at this
checkpoint, set `advance_mode: external`, and set
`terminator.variant: image_only` in `stage1_eval_config.yaml`. A model entry may
override the global choice with `terminator_variant: image_only`, which also
allows normal and image-only terminator panels in one comparison run.
