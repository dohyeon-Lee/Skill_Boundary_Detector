# Multi-terminator skill evaluation

This evaluator is the episode-exact, single-skill evaluation separated from
`stage1_skill_eval`. It loads one controlling terminator and any number of
independent display-only terminators:

- `model.terminator_source` selects the **MAIN** terminator that controls
  rollout stopping: `own` uses the Stage-1 checkpoint copy, `original` follows
  that checkpoint's `fsq_path` and reconstructs the co-trained FSQ terminator,
  and `external` uses `external_skill_model`. The external mapping
  can be `{variant: fsq_initial}` to reconstruct the pristine terminator from
  the selected action policy's raw `FSQ.pt`, or can select a trained overlay
  using `variant: state_image|image_only|wrist_only|state_only|state_rnn` plus
  `group`, `model_dir`, and `checkpoint`.
- `terminator_models` selects `state_image`, `image_only`, `wrist_only`,
  `state_only`, `state_rnn`, or `fsq_initial` for each display row. Trained
  entries use `{label, variant, model_dir, checkpoint}` and resolve under the fixed
  `skillVLA_terminator` group plus top-level `outputs_root`. `fsq_initial`
  accepts no checkpoint fields: it loads the normal state+top+wrist terminator
  directly from the `fsq_path` recorded by the selected action policy, before
  auxiliary terminator training. It is independent of `external_skill_model`,
  which may still supply the MAIN terminator. Every entry is **display-only**.
  The legacy `image_only_terminator_model` key/path format remains accepted as
  `image_only` for existing configs.

`state_only` and `state_rnn` consume the current normalized proprio state and
skill code; they never receive camera tensors. The recurrent variant receives
one current state per evaluator step and carries its returned RNN hidden state
to the next step. MAIN and display hidden states are independent and are reset
to `None` at the start of every GT/policy branch (therefore every evaluated
skill).

Each branch panel contains only full-width signal bars:

```text
camera frame
     [ termination bar | progress bar ]
     [ termination bar | progress bar ]
     [ termination bar | progress bar ]
```

The synchronized comparison adds one shared label gutter at the far left:

```text
TERM1 | GT bars | exact bars | alt-noise bars | early bars | late bars
TERM2 | GT bars | exact bars | alt-noise bars | early bars | late bars
MAIN  | GT bars | exact bars | alt-noise bars | early bars | late bars
```

No percentage text is drawn. Termination and progress each use one full
half-frame width and retain only the threshold marker. A termination bar
freezes at its first `end_threshold` crossing for the rest of that rollout
branch, independently for each terminator. This latch is display-only: raw
signals remain in the manifest and MAIN alone controls rollout stopping.

The HTML keeps the five synchronized rollout branches, start-frame poster, and
final-frame comparison provided by `stage1_skill_eval`.

Edit `terminator_eval_config.yaml`, then submit with:

```bash
./submit_terminator_eval.sh
```

`eval_num_gpus` is a physical-GPU ceiling and
`eval_max_workers_per_gpu` controls how many independent evaluator processes
share each GPU (maximum 4). The planner requests only the number of GPUs needed
for the selected episodes; for example, six episode workers with a limit of
four workers per GPU request two GPUs, not six or the full configured ceiling.

Outputs, logs, and config snapshots remain under this `terminator_eval`
directory and do not overlap `stage1_skill_eval` artifacts.
