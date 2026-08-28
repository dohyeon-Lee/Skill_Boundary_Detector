# FSQ terminator probe (GT skills)

Scores the terminator that was **co-trained inside an FSQ checkpoint** against the
GT skill boundaries, for several checkpoints at once.

Nothing here rolls out a policy. The skillset already fixes each skill's start and
end frame, so there is no action model to run and no MAIN terminator to pick a stop
point — the only question is *when does this terminator fire relative to the known
end?* That is why `train_skillVLA/terminator_eval`'s `model:` and
`external_skill_model:` have no counterpart in this config.

## What is comparable, and what is not

All FSQ runs built on one `skillset_seg_name` share the **same skills and the same
GT end frames**, so the models are directly comparable skill by skill. The resolver
refuses a config whose entries come from different skillsets.

What is *not* shared is the code space: token 5 of one run is unrelated to token 5
of another. The codebook grid in the report therefore belongs to the **first listed
model** and is only a way to navigate to skills.

## Variants

The terminator's input contract is read from the checkpoint, never configured:

| checkpoint cfg | module | reads frames |
|---|---|---|
| `state_rnn_terminator` | `FSQStateRNNTerminator` | no |
| `terminator_input_space: state` | `FSQStateMLPTerminator` | no |
| `terminator_input_space: image` | `FSQImageOnlyQueryTerminator` | yes |
| otherwise | `FSQQueryTerminator` | yes |
| `reconstructor_only` | none — the job fails with that message | — |

The recurrent variant carries hidden state across a skill's frames, so skills are
stepped in lockstep rather than shuffled into frame batches.

## Metrics

Per skill: the first frame whose termination probability crosses `end_threshold`
(or the arg-max when it never fires), against `gt_end = length - 1`.

Per model: `|err|` mean/median, share within 3 and 5 frames, early / late / exact
rates, no-fire rate, and a clipped signed-error histogram.

## Output

`outputs/fsq_terminator_eval/<output_name>/`

```
models/<label>/metrics/manifest.json   per-model signals and per-skill timings
media/task_XX/epNNNNN_skillNN/         gt.mp4 + start.jpg + end.jpg (shared)
metrics/compare.json                   joined payload
index.html                             summary table + codebook grid + skill panels
```

Frames are decoded once per skill and reused for every model, since only the
overlaid signals differ. `max_plot_samples` (and `max_plot_entries`) is the cost
knob: every selected skill is *scored*, but only these get a video.

## Run

Edit `fsq_terminator_eval_config.yaml`, then:

```bash
./submit_fsq_terminator_eval.sh
```

One array task per model. The last one to finish builds the comparison page; to
rebuild it by hand later:

```bash
python src/fsq_terminator_eval_report.py \
  --collection-dir outputs/fsq_terminator_eval/<output_name>
```
