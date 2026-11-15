# ABC FSQ codebook evaluation

Run from this directory:

```bash
./submit_eval.sh
```

The GPU prepass encodes each available FSQ checkpoint once. The dependent
CPU-only replay reads the official ABC native videos, groups skill start/end
frames by FSQ token and language-task ID, and writes the combined report to:

```text
outputs/abc_grab_clutter_absolute/index.html
```

The configured checkpoint list may include epochs that training has not
reached yet. They are skipped safely; rerunning the command later evaluates
only new checkpoints because `resume: true` is enabled.
