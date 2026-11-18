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

Each occurrence card includes a lazy top-view skill player. Clicking the play
button opens one shared modal, seeks the official episode MP4 to the skill's
frame interval, and loops only that interval. The report creates relative
symbolic links to the source videos; it does not copy or transcode clips.

The configured checkpoint list may include epochs that training has not
reached yet. They are skipped safely; rerunning the command later evaluates
only new checkpoints because `resume: true` is enabled.
