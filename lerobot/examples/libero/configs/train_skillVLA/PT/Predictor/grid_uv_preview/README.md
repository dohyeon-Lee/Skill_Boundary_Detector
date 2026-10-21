# First-skill endpoint UV grid preview

CPU-only analysis of the current PT GT targets for LIBERO-90 Tasks 3, 4, 5.
No simulator/model weights are loaded; no training/evaluation config is changed.

From the repository root:

```bash
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/grid_uv_preview/preview_first_skill_uv.py
```

`outputs/endpoints_tasks_3_4_5.png` places the first exact demonstration of each
task side by side at the first skill's endpoint frame. Red is the exact GT UV;
yellow is the center of its existing vision-aux patch cell. Cyan axes show the
image-center origin; positive v points downward.

`outputs/comparison_all.png` shows each task's original skill-start image,
Mode3's 1.1x centered crop of that image, and the original endpoint image. All
three are annotated with the appropriate transformed first-skill endpoint UV.

The default grid comes from the current DINOv3-S/16 config: 224/16 = 14 cells
per axis, UV cell width 2/14. Centers do not include zero for this even grid.
This visualizes the EXISTING aux grid; it does not select or implement a new
Predictor UV quantization interval. `--grid 21` can inspect a denser cell grid,
but note that cell-center spacing is not the same as rounding UV to 0.1.

The canonical endpoint uses `min(frame_end, episode_length-1)`, matching the
dataset focus artifact (not frame_end-1). Skill code/target identity uses the
first skill even if its endpoint frame is the next skill's first frame.
Task IDs follow LIBERO benchmark order, joined via `eval_init_states.npz`,
not the dataset's compact task IDs. Episode ordering matches PT exact eval.

The CSV and manifest include coordinate-only grid occupancy for the first ten
exact demos per task, to avoid treating one image as proof of separability.
Image zoom/resize uses PIL bilinear for display; UV transforms mirror the exact
integer zoom/crop and half-pixel geometry of PredictorTopCrop. These previews
are not bit-exact torch image-interpolation tests.
