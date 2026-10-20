# Temporary Top-view crop preview

CPU-only preview on a real skill-start frame from the currently used PT dataset.
This folder does not change Predictor/VSA training, evaluation, or their YAMLs.

From the repository root:

```bash
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/crop_preview_tmp/preview_topview_crop.py
```

Default: episode 0, second skill's start frame, zoom 1.10, six distinct random
crops, seed 7. The native 256×256 frame becomes 282×282; a 256×256 window is
cropped with integer offsets independently sampled in each axis. No padding,
flip, color adjustment, XYZ adjustment, or Wrist-view modification is applied.
The future model-input resize is not part of this preview.

`outputs/zoom_110/comparison.png` shows the original, centered zoom, and six
random crops. Individual images and exact source/crop metadata are saved too.

Try another scale or frame:

```bash
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/crop_preview_tmp/preview_topview_crop.py --zoom 1.15
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/crop_preview_tmp/preview_topview_crop.py --episode 0 --frame 0
```

Use `--output-dir` to keep multiple previews at the same scale; repeating the
same output directory refreshes these generated preview files.

## Multiple scenes and zoom scales, center crops only

```bash
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/crop_preview_tmp/preview_zoom_sweep.py
```

This selects six actual PT skill-start frames from six distinct scene layouts
(verified using the episode-exact scene mapping): study/caddy, living-room/basket,
kitchen/wine rack, kitchen/stove, kitchen/microwave, and kitchen/drawers.
Each row compares the same original against 1.10x, 1.20x, 1.30x, and 1.40x zooms.
All displayed inputs finish at 224x224; the source videos remain 256x256.

For each zoom, the order is native-frame resize, native-size center crop, then
resize to 224x224. No padding or random crops are used. Bilinear image previews
match model input dimensions and field of view, not bit-exact PyTorch tensor
interpolation. XYZ values and training/evaluation configuration are untouched.

`outputs/center_sweep_110_120_130_140/` contains two three-task comparison sheets,
a combined overview, per-task original/crop pairs, individual inputs, and a
manifest recording episode, frame, scene layout, video timestamp, and crop bounds.

```bash
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/crop_preview_tmp/preview_zoom_sweep.py --zooms 1.1 1.2 1.3 1.4 1.5
```

## Six random crops per scene at 1.20x

```bash
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/crop_preview_tmp/preview_randomized_scenes.py
```

Uses the same six scenes and frames as the centered sweep. Each native 256x256
frame is enlarged to 307x307, cropped with six distinct uniformly sampled offsets
in the inclusive range 0..51 on each axis, and resized to 224x224.
Each scene uses seed `7 + episode_index` for reproducibility.

`outputs/random_zoom_120_multi_env/` contains a comparison sheet for every scene:
original, centered reference, and six random crops. All displayed inputs are
224x224. Individual frames, expanded canvases, and exact crop metadata are also
saved. This remains preview-only and changes no training/evaluation pipeline.

## Eight extreme directions at 1.15x, all scenes in one PNG

```bash
.venv/bin/python lerobot/examples/libero/configs/train_skillVLA/PT/Predictor/crop_preview_tmp/preview_extreme_directions.py
```

The same six sources each occupy one row in `outputs/extreme_zoom_115_multi_env/comparison_all.png`.
Columns are original, centered zoom, up, down, left, right, up-left, up-right,
down-left, down-right. Directions describe image-content movement relative to the
centered zoom, NOT movement of the crop window, which goes the opposite way.

For 256x256 inputs, 1.15x rounds to a 294x294 canvas. Every crop is 256x256,
with offsets spanning 0..38 in each axis. The centered offset is (19,19), and
the eight examples use the four edge midpoints and four corners of this range.
Thus content moves exactly +/-19 native pixels along each applicable axis
(+/-16.625 pixels at the final 224x224 input size). No randomness or padding is used.
Individual PNGs and exact offsets remain available alongside the combined PNG.
