# Initial top-view capture

Uses OSMesa CPU rendering by default; no EGL display/GPU allocation is needed.
Requires system libOSMesa (available on this server). This does not submit a
Slurm job, load a model, or execute policy actions. To use GPU rendering on a
GPU-allocated node, prefix the command with `MUJOCO_GL=egl`.

```bash
cd /scratch2/mdorazi/Skill_Boundary_Detector/lerobot/examples/libero/configs/train_pi05/FT/eval
bash capture_initial_frames.sh --label baseline
```

Defaults follow `pi05_eval_config.yaml`: currently 10 tasks × 10 episodes.
Captures use `LiberoEnv.reset` followed by `LiberoEnv.render`, including the same
10 no-op settling steps and display orientation as eval videos. Default image
resolution is pi05 eval's 360×360; no resizing is applied to individual PNGs.

Each run creates `initial_frames/<timestamp>_<label>/` containing `index.html`,
`overview.png` (task rows / episode columns), original per-episode PNGs,
`eval_config.yaml`, and `manifest.json` (seeds, init indices, BDDL hashes and
object positions). Original runs are never overwritten. Compare the same task,
episode, seed and init source across labeled runs.

```bash
# Resolve settings only; no simulator or rendering.
bash capture_initial_frames.sh --dry-run

# Fresh placements sampled from the BDDL, NOT the stored benchmark states.
bash capture_initial_frames.sh --init-source random --label sampled --seed 1000

# After making a separate BDDL tree with wider placement ranges:
bash capture_initial_frames.sh --init-source random --bddl-root /absolute/path/to/wider_bddl_files --label wider

# Or inspect a different bank of pre-generated benchmark states:
bash capture_initial_frames.sh --init-source benchmark --init-states-root /absolute/path/to/new_init_files --label new_bank
```

`--init-source random` alone does **not** enlarge placement ranges. Changing BDDL
ranges while still using stored benchmark states does not widen those stored
layouts either. Use random mode with the modified BDDL, or generate a new state
bank. Keep task definitions/assets compatible with the chosen state bank.

Other overrides: `--config <yaml>`, `--task-ids 0 1`, `--episodes 10`,
`--episode-offset 10` (benchmark only), `--settle-steps 10`, `--size 360`,
`--output-dir <new-or-empty-directory>`, `--libero-root <libero/libero>`.
The default seed 1000 matches the current pi05 eval launcher; episode k uses
1000+k. A changed eval seed/resolution/settle policy must be passed here too.
Episode-exact mode follows the YAML archive selection and repeat setting; unlike
the evaluator's dropping of insufficiently populated tasks, this diagnostic
fails explicitly if a requested task cannot supply the requested episode count.

## YAML-controlled widened evaluation

Set `random_range: 2.0` in `pi05_eval_config.yaml` (with
`oracle.episode_exact: false` and `episode_offset: 0`), then use the usual
`./submit_eval.sh`. `1.0` leaves the existing benchmark/demo evaluation unchanged;
values greater than one scale each movable workspace object's X/Y interval
around its original center. Rotation, Z offsets, fixtures, goal regions and
conditional inside/on-object placement regions are not expanded. All movable
workspace objects are included, not only the task's target objects. Zero-width
intervals remain fixed. Unsupported tasks fail explicitly.

Each worker generates a persistent initial-state bank **before model loading**,
without cameras/GPU rendering. Every checkpoint restores the identical bank.
Sampling seeds are 1000+k, with deterministic retries for invalid placements.
The normal 10 settling steps run after bank restoration. Fixture model-body
poses are stored too (not just qpos/qvel), so replay retains the sampled scene.
Placement sampling keeps LIBERO's collision checks; initial success, nonfinite
states and objects falling more than 10cm during settling are rejected. This is
not a guarantee of reachability or task solvability: inspect the gallery before
using a large scale. The bank manifest records sampling attempt counts/ranges.

Output folders receive `_range2`, `_range1p5`, etc. automatically, even when
`output_name` is set. `initial_states/<worker>.json` inside each eval output
records its bank path/hash; changing a bank on resume is refused. Banks live in
`initial_state_banks/`, with keys including scale, episodes, task set, seed,
source BDDL hashes and generator version. Keep banks for paired comparisons.

The capture script follows the same YAML setting and generator:

Its default output folder also gains `_range2`, `_range1p5`, etc. automatically;
an already matching label is not duplicated. An explicit `--output-dir` is
treated as a literal path and is not renamed.

```bash
./capture_initial_frames.sh --label range2
# Or preview a scale without changing the eval YAML:
./capture_initial_frames.sh --random-range 2.0 --label range2_preview
```

At the same task/episode/seed/scale the capture and evaluation use the same
initial scene, regardless of GPU worker packing. Changing the episode count or
task subset may create a different bank file, but sampling each task starts
from the same seeds. Range expansion includes the original region; it does not
restrict samples to outside the original support or prove OOD relative to all
pretraining data.
