# LIBERO anchor-relative EEF dataset

This is an opt-in conversion path. Existing LIBERO datasets, ABC datasets,
`meta/stats.json`, and the default Diffusion Policy action pipeline are not modified.

## Representation and alignment

The canonical LIBERO action is the normalized 7D robosuite OSC command
`[delta_position(3), delta_axis_angle(3), gripper]`. The configured controller applies
position scale `0.05 m`, rotation scale `0.5 rad`, and left/world rotation composition:

```text
p_target = p_current + 0.05 * delta_position
R_target = Exp(0.5 * delta_axis_angle) @ R_current
```

The source collector steps the simulator first and then records the returned observation
with that action. Thus source row `t` is `(post_action_observation[t], action[t])`. The
builder makes a behavior-cloning row from source `observation[t]` and source `action[t+1]`;
it drops the first source action and final source observation in every episode. This shift
provides the actual pre-action pose, so simulator replay is not needed.

The derived dataset stores the absolute command
`[p_target(3), target_axis_angle_rotation_vector(3), gripper]`. At sampling time the policy
preprocessor maps every future target in a chunk to one current-observation anchor:

```text
delta_p = p_target - p_anchor
delta_R = Log(R_target @ R_anchor.T)
```

`meta/action_contract.json` records these conventions. The separate
`meta/relative_action_stats.json` contains the training distribution across all valid
chunk offsets. The ordinary dataset stats remain available for non-relative consumers.

## Build

Edit `libero_anchor_relative_dataset_config.yaml`, then run:

```bash
./build_libero_anchor_relative_dataset.sh
```

For a small validation build or one configured dataset:

```bash
ANCHOR_RELATIVE_ONLY=libero_10_full_full_rel MAX_EPISODES=2 \
  ./build_libero_anchor_relative_dataset.sh
```

`MAX_EPISODES` writes a deliberately partial dataset under the configured output name;
run the later full build with `FORCE=1`.

`FORCE=1` rebuilds the selected derived output. It never deletes the source dataset.
Use the Slurm submitter for a full build:

```bash
./submit_build_libero_anchor_relative.sh

# Or only one configured output
ANCHOR_RELATIVE_ONLY=libero_10_full_full_rel \
  ./submit_build_libero_anchor_relative.sh
```

The submitter creates one balanced episode array with `convert_num_shards` tasks per
dataset. Each array task persists a valid LeRobot checkpoint every
`convert_checkpoint_episodes` episodes (20 by default), then packs those checkpoints into
one array shard. A dependent `afterok` job starts only when every shard succeeds, remuxes
the shards without re-encoding, validates the episode/frame/video counts, and computes
`relative_action_stats.json` once. The final output names remain
`libero_10_full_full_rel` and `libero_90_full_full_rel`. Intermediate data lives under
`_libero_anchor_relative_checkpoints/<output-name>/` and
`_libero_anchor_relative_shards/<output-name>/`. Checkpoints are deleted only after their
array shard validates, and shards are deleted only after the final aggregate validates;
the `convert_keep_checkpoints` and `convert_keep_shards` settings can retain them.

`portable_h264` performs an actual 256x256 one-frame encode on the GPU assigned to each
array task. It selects `h264_nvenc` only when that encode and flush succeed; otherwise it
falls back to software `h264`. Both paths emit the same canonical H.264 dataset feature,
so shards built on different NVIDIA generations remain merge-compatible. Streaming
encoding removes the temporary PNG write/read round trip, while a queue sized above the
largest selected episode prevents frame drops. Stored video durations are checked against
every episode length before a shard receives its completion manifest.

The jobs are submitted with Slurm requeue enabled. If a GPU job is requeued, completed
episode checkpoints are detected by their manifests and only its interrupted checkpoint
is rebuilt. If the cluster ends the job instead of requeueing it, submit the same command
again; completed array shards and checkpoints are skipped automatically. With
`convert_replace_incomplete_output: true`, an old/partial final folder is automatically
replaced only after all new shards validate, so the normal submit command does not need
`FORCE=1` and the partial folder does not need to be manually deleted. Explicit `FORCE=1`
still requests a hard final-output rebuild. Use `RESET_SHARDS=1` only when an intentional
full rebuild of all intermediate shards is required.

The submitter refuses to overlap another LIBERO-relative builder by default because two
builders can target the same final directory. `DRY_RUN=1` prints the array/dependency
commands without submitting them. Neither a checkpoint, shard, nor final dataset is
accepted as complete until its manifest and frame/video counts validate.

## Diffusion Policy training

Select a derived dataset and enable `dp_eef_relative: true` in
`configs/train_skills/DP/dp_config.yaml`. This mode requires `dp_n_action_steps: 1`: each
new model call anchors the predicted chunk to the current live EEF pose, and its first
prediction is converted back to a normalized OSC input. `dp_relative` is the independent
ABC joint-relative path and cannot be enabled at the same time.
