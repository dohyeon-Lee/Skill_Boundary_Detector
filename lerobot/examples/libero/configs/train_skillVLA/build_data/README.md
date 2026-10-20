# SkillVLA data build

`./submit_build_all.sh` builds the selected dataset. Optional Predictor relabeling
is a separate command and never changes the normal build.

## Optional camera UV targets

Set `focus_uv.enabled: false` to build without `eval_init_states.npz` or original
camera XML. Skill codes, boundaries, ISS, proprio/XYZ, and action/image data are
still built normally. XYZ/full-state Predictor targets and goal-XYZ Terminators
read canonical boundaries and endpoint states directly from the dataset parquet;
they do not require `skill_focus_uv.npz` or retained intermediate latents.

Keep `focus_uv.enabled: true` for UV prediction, foveated inputs, or Top-camera
patch supervision (including Both3 and Direct/Residual patch variants). These
objectives genuinely need camera calibration; disabling UV does **not** silently
disable their losses. Existing UV-enabled datasets retain their original target
lookup. Evaluation's episode-exact scene mapping is a separate requirement.

### Build the episode-exact map before enabling UV

The restored matcher lives in `oracle_matching/`. Set its
`oracle_matching_config.yaml` to the same `source_dataset` as the build YAML
(currently `libero_10_full_full`), then, from this `build_data` directory:

```bash
./oracle_matching/submit_oracle_matching.sh
```

After it finishes, the map is at
`<dataset_root>/skillvla_dataset/<source_dataset>/eval_init_states.npz`.
Set `focus_uv.enabled: true` in the build YAML and run `./submit_build_all.sh`.
Existing complete dataset/ISS/FSQ stages are reused; only missing UV targets
are generated. A LIBERO map is built by matching actions against the original
HDF5 demonstrations, without rendering. For an intentional foreground run use
`./oracle_matching/run_oracle_matching.sh`; LangGap matching needs a GPU
allocation because it renders candidate scenes. Existing maps are preserved
unless the matching YAML explicitly sets `overwrite: true`.

## Relabel with a standalone or Joint Predictor

In `train_skillVLA_config.yaml`, set the dataset family with `source_dataset` and
select an already-built dataset and a Predictor checkpoint:

```yaml
skill_relabel:
  source_run: FSQ333_contrastive_2000_pt_grounded_fov_new
  predictor_model: bs16_FSQ333_contrastive_2000_pt_grounded_fov_new_libero_90_full_full_predictor_xyz_fullvlm_pro
  checkpoint: "100000"  # or last: greatest complete numeric checkpoint
  batch_size: 4
```

Run `./submit_relabel.sh`. For a Joint model, use its **Joint run folder name**
in `predictor_model` and the desired Joint checkpoint. The resolver searches
`skillVLA_PT/Predictor`, `skillVLA_FT/Predictor`,
`skillVLA_FT/Joint`, and historical output layouts. Only the saved
Predictor submodule is loaded, not the VSA/Direct/Residual/Terminator networks.
The command follows the server's scheduler setting (Slurm or local/RunPod).

The dataset and Predictor must use the same FSQ code space. Different PT/FT
dataset tags are accepted only when their `fsq_source.json` records identify
the same labeling FSQ model and checkpoint; the source dataset's identity is
preserved. Paths recorded on another server are rebased using the dataset/run
tail. An explicit
`source_run` needs no current FSQ build metadata. The checkpoint's own
preprocessor/normalizer and local tokenizer are used. Predictors with a
skill-start proprio adapter receive the normalized boundary-frame state;
older image/language-only Predictors keep their original input contract.

Each canonical GT boundary is queried once. Only real entries in
`skill_sequence` and the matching `skill_latents.npz` codes/grid coordinates
are replaced. GT boundaries, skill counts/lengths, XYZ/UV targets, proprio,
actions, images, EOS/PAD, and normalization statistics are unchanged. This is
**not** segmentation randomization or replacement of XYZ by Predictor output.

The original dataset is never overwritten. The sibling output is named
`<source_run>_relabeled_<step>` (e.g. `_relabeled_100k`) and includes prediction
audit data and `relabel_provenance.json`. A matching completed output is
skipped; a conflicting existing output is rejected rather than overwritten.
Two different Predictor runs at the same step therefore cannot overwrite
each other's relabeled dataset.

To select the result for VSA/Terminator training, keep the original
`dataset.run` and set `dataset.relabeled: relabeled_100k` in the relevant
common/component YAML. Predictor training intentionally keeps original labels.
