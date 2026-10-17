# SkillVLA data build

`./submit_build_all.sh` builds the selected dataset. Optional Predictor relabeling
is a separate command and never changes the normal build.

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
