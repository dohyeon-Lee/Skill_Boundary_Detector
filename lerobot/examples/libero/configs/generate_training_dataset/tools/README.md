# Dataset conversion and inspection tools

Standalone tools formerly located directly in `examples/libero/`:

- `convert_libero_data_to_lerobot_dp.py`: RLDS to LeRobot with raw OSC actions.
- `convert_libero_stateaction_to_lerobot_dp.py`: RLDS to LeRobot with state-delta actions.
- `convert_lerobot_to_stateaction.py`: convert an existing LeRobot dataset to state-delta actions.
- `inspect_dataset.py`: inspect dataset metadata, statistics and example episodes.

Run from `lerobot/` with the project environment, for example:

```bash
../.venv/bin/python examples/libero/configs/generate_training_dataset/tools/inspect_dataset.py \
  --dataset_dir /path/to/dataset
```

Moving these tools does not change their conversion algorithms, defaults or
output conventions. The RLDS converters still require TensorFlow Datasets.
