# ABC workflows

ABC is a sibling of LIBERO under `lerobot/examples`. Its configuration tree owns
the ABC real-world/simulator dataset preparation, PI0.5 training and ABC Simulator
evaluation workflows.

- `configs/global_config.yaml`: ABC dataset/output roots
- `configs/servers/`: ABC-specific machine and Slurm settings
- `configs/generate_training_dataset/`: ABC downloads, native views and previews
- `configs/train_pi05/PT/`: PI0.5 post-training and simulator evaluation
- `configs/src/simulator/`: internal pinned simulator environment and asset installer
