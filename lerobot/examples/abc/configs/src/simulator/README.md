# ABC Simulator environment

The official `amazon-far/abc` source is pinned in `third_party/abc` and installed
editable into `.venv_abc_sim`. This environment is isolated from the main SBD,
GR00T, and MolmoAct2 environments because the official simulator pins its own
PyTorch 2.11 / CUDA 12.8 / MuJoCo-Warp stack.

From the project root:

```bash
bash setup_env.sh --abc-sim-only
bash setup_env.sh --abc-sim-only --check

# Include ABC Sim while rebuilding the main environment on a new server.
SBD_SETUP_ABC_SIM=1 bash setup_env.sh
```

The installer prepares code, Python dependencies, the minimal LeRobot PI0.5
inference runtime, all standard simulator asset packages, and both RoboCasa
object packs. A successful `--check` verifies the official package markers,
PI0.5 imports, extracted paths, RoboCasa directories, and an actual asset-free
MuJoCo step.

Episode datasets, policy checkpoints, and standalone DINO weights remain
explicit operations. For example, a task dataset can be prepared with:

```bash
cd third_party/abc
../../.venv_abc_sim/bin/python prepare.py --sim-data sim_put_the_plastic_bottles_in_the_bin
```

The source revision and every non-editable Python package are pinned, while an
existing dirty checkout at another revision is rejected instead of overwritten.

`resolver.constraints.txt` keeps `mjviser` at 0.0.16. Its next release requires
the MuJoCo 4 `mujoco-core` distribution, which overwrites files from ABC's
explicitly pinned MuJoCo 3.8 package and breaks its Python ABI. The environment
checker performs a real asset-free MuJoCo step so this conflict cannot pass as a
successful installation.

## Closed-loop evaluation

`../../train_pi05/PT/eval/` contains the ABC Sim rollout runner for PI0.5. It supports the
PI0.5 adapter and a `hold` diagnostic policy. The PI0.5 adapter applies the ABC
dataset's 14D state/action quantile statistics, reproduces PI0.5 state-prompt
tokenization, maps the top/left/right cameras to the pretrained camera names,
and executes absolute 14D joint-action chunks.

```bash
cd lerobot/examples/abc/configs/train_pi05/PT/eval
./submit_eval.sh

# Resolve config, including the selected server profile, without submitting.
ABC_EVAL_DRY_RUN=1 ./submit_eval.sh

# Local config inspection only (does not construct a simulator).
../../../../../../../.venv_abc_sim/bin/python src/run_eval.py --dry-run
```

Edit `abc_sim_eval_config.yaml` to select explicit task aliases or `tasks: all`.
`all` is resolved from `dataset_ABC/abc_sim_all/native_dataset.json`, so it
means the 24 downloaded training tasks rather than unrelated simulator-only
tasks. `backend: auto` uses MJWarp where supported and falls back to sequential
MuJoCo for tasks with dynamic runtimes. Slurm partition, QoS, node list, and
excluded nodes always come from the active server profile (for Yonsei,
`lerobot/examples/abc/configs/servers/yonsei.yaml`).

The shipped config points at `models/pi05_base` for an integration smoke test.
That base model has not learned ABC tasks, so its success rate is not a useful
quality metric. After ABC post-training, replace `policy.checkpoint` with the
resulting `pretrained_model` directory; the observation/action interface stays
the same. `model_chunk_size: 30` and `action_chunk_size: 15` match the ABC
training/evaluation convention used by this repository.
