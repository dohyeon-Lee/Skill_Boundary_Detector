# MimicGen Coffee simulator setup

From the repository root:

```bash
bash setup_env.sh --mimicgen-only
bash setup_env.sh --mimicgen-only --check
```

The setup uses Python 3.12 in `.venv_mimicgen`, the official MimicGen revision
in `source_revision.txt` under `third_party/mimicgen`, and the exact package
versions in `requirements.lock.txt`. The dataset's robosuite 1.4.1 OSC interface
is retained. MuJoCo 3.6.0 matches this project's working modern simulator runtime;
this is not a claim of exact dynamics parity with the original dataset's engine.

The default check runs on CPU without creating an OpenGL context: it validates
the source, package versions and Coffee assets, then constructs Coffee D0 with
Panda/PandaGripper, resets it, steps twice, and checks the two camera calibration
matrices and observation/action shapes. It does not claim visual alignment
between reconstructed cameras and the Hugging Face dataset; that check belongs
to the subsequent data-conversion integration.

To additionally test actual 256x256 RGB images, run inside an allocated GPU shell:

```bash
MUJOCO_GL=egl bash setup_env.sh --mimicgen-only --check --render
```

Software rendering is also supported with `MUJOCO_GL=osmesa` when the system has
OSMesa installed. Rendering failures report an error; the checker does not
silently substitute a different backend or allocate a Slurm GPU.

`--check` does not clone, install, or download packages. It runs the physics check,
so normal third-party runtime/JIT caches may be created. Existing source edits
are preserved and reported as an error. Existing virtual environments are reused;
none of the main, ABC, GR00T, or Molmo environments is changed.

This environment provides the simulator only. Coffee/CoffeePreparation source
and bundled assets are present, but the smoke test currently exercises Coffee D0.
It does not download episode data or policy checkpoints, install a VLA trainer,
or implement data conversion, DP/FSQ processing, or evaluation dispatch.
The direct Coffee simulator does not need robomimic, robosuite-task-zoo, or Torch;
MimicGen's full data-generation/policy-learning workflows need additional packages.

To include this optional simulator during a full base-environment setup, set
`SBD_SETUP_MIMICGEN=1`. Running plain `setup_env.sh` follows the existing behavior
of rebuilding the main environment; use `--mimicgen-only` to prepare only this one.

References:

- [MimicGen installation](https://mimicgen.github.io/docs/introduction/installation.html)
- [Coffee environment](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/envs/robosuite/coffee.py)
- [Coffee D0 dataset](https://huggingface.co/datasets/OliverHausdoerfer/demo_src_coffee_task_D0_robot_Panda_gripper_PandaGripper)
