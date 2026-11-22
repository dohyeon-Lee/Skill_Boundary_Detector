# Yonsei workflow dispatch

Run any executable shell workflow under `lerobot/skillVLA` with `--yonsei`:

```bash
./submit_eval.sh --yonsei
./submit_train.sh --yonsei
./submit_build_all.sh --yonsei
```

The option is handled before local argument parsing, config snapshots, or Slurm
submission. Ordinary commands and sourced shell helpers keep their existing
behavior. Existing workflow arguments and config environment variables still
work; `--` ends remote-option parsing. `--yonsei --12345` selects an allocation
**on Yonsei**, rather than an allocation on the sending server.

Connection and GPU/partition/QoS settings come from
`lerobot/skillVLA/servers/yonsei.yaml`: `remote_host`, `remote_project_root`,
`train_gres`, and `slurm_profile`. Existing transfer overrides `YONSEI_HOST`,
`YONSEI_SCRATCH2_BASE`, or `YONSEI_PROJECT_ROOT` can select another workspace.
The target workspace must already have a working `.venv`, SSH access, rsync,
and Slurm. Model-specific environments such as `.venv_groot_lerobot` are reused
when installed there.

SkillVLA PT/FT simulator evaluation also reads `server_eval_num_gpus` and
`server_eval_max_workers_per_gpu` from this server profile, taking priority over
the experiment YAML's `eval_num_gpus` / `eval_max_workers_per_gpu`. Yonsei defaults
to four separate one-GPU jobs with one worker each; the experiment's local packing
remains unchanged. `--yonsei --JOB_ID` uses that allocation's single GPU instead.
These server overrides do not change episode count, replanning or training.

Before executing, the sender freezes the current code (including uncommitted
changes), YAMLs, experiment arguments, and selected input paths. The native
training/evaluation config resolvers identify required datasets/checkpoints.
Utility/report YAMLs and explicit file arguments supply additional input paths.
Inputs already in the remote workspace are reused; missing inputs are copied
with rsync through a staging directory and published without replacing an
existing destination. Checkpoint selectors such as `last` retain their native
meaning; use a numeric step to pin a specific save. Inputs should be complete
before starting a transfer.

Every launch uses its own code checkout under
`<remote_project_root>/.cache/yonsei_launches/<id>/project`, so changing local or
remote source files cannot move a queued shell reader's offsets. Data, models,
and training outputs use the destination workspace through links. Eval reports
written beside source files live in that launch's `project/.../eval/outputs`.
Its `console.log` and `status.json` track the launcher; Slurm tracks the jobs
it submits. Launcher completion does not mean training/evaluation completion.
Local receipts are under `.cache/yonsei_launches/<id>/`.

## Automatic result collection and Yonsei Desk

For resolved evaluation workflows, `--yonsei` automatically starts a persistent
local watcher. It tracks actual Slurm jobs (including all array elements), not
just the submitting shell. After successful completion it copies that run's
HTML, videos and statistics back to the corresponding local eval output folder.
The received folder always ends in `_yonsei`. If that name already belongs to
another run, a launch ID is inserted before `_yonsei`; existing results are
preserved. Training checkpoints and datasets are not automatically downloaded.

Transfers resume after connection failures. Results are staged in a hidden
directory, verified with rsync checksums, then published atomically. Absolute
references within reports/JSON are relocated, and `yonsei_origin.json` records
the source. Remote results remain in place. A failed evaluation is shown as
failed and is not automatically resubmitted. Transfer failures show a retry
status and preserve their partial download. The watcher survives terminal and
browser closure and catches up when this server reconnects.

The unified Desk exposes **Yonsei Desk** at `/yonsei/` (default:
`http://127.0.0.1:8765/yonsei/`). It shows submitted Slurm IDs, queue/execution
state, completed episode counts, current success rate, per-model/task outcomes,
and result collection status. Completed local reports can be opened directly;
video range requests are supported. Episode statistics come from explicit
completed outcomes, not from the rollout's frame counter. Refreshing the Desk
does not restart an evaluation or submit GPU jobs.

Watcher receipts are in `.cache/yonsei_launches/<id>/`; consolidated state and
daemon logs are in `.cache/yonsei_desk/`. Only workflows launched with `--yonsei`
are tracked, not unrelated jobs belonging to other users or old manual runs.

The snapshot replaces sending-server GPU types, partition/QoS/node constraints,
scheduler mode, and project paths with Yonsei settings. CPU stages retain CPU
resources from the Yonsei profile. Checkpoint offloading back to Yonsei is
disabled on Yonsei itself. Credentials, local CUDA assignments, and the sending
server's Slurm allocation are not forwarded. No Pod is created and no RunPod/HF
upload is needed.

To inspect without launching:

```bash
./submit_eval.sh --yonsei-plan   # local resolution/transfer plan only; no SSH
./submit_eval.sh --yonsei-check  # also check SSH/environment/missing inputs; no writes/jobs
```

A dry-run option belonging to the underlying workflow still works remotely.
For example `STAGE1_EVAL_DRY_RUN=1 ./submit_eval.sh --yonsei` stages the recipe
and runs the native evaluator's submission dry run on Yonsei.

New `.sh` workflows should source `lerobot/skillVLA/remote.sh` at the beginning,
inside `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then ...; fi`. The coverage check
in `tools/remote/tests/test_yonsei.py` catches omitted hooks.
