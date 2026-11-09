# Slurm Desk

Slurm Desk is part of the unified Sync UI. Run it on a Slurm login server and
open the **Slurm Desk** tab alongside server transfers and Hugging Face.

```bash
tmux new -s sbd-desk
bash sync_server_ui.sh
# Ctrl-b d detaches; tmux attach -t sbd-desk returns to the server.
```

Forward **8765** and open `http://127.0.0.1:8765/slurm/`. To select another port,
use `bash sync_server_ui.sh --port 8770`. The UI binds only to loopback.
`slurm_monitor_ui.sh` remains a compatibility alias for the same unified app.
Only one monitor owns `.slurm_monitor`; prompts and receipts survive restarts.
An existing standalone monitor can be reused with `--reuse-slurm-port 8767`.

## Jobs and completion instructions

Running allocation and individual training cards show used, free and total VRAM
(GiB) for their allocated GPUs, including the node and physical GPU index. Each
card explicitly labels this as the GPU total including all shared trainings,
rather than the individual trainer's memory. Finished and pending cards do not
display current GPU readings. It refreshes
about every 15 seconds in a separate background worker so slow telemetry does
not block the job list. Read-only `scontrol` and SSH `nvidia-smi` queries require
compute-node SSH access; no extra `srun` step or GPU allocation is created.
Failed node queries retain the last reading marked as delayed with its timestamp.

- **개별 학습** cards track trainer attempts separately from the Slurm GPU
  allocation cards. Each trainer has its own saved prompt, receipt and automatic
  completion notification, even while its srun allocation remains RUNNING.
  Grouped VSA/Predictor/Terminator workers report independently. Retrying creates
  a new attempt; earlier prompts and receipts remain attached to the old one.
- Trainer exits are recorded under `.cache/training_events`. Existing persistent
  shell runs are also discovered from their `training_status.json` without
  restarting training. A confirmed allocation end without a trainer exit record
  is **INTERRUPTED**, never assumed successful. Missing logs or a failed Slurm
  query alone do not establish completion. Training cards do not cancel a shared
  GPU allocation; use its separate allocation card for that action.

- Running and pending allocations for the current Unix user appear first;
  completed, failed and cancelled allocations appear below. By default the
  initial history query covers three days. Observed jobs remain in local state.
- Titles use the output-folder basename from job logs. A grouped training job
  lists all discovered model folders. A pending eval without a log can use its
  unique frozen config snapshot within five seconds before submission; the UI
  identifies that source. It never guesses from today's edited YAML. If a name
  cannot be established, the job name/ID appears and a manual label is available.
- New jobs and individual training attempts receive this saved default prompt,
  with **종료 시 자동 전송** already enabled:
  `실패 시 원인 분석 후 해결한 뒤 이어학습 혹은 새로 학습할 수 있도록 재제출 (생성물이 방해되면 생성물도 삭제후 재제출). 저장공간부족으로 인한실패일 경우 재제출 금지.`
  Edit and save any job's instructions or turn off automatic delivery to override
  its preset. Explicit edits persist across polling and restarts.
  Empty prompts never send. All terminal outcomes can trigger the saved
  prompt, including failures; the actual Slurm state and exit code are attached.
- Historical jobs do not auto-send merely because a prompt is saved. Use their
  explicit **지금 전송** button. Failed/ambiguous sends require manual retry.
- The original prompt is passed verbatim, followed by a clearly labelled JSON
  context containing the job ID, state, exit code, output paths and log paths.
  Logs and scheduler fields are reference data, not executable shell text.
- **접수 성공** means `codex queue` returned success. It establishes queue
  acceptance, not proof that the model read the message or finished the work.
  Read the target Codex conversation for its response and results.
- **Job 취소** asks for confirmation, rechecks the job's owner and submission
  identity in the live queue, then calls `scancel` for that individual job only.
  Its state changes only after Slurm confirms the outcome. Cancelling does not
  change saved instructions or auto-send settings: cancellation can still trigger
  the saved completion prompt, with `CANCELLED` attached as the actual state.
- Training cards show the latest step and target from training logs and small
  checkpoint JSON files. Abbreviated logs such as `60K` are labelled **약**;
  they do not establish an exact step count. Grouped models use separate child
  logs. Pending jobs label any prior training record; evaluation jobs omit this
  training display. Hover over a step to see its source and record time.

## Shared GPU memory protection

New common LeRobot trainers (VSA / pi05 PT and FT, cyclic pi05, Predictor and
Terminator) launched through the current srun launcher use newest-first GPU
protection. This also applies to new training inside an older held allocation;
already running Python trainers retain their original behavior and are not
restarted or signalled by installing this code. Arbitrary external programs and
multi-GPU/DDP/MIG launches are not covered by this trainer integration.

Enable protection in `lerobot/examples/libero/configs/servers/rllab.yaml`:

```yaml
srun_gpu_memory_guard: true
```

No other settings are required. Built-in defaults reserve 8 GiB, stop the newest
managed trainer below 4 GiB free, require an 8 GiB minimum startup budget, and
check every second. Retries require at least 2 GiB more budget (or 10% more,
whichever is larger). Advanced YAML overrides remain supported internally.

The supervisor leaves reserved headroom, applies a PyTorch allocator ceiling to
the actual trainer, and serializes initial startup until the previous trainer's
first optimizer update. Under memory pressure it stops only the newest managed
trainer's process group, including its DataLoader workers. The GPU allocation
and its interactive shell stay held. Other users' processes are never targets.

The individual training card becomes **VRAM 대기** while the supervisor waits.
This is nonterminal, so completion prompts do not cause another submission.
Retries require a strictly larger budget, preventing immediate OOM loops. A
complete local checkpoint restores model, optimizer, scheduler and RNG state;
the submitted step goal, dataset path and operational settings stay intact.
Without a checkpoint, only a fresh output owned by this attempt can be renamed
to `.gpu-paused-…` and restarted. Files are preserved. Unowned/incomplete recovery
state requires attention rather than deletion. Storage-full errors never retry.

This is best-effort protection, not hardware memory partitioning. CUDA/library
allocations outside the PyTorch allocator and bursts faster than polling can
still exhaust memory. Conservative headroom may delay a model that would fit
with a smaller reserve. Neither batch size nor checkpointing is changed.

Guard status and attempts live in `.cache/gpu_memory_guard`; coordination uses
a private per-user, per-allocation/GPU directory under `/tmp`. For a one-off
opt-out use `SBD_GPU_MEMORY_GUARD=0 ./submit_train.sh` in the training directory.

## Target Codex session

Use **연결 설정** to select the session UUID, `CODEX_HOME`, CLI executable and,
if needed, a `unix://`, `ws://` or `wss://` app-server endpoint. The CLI must
reach the Codex server that owns the target conversation. A local session-record
check is shown separately from message delivery.

On the first launch, `CODEX_THREAD_ID` and `CODEX_HOME` are used when available.
Saved settings take precedence on later launches, so restarting from another
terminal does not silently reroute armed notifications. Explicit CLI overrides:

```bash
bash sync_server_ui.sh \
  --thread-id '<session UUID>' \
  --codex-home /path/to/codex-home \
  --codex-command /path/to/codex
```

Set the desired actions in each job's prompt, for example the exact Notion
database and SR column to update, how to verify all expected task results, and
what to report on failure. This tool does not independently modify Notion or
launch training/evaluation. Cancellation requires the explicit UI action above.

## Persistence and delivery

The private `.slurm_monitor/state.json` stores connection settings, prompts,
armed jobs and delivery receipts; it is ignored by Git. To use another state
directory, pass `--slurm-state-dir` (`--state-dir` is also accepted). One process may own a state directory at a time.
Before a queue call, the attempt is persisted. On a crash during delivery, that
attempt becomes **접수 여부 확인 필요** on restart; automatic retries are disabled
because the queue may already have accepted the message. Keep the monitor and
the target Codex server available. Closing the browser does not stop monitoring.

Status polling uses `squeue` plus `sacct`; a missing queue entry or a failed
query is never interpreted as job completion. A running/pending queue entry
overrides a stale accounting terminal state, including requeued jobs.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --noconftest -q tools/slurm_monitor/tests
```
