# Slurm Desk

Run on a Slurm login server and open the HTML UI through SSH / VSCode port
forwarding. Python 3.10+ and the existing Slurm/Codex CLIs are sufficient; no
extra Python packages, GPU allocation or AI polling calls are needed.

```bash
tmux new -s slurm-monitor
bash slurm_monitor_ui.sh
# Ctrl-b d detaches; tmux attach -t slurm-monitor returns to the server.
```

Forward **8767** and open `http://127.0.0.1:8767`. To select another port, use
`bash slurm_monitor_ui.sh --port 8768`. The UI binds only to loopback.

## Jobs and completion instructions

- Running and pending allocations for the current Unix user appear first;
  completed, failed and cancelled allocations appear below. By default the
  initial history query covers three days. Observed jobs remain in local state.
- Titles use the output-folder basename from job logs. A grouped training job
  lists all discovered model folders. A pending eval without a log can use its
  unique frozen config snapshot within five seconds before submission; the UI
  identifies that source. It never guesses from today's edited YAML. If a name
  cannot be established, the job name/ID appears and a manual label is available.
- Expand a job, enter its completion instructions, enable **종료 시 자동 전송**
  and save. Empty prompts never send. All terminal outcomes can trigger the saved
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

## Target Codex session

Use **연결 설정** to select the session UUID, `CODEX_HOME`, CLI executable and,
if needed, a `unix://`, `ws://` or `wss://` app-server endpoint. The CLI must
reach the Codex server that owns the target conversation. A local session-record
check is shown separately from message delivery.

On the first launch, `CODEX_THREAD_ID` and `CODEX_HOME` are used when available.
Saved settings take precedence on later launches, so restarting from another
terminal does not silently reroute armed notifications. Explicit CLI overrides:

```bash
bash slurm_monitor_ui.sh \
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
directory, pass `--state-dir`. One process may own a state directory at a time.
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
