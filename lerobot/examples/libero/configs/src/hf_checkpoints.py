#!/usr/bin/env python3
"""Upload finished training checkpoints to a public Hugging Face model repo (``hf_checkpoint_repo``).

Run it next to training, e.g. in tmux on RunPod:  ``bash hf_sync.sh watch [--keep 3 --protect 050000,100000]``

* Scans every ``outputs`` / ``outputs_*`` folder of the checkout (or ``--root``) for runs, i.e.
  ``<root>/.../<run>/checkpoints/<step>/``. Symlinked runs (start checkpoints linked in from the
  storage volume) are not ours and are skipped.
* A step is uploaded once it is complete: LeRobot re-points ``checkpoints/last`` only after a save has
  fully finished, so every numeric step up to the one ``last`` points to is complete. The repo path
  mirrors the checkout: ``outputs_filtered/pi05_PT/<run>/checkpoints/030000/{pretrained_model,training_state}``
  (``last`` itself, wandb/ and logs stay local). What is already on the Hub is never re-sent.
* The first upload also writes the model card and license files (Gemma notice, DINOv3 license).
* Steps pulled from the Hub (``hf_sync.sh pull --checkpoints``, recorded in
  ``checkpoints/.hf_pulled`` by ``mark_pulled``, which also points ``last`` at the newest one so the
  run can be resumed) are never uploaded or deleted, locally or on the Hub. A run is ours once this
  machine has written a step of its own into it (a fresh run, or a pulled run that was resumed).
* ``--keep N`` keeps only the latest N steps per run **on the Hub** (``--protect`` steps always kept),
  for our own runs only: runs that are only on the Hub (other machines) or only pulled are left alone.
  The window is computed over local + Hub steps, so older steps are deleted from the Hub and never
  uploaded again (also after a restart). Hub history still holds deleted files; ``--squash`` rewrites
  history to free the space (irreversible).
* ``prune_local`` (``--prune-local``; on by default on RunPod via ``hf_watch_prune_local``) applies the
  same window to the local disk: older complete steps of our own runs are removed there too. The
  newest step (resume), steps still being written, protected steps, pulled steps and linked start
  checkpoints are never removed. Without ``--keep`` nothing is pruned.
* The first scan lists everything it will upload, delete on the Hub and delete locally, and asks once.
* Pod auto-terminate (RunPod only) is on when ``done_step`` (``--done-step``) and/or ``done_time``
  (``--done-time``) is given. The pod is terminated (``runpodctl``) when, on two scans in a row, the
  newest step of every run trained here is on the Hub and any of these holds:
  (A) every job submitted through src/submit_job.sh has stopped (finished or failed),
  (B) every run trained here has a step >= ``done_step`` on the Hub (0 = only (A)),
  (C) ``done_time`` has passed since the first local training job started.
  Training still running at (B) or (C) is cut off. Just before, the job logs and a summary (exit
  codes) go to the private ``log_repo`` (the dataset repo) under runpod_logs/<time>_<pod>/; after
  three failed log uploads the pod is terminated anyway.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

STEP = re.compile(r"^\d+$")
PULLED_MARKER = ".hf_pulled"          # in <run>/checkpoints/: the steps that came from the Hub
REMOTE_STEP = re.compile(r"^(outputs(?:_[^/]*)?/.+/checkpoints/\d+)/")
_SKIP_DIRS = {"wandb", "logs", "videos", "eval", "__pycache__"}

GEMMA_NOTICE = (
    "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n\n"
    "These checkpoints are Model Derivatives of lerobot/pi05_base (PaliGemma + Gemma action expert).\n"
    "Use is subject to the Gemma Terms of Use and the Gemma Prohibited Use Policy:\n"
    "  https://ai.google.dev/gemma/terms\n"
    "  https://ai.google.dev/gemma/prohibited_use_policy\n"
)


def is_outputs_root(name: str) -> bool:
    return name == "outputs" or name.startswith("outputs_")


def output_roots(project_root: Path, only: str | None = None) -> list[str]:
    if only:
        return [only]
    return sorted(path.name for path in project_root.iterdir() if path.is_dir() and is_outputs_root(path.name))


def find_runs(project_root: Path, roots: list[str], max_depth: int = 5):
    """Yield (root, run_dir) for every real (non-symlinked) run with a checkpoints/ folder."""
    for root in roots:
        top = project_root / root
        if not top.is_dir():
            continue
        # followlinks=False: symlinked run dirs (start checkpoints from the volume) are not entered.
        for dirpath, dirnames, _ in os.walk(top):
            if "checkpoints" in dirnames:
                yield root, Path(dirpath)
                dirnames[:] = []
                continue
            depth = len(Path(dirpath).relative_to(top).parts)
            dirnames[:] = [] if depth >= max_depth else [
                name for name in dirnames if not name.startswith(".") and name not in _SKIP_DIRS
            ]


def complete_steps(run_dir: Path) -> list[Path]:
    checkpoints = run_dir / "checkpoints"
    last = checkpoints / "last"
    if not last.is_symlink():
        return []
    newest = Path(os.readlink(last)).name
    if not STEP.match(newest):
        return []
    return sorted(
        path for path in checkpoints.iterdir()
        if STEP.match(path.name) and path.is_dir() and not path.is_symlink() and int(path.name) <= int(newest)
    )


def pulled_steps(run_dir: Path) -> set[str]:
    marker = run_dir / "checkpoints" / PULLED_MARKER
    return set(marker.read_text().split()) if marker.is_file() else set()


def scan(project_root: Path, roots: list[str]) -> tuple[dict[str, Path], set[str]]:
    """({repo path: step dir} of our own complete steps, repo paths of pulled steps)."""
    own, pulled = {}, set()
    for root, run_dir in find_runs(project_root, roots):
        prefix = f"{root}/{run_dir.relative_to(project_root / root).as_posix()}/checkpoints/"
        from_hub = pulled_steps(run_dir)
        pulled |= {prefix + name for name in from_hub}
        for step in complete_steps(run_dir):
            if step.name not in from_hub:
                own[prefix + step.name] = step
    return own, pulled


def local_steps(project_root: Path, roots: list[str]) -> dict[str, Path]:
    """{repo path: step dir} for every complete local step this machine wrote."""
    return scan(project_root, roots)[0]


def mark_pulled(project_root: Path, steps: set[str]) -> None:
    """After a checkpoint pull: record the pulled steps of each run (watch never uploads or deletes
    them) and point ``checkpoints/last`` at the newest step so the run can be resumed. A run that
    already has steps of its own keeps its ``last``."""
    by_run: dict[Path, set[str]] = {}
    for path in steps:
        step_dir = project_root / path
        if step_dir.is_dir():
            by_run.setdefault(step_dir.parent, set()).add(step_dir.name)
    for checkpoints, names in sorted(by_run.items()):
        marker = checkpoints / PULLED_MARKER
        recorded = set(marker.read_text().split()) if marker.is_file() else set()
        recorded |= names
        marker.write_text("".join(f"{name}\n" for name in sorted(recorded, key=int)))
        present = sorted((path.name for path in checkpoints.iterdir() if STEP.match(path.name) and path.is_dir()), key=int)
        last = checkpoints / "last"
        if set(present) - recorded and (last.is_symlink() or last.exists()):
            print(f"  {checkpoints.parent}: has steps trained here; checkpoints/last left as is")
            continue
        if last.exists() and not last.is_symlink():
            print(f"  {checkpoints.parent}: checkpoints/last is not a link; left as is")
            continue
        if last.is_symlink():
            last.unlink()
        last.symlink_to(present[-1])
        print(f"  {checkpoints.parent}: checkpoints/last -> {present[-1]} (resume starts here)")


def remote_steps(api, repo: str) -> set[str]:
    try:
        files = api.list_repo_files(repo, repo_type="model")
    except Exception:  # noqa: BLE001 - a repo that does not exist yet has no steps
        return set()
    return {match.group(1) for path in files if (match := REMOTE_STEP.match(path))}


def plan_cleanup(steps: set[str], keep: int, protect: set[int]) -> list[str]:
    """Steps outside the window: per run keep the latest ``keep`` plus every protected step number."""
    if keep <= 0:
        return []
    by_run: dict[str, list[str]] = {}
    for path in steps:
        by_run.setdefault(path.rsplit("/", 1)[0], []).append(path)
    doomed = []
    for paths in by_run.values():
        ordered = sorted(paths, key=lambda path: int(path.rsplit("/", 1)[1]))
        for path in ordered[:-keep]:
            if int(path.rsplit("/", 1)[1]) not in protect:
                doomed.append(path)
    return sorted(doomed)


def plan_sync(local: set[str], remote: set[str], keep: int, protect: set[int],
              pulled: set[str] = frozenset()) -> tuple[list[str], list[str], list[str]]:
    """(upload, delete on the Hub, prune locally) for our own runs, i.e. the runs of ``local`` (steps
    this machine wrote). The per-run window spans local and Hub steps, so a step dropped for age is
    not re-uploaded on the next scan or after a restart, and both sides keep the same steps. Hub-only
    runs and ``pulled`` steps are never deleted."""
    own_runs = {path.rsplit("/", 1)[0] for path in local}
    remote_own = {path for path in remote if path.rsplit("/", 1)[0] in own_runs}
    outside = set(plan_cleanup(local | remote_own, keep, protect)) - set(pulled)
    return sorted(local - remote - outside), sorted(remote_own & outside), sorted(local & outside)


def _jobs_dir(project_root: Path) -> Path:
    return Path(os.environ.get("SBD_LOCAL_JOBS_DIR") or project_root / ".cache" / "local_jobs")


def _alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
    except (OSError, ValueError) as error:
        return isinstance(error, PermissionError)       # alive, owned by someone else
    return True


def local_jobs(project_root: Path) -> dict[str, int]:
    """Jobs started by src/submit_job.sh (scheduler: local): recorded / running / waiting."""
    counts = {"recorded": 0, "running": 0, "pending": 0}
    for record in _jobs_dir(project_root).glob("*.job"):
        counts["recorded"] += 1
        if _alive(record.with_suffix(".pid")):
            counts["pending" if record.with_suffix(".pending").exists() else "running"] += 1
    return counts


_DURATION = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$")


def parse_duration(text: str) -> float:
    """Seconds from ``48:00:00`` / ``2-00:00:00`` (like Slurm's --time) or ``12h``, ``90m``, ``1.5d``, ``30s``."""
    value = text.strip().lower()
    seconds = None
    if match := _DURATION.match(value):
        seconds = float(match.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    else:
        days, _, clock = value.rpartition("-")
        parts = clock.split(":")
        if len(parts) == 3 and all(part.isdigit() for part in parts) and (not days or days.isdigit()):
            seconds = int(days or 0) * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if not seconds or seconds <= 0:
        raise ValueError(f"--done-time {text!r}: use e.g. 48:00:00, 2-00:00:00, 12h, 90m or 1d")
    return seconds


def training_started(project_root: Path) -> float | None:
    """When the first job submitted through src/submit_job.sh started (epoch seconds)."""
    starts = []
    for record in _jobs_dir(project_root).glob("*.job"):
        for line in record.read_text().splitlines():
            if line.startswith("started="):
                try:
                    starts.append(datetime.fromisoformat(line.split("=", 1)[1].strip()).timestamp())
                except ValueError:
                    pass
    return min(starts, default=None)


def pod_done(found: dict[str, Path], uploaded: set[str], done_step: int, jobs: dict[str, int], *,
             elapsed: float = 0.0, done_time: float = 0.0) -> str | None:
    """Why the pod may be terminated now, or None (conditions A-C in the module docstring)."""
    runs: dict[str, list[int]] = {}
    for path in found:
        run, step = path.rsplit("/", 1)
        runs.setdefault(run, []).append(int(step))
    on_hub: dict[str, list[int]] = {}
    for path in uploaded:
        run, step = path.rsplit("/", 1)
        on_hub.setdefault(run, []).append(int(step))
    newest_uploaded = all(max(steps) in on_hub.get(run, []) for run, steps in runs.items())
    if jobs["recorded"] and not jobs["running"] and not jobs["pending"] and newest_uploaded:
        return "training has stopped (finished or failed) and its checkpoints are on the Hub"
    if (done_step > 0 and runs and not jobs["pending"] and newest_uploaded
            and all(max(on_hub[run]) >= done_step for run in runs)):
        return f"every run trained here has step {done_step:06d} on the Hub"
    if done_time > 0 and elapsed >= done_time and newest_uploaded:
        return f"{done_time / 3600:g} h of training have passed"
    return None


def upload_logs(api, repo: str, project_root: Path) -> str:
    """Copy the local jobs' logs and a summary to the private ``repo`` (runpod_logs/<time>_<pod>/)."""
    dest = f"runpod_logs/{time.strftime('%Y%m%d-%H%M%S')}_{os.environ.get('RUNPOD_POD_ID', 'local')}"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        summary = ["id\tname\tstate\tlog"]
        for record in sorted(_jobs_dir(project_root).glob("*.job")):
            fields = dict(line.split("=", 1) for line in record.read_text().splitlines() if "=" in line)
            exit_file = record.with_suffix(".exit")
            state = ("running" if _alive(record.with_suffix(".pid"))
                     else f"exit={exit_file.read_text().strip()}" if exit_file.is_file() else "stopped")
            summary.append(f"{record.stem}\t{fields.get('name', '')}\t{state}\t{fields.get('output', '')}")
            for key in ("output", "error"):
                log = Path(fields.get(key, ""))
                if log.name and log.is_file() and not (folder / log.name).exists():
                    shutil.copyfile(log, folder / log.name)
        (folder / "jobs.tsv").write_text("\n".join(summary) + "\n")
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=dest, repo_type="dataset",
                          commit_message=f"Add {dest}")
    return dest


def terminate_pod() -> None:
    """Terminate this RunPod pod (everything but network volumes is deleted)."""
    pod = os.environ["RUNPOD_POD_ID"]
    for command in (["runpodctl", "pod", "delete", pod], ["runpodctl", "remove", "pod", pod]):
        if subprocess.run(command, check=False).returncode == 0:
            return
    raise RuntimeError("runpodctl could not terminate this pod; terminate it from the RunPod console.")


def model_card(repo: str) -> str:
    return f"""---
license: other
license_name: gemma-terms-of-use-and-dinov3-license
license_link: LICENSE.md
tags:
- robotics
- lerobot
- vision-language-action
---

# {repo.split("/", 1)[-1]}

Training checkpoints of the SkillVLA project (pi0.5 baselines and SkillVLA models), uploaded as they
are written so that runs can be inspected or resumed on another machine.

Layout mirrors the training output tree:

```
<outputs_root>/<group>/<run>/checkpoints/<step>/
├── pretrained_model/   weights, policy/train configs, normalization stats
└── training_state/     optimizer, scheduler, RNG state and step (for resuming)
```

## License

These are derivative works; the upstream terms apply and travel with every copy:

* **Gemma Terms of Use** — every checkpoint is fine-tuned from `lerobot/pi05_base`
  (PaliGemma + Gemma action expert). See `NOTICE` and https://ai.google.dev/gemma/terms,
  including the Prohibited Use Policy.
* **DINOv3 License** — SkillVLA checkpoints also contain DINOv3 vision weights
  (`facebook/dinov3-*-pretrain-lvd1689m`). See `LICENSE-DINOv3.md`; publications using them must
  acknowledge DINOv3.
* Training data: LIBERO (CC BY 4.0) — Liu et al., "LIBERO: Benchmarking Knowledge Transfer for
  Lifelong Robot Learning", 2023.
"""


def ensure_card(api, repo: str, project_root: Path) -> None:
    """Create the public repo and, once, its card + license files."""
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    if "README.md" in set(api.list_repo_files(repo, repo_type="model")):
        return
    dinov3 = next((path for path in sorted((project_root / "models").glob("dinov3-*/LICENSE.md"))), None)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "README.md").write_text(model_card(repo))
        (folder / "NOTICE").write_text(GEMMA_NOTICE)
        license_text = "See NOTICE (Gemma Terms of Use) and LICENSE-DINOv3.md (DINOv3 License).\n"
        (folder / "LICENSE.md").write_text(license_text)
        if dinov3 is not None:
            shutil.copyfile(dinov3, folder / "LICENSE-DINOv3.md")
        else:
            (folder / "LICENSE-DINOv3.md").write_text(
                "DINOv3 License: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/blob/main/LICENSE.md\n"
            )
        api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                          commit_message="Add model card and license files")
    print(f"  model card + licenses → https://huggingface.co/{repo}")


def _size_gb(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1e9


def watch(project_root: Path, repo: str, *, root: str | None = None, interval: int = 300, once: bool = False,
          keep: int = 0, protect: set[int] | None = None, squash: bool = False, prune_local: bool = False,
          done_step: int | None = None, done_time: float | None = None, log_repo: str = "", dry_run: bool = False,
          assume_yes: bool = False, ask=input, api=None, jobs_state=None, terminate=None) -> None:
    terminate_when_done = done_step is not None or done_time is not None
    if terminate_when_done and terminate is None:
        if not os.environ.get("RUNPOD_POD_ID") or shutil.which("runpodctl") is None:
            raise SystemExit("--done-step/--done-time terminate the pod: only inside a RunPod pod ($RUNPOD_POD_ID, runpodctl).")
        terminate = terminate_pod
    done_step, done_time = done_step or 0, done_time or 0.0
    watch_started = time.time()
    jobs_state = jobs_state or (lambda: local_jobs(project_root))
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    protect = protect or set()
    roots = output_roots(project_root, root)
    print(f"watch: repo=https://huggingface.co/{repo} (public)  roots={roots}  every {interval}s"
          f"{f'  keep={keep}' if keep else ''}{f'  protect={sorted(protect)}' if protect else ''}"
          f"{'  local-prune=on' if prune_local and keep else ''}")
    if prune_local and not keep:
        print("  (local prune needs --keep N; nothing will be removed locally)")
    if terminate_when_done:
        print(f"  pod auto-terminate once uploaded: training stopped{f' | step {done_step:06d}' if done_step else ''}"
              f"{f' | {done_time / 3600:g} h of training' if done_time else ''}"
              f"; logs -> {f'{log_repo} (private)' if log_repo else 'nowhere (hf_dataset_repo unset)'}")
    if not dry_run:
        ensure_card(api, repo, project_root)
    uploaded = remote_steps(api, repo)
    first = True
    done_seen = False
    log_failures = 0
    while True:
        found, pulled = scan(project_root, roots)
        to_upload, to_delete, to_prune = plan_sync(set(found), uploaded, keep, protect, pulled)
        if not prune_local:
            to_prune = []
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if to_upload:
            print(f"[{stamp}] {len(to_upload)} new checkpoint(s):")
            for path in to_upload:
                print(f"    {path}")
        if to_delete:
            print(f"[{stamp}] {len(to_delete)} old Hub checkpoint(s) outside keep={keep}:")
            for path in to_delete:
                print(f"    {path}")
        if to_prune:
            print(f"[{stamp}] {len(to_prune)} old local checkpoint(s) outside keep={keep}:")
            for path in to_prune:
                print(f"    {found[path]}")
        if first and (to_upload or to_delete or to_prune) and not assume_yes and not dry_run:
            total = sum(_size_gb(found[path]) for path in to_upload)
            parts = [f"{total:.1f} GB 를 공개 저장소에 올리기"] if to_upload else []
            parts += [f"Hugging Face 체크포인트 {len(to_delete)}개 지우기"] if to_delete else []
            parts += [f"로컬 체크포인트 {len(to_prune)}개 지우기"] if to_prune else []
            question = f"  {' + '.join(parts)}: 진행할까요? [y/N]: "
            if ask(question).strip().lower() not in {"y", "yes"}:
                print("취소했습니다.")
                return
        for path in to_upload:
            if dry_run:
                print(f"  [dry-run] upload_folder({found[path]} -> {repo}:{path})")
                continue
            print(f"  uploading {path} ...")
            api.upload_folder(repo_id=repo, folder_path=found[path], path_in_repo=path, repo_type="model",
                              commit_message=f"Add {path}")
            uploaded.add(path)
        for path in to_delete:
            if dry_run:
                print(f"  [dry-run] delete {repo}:{path}")
                continue
            print(f"  deleting old step on the Hub: {path}")
            api.delete_folder(path_in_repo=path, repo_id=repo, repo_type="model", commit_message=f"Remove {path}")
            uploaded.discard(path)
        if to_delete and squash and not dry_run:
            api.super_squash_history(repo_id=repo, repo_type="model")
            print("  squashed Hub history (deleted steps no longer use storage)")
        for path in to_prune:
            if dry_run:
                print(f"  [dry-run] rm -r {found[path]}")
                continue
            print(f"  removing old local step: {found[path]}")
            shutil.rmtree(found[path])
        first = False
        if terminate_when_done and not dry_run:
            elapsed = time.time() - (training_started(project_root) or watch_started)
            reason = pod_done(found, uploaded, done_step, jobs_state(), elapsed=elapsed, done_time=done_time)
            if reason and done_seen:
                logs_ok = True
                if log_repo:
                    try:
                        print(f"[{stamp}] logs -> {log_repo}:{upload_logs(api, log_repo, project_root)}")
                    except Exception as error:  # noqa: BLE001 - never keep a finished pod up forever
                        log_failures += 1
                        logs_ok = log_failures >= 3
                        print(f"  log upload failed ({error}); "
                              f"{'terminating anyway' if logs_ok else 'retrying at the next scan'}")
                if logs_ok:
                    print(f"[{stamp}] {reason}: terminating this pod.", flush=True)
                    terminate()
                    return
            elif reason:
                done_seen = True
                print(f"[{stamp}] {reason}; the pod is terminated at the next scan.")
            else:
                done_seen = False
        if once or dry_run:
            return
        time.sleep(interval)
