#!/usr/bin/env bash
# Safety net for gpu_guard.sh: release jobs stuck in "job requeued in held
# state". The in-job release in gpu_guard.sh races against Slurm killing the
# batch script, so a job can end up held with nobody left to release it. This
# script runs OUTSIDE the job (cron on a login node), merges RECENT entries of
# the bad-node log into the job's exclude list, and releases it. Only recent
# entries count (GPU_GUARD_BAD_NODE_TTL_HOURS, default 6): nodes sometimes
# recover, and the append-only log must not become a permanent blacklist.
#
# Install (any login node that can run squeue/scontrol):
#
#     crontab -e
#     */2 * * * * /bin/bash /scratch2/mdorazi/Skill_Boundary_Detector/lerobot/examples/libero/configs/gpu_guard_watchdog.sh >> $HOME/.gpu_guard_watchdog.log 2>&1
#
# Can also be run once by hand to unstick currently-held jobs.
#
# Only touches jobs whose pend reason is job_requeued_in_held_state, so jobs
# held on purpose (scontrol hold -> JobHeldUser) are never released.

set -u
PATH=/usr/bin:/bin

BAD_NODE_FILE="${GPU_GUARD_BAD_NODE_FILE:-/scratch2/mdorazi/Skill_Boundary_Detector/.slurm_bad_gpu_nodes}"
BAD_NODE_TTL_HOURS="${GPU_GUARD_BAD_NODE_TTL_HOURS:-6}"

cutoff="$(( $(date +%s) - BAD_NODE_TTL_HOURS * 3600 ))"
bad_nodes=""
if [ -r "${BAD_NODE_FILE}" ]; then
  bad_nodes="$(while IFS="$(printf '\t')" read -r ts node _; do
      t="$(date -d "${ts}" +%s 2>/dev/null)" || continue
      [ "${t}" -ge "${cutoff}" ] && echo "${node}"
    done <"${BAD_NODE_FILE}" | sort -u | paste -sd, -)"
fi

squeue -h -u "${USER}" -t PD -o "%i %r" 2>/dev/null | while read -r jobid reason; do
  [ "${reason}" = "job_requeued_in_held_state" ] || continue

  current="$(scontrol show job "${jobid}" 2>/dev/null |
    tr ' ' '\n' | sed -n 's/^ExcNodeList=//p' | head -1)"
  [ "${current}" = "(null)" ] && current=""

  # `scontrol show hostnames` expands bracket ranges like node[04-05,13-14],
  # which makes the merge a plain sort -u instead of substring guesswork.
  merged="$({ [ -n "${current}" ] && scontrol show hostnames "${current}" 2>/dev/null
              [ -n "${bad_nodes}" ] && scontrol show hostnames "${bad_nodes}" 2>/dev/null
            } | sort -u | paste -sd, -)"

  if [ -n "${merged}" ]; then
    scontrol update JobId="${jobid}" ExcNodeList="${merged}" 2>/dev/null ||
      echo "$(date -Is) WARN: could not update ExcNodeList of ${jobid}"
  fi

  if scontrol release "${jobid}" 2>/dev/null; then
    echo "$(date -Is) released ${jobid} (exclude: ${merged:-unchanged})"
  else
    echo "$(date -Is) WARN: release of ${jobid} failed"
  fi
done
