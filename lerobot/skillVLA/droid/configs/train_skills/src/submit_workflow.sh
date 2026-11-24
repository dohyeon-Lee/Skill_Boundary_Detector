#!/usr/bin/env bash
# Source with: droid_submit DP|FSQ DEFAULT_CONFIG [--JOB_ID] [--config YAML] [--dry-run]
droid_submit() {
  local stage="$1" config="$2"; shift 2
  local project shared configs resolver plan=false
  project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
  configs="${project}/lerobot/skillVLA/droid/configs"
  shared="${project}/lerobot/skillVLA/libero/configs/src"
  resolver="${configs}/train_skills/src/train_skills_config.py"
  export SBD_GLOBAL_CONFIG="${configs}/global_config.yaml"
  source "${shared}/submit_args.sh"
  while (( $# )); do
    case "$1" in
      --config) config="${2:?missing config YAML}"; shift 2 ;;
      --dry-run) plan=true; shift ;;
      *) echo "Unknown DROID training option: $1" >&2; return 2 ;;
    esac
  done
  export PYTHONPATH="${project}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ "${plan}" == true ]]; then
    "${project}/.venv/bin/python" "${resolver}" --stage "${stage}" --config "${config}"
    return
  fi
  # Refuse missing prerequisites before acquiring or enqueueing GPU work.
  "${project}/.venv/bin/python" "${resolver}" --stage "${stage}" --config "${config}" --check >/dev/null
  source "${shared}/snapshot_config.sh"
  config="$(snapshot_config "${config}")"
  export SBD_GLOBAL_CONFIG="$(dirname "${config}")/global_config.yaml"
  local resolved
  resolved="$("${project}/.venv/bin/python" "${resolver}" --stage "${stage}" --config "${config}" --shell)" || return
  eval "${resolved}"
  source "${shared}/submit_job.sh"
  local folder="${configs}/train_skills/${stage}" script
  if [[ "${stage}" == DP ]]; then script="${folder}/src/train_dp.sbatch"; else script="${folder}/src/train_fsq.sbatch"; fi
  local -a options=(--job-name="DROID_${stage}" --gres=gpu:1 --cpus-per-task="${DROID_CPUS}" --mem="${DROID_MEM}" --time="${DROID_TIME}")
  [[ -z "${DROID_PARTITION}" ]] || options+=(--partition="${DROID_PARTITION}")
  [[ -z "${DROID_QOS}" ]] || options+=(--qos="${DROID_QOS}")
  [[ -z "${DROID_NODELIST}" ]] || options+=(--nodelist="${DROID_NODELIST}")
  [[ -z "${DROID_EXCLUDE}" ]] || options+=(--exclude="${DROID_EXCLUDE}")
  mkdir -p "${folder}/logs"
  cd "${folder}"
  options+=(--output="${folder}/logs/%x_%j.out" --error="${folder}/logs/%x_%j.err")
  echo "DROID ${stage}: ${DROID_OUTPUT_DIR}"
  submit_job "${options[@]}" "${script}" "${config}"
}
