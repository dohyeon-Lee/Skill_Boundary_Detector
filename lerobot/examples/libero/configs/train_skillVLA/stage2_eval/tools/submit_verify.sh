#!/usr/bin/env bash
# Submit the VSA≡stage1 open-loop forward-equivalence test (mirrors stage2_eval/submit_eval.sh).
#   (login) resolve config + stage1 parent from the stage2 ckpt config.json → sbatch verify.sbatch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"                 # stage2_eval/tools
CONFIG_PATH="${VERIFY_CONFIG:-${HERE}/verify_equiv_config.yaml}"
GLOBAL_CONFIG="$(cd "${HERE}/../../.." && pwd)/global_config.yaml"   # configs/global_config.yaml
PROJECT_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"           # 7 levels up → SBD root
PY="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || { echo "[error] project venv python not found: ${PY}" >&2; exit 1; }

# Resolve paths + slurm knobs with the venv python (yaml + ckpt config.json) → shell exports.
eval "$("${PY}" - "${CONFIG_PATH}" "${GLOBAL_CONFIG}" "${PROJECT_ROOT}" <<'PY'
import sys, json, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1])) or {}
gc  = yaml.safe_load(open(sys.argv[2])) or {}
proj = Path(sys.argv[3])
outputs_root = gc.get("outputs_root", "outputs_filtered")

model_dir = cfg["model_dir"]; ckpt = str(cfg.get("checkpoint", "last"))
s2 = proj / outputs_root / "skillVLA_stage2" / model_dir / "checkpoints" / ckpt / "pretrained_model"
c  = json.loads((s2 / "config.json").read_text()) if (s2 / "config.json").is_file() else {}
s1 = c.get("stage1_checkpoint_path") or ""

# raw dataset dir (for --source dataset), derived from fsq_path like stage2_eval_config._resolve_model
raw = cfg.get("raw_dataset_dir") or ""
if not raw and (c.get("fsq_path")):
    fp = Path(c["fsq_path"])
    try: raw = str(fp.parents[3] / fp.parents[1].name)
    except IndexError: raw = ""

part = gc.get("train_partition") or ""
if isinstance(part, list): part = ",".join(part)
excl = gc.get("train_exclude_nodes") or ""
if isinstance(excl, list): excl = ",".join(excl)

out = {
    "STAGE2_PATH": s2, "STAGE1_PATH": s1, "RAW_DATASET_DIR": raw,
    "MODEL_TAG": f"{model_dir}_{ckpt}",
    "SOURCE": cfg.get("source", "synthetic"), "CODES": cfg.get("codes", "0,31,62,93,124"),
    "NUM_STEPS": cfg.get("num_steps", 10), "DTYPE": cfg.get("dtype", "asis"),
    "ADAPTER_PROBES": str(bool(cfg.get("adapter_probes", True))).lower(),
    "V_GRES": cfg.get("verify_gres", "gpu:1"), "V_CPUS": cfg.get("verify_cpus_per_task", 8),
    "V_MEM": cfg.get("verify_mem", "32G"), "V_TIME": cfg.get("verify_time", "0:20:00"),
    "V_PARTITION": part, "V_QOS": str(gc.get("train_qos", "base_qos")).strip(),
    "V_NODELIST": (gc.get("train_nodelist") or ""), "V_EXCLUDE": excl,
}
for k, v in out.items():
    print(f"export {k}='{str(v).replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'")
PY
)"

[ -d "${STAGE2_PATH}" ] || { echo "[error] stage2 ckpt not found: ${STAGE2_PATH}" >&2; exit 1; }
[ -n "${STAGE1_PATH}" ] && [ -d "${STAGE1_PATH}" ] || {
  echo "[error] stage1 parent not found (from ckpt config.json stage1_checkpoint_path): '${STAGE1_PATH}'" >&2
  echo "        scratch 학습이면 VSA≡stage1 대조가 성립하지 않습니다." >&2; exit 1; }

mkdir -p "${HERE}/logs"
# 결과물 저장 위치 (stage2_eval/outputs/verify_equiv/<model>_<ckpt>_<source>/): result.txt + result.json
RESULT_DIR="$(cd "${HERE}/.." && pwd)/outputs/verify_equiv/${MODEL_TAG}_${SOURCE}_${DTYPE}"

echo "Submit VSA≡stage1 forward-equivalence test"
echo "  stage2 : ${STAGE2_PATH}"
echo "  stage1 : ${STAGE1_PATH}"
echo "  source : ${SOURCE}  codes=${CODES}  num_steps=${NUM_STEPS}"
echo "  slurm  : partition=${V_PARTITION} qos=${V_QOS} gres=${V_GRES} mem=${V_MEM}"
echo "  result : ${RESULT_DIR}/result.txt (+ .json)"

SBATCH_ARGS=(
  --partition="${V_PARTITION}" --qos="${V_QOS}"
  --gres="${V_GRES}" --cpus-per-task="${V_CPUS}" --mem="${V_MEM}" --time="${V_TIME}"
)
[ -n "${V_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${V_NODELIST}")
[ -n "${V_EXCLUDE}"  ] && SBATCH_ARGS+=(--exclude="${V_EXCLUDE}")

# env carried to the job via --export=ALL (slurm default).
STAGE2_PATH="${STAGE2_PATH}" STAGE1_PATH="${STAGE1_PATH}" RAW_DATASET_DIR="${RAW_DATASET_DIR}" \
SOURCE="${SOURCE}" CODES="${CODES}" NUM_STEPS="${NUM_STEPS}" DTYPE="${DTYPE}" ADAPTER_PROBES="${ADAPTER_PROBES}" PROJECT_ROOT="${PROJECT_ROOT}" \
TOOLS_DIR="${HERE}" RESULT_DIR="${RESULT_DIR}" \
  sbatch --job-name="VSAeq" \
         --output="${HERE}/logs/%x_%j.out" --error="${HERE}/logs/%x_%j.err" \
         "${SBATCH_ARGS[@]}" "${HERE}/verify.sbatch"
