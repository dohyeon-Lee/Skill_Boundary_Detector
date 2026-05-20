#!/usr/bin/env bash
# 다른 서버로 아티팩트를 rsync하는 스크립트.
# 사용법: ./sync_to_server.sh user@otherserver
set -euo pipefail

REMOTE=mdorazi@165.132.142.207
REMOTE_BASE=/scratch/mdorazi/Skill_Boundary_Detector

# ── pipeline_config에서 경로 로드 ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PY="${SCRIPT_DIR}/lerobot/examples/libero/configs/data_generation/pipeline_config.py"

source "${SCRIPT_DIR}/.venv/bin/activate"
eval "$(python "${CONFIG_PY}" --shell)"

LOCAL_BASE="${HOMEDIR}${PROJDIR}"   # /data2/dohyeon/SBD

# ── 전송 항목 정의 ────────────────────────────────────────────
declare -a LABELS PATHS

LABELS+=("DINO backbone 모델")
PATHS+=("${IMAGE_MODEL_PATH}")

LABELS+=("SAM2 모델")
PATHS+=("$(dirname "${SAM2_CHECKPOINT}")")   # models/sam2/ 디렉토리째

LABELS+=("DP policy 체크포인트")
PATHS+=("${DP_POLICY_PATH}")

LABELS+=("FSQ source 체크포인트 (.pt)")
PATHS+=("${FSQ_SOURCE_CKPT}")

LABELS+=("FSQ 패키지 디렉토리 (FSQ.pt + skill_latents.npz)")
PATHS+=("${FSQ_PACKAGE_DIR}")

LABELS+=("DINO precomputed tokens (.npz)")
PATHS+=("${DINO_TOKENS_PATH}")

LABELS+=("Skillset 디렉토리")
PATHS+=("${SKILLSET_DIR}")

LABELS+=("SkillVLA 데이터셋")
PATHS+=("${SKILLVLA_DATASET_DIR}")

LABELS+=("Raw 원본 데이터셋")
PATHS+=("${RAW_DATASET_DIR}")

# ── 메뉴 출력 ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  전송 대상 서버: ${REMOTE}"
echo "  데이터셋      : ${DATA}  /  FSQ epoch: ${FSQ_EPOCH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "전송할 항목 번호를 입력하세요 (예: 1 3 5  또는  all)"
echo ""

for i in "${!LABELS[@]}"; do
    printf "  [%d] %-40s  %s\n" "$((i+1))" "${LABELS[$i]}" "${PATHS[$i]}"
done

echo ""
read -r -p "선택: " INPUT

# ── 선택 파싱 ─────────────────────────────────────────────────
declare -a SELECTED
if [ "${INPUT}" = "all" ]; then
    for i in "${!LABELS[@]}"; do SELECTED+=("$i"); done
else
    for tok in $INPUT; do
        idx=$(( tok - 1 ))
        if [ $idx -lt 0 ] || [ $idx -ge ${#LABELS[@]} ]; then
            echo "잘못된 번호: $tok (무시)" >&2
            continue
        fi
        SELECTED+=("$idx")
    done
fi

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "선택된 항목 없음. 종료."
    exit 0
fi

# ── rsync 실행 ────────────────────────────────────────────────
echo ""
for idx in "${SELECTED[@]}"; do
    SRC="${PATHS[$idx]}"
    # 로컬 base를 제거하고 리모트 base로 교체
    REL="${SRC#${LOCAL_BASE}/}"
    DST="${REMOTE}:${REMOTE_BASE}/${REL}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${LABELS[$idx]}]"
    echo "  ${SRC}"
    echo "  → ${DST}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "${SRC}" ]; then
        ssh "${REMOTE}" "mkdir -p $(dirname "${REMOTE_BASE}/${REL}")" 2>/dev/null || true
        rsync -avzh --progress "${SRC}" "${DST}"
    elif [ -d "${SRC}" ]; then
        rsync -avzh --progress "${SRC}/" "${DST}/"
    else
        echo "  경고: 경로가 존재하지 않음, 건너뜀: ${SRC}" >&2
    fi
done

echo ""
echo "완료."
