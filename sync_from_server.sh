#!/usr/bin/env bash
# Yonsei 서버에서 현재 서버로 아티팩트를 rsync하는 스크립트.
# 사용법: ./sync_from_server.sh
set -euo pipefail

REMOTE=mdorazi@165.132.142.207
REMOTE_BASE=/scratch/mdorazi/Skill_Boundary_Detector

# ── pipeline_config에서 로컬 경로 로드 ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PY="${SCRIPT_DIR}/lerobot/examples/libero/configs/data_generation/pipeline_config.py"

source "${SCRIPT_DIR}/.venv/bin/activate"
eval "$(python "${CONFIG_PY}" --shell)"

LOCAL_BASE="${HOMEDIR}${PROJDIR}"   # /data2/dohyeon/SBD

# ── 수신 항목 정의 ────────────────────────────────────────────
declare -a LABELS PATHS RENAMEABLE

add_item() {
    LABELS+=("$1")
    PATHS+=("$2")
    RENAMEABLE+=("${3:-0}")  # 1이면 remote 폴더명 변경 prompt 제공
}

add_item "DINO backbone 모델" "${IMAGE_MODEL_PATH}"

add_item "SAM2 모델" "$(dirname "${SAM2_CHECKPOINT}")"   # models/sam2/ 디렉토리째

add_item "DP policy 체크포인트" "${DP_POLICY_PATH}"

add_item "FSQ source 체크포인트 (.pt)" "${FSQ_SOURCE_CKPT}"

add_item "FSQ 패키지 디렉토리 (FSQ.pt + skill_latents.npz)" "${FSQ_PACKAGE_DIR}"

add_item "DINO precomputed tokens (.npz, FSQ용)" "${DINO_TOKENS_PATH}"

add_item "SAM2 mask 디렉토리 (FSQ dino_flags용)" "${SAM2_MASKS_DIR}"

add_item "SAM2 patch flags (.npz, FSQ dino_flags용)" "${SAM2_FLAGS_PATH}"

add_item "DINO feature 디렉토리 (에피소드별 npz, DP eval용)" "${DINO_FEATURE_DIR}"

add_item "Skillset 디렉토리" "${SKILLSET_DIR}"

add_item "SkillVLA 데이터셋" "${SKILLVLA_DATASET_DIR}"

add_item "Raw 원본 데이터셋" "${RAW_DATASET_DIR}"

add_item "전체 ${DATA}_DINO 디렉토리" "${DATA_DIR}/${DATA}_DINO" 1

add_item "전체 ${DATA}_for_FSQ 디렉토리" "${FSQ_PRECOMPUTE_DIR}" 1

add_item "전체 ${DATA}_for_skillVLA 디렉토리" "${SKILLVLA_ROOT}" 1

# ── 메뉴 출력 ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  수신 원격 서버: ${REMOTE}"
echo "  데이터셋      : ${DATA}  /  FSQ epoch: ${FSQ_EPOCH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "가져올 항목 번호를 입력하세요 (예: 1 3 5  또는  all)"
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
    DST="${PATHS[$idx]}"
    REL="${DST#${LOCAL_BASE}/}"
    REMOTE_REL="${REL}"

    if [ "${RENAMEABLE[$idx]}" = "1" ]; then
        DEFAULT_NAME="$(basename "${DST}")"
        read -r -p "  remote 폴더명 [${DEFAULT_NAME}] (${LABELS[$idx]}): " REMOTE_NAME
        if [ -n "${REMOTE_NAME}" ]; then
            REMOTE_PARENT="$(dirname "${REL}")"
            REMOTE_REL="${REMOTE_PARENT}/${REMOTE_NAME}"
        fi
    fi

    SRC="${REMOTE}:${REMOTE_BASE}/${REMOTE_REL}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${LABELS[$idx]}]"
    echo "  ${SRC}"
    echo "  → ${DST}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if ssh "${REMOTE}" "test -f \"${REMOTE_BASE}/${REMOTE_REL}\""; then
        mkdir -p "$(dirname "${DST}")"
        rsync -avzh --progress "${SRC}" "${DST}"
    elif ssh "${REMOTE}" "test -d \"${REMOTE_BASE}/${REMOTE_REL}\""; then
        mkdir -p "${DST}"
        rsync -avzh --progress "${SRC}/" "${DST}/"
    else
        echo "  경고: 원격 경로가 존재하지 않음, 건너뜀: ${REMOTE_BASE}/${REMOTE_REL}" >&2
    fi
done

echo ""
echo "완료."
