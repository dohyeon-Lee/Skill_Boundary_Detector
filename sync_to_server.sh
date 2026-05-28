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
declare -a LABELS PATHS RENAMEABLE TYPES

add_item() {
    LABELS+=("$1")
    PATHS+=("$2")
    RENAMEABLE+=("${3:-0}")  # 1이면 remote 폴더명 변경 prompt 제공
    TYPES+=("${4:-normal}")
}

VLA_OUTPUTS_DIR="${LOCAL_BASE}/VLA_outputs"

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

add_item "SkillVLA checkpoint  [eval_exp=${EVAL_EXP}  step=${EVAL_CHECKPOINT}]" "${VLA_OUTPUTS_DIR}" 0 "vla_ckpt"

add_item "SkillVLA checkpoint  [직접 입력: 폴더명 + step]" "${VLA_OUTPUTS_DIR}" 0 "vla_ckpt_manual"

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

    # ── VLA checkpoint 특수 처리 ──────────────────────────────
    if [ "${TYPES[$idx]}" = "vla_ckpt" ]; then
        # yaml eval 파라미터 기반 자동 탐색
        VLA_LANG_TAG="allowlang"
        [ "${EVAL_BLOCK_LANG}" = "true" ] && VLA_LANG_TAG="blocklang"
        mapfile -t VLA_MATCHES < <(ls -1 "${VLA_OUTPUTS_DIR}" 2>/dev/null \
            | grep "_${VLA_LANG_TAG}_" | grep "_${EVAL_EXP}$" | grep -v "connectgrad")

        if [ ${#VLA_MATCHES[@]} -eq 0 ]; then
            echo "  오류: eval_exp='${EVAL_EXP}'로 끝나는 폴더 없음 (VLA_outputs/)" >&2
            continue
        elif [ ${#VLA_MATCHES[@]} -eq 1 ]; then
            VLA_FOLDER="${VLA_MATCHES[0]}"
        else
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [SkillVLA checkpoint] '${EVAL_EXP}' 매칭 폴더 여러 개:"
            echo ""
            for i in "${!VLA_MATCHES[@]}"; do
                printf "    [%d] %s\n" "$((i+1))" "${VLA_MATCHES[$i]}"
            done
            echo ""
            read -r -p "  폴더 번호: " VLA_FOLDER_NUM
            VLA_FOLDER="${VLA_MATCHES[$((VLA_FOLDER_NUM - 1))]}"
        fi

        VLA_STEP="${EVAL_CHECKPOINT}"
        if [ ! -d "${VLA_OUTPUTS_DIR}/${VLA_FOLDER}/checkpoints/${VLA_STEP}" ]; then
            echo "  오류: 체크포인트 없음 '${VLA_FOLDER}/checkpoints/${VLA_STEP}'" >&2
            echo "  사용 가능한 스텝: $(ls "${VLA_OUTPUTS_DIR}/${VLA_FOLDER}/checkpoints/" 2>/dev/null | tr '\n' ' ')" >&2
            continue
        fi

        SRC="${VLA_OUTPUTS_DIR}/${VLA_FOLDER}/checkpoints/${VLA_STEP}"
        REL="VLA_outputs/${VLA_FOLDER}/checkpoints/${VLA_STEP}"
        REMOTE_REL="${REL}"
        DST="${REMOTE}:${REMOTE_BASE}/${REMOTE_REL}"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [SkillVLA checkpoint]"
        echo "  ${SRC}"
        echo "  → ${DST}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh "${REMOTE}" "mkdir -p \"${REMOTE_BASE}/${REMOTE_REL}\""
        rsync -avzh --progress "${SRC}/" "${DST}/"
        continue
    fi

    # ── VLA checkpoint 직접 입력 ──────────────────────────────
    if [ "${TYPES[$idx]}" = "vla_ckpt_manual" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [SkillVLA checkpoint 직접 입력]"
        echo ""
        echo "  사용 가능한 폴더 목록:"
        ls -1 "${VLA_OUTPUTS_DIR}" 2>/dev/null | sed 's/^/    /'
        echo ""
        read -r -p "  폴더명 (VLA_outputs/ 아래): " VLA_FOLDER
        if [ -z "${VLA_FOLDER}" ] || [ ! -d "${VLA_OUTPUTS_DIR}/${VLA_FOLDER}" ]; then
            echo "  오류: 폴더 없음 '${VLA_FOLDER}'" >&2
            continue
        fi
        echo ""
        echo "  사용 가능한 체크포인트 스텝:"
        ls -1 "${VLA_OUTPUTS_DIR}/${VLA_FOLDER}/checkpoints/" 2>/dev/null | sed 's/^/    /'
        echo ""
        read -r -p "  checkpoint step (예: 060000): " VLA_STEP
        if [ -z "${VLA_STEP}" ] || [ ! -d "${VLA_OUTPUTS_DIR}/${VLA_FOLDER}/checkpoints/${VLA_STEP}" ]; then
            echo "  오류: 체크포인트 없음 '${VLA_FOLDER}/checkpoints/${VLA_STEP}'" >&2
            continue
        fi

        SRC="${VLA_OUTPUTS_DIR}/${VLA_FOLDER}/checkpoints/${VLA_STEP}"
        REL="VLA_outputs/${VLA_FOLDER}/checkpoints/${VLA_STEP}"
        DST="${REMOTE}:${REMOTE_BASE}/${REL}"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  ${SRC}"
        echo "  → ${DST}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh "${REMOTE}" "mkdir -p \"${REMOTE_BASE}/${REL}\""
        rsync -avzh --progress "${SRC}/" "${DST}/"
        continue
    fi

    # 로컬 base를 제거하고 리모트 base로 교체
    REL="${SRC#${LOCAL_BASE}/}"
    REMOTE_REL="${REL}"
    if [ "${RENAMEABLE[$idx]}" = "1" ] && [ -d "${SRC}" ]; then
        DEFAULT_NAME="$(basename "${SRC}")"
        read -r -p "  remote 폴더명 [${DEFAULT_NAME}] (${LABELS[$idx]}): " REMOTE_NAME
        if [ -n "${REMOTE_NAME}" ]; then
            REMOTE_PARENT="$(dirname "${REL}")"
            REMOTE_REL="${REMOTE_PARENT}/${REMOTE_NAME}"
        fi
    fi
    DST="${REMOTE}:${REMOTE_BASE}/${REMOTE_REL}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${LABELS[$idx]}]"
    echo "  ${SRC}"
    echo "  → ${DST}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "${SRC}" ]; then
        ssh "${REMOTE}" "mkdir -p \"$(dirname "${REMOTE_BASE}/${REMOTE_REL}")\""
        rsync -avzh --progress "${SRC}" "${DST}"
    elif [ -d "${SRC}" ]; then
        ssh "${REMOTE}" "mkdir -p \"${REMOTE_BASE}/${REMOTE_REL}\""
        rsync -avzh --progress "${SRC}/" "${DST}/"
    else
        echo "  경고: 경로가 존재하지 않음, 건너뜀: ${SRC}" >&2
    fi
done

echo ""
echo "완료."
