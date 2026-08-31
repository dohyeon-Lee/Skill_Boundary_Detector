# LangGap dataset (download + canonical rewrite)

LangGap(YC11Hou, arXiv 2603.00592)은 동일 장면에서 언어 지시만 바꾼 LIBERO 확장
(same-scene semantic perturbation) 데이터셋이다. HF에 LeRobot v3.0으로 배포되지만
로컬 canonical 컨벤션과 네 가지가 다르므로 여기서 재작성해 맞춘다:

| 항목 | LangGap HF 배포본 | 로컬 canonical | 처리 |
|---|---|---|---|
| fps 메타데이터 | 10 (라벨 오류) | 20 | 20 Hz로 재작성 (수집은 실제 control_freq=20, frame_skip=1) |
| wrist 카메라 키 | `observation.images.image2` | `observation.images.wrist_image` | rename |
| 이미지 방향 | 공식 prefix는 canonical, 저자 ext는 raw에 `[::-1]`만 적용 | raw에 `[::-1, ::-1]` | ext task에만 추가 W flip |
| `observation.states.*` | 없음 (joint_state 복원 불가) | 있음 | 생략 — 학습 미사용 확인됨 |

action(OSC delta EEF 7D, gripper ±1)과 state(eef pos3+axis-angle3+gripper2 = 8D)는
의미론이 동일해 그대로 통과한다(gripper만 기존 `normalized_action` 규칙 적용).

## Build

```bash
./submit_build_langgap.sh                      # 로그인 노드에서 다운로드+방향판정 → Slurm으로 재작성+stats
LANGGAP_ONLY="langgap_6_smoke" ./build_langgap_dataset.sh   # 스모크(300ep)를 이 노드에서 직접

# ext 16개가 파생된 source env의 공식 9 task만 다운로드+변환
LANGGAP_ONLY="langgap_preext_full_full" ./submit_build_langgap.sh
```

단계: ① download(HF) → ② orientation verify → ③ full rewrite(20 Hz) → ④ quantile stats.
최종 산출물은 `{langgap_root}/{name}` (blank root → global `dataset_root`), HF 원본은
`{langgap_root}/_hf/{name}`에 보존된다(`CLEAN_HF=1`로 삭제).

`langgap_preext_full_full`은 `YC11Hou/langgap_full`의 source task index
`10,17,18,21,22,26,31,34,38`만 선택한다. 먼저 meta/data를 내려받아 이 episode들이 쓰는
최소 packed MP4 목록을 계산하므로, 현재 배포본 기준 전체 21개 대신 8개 video shard만
받는다. 다만 LangGap MP4는 여러 task가 한 shard를 공유하므로 staging MP4 안에는 선택하지
않은 episode의 byte range가 일부 함께 존재한다. staging의 parquet/task/episode metadata와
최종 canonical 데이터셋은 정확히 9 task(출력 task index `0..8`, 413 episodes)만 포함한다.
설정은 `langgap_dataset_config.yaml`의 `download_task_ids_by_set`에서 관리한다.

## Split an existing full dataset

이미 canonical `langgap_56_full_full`이 있으면 공식 40과 확장 16을 다시 다운로드하거나
재인코딩하지 않고 분리할 수 있다. 두 그룹이 서로 다른 packed MP4 shard를 사용하므로
영상은 whole-file hardlink하고 parquet/metadata/global non-video stats만 새로 만든다.

```bash
# 현재 경로와 shard 경계를 쓰기 전에 확인만
python src/split_langgap_full.py --root dataset --dry-run

# yaml의 langgap_derived_sets 두 개를 모두 생성
python src/split_langgap_full.py --root dataset

# 하나만 생성
python src/split_langgap_full.py --root dataset --set langgap_original_full_full
```

기본 출력은 `langgap_original_full_full`(원본 task 0..39 → 출력 0..39)과
`langgap_ext_full_full`(원본 task 40..55 → 출력 0..15)이다. hardlink이므로 같은
파일시스템에서는 영상 추가 공간이 들지 않고 원본 폴더를 지워도 파생 링크는 유지된다.
state/action/index 계열 global stats와 quantile은 분할 parquet에서 정확히 재계산하며,
학습에서 사용하지 않는 video stats는 기본적으로 원본 값을 계승한다. 영상 통계까지
다시 계산하려면 `--include-videos`를 붙일 수 있지만 모든 영상을 디코딩하므로 오래 걸린다.

## Orientation verify (③ 전 필수 게이트)

`src/verify_image_orientation.py`가 LangGap과 로컬 데이터셋이 공유하는 공식 task
(예: libero_goal의 "put the bowl on the plate")의 첫 프레임을 네 방향 변형으로 비교해
MSE 최소 변형을 verdict.json에 기록하고, `flip=auto`인 변환기가 이를 읽는다.
verdict가 unknown이면 변환이 중단된다 — `_hf/{name}/.orientation/*_compare.png`를
눈으로 확인하고 yaml의 `convert_flip_image/wrist`에 none|h|w|hw를 명시할 것.
자동 판정이 나와도 PNG는 반드시 한 번 눈으로 확인하는 것을 전제로 한다. 이 verdict는
공유되는 공식 task의 base 방향만 판정한다. `langgap_56`은 서로 다른 두 변환 파이프라인을
합친 데이터셋이므로 ext 방향까지 대표하지 않는다. 변환기는 yaml의
`convert_ext_task_start_by_set`부터 추가 W flip을 별도로 적용한다.

`langgap_ext_full_full`과 `langgap_6_smoke`는 ext-only이므로 task 0부터 추가 W flip을
적용한다.

## Repair an existing conversion

이미 만들어진 `langgap_56_full_full`은 전체 state/action 변환을 다시 할 필요가 없다.
다음 명령은 원본을 보존한 hardlink clone을 만든다. 공식 40 task의 최종 비디오는
bitstream 그대로 재사용하고, ext 비디오만 HF staging 원본에서 직접 좌우 flip+20 Hz PTS
보정하여 정상 변환과 동일하게 한 번 인코딩한다.

```bash
python src/repair_ext_orientation.py --dataset langgap_56_full_full --dry-run
./submit_repair_ext_orientation.sh
```

기본 출력은 `langgap_56_full_full_canonical_orientation`이다. 현재 데이터 기준 ext는
episode 1693부터 2400개이며, staging의 ext-only packed video 10개만 새로 만든다. ext는
현재 잘못된 최종 MP4를 다시 압축하지 않으므로 추가 인코딩 세대가 없고, 공식 task는
재인코딩조차 없다. 공간 flip은 채널별 통계를 바꾸지 않으므로 stats는 그대로 재사용하며,
episode video 위치/PTS 메타데이터만 새 ext 파일에 맞게 갱신한다.

## Why full rewrite (metadata patch가 아니라)

fps 10→20을 라벨만 고치면 frame `timestamp` 컬럼과 비디오 컨테이너 PTS가 어긋나
timestamp 기반 프레임 조회가 깨진다. 방향 flip도 재인코딩 필수. 그래서
`src/convert_langgap_to_canonical.py`가 소스 LeRobot을 프레임 단위로 디코드해
canonical writer(`LeRobotDataset.create`, AV1)로 다시 쓴다. langgap_56(4,093 에피소드,
~55만 프레임)은 GPU 노드에서 수 시간짜리 작업이다.

## Which set to use

- **`langgap_ext_full_full` (기본)** — 확장 16 task, 2,400 에피소드, 269,490 프레임.
  공식 40 task는 로컬 자체 변환본(`libero_*_full_full`)을 학습 믹싱에서 조합한다.
  → 공식 데이터의 출처/컨벤션이 항상 로컬 것으로 통일되고 중복이 없다.
- `langgap_56_full_full` (extra) — 공식 40(저자 재수집 replay) + 확장 16, 4,093 에피소드.
  논문 56-task 실험을 배포본 그대로 재현할 때만. 공식/확장 구분은 task 문자열로 가능
  (56개 중 뒤 16개 task 문자열이 langgap_ext의 16개와 정확히 일치함을 확인).
- `langgap_preext_full_full` — 확장 16 task가 변형된 source environment에 해당하는 공식
  9 task만 모은 세트. `langgap_full`의 source task index는
  `10,17,18,21,22,26,31,34,38`이고, 출력에서는 `0..8`로 재매핑된다.

## Downstream

산출물 이름은 `{suite}_full_full` 컨벤션을 따르므로 `split_dataset/build_training_dataset.py`
믹싱/학습 설정에 다른 스위트처럼 이름만 추가하면 된다.

## Eval

`langgap_ext` 스위트(확장 59 task)가 vendored libero에 등록되어 있어 기존 평가
스크립트에 `--env.task=langgap_ext`로 바로 돌아간다 (libero_90 평가와 동일한 경로).
- BDDL/init: `tools/lerobot-libero/libero/libero/{bddl_files,init_files}/langgap_ext/`
- 지시문은 BDDL의 `(:language ...)`에서 파싱 (langgap 스위트 한정; 기존 스위트 불변)
- task_index ↔ 지시문 ↔ trained(16)/held-out(43) 매핑: [LANGGAP_EXT_TASKS.md](LANGGAP_EXT_TASKS.md)
- 스텝 예산: `TASK_SUITE_MAX_STEPS["langgap_ext"] = 400` (lerobot/src/lerobot/envs/libero.py)
- 저장소에서 init이 누락된 goal 확장 3개 task는 LangGap 컨벤션(확장 init = 소스 공식
  task init의 바이트 동일 복사본; md5 검증)에 따라 소스 task init으로 대체함.
