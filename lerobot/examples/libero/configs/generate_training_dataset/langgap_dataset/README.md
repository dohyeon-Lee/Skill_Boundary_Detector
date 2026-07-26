# LangGap dataset (download + canonical rewrite)

LangGap(YC11Hou, arXiv 2603.00592)은 동일 장면에서 언어 지시만 바꾼 LIBERO 확장
(same-scene semantic perturbation) 데이터셋이다. HF에 LeRobot v3.0으로 배포되지만
로컬 canonical 컨벤션과 네 가지가 다르므로 여기서 재작성해 맞춘다:

| 항목 | LangGap HF 배포본 | 로컬 canonical | 처리 |
|---|---|---|---|
| fps 메타데이터 | 10 (라벨 오류) | 20 | 20 Hz로 재작성 (수집은 실제 control_freq=20, frame_skip=1) |
| wrist 카메라 키 | `observation.images.image2` | `observation.images.wrist_image` | rename |
| 이미지 방향 | raw robosuite에 `[::-1]` | 원본 HDF5에 `[::-1, ::-1]` | verify로 판정 후 flip |
| `observation.states.*` | 없음 (joint_state 복원 불가) | 있음 | 생략 — 학습 미사용 확인됨 |

action(OSC delta EEF 7D, gripper ±1)과 state(eef pos3+axis-angle3+gripper2 = 8D)는
의미론이 동일해 그대로 통과한다(gripper만 기존 `normalized_action` 규칙 적용).

## Build

```bash
./submit_build_langgap.sh                      # 로그인 노드에서 다운로드+방향판정 → Slurm으로 재작성+stats
LANGGAP_ONLY="langgap_6_smoke" ./build_langgap_dataset.sh   # 스모크(300ep)를 이 노드에서 직접
```

단계: ① download(HF) → ② orientation verify → ③ full rewrite(20 Hz) → ④ quantile stats.
최종 산출물은 `{langgap_root}/{name}` (blank root → global `dataset_root`), HF 원본은
`{langgap_root}/_hf/{name}`에 보존된다(`CLEAN_HF=1`로 삭제).

## Orientation verify (③ 전 필수 게이트)

`verify_image_orientation.py`가 LangGap과 로컬 데이터셋이 공유하는 공식 task
(예: libero_goal의 "put the bowl on the plate")의 첫 프레임을 네 방향 변형으로 비교해
MSE 최소 변형을 verdict.json에 기록하고, `flip=auto`인 변환기가 이를 읽는다.
verdict가 unknown이면 변환이 중단된다 — `_hf/{name}/.orientation/*_compare.png`를
눈으로 확인하고 yaml의 `convert_flip_image/wrist`에 none|h|w|hw를 명시할 것.
자동 판정이 나와도 PNG는 반드시 한 번 눈으로 확인하는 것을 전제로 한다.

`langgap_6_smoke`는 확장 task 위주라 공통 task가 없어 verdict가 unknown일 수 있다 —
스모크는 `--flip-image/--flip-wrist`를 CLI로 명시해 돌린다.

## Why full rewrite (metadata patch가 아니라)

fps 10→20을 라벨만 고치면 frame `timestamp` 컬럼과 비디오 컨테이너 PTS가 어긋나
timestamp 기반 프레임 조회가 깨진다. 방향 flip도 재인코딩 필수. 그래서
`convert_langgap_to_canonical.py`가 소스 LeRobot을 프레임 단위로 디코드해
canonical writer(`LeRobotDataset.create`, AV1)로 다시 쓴다. langgap_56(4,093 에피소드,
~55만 프레임)은 GPU 노드에서 수 시간짜리 작업이다.

## Which set to use

- **`langgap_ext_full_full` (기본)** — 확장 16 task, 2,400 에피소드, 269,490 프레임.
  공식 40 task는 로컬 자체 변환본(`libero_*_full_full`)을 학습 믹싱에서 조합한다.
  → 공식 데이터의 출처/컨벤션이 항상 로컬 것으로 통일되고 중복이 없다.
- `langgap_56_full_full` (extra) — 공식 40(저자 재수집 replay) + 확장 16, 4,093 에피소드.
  논문 56-task 실험을 배포본 그대로 재현할 때만. 공식/확장 구분은 task 문자열로 가능
  (56개 중 뒤 16개 task 문자열이 langgap_ext의 16개와 정확히 일치함을 확인).

## Downstream

산출물 이름은 `{suite}_full_full` 컨벤션을 따르므로 `build_training_dataset.py`
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
