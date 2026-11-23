# MimicGen Coffee D0

LIBERO와 모델·학습 구현을 공유하고, 데이터·설정·결과 경로를 분리한다.
초기 설정은 **LIBERO DP 10k + FSQ 2000 → Coffee 전처리 → LIBERO VSA/PT에서 Coffee FT**다.
DP 10k는 기존 표준 FSQ와 짝을 이루는 체크포인트이며, 새 `fresh` DP 100k와 구분한다.
기존 PT/FT의 `fsq_source.json`에 기록된 `FSQ333_contrastiveON_routeON_loss_new` epoch 2000을 사용한다.
Coffee FT 빌드도 기존 PT의 지원 코드 분포(min frequency 10), max_length 500, jitter storage 20/30% 설정을 따른다.

## 폴더와 실행 순서

각 폴더에서 YAML을 수정하고 SH를 실행한다. venv 활성화는 필요 없다.
`--74714`처럼 이미 확보한 allocation을 지정할 수 있다. 지정하지 않으면 공통 서버 설정을 따른다.
Coffee 5개 에피소드의 DP/SBD 부분 평가는 완료했다. 전체 SBD·SkillVLA 빌드와 학습/rollout은 아직 실행하지 않았다.

| 순서 | 폴더 (`configs/` 기준) | 실행 | 결과 |
|---|---|---|---|
| 1 | `generate_training_dataset/download_dataset` | `./prepare_dataset.sh` | `dataset_mimicgen/coffee_d0_full_full` (이미 준비됨) |
| 2 | `train_skillVLA/build_data` | `./submit_build_all.sh --JOBID` | 전체 SBD → FSQ 코드 → SkillVLA parquet, ISS, 시작/종료 UV |
| 4 | `train_skillVLA/FT/VSA` | `./submit_train.sh --JOBID` | LIBERO VSA/PT에서 Coffee만으로 FT |
| 4 | `train_skillVLA/FT/Predictor`, `FT/Terminator` | 각각 `./submit_train.sh --JOBID` | Coffee Predictor/Terminator FT |
| 5 | `train_skillVLA/FT/eval` | `./submit_eval.sh --JOBID` | Coffee 시뮬레이터 평가 + HTML/영상/통계 |

`submit_build_all.sh --74714` 한 번이면 기존 allocation의 GPU 한 개에서 모든 단계가 순서대로 실행된다.
SBD는 전체 에피소드 완료 여부와 검출기 설정을 검증한 뒤 재사용하고, 미완료이면 같은 GPU에서 이어 빌드한다.
SBD 실패 시 FSQ와 SkillVLA 단계는 시작하지 않는다. 같은 설정의 FSQ 인코딩과 최종 결과도 완료 시 건너뛴다.
앞단계 SBD를 별도로 실행할 필요는 없다. 필요한 경우 `train_skills/build_data/submit_build_data.sh`로 SBD만 실행할 수 있다.
제출 시 SBD recipe를 SkillVLA YAML snapshot 안에 함께 저장하므로, 대기 중 live YAML 변경의 영향을 받지 않는다.
FT eval은 기본적으로 세 컴포넌트의 **5k 체크포인트**를 쓰고, run 이름은 해당 FT 학습 YAML에서 계산한다.
`PT/eval`은 기본 LIBERO PT 모델 세 개를 Coffee에서 평가하는 zero-shot 구성이다.
Coffee 자체 PT 모델을 평가하려면 이 폴더의 기본 YAML에서 모델 경로를 선택한다.

추가 워크플로:

- `train_skills/DP/submit_train_dp.sh`: Coffee 자체 DP 학습. Transformer / GRU_iid / GRU_epi, small/base 설정은 공통 학습기와 동일.
- `train_skills/FSQ/submit_train_fsq.sh`: Coffee SBD skillset으로 자체 FSQ 학습.
- `train_skills/DP/eval/submit_eval.sh`: 선택한 Coffee 에피소드의 SBD 부분빌드 및 영상 HTML.
- `train_skills/FSQ/eval/submit_eval.sh`: Coffee에서 학습한 FSQ의 코드북·실제 데이터 영상 갤러리.
- `train_skillVLA/PT/{VSA,Predictor,Terminator}/submit_train.sh`: Coffee 자체 PT.
- 위 학습/빌드/평가 SH의 `--dry-run`: 설정과 경로 확인. 필요한 이전 단계 결과물은 존재해야 한다.
- `--yonsei-plan`: 전송/실행 계획만 확인. `--yonsei`는 공통 Yonsei 전송/실행/결과 회수 기능을 사용한다. 원격에도 `bash setup_env.sh --mimicgen-only`가 선행되어야 한다.

## 설정과 저장 경로

전역 저장 경로는 `global_config.yaml`의 `dataset_mimicgen`, `outputs_mimicgen`이다.
모델 구현과 실행 배치는 LIBERO 쪽 공통 파일을 사용한다. MimicGen에는 별도 base YAML과 진입점만 둔다.

- SBD: `dataset_mimicgen/FSQ_dataset/coffee_d0_full_full/FSQ_inputs/seg_.../skillset`
- SkillVLA: `dataset_mimicgen/skillvla/coffee_d0_full_full/FSQ333_libero2000_ft_coffee_d0/`
  - `skillvla/`: 학습 parquet와 메타데이터
  - `FSQ.pt`, `fsq_source.json`, `skill_latents.npz`, `skill_initial_state.npz`, `skill_focus_uv.npz`: 같은 run의 companion 파일
- 모델: `outputs_mimicgen/{DP,FSQ,skillVLA_PT,skillVLA_FT}/...`
- 평가: 각 `eval/outputs/`, DP 부분빌드: `DP/eval/partial_build/`

SBD와 DP eval의 외부 DP 선택은 `dp_outputs_root`, `dp_run_name`, `dp_checkpoint`다.
`DP/checkpoints/.../pretrained_model`은 자동 구성한다. Coffee 자체 모델을 쓰면 root를 생략하고 global을 따른다.
SkillVLA 빌드는 `fsq_outputs_root`, `fsq_run_name`, `fsq_checkpoint`(epoch)로 선택한다.
저장된 FSQ 인코더 정규화를 사용하며 Coffee 통계로 다시 fit하지 않는다.
FT warm start도 `warm_start.outputs_root`와 각 컴포넌트의 `{run, checkpoint}`만 지정한다.
FT eval의 `auto`는 각 컴포넌트의 현재 FT 학습 YAML에서 모델 이름을 가져온다.

기존 LIBERO PT·FT 전처리 기록에 맞춰 DP 10k, current GT 중심, history 20 / prediction 24,
replan 5, GMM 5, NMS 15, SG 7/4, power 2, initial mask OFF, terminal mask 24를 사용한다.
PCA는 std/variance 0.95/stride 3으로 Coffee 전체 데이터에 fit한다.
부분 평가의 평균 임계값은 선택한 5개 에피소드에서 계산한다. 전체 빌드는 현재 `ft`이므로
기존 LIBERO PT의 임계값 0.003306652329313102를 재사용한다. PCA 행렬은 Coffee 전체에서 새로 fit한다.

전체 자동 빌드는 SkillVLA YAML의 `target_dataset`, 저장 경로, `skillvla_data_mode`, PT 참조가 우선한다.
DP/검출기와 SBD 워커 수는 `train_skills/build_data/build_data_config.yaml`에서 가져온다.
SBD만 별도로 실행할 때는 해당 SBD YAML의 모드와 PT 참조를 사용한다.
`pt`/`ft_own`은 해당 데이터 전체 평균, `ft`는 선택한 PT의 평균 임계값, `ft_epi`는 에피소드별 평균을 쓴다.
PT 참조는 SBD의 `skillvla_pt_reference`, SkillVLA의 `fsq_snap_reference`에서 선택한다.
임계값 유형과 모드는 폴더명에 자동 반영되며, SkillVLA 빌드는 선택한 SBD가 실제로 그 모드인지 검사한다.
Snap을 켠 경우 `pt`는 자기 데이터 분포, FT 계열은 선택한 PT의 지원 코드를 기준으로 보정한다.

이 모드는 q01/q99 정규화 통계 선택과 별개다. Coffee 데이터에는 자체 통계를 기록하며,
VSA FT는 학습 시 warm-start PT 체크포인트의 state/action 통계를 상속한다.
Predictor FT도 warm-start Predictor의 정규화 통계를 상속한다. 재개 시에는 저장된 FT 통계를 유지한다.
자체 FSQ를 새로 학습하면 코드 의미가 달라질 수 있으므로, 그 FSQ에 맞는 Coffee PT 체크포인트를 FT의 warm start로 선택해야 한다.
FSQ 모델이 다른 LIBERO PT와의 잘못된 혼용은 lineage 검증에서 거부한다.

## 좌표와 카메라

- State: EEF XYZ + axis-angle radian + finger qpos 2개 = 8D.
- Quaternion 변환은 LIBERO/robosuite와 동일하며 음수 w를 임의로 뒤집지 않는다.
- Action: normalized OSC_POSE delta 7D. gripper는 원래의 `-1=open`, `+1=close`를 유지한다.
- Video: 원본 Coffee 영상 방향 유지. LIBERO 전용 180° flip을 적용하지 않는다.
- UV: 저장된 고정 agentview calibration과 실제 EEF XYZ에서 시작/종료점을 투영한다.
  종료 인덱스는 기존 코드와 동일한 `min(frame_end, episode_length - 1)`이다.
- Top projection matrix에는 공통 LIBERO projection 함수의 수평 mirror를 상쇄하는 변환을 넣는다.
  영상이나 정답 점을 수동으로 이동하지 않는다. Episode grounding은 matrix에도 반영한다.
- Wrist label/overlay는 Coffee의 `mirror_x=false`를 사용한다. LIBERO의 기존 기본값은 true다.
  학습 batch별 카메라 방향도 지원한다.

`generate_training_dataset/visualized_dataset/visualize_uv.sh`는 CPU에서 실제 녹화 프레임을 읽어
초록색 현재 EEF와 하늘색 20프레임 뒤 EEF를 표시한다. 예시 4장은
`visualized_dataset/outputs/coffee_d0_uv_check/`에 저장되며 실제 frame timestamp와 투영 좌표도 JSON으로 남는다.
이는 카메라 정합 확인용이며 SBD로 계산한 스킬 경계 예시는 아니다.

## 검증과 현재 범위

실제 Coffee 샘플에서 LIBERO FSQ 인코딩, SkillVLA annotation/ISS/UV 생성,
VSA·Predictor·Terminator의 PT/FT 설정 해석을 확인했다. 별도 MimicGen Python 프로세스의 reset/step도 확인했다.
기존 LIBERO projection과 새 Coffee projection, 이미지 방향, 음수-w quaternion, 원격 resolver 선택에 대한 CPU 검사를 포함한다.

Coffee HF 데이터에는 각 데모의 완전한 simulator 초기 상태가 없다. 따라서
**demo-exact reset, LIBERO initial-state bank, demo GT action으로 성공률 재현**은 제공하지 않는다.
평가는 같은 seed의 Coffee_D0 random reset과 학습된 Predictor/Terminator를 사용한다.
전체 Coffee 학습/rollout 성공률과 대규모 성능은 아직 측정하지 않았다.
CoffeePreparation의 simulator 클래스는 소스에 존재하지만, 별도 데이터·task instruction 전처리는 이번 Coffee_D0 통합 범위에 포함하지 않는다. VLA 환경 설정은 Coffee_D0만 허용한다.
