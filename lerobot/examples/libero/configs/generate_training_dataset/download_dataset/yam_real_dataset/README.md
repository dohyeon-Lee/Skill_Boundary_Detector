# YAM real recorder → Pi0.5 canonical LeRobot v3

이 폴더는 i2rt의 `workstation/lerobot_recorder`가 직접 만든 LeRobot v3 데이터셋을
Pi0.5 학습용 canonical 데이터셋으로 재작성한다. `abcdl`은 거치지 않는다.

```text
i2rt YAM recorder (LeRobot v3)
  observation.state 42D = arm별 pos7 + vel7 + effort7
  action 14D = left7 + right7 absolute target
  agentview / wrist_left / wrist_right
        ↓
yam_real_dataset converter
        ↓
Pi0.5 canonical LeRobot v3
  observation.state 14D = left pos7 + right pos7
  action 14D absolute
  top / left_wrist / right_wrist
  task + normalization quantile stats
```

## 중요한 동작

- 원본 폴더는 수정하지 않는다. 출력은 기본적으로 `{project_root}/dataset_YAM/`에 새로 만든다.
- `outcomes.jsonl`을 읽어 기본적으로 `success` episode만 포함한다.
- 상태 인덱스는 recorder의 `left.pos.*`, `right.pos.*` names에서 찾는다. names가 없는
  구형 42D recorder 데이터만 `[0:7, 21:28]` fallback을 사용한다.
- 카메라는 aspect ratio를 보존하고 zero-pad하여 기본 256×256으로 만든다.
- 홈 복귀/홈 포징 프레임은 자르지 않는다. episode 전체가 그대로 학습 데모가 된다.
- output의 `meta/yam_conversion.json`에 source episode 대응과 변환 규약을 기록한다.

## 준비

workstation에서 수집한 dataset 폴더 전체를 `outcomes.jsonl`까지 함께 복사한다.
recorder 설정이 아래와 같았다면:

```yaml
root: /data/yam_recordings
repo_id: user/yam_bimanual_raw
format: lerobot
```

실제 원본은 `/data/yam_recordings/yam_bimanual_raw`이다. 기본 설정을 사용할 경우 이를 다음
위치에 둔다.

```text
{project_root}/dataset_YAM_raw/yam_bimanual_raw/
├── data/
├── meta/
├── videos/
└── outcomes.jsonl
```

다른 위치를 쓸 경우 `yam_real_dataset_config.yaml`의 `yam_raw_root`와 `yam_sets.*.source`를
수정하거나 CLI `--source`를 사용한다.

## 실행

```bash
cd lerobot/examples/libero/configs/generate_training_dataset/download_dataset/yam_real_dataset

# 스키마와 episode 선택만 검사. 파일을 쓰지 않는다.
python src/convert_yam_lerobot.py --set yam_pi05 --dry-run

# 실제 변환
./build_yam_real_dataset.sh

# 소규모 smoke build
MAX_EPISODES=2 YAM_ONLY=yam_pi05 ./build_yam_real_dataset.sh

# 기존 output을 명시적으로 재생성
FORCE=1 ./build_yam_real_dataset.sh
```

직접 경로를 지정할 수도 있다.

```bash
python src/convert_yam_lerobot.py \
  --set yam_pi05 \
  --source /data/yam_recordings/yam_bimanual_raw \
  --output /data/pi05_datasets/yam_pi05
```

실패 episode까지 모두 보존하려면 YAML의 `include_outcomes: []`로 바꾸거나 다음처럼 실행한다.

```bash
python src/convert_yam_lerobot.py --set yam_pi05 --include-outcome success --include-outcome fail
```

## 출력 검증

변환기는 완료 전에 다음을 확인한다.

- LeRobot `codebase_version == v3.0`
- state/action shape가 각각 14D
- canonical 카메라 세 개와 동일 해상도
- episode/frame 총합
- state/action의 `q01/q10/q50/q90/q99`

학습 시에는 `global_config.yaml`의 dataset root를 출력 폴더로 바꾸고, Pi0.5 설정에서
source dataset 이름을 `yam_pi05`로 지정한다.
