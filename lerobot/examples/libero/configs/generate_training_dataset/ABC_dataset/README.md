# ABC Dataset Download & Convert (XDOF/ABC-130k → LeRobot v3)

`filtered_dataset/`이 필터링된 LIBERO(LeRobot v2.1)를 받아 v3로 변환하듯, 이 폴더는
**ABC**(bimanual YAM 실로봇 teleop, 14D state/action, 30Hz, 카메라 top/left_wrist/right_wrist)를
받아 변환합니다. HF에는 **풀해상도 MCAP 원본**(>1TB, `data/{train,val}/<task>/<episode>/episode.mcap`)으로
배포되므로 subset만 골라 받아 2단 변환합니다:

| 단계 | 무엇 | 코드 |
|---|---|---|
| ① download | mcap subset → `_mcap/{name}/` | **`{abcdl_repo}/download` 선택형 다운로더** (그룹/태스크/에피소드 지정; `src/download_abc_subset.py`는 subset별 config를 생성해 위임) |
| ② mcap→abcdl | 30Hz 고정클록 리샘플 + 정방 다운스케일(256) + 스택 mp4 캐시 → `_abcdl/{name}/` | `abcdl_RLLAB` 패키지 (`mcap_to_abcdl`, 에피소드 병렬) |
| ③ abcdl→v3 | per-camera mp4 + parquet **진짜 v3** → `dataset_ABC/{name}/` | `src/convert_abc_dataset.py` (pyav 기반 — **torchcodec 불필요**) |
| ④ stats | quantile(q01..q99) 보장 | `../filtered_dataset/ensure_quantile_stats.py` 재사용 |

최종 산출물은 LIBERO와 동일한 v3 레이아웃 → 하류 파이프라인(DP/SBD/FSQ/skillvla)이 **무수정 소비**.
v3 feature 키는 `observation.images.{top,left_wrist,right_wrist}` 로 생성됩니다
(하류 3-카메라 슬롯 매핑은 별도 작업 항목).

`_abcdl/` 캐시는 지우지 마세요 — v3 재빌드 소스이자, 추후 `AbcdlDataset` 직결
(memmap/스트리밍 고속로더)을 쓰게 되면 그때의 데이터 소스입니다.

## Subset 정의 (그룹/태스크/에피소드 지정)

다운로드 엔진은 `{abcdl_repo}/download/src/download_abc.py` — **논문 Appendix A의 7개
primitive 카테고리 → 197개 태스크 매핑**(`{abcdl_repo}/download/config.yaml`의 `groups:`)을
단일 소스로 씁니다: `pick_and_place / fine_pick_and_place / folding / insertion_ejection /
tool_use / sorting / tying_untying`.

`ABC_dataset_config.yaml`의 `abc_subsets`에서 로컬 이름 → 엔진 스키마 그대로:

```yaml
abc_subsets:
  abc_toy:                       # {project_root}/dataset_ABC/abc_toy 로 생성됨
    split: train                 # 기본 split (entry별 split: 오버라이드 가능)
    downloads:
      - group: tool_use          # 카테고리 단위
        episodes: 2              #   태스크당 앞에서 2개 (uuid 정렬, 결정적)
      - task: tie_the_shoes      # 단일 태스크
        episodes: all            #   전부  ([0,1,2] 인덱스, ["uuid부분"] 매칭도 가능)

# extra_groups:                  # 공식 7그룹에 병합되는 커스텀 버킷 (선택)
#   my_pilot: [open_the_umbrella, tie_the_shoes]
```

로컬 이름이 곧 하류 `source_dataset` 이름입니다 (LIBERO의 `libero_90_full_full` 자리).
태스크 폴더명 확인: `./download_ABC.sh --list-tasks` (`--counts`로 에피소드 수까지, 느림).

## Run

```bash
cd lerobot/examples/libero/configs/generate_training_dataset/ABC_dataset

# 0) (1회) gated 데이터셋 접근 + mcap 읽기용 의존성:
#    https://huggingface.co/datasets/XDOF/ABC-130k 에서 라이선스 수락 후
huggingface-cli login
uv pip install mcap "mcap-protobuf-support>=0.5,<0.6" foxglove-schemas-protobuf

# 1) 태스크 목록/계획 확인 (다운로드 없음)
./download_ABC.sh --list-tasks
./download_ABC.sh --dry-run

# 권장 (원샷): 다운로드는 현재 노드(로그인)에서, 변환은 CPU Slurm 잡으로
./submit_build_ABC.sh
ABC_ONLY="abc_toy" ./submit_build_ABC.sh             # 일부만

# 또는 전부 현재 노드에서 직접:
./download_ABC.sh && ./build_ABC_dataset.sh
FORCE=1 ./build_ABC_dataset.sh                       # 최종 v3 재빌드 (abcdl 캐시 재사용)
```

**별도 파이프라인으로 이미 받은 데이터 진입점**: mcap이면 `dataset_ABC/_mcap/{name}/` 아래
(하위 구조 무관, `**/episode.mcap` 글롭), 이미 abcdl 에피소드 디렉토리면
`dataset_ABC/_abcdl/{name}/<ep>/` 에 놓고 `./build_ABC_dataset.sh` — mcap 스테이징이 없으면
②를 자동 스킵하고 ③(v3 변환)부터 진행합니다.

ffmpeg CLI가 노드에 없어도 됩니다 — `.venv`의 `imageio_ffmpeg` 번들 바이너리를
`dataset_ABC/_tools/ffmpeg` 로 자동 shim 합니다. torchcodec도 불필요합니다
(③이 pyav로 직접 디코드 — 클러스터에 시스템 libav가 없어 torchcodec은 로드 불가).

## 하류 파이프라인 전환

이 데이터로 전체 파이프라인(DP/FSQ/skillvla/stage1/2)을 돌리려면 `configs/global_config.yaml`에서:

```yaml
dataset_root: dataset_ABC   # 데이터 + 파생물(FSQ_dataset, skillvla_dataset, ...)
outputs_root: outputs_ABC   # 모든 학습 출력
```

두 줄만 바꾸면 모듈 yaml 수정 없이 전환됩니다 (filtered와 동일 컨벤션).
이후 각 모듈 yaml의 `source_dataset`/`pt_dataset` 등을 subset 이름(`abc_toy` 등)으로.

## LIBERO와 다른 점 (하류 작업 예고)

- **카메라 3개**: `observation.images.{top,left_wrist,right_wrist}` — 현 파이프라인의
  2-슬롯(`image`/`wrist_image`) 규약을 3-슬롯으로 확장 필요 (dataset_skillVLA CAM_*,
  FSQ terminator, transition_pack, 모델 forward).
- **bimanual 14D**: `[L_arm(6), L_grip(1), R_arm(6), R_grip(1)]` — `max_*_dim=32` 패딩은
  그대로 통과하지만, `FSQ.py`의 `N_GRIPPER_DIMS=2`(마지막 2dim=gripper) + zero-grounding은
  단일팔 전제라 재설계 필요. SBD 구면 probe도 동일.
- **fps 30** (LIBERO 20): obs window/horizon 등 시간 관련 config 재튜닝.
- **sim eval 없음**: 실로봇 배포 eval(별도 인프라). 오프라인 HTML eval(build_data_eval)은 그대로 동작.
