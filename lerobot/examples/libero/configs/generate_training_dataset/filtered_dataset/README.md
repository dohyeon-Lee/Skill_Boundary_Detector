# Filtered LIBERO Download (OpenVLA no_noops)

`original_dataset/`이 **원본** LIBERO HDF5(4500 demos, 실패 replay 포함)를 받아 변환하는 반면,
이 폴더는 **필터링된** 공개본을 받습니다 — OpenVLA 전처리(실패 replay 제거 + no-op 프레임
제거)를 거쳐 IPEC-COMMUNITY가 LeRobot v2.1로 올려둔 suites:

| 로컬 이름 | HF repo | eps |
|---|---|---|
| libero_90 | IPEC-COMMUNITY/libero_90_no_noops_lerobot | 3921 |
| libero_10 | IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot | 379 |
| libero_spatial | IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot | 434 |
| libero_object | IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot | 452 |
| libero_goal | IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot | ~430 |

## 파이프라인 (build_filtered_dataset.sh가 전부 수행)

```text
① download  : v2.1 원본 → dataset_filtered/_v21/{name}
② migrate   : lerobot/scripts/convert_dataset_v21_to_v30.py
              (upstream 변환기는 완료 시 root를 v3.0으로 제자리 교체 + 원본을 {root}_old로 이동)
③ 정리       : 변환본 → dataset_filtered/{name} (최종, 바로 사용)
              v2.1 원본 → _v21/{name}로 복원·보존 (CLEAN_V21=1이면 삭제)
④ remap     : gripper action 0/1(0=close) → ±1(+1=close) — "zero_close" (new = 1−2·old)
              기존 libero_dataset/libero_90과 동일 컨벤션 (state는 변경 불필요 — hub와 동일)
⑤ stats     : meta/stats.json quantile(q01..q99) 보장
```

이미지 180° 회전은 **불필요**합니다 (OpenVLA 재생성 단계에서 이미 바로잡힘 —
`original_dataset/rotate180_postprocess.py`는 원본 HDF5 경로에만 해당).

## Run

```bash
cd lerobot/examples/libero/configs/generate_training_dataset/filtered_dataset

# 권장 (원샷): 다운로드는 현재 노드(로그인)에서, 변환+remap+stats는 CPU Slurm 잡으로
./submit_build_filtered.sh
FILTERED_ONLY="libero_90" ./submit_build_filtered.sh             # 일부만

# 또는 전부 현재 노드에서 직접:
./build_filtered_dataset.sh                          # 전체 suite (이미 빌드된 것은 스킵)
FILTERED_ONLY="libero_10 libero_goal" ./build_filtered_dataset.sh   # 일부만
RECOMPUTE_STATS=1 ./build_filtered_dataset.sh        # remap의 해석적 stats 대신 데이터에서 재계산
```

## 하류 파이프라인 전환

이 데이터로 전체 파이프라인(DP/FSQ/skillvla/stage1/2)을 돌리려면 `configs/global_config.yaml`에서:

```yaml
dataset_root: dataset_filtered   # 데이터 + 파생물(FSQ_dataset, skillvla_dataset, ...)
outputs_root: outputs_filtered   # 모든 학습 출력
```

두 줄만 바꾸면 모듈 yaml 수정 없이 전환됩니다.
