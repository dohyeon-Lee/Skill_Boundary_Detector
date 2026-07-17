# ABC Dataset 이식 — 작업 핸드오프

> LIBERO(시뮬) 기반 SkillVLA 파이프라인에 **ABC**(HF `XDOF/ABC-130k`, bimanual YAM 실로봇
> teleop) 데이터셋을 연결하는 작업. 이 문서 하나로 다른 서버/새 세션에서 이어서 작업 가능.
> 브랜치: `splitVLA_ABC_dinoX`.  작성 시점: 2026-07-16.

---

## 0. TL;DR — 지금 상태 / 다음 할 일

- **데이터 다운로드 완료**: `XDOF/ABC-130k` 중 `abc_toy` subset (pick_and_place 81태스크 × 20ep = 1620 에피소드, ~270GB mcap). `dataset_ABC/_mcap/abc_toy/` 에 있음.
- **✅ v3 빌드 완료** (2026-07-17): `dataset_ABC/abc_toy/` = **1620 에피소드 / 4,973,747 프레임, fps 30, 3캠(top/left_wrist/right_wrist), 20GB**. ZED-X는 정규화(top_left→top, top_right 드롭)로 통합. stats 2종(absolute quantile + `relative_action_stats.json` chunk50/gripper 제외) 완비, per-dim names(left_gripper@6, right_gripper@13) 포함. ee_pose는 설계 제외. **바로 DP 학습 가능 상태.**
- ③은 **16샤드 병렬 + aggregate 병합**으로 재구현됨(1샤드 순차 ~20h → ~2.5h; 등가성 비트단위 검증, 완성 샤드 재사용으로 재개 가능). `_mcap/`(266G)은 이제 지워도 됨(재변환 소스는 `_abcdl/` 28G로 충분).
- **다음 코드 작업(본론, 아직 미착수)**: 카메라 3슬롯 일반화 → FSQ bimanual → SBD probe 재설계. (§6)
- **파이프라인 전환**: `configs/global_config.yaml` 의 `dataset_root: dataset_ABC` / `outputs_root: outputs_ABC` 두 줄로 전환 (LIBERO와 동일 컨벤션). DP 학습은 코드 수정 0, config만 연결.

---

## 1. 배경 / 설계 결정

### 통합 경로 = A (물리 변환), B(직결) 아님
- ABC는 HF에 **풀해상도 MCAP 원본**(>1TB)으로 배포. 그대로는 못 쓴다.
- **경로 A 확정**: `mcap → abcdl(캐시) → LeRobot v3(디스크)` 물리 변환 후, 기존 파이프라인이 v3를 **무수정 소비**.
- 경로 B(AbcdlDataset 직결)를 안 쓴 이유: 우리 파이프라인의 SBD·FSQ warm-pass·build·stage1~3이 `Dataset` 클래스가 아니라 **v3 on-disk 레이아웃(parquet/mp4)을 직접** 읽음. 또 `AbcdlDataset`은 `meta.stats`(정규화 통계)·episodes parquet·is_pad 마스크가 전무 → 어댑터로 못 메움. abcdl 캐시(`_abcdl/`)는 보존 (추후 B 소스).

### action space = absolute joint (delta 아님)
- ABC action = **팔당 관절 6 + gripper 1 = 14D absolute joint** (ALOHA식 leader-follower teleop; mcap `-arm-action` 토픽 = `RobotState.position`). LIBERO의 delta-EEF와 다름.
- **relative action 채택**: pi 계열 학습은 relative(`action − state(anchor)`, 청크당 anchor 1개, UMI 용어)로. HF 문서 `action_representations` 기준, `RelativeActionsProcessorStep`(fork에 이미 포팅됨). 데이터는 absolute 저장 유지(relative는 anchor 종속이라 컬럼화 원리적 불가 → 학습 시 on-the-fly 변환). **gripper(dim 6,13)는 exclude** (absolute 유지).

### 카메라 / state 차원
- state/action 14D: `[L_arm(6), L_grip(1), R_arm(6), R_grip(1)]`. `max_*_dim=32` 패딩 무수정 통과.
- **⚠️ ABC는 스테이션이 섞여 있다** (③에서 발견): **RealSense 1380개(3캠: top, left_wrist, right_wrist)** + **ZED-X 240개(4캠 스테레오: top_left, top_right, left_wrist, right_wrist)**. state/action은 둘 다 14D 동일, 카메라만 다름. **해상도는 ②에서 전부 256×256으로 통일**됨.
- **결정(user): 전체 1620개 사용 (정규화)** — ZED-X는 스테레오 오른쪽 눈(top_right) 버리고 top_left→top rename → RealSense와 동일 `{top, left_wrist, right_wrist}` 3캠 스키마. (depth는 어차피 mono만 써서 손실 무의미. RealSense-only 1380도 가능했으나 15% 더 확보 위해 통합.) 픽셀 레벨 검증 완료(v3 top = 원본 top_left).
  - yaml: `v3_cameras: [top, left_wrist, right_wrist]`, `camera_rename: {top_left: top}`, `camera_drop: [top_right]`.
- LIBERO는 2슬롯(`image`/`wrist_image`) → **3슬롯 확장이 남은 본론**(§6).
- fps 30 (LIBERO 20).

### ee_pose — 설계에서 제외 (user 결정)
- 초기엔 mcap `RobotState.pose`(팔별 EE 4×4)를 v3에 보존하려 했으나, **849/1620(52%)만 존재**해 스키마 일관성이 깨지고 SBD probe도 미확정이라 **설계에서 완전히 뺐다**(안 받음). `mcap_abcdl.py`의 보존 코드도 되돌림(커밋 `a8c1847`). 이미 만들어진 abcdl 캐시엔 849개에 `ff_ee_pose_*.bin`이 남아있지만 ③이 무시(무해). SBD probe(§6-C)를 EEF-space로 갈 거면 그때 재도입.

### eval
- ABC는 실로봇 데이터 → LIBERO식 sim closed-loop eval **불가**. **실로봇 배포 eval**(별도 인프라). 오프라인 HTML eval(build_data_eval)은 그대로 동작.

---

## 2. 만든 것 — `ABC_dataset` 모듈

위치: `lerobot/examples/libero/configs/generate_training_dataset/ABC_dataset/`
(`filtered_dataset` 미러 — 사용자 입장에서 LIBERO 다운로드와 동일한 UX)

| 파일 | 역할 |
|---|---|
| `ABC_dataset_config.yaml` | subset 정의(그룹/태스크/에피소드) + 해상도/fps/병렬도/안정성 노브 |
| `src/ABC_dataset_config.py` | config 리졸버(`--shell`로 bash export). global_config 병합. `abcdl_repo`는 상대경로→project_root 기준 |
| `src/download_abc_subset.py` | 다운로드 래퍼 — subset별 config 생성해 **abcdl_RLLAB/download 엔진**에 위임 |
| `download_ABC.sh` | ① 다운로드 (hang 워치독 포함, §4) |
| `src/convert_abc_dataset.py` | ② mcap→abcdl(에피소드 병렬) → ③ abcdl→v3(**16샤드 병렬+aggregate**, 카메라 정규화) → ④ stats. 핵심 변환 로직 |
| `src/compute_relative_action_stats.py` | ④-b relative action 통계 (§5) |
| `build_ABC_dataset.sh` | ②③④ 직접 실행 (로그인 노드) |
| `submit_build_ABC.sh` | ① download + ②③④를 Slurm 잡으로 (권장) |
| `README.md` | 사용법 전체 |

### 파이프라인 4단계 (`convert_abc_dataset.py`)
```
① download : HF mcap subset      → dataset_ABC/_mcap/{name}/data/{split}/<task>/<ep>/episode.mcap
② mcap→abcdl: 30Hz 리샘플+256px   → dataset_ABC/_abcdl/{name}/<task>__<ep>/  (abcdl_RLLAB 패키지, 에피소드 병렬)
③ abcdl→v3 : per-camera mp4+parquet → dataset_ABC/{name}/  (pyav 직접구현 — torchcodec 불필요)
              ⚡ 16샤드 병렬 + lerobot aggregate_datasets 병합 (v3 writer가 single-writer라
              에피소드 병렬 불가 → 샤드별 독립 writer 후 병합; 완성 샤드 재사용=재개 가능;
              streaming_encoding+encoder_threads로 인코딩 가속. _v3_shards/는 병합검증 후 자동삭제)
④ stats    : ④-a absolute quantile(ensure_quantile_stats 재사용) + ④-b relative action stats
```

### 실행 명령
```bash
cd lerobot/examples/libero/configs/generate_training_dataset/ABC_dataset

# (1회) gated 접근 + 의존성
huggingface-cli login   # https://huggingface.co/datasets/XDOF/ABC-130k 라이선스 수락
uv pip install mcap "mcap-protobuf-support>=0.5,<0.6" foxglove-schemas-protobuf

# 태스크 목록/계획 (다운로드 X)
./download_ABC.sh --list-tasks
./download_ABC.sh --dry-run

# 다운로드 (워치독 자동복구)
./download_ABC.sh

# 빌드 (권장: Slurm 잡)  — 다운로드 완료됐으면 ① 즉시 스킵되고 ②③④만 잡으로
./submit_build_ABC.sh
# 또는 직접: ./build_ABC_dataset.sh
```

---

## 3. abcdl_RLLAB 패키지 (동료 jellyho 코드)

- 위치: `{project_root}/abcdl_RLLAB/` (구 `/data2/dohyeon/abcdl_RLLAB`에서 SBD 안으로 복사, 2026-07-13). `ABC_dataset_config.yaml`의 `abcdl_repo: abcdl_RLLAB` (상대경로).
- **ABC 데이터 레이어 패키지** (변환기 아님): mcap I/O, abcdl 포맷 I/O, LeRobot 변환, HF pull. `download/` 서브폴더에 **선택형 다운로더**(논문 7개 primitive 카테고리→197태스크 taxonomy 내장) — 우리 다운로드가 이걸 씀.
- **abcdl_RLLAB은 SBD에 vendoring됨** (2026-07-16, 커밋 `a29be99`): 원래 bare gitlink(submodule)였으나 자기 `.git`이 없어 내부 파일이 SBD git에 안 잡히던 문제 → gitlink 풀고 일반 파일 50개로 편입 → 다른 서버는 `git pull`만으로 받음 (별도 동기화 불필요). (a29be99에 들어갔던 EE pose 보존은 이후 `a8c1847`에서 설계 제외로 **되돌림** — 현재 mcap_abcdl.py는 ee_pose 코드 없음.)

---

## 4. 다운로드 안정화 (겪은 문제와 해결 — 재현될 수 있으니 기록)

ABC mcap은 큰 파일(190~320MB)이라 HF 전송이 자주 끊긴다. 3중 방어:

1. **Xet 비활성** (`download_ABC.sh`가 `export HF_HUB_DISABLE_XET=1`): HF의 Xet(CAS 청크) 경로가 대용량에서 자주 스톨(100초+ 하드행). Xet을 끄면 재래식 LFS로 되돌아가고 타임아웃이 먹힌다. `.incomplete` 파일명에 `r2MXud…=` 토큰 = Xet 사용 신호. **가장 결정적**. (`original_dataset`이 검증한 레시피)
2. **워커 감소** (`download_workers: 2`, hf_transfer off, timeout 60): 동시연결 과다 시 HF가 연결 끊음.
3. **hang 워치독** (`download_ABC.sh:run_with_watchdog`): 스테이징 용량 무증가를 감지. **완료 판정 = `.incomplete`/`.lock` 없음**(filtered_dataset 기준). `.incomplete` 있는데 무증가 → 진짜 hang → 그룹 kill+재시작. 없으면 → 완료. 이미 다 받았으면 엔진 안 띄우고 즉시 통과.
   - `set -m` job-control로 자기 프로세스 그룹만 kill (남의 프로세스 불침해).
- 노브: `STALL_TIMEOUT=180`, `LISTING_GRACE=600`, `COMPLETE_CONFIRM=45`, `WATCHDOG=0`(끄기).

---

## 5. 데이터 표현 관련 수정 (v3 스키마)

### 카메라 정규화/필터 (`convert_abc_dataset.py:stage_abcdl_to_v3`, `_canon_map`)
- 스테이션 혼재 통합: `camera_rename`(raw→canonical) + `camera_drop`(버릴 raw)으로 각 에피소드 카메라를 정규화 → `v3_cameras`와 일치하는 것만 담음. 현재 ZED-X top_left→top rename, top_right drop → 전체 1620개가 3캠으로 통일.
- 프레임 emit 시 drop된 raw(top_right)는 스킵, rename된 건 canonical 키로 방출. 해상도는 canonical별로 첫 에피소드에서.
- `v3_cameras: []`(빈 값)이면 첫 에피소드 카메라 사용(구 동작; 혼재 시 heterogeneous 에러).

### ee_pose — 제거됨 (설계에서 안 받음)
- 위 "ee_pose 설계에서 제외" 참조. `mcap_abcdl.py`(②)의 보존 코드와 `convert_abc_dataset.py`(③)의 emission 모두 제거(`a8c1847`). v3에 ee_pose 컬럼 없음.

### per-dim names (`convert_abc_dataset.py:_joint_names`)
- ③이 state/action feature에 `names`(…, `left_gripper`@6, `right_gripper`@13) 심음. `RelativeActionsProcessorStep`이 exclude_joints를 **이름으로** 매칭하기 때문.

### relative action stats (`compute_relative_action_stats.py`, ④-b)
- pi 파이프라인은 `relative → normalize` 순서 → normalizer가 relative 분포 통계 필요.
- 공식 `to_relative_actions` + 이름기반 exclude(gripper) 그대로 사용 → `meta/relative_action_stats.json` **별도 파일**에 선계산. **stats.json 무접촉** → DP(absolute+MIN_MAX) 소비자와 구조적 격리.
- min/max/mean/std 전수 스트리밍, q01/q99 stride-anchor(≤20M). yaml: `relative_chunk_size: 50`, `relative_exclude_joints: [gripper]`.
- **데이터/stats.json 무변형** (relative는 학습 시 on-the-fly).

---

## 6. 남은 본론 (아직 미착수) — LIBERO 하드코딩을 ABC에 일반화

### A. 카메라 2→3 슬롯 (가장 먼저, 기계적, LIBERO 회귀검증 가능)
LIBERO는 primary(`observation.images.image`)+wrist(`wrist_image`) 2슬롯. ABC는 top+양wrist 3슬롯.
하드코딩 지점:
- `lerobot/src/lerobot/policies/skillVLA/dataset_skillVLA.py:39-40` `CAM_3RD`/`CAM_WRIST` (모든 stage 소비)
- `dataset_transitions.py` transition-pack 2캠 고정
- FSQ terminator `use_third`/`use_wrist` 2슬롯 (`FSQ.py`, `train_fsq.sbatch` 키)
- 모델 forward: `modeling_skillVLA.py`, `modeling_skill_expert.py`의 `.image`/`.wrist_image`
- `processor_skillVLA.py` 카메라 키

### B. FSQ bimanual (`FSQ.py`)
- `FSQ.py:54 N_GRIPPER_DIMS=2` + `zero_ground_trajectory`가 "마지막 2dim=gripper, 나머지=단일 pose" 전제 → 단일팔용. ABC는 gripper 인덱스 **{6,13}**, pose 2벌. 일반화 필요.
- B-spline fitting은 차원 불가지론적이라 OK지만 gripper 인덱싱만 재설계.

### C. SBD 구면 probe (`skill_divider.py`, 최난·연구항목)
- `skill_divider.py:_generate_spherical_samples`가 `gt_chunk[:,:3]`을 **단일 EEF xyz 변위**로 가정하고 구면 회전. ABC는 `[:3]`이 왼팔 관절 3개일 뿐 → 물리적 의미 없음.
- 선택지: (a) EEF-space 유지 — 단 **ee_pose를 v3에서 뺐으므로**(설계 제외) 다시 도입하거나 FK로 관절→EEF 계산 필요 + probe를 joint로 되돌리는 IK 필요 / (b) joint-space 재정의(IK 불필요, 구면 직관 재정의) / (c) relative-joint 관점. **abc_toy 실데이터로 관절 궤적 보고 결정.** ee_pose가 필요하면 그때 재도입(abcdl 캐시 849개엔 남아있고, mcap_abcdl.py 보존코드 되살리면 됨).
- ⚠️ DP action space는 어느 선택이든 **joint 유지** — 바뀌는 건 SBD probe 좌표계뿐. (next-state를 action으로 쓰는 것 아님)

### 기타
- relative action **배선**: **DP는 완료(2026-07-17)** — user 결정("probe를 VLA와 같은 relative 공간에서")에 따라 DP도 relative로 학습. 구현: `DiffusionConfig.use_relative_actions`/`relative_stats_path` + `processor_diffusion.py`의 `DiffusionRelativeActionsProcessorStep`(공식 스텝 서브클래스 — **DP는 state가 히스토리 윈도우 (B,n_obs,D)라 anchor=윈도우 마지막(현재) state로 축약** 후 위임) + normalizer action stats를 ④-b relative stats로 스왑 + 짝 `AbsoluteActionsProcessorStep`(역변환) + horizon>chunk 가드. 수치검증: 변환·gripper-absolute·round-trip 전부 0오차. 배선: `dp_config.yaml dp_relative: true` → resolver `DP_RELATIVE` → sbatch가 `--policy.use_relative_actions` + stats 경로(fail-fast). **pi05/skillVLA/skill_expert 배선은 아직** (stage1 진입 시; 같은 ④-b stats 사용, chunk50=pi05 호환).
- DP config: `dp_config.yaml` `target_dataset: abc_toy`, `train_DP: true`, `dp_relative: true`(신설, 기본 on — LIBERO 돌릴 땐 false로), `dp_batch_size: 256` 권장. 시간창 `n_obs/horizon`은 프레임수라 30Hz에서 시간 짧아짐 — 프레임수 유지(20/24)로 1차 (SBD 경계 이상하면 1순위 튜닝). **DP는 state-only라 카메라 무관.**

---

## 7. 겪은 버그 (재발 방지 기록)

1. **ffprobe 누락** (빌드 0완료의 원인): `_ensure_ffmpeg`가 `ffmpeg`만 shim하고 `ffprobe` 누락. abcdl `encode_strict_h264`가 mp4 후 `ffprobe`로 프레임수 검증하는데 없어서(FileNotFoundError) 매 에피소드가 mp4까지만 만들고 실패 → 수백 .tmp, 완료 0. **수정**: `_ensure_ffmpeg`가 ffprobe도 shim (시스템 우선, 없으면 imageio ffmpeg 기반 wrapper — conda ffprobe는 lib 의존 회피). imageio 번들엔 ffmpeg만 있고 ffprobe 없음이 함정.
2. **빌드 속도 오판**: 4워커 초반 측정(오염)으로 19시간 추정했으나, 단일 에피소드 프로파일 = ~14초(병목=단일스레드 인코딩 ~10s, ffprobe 아님). 실측 16워커 = **23 ep/분 → ②단계 ~1시간**. `-threads 1`+다중워커가 처리량 정답.
3. **워치독 완료 오판**: 초기 워치독이 "다 받아서 무증가"를 hang으로 오판→무한 재시작. `.incomplete` 유무로 완료/hang 구분하게 수정(§4).
4. **submit이 로그인 노드에서**: `build_ABC_dataset.sh`(직접)와 `submit_build_ABC.sh`(Slurm) 혼동. submit은 ① download를 먼저 로그인노드에서 실행 후 sbatch. 다운로드 완료면 즉시 스킵.
5. **③ 순차 20시간 사건**: ③이 single-writer 순차 루프라 1워커 0.5ep/분 → ~52h(>walltime)로 잡 2회 낭비. 원인=②의 16워커 처리량을 ③ 추정에 잘못 적용(프레임당 비용은 ②≈③ ~11-13ms/f, 병렬도만 16배 차이). 1차 완화=streaming_encoding+encoder_threads(4×), 근본 해결=**16샤드 병렬+aggregate**(13.6배, 17.7ep/분, 총 ~2.5h 완주). 교훈: (a) 순차/병렬 구조 확인 없이 처리량 외삽 금지, (b) v3 writer 병렬화의 정석=샤드+aggregate_datasets, (c) 재개 불가 장시간 작업엔 walltime 여유(48h) 필수.

---

## 8. 성능/리소스 설정 (현재)
- ②: `convert_workers: 16` (에피소드 프로세스 병렬, ffmpeg CLI).
- ③: `v3_shards: 16` + `v3_encoder_threads: 2` (16 독립 writer × h264 2스레드; `KEEP_SHARDS=1`로 샤드 보존 가능). 실측 17.7 ep/분.
- submit: `--cpus-per-task=32`, `--mem=96G`, `--time=48:00:00`. GPU는 base_qos가 gpu:1 강제하지만 **실제 미사용**(CPU/ffmpeg; nvenc은 pyav 빌드에서 hang이라 배제).
- 실측 총 소요: 다운로드 ~9h(11MB/s 서버상한) + ② ~1.5h + ③ ~2.2h + aggregate/④ ~15분.

---

## 9. 다른 서버 동기화 체크리스트
1. 코드: 브랜치 `splitVLA_ABC_dinoX` 커밋/푸시 (ABC_dataset 폴더 + `global_config.yaml`).
2. **abcdl_RLLAB**: 이제 SBD에 vendoring되어 커밋됨(`a29be99`, ee_pose는 `a8c1847`에서 제거) → 코드는 `git pull`로 따라옴. 별도 동기화 불필요. `abcdl_repo`는 상대경로라 `{project_root}/abcdl_RLLAB`면 자동.
3. 데이터: `dataset_ABC/`(대용량)는 재다운로드하거나 `sync_server.sh`로 전송. 재다운로드가 안전(idempotent, 워치독).
4. 의존성: `uv pip install mcap "mcap-protobuf-support>=0.5,<0.6" foxglove-schemas-protobuf` + gated `huggingface-cli login`.
5. ffprobe: 시스템에 없어도 됨(imageio ffmpeg 기반 wrapper 자동 생성). 단 `imageio-ffmpeg` 설치 필요.

---

## 10. 핵심 참조 (file:line)
- `abcdl_RLLAB/abcdl/convert/mcap_abcdl.py` — mcap→abcdl (30Hz 리샘플·다운스케일; ee_pose 없음)
- `ABC_dataset/src/convert_abc_dataset.py:_canon_map` — 카메라 정규화 (ZED top_left→top, top_right 드롭)
- `abcdl_RLLAB/abcdl/format/encode.py:28` `probe_frame_count` (ffprobe 사용처)
- `ABC_dataset/src/convert_abc_dataset.py:53` `_ensure_ffmpeg` (ffmpeg+ffprobe shim)
- `ABC_dataset/src/convert_abc_dataset.py` `stage_mcap_to_abcdl`/`stage_abcdl_to_v3`/`stage_stats`
- `ABC_dataset/src/convert_abc_dataset.py:_build_v3_shard`/`_build_v3_dataset` — ③ 샤드 워커 (single-writer 코어 루프)
- `lerobot/src/lerobot/datasets/aggregate.py:aggregate_datasets` — 샤드 병합 (episode/task 인덱스 재매핑 + stats 병합)
- `ABC_dataset/src/compute_relative_action_stats.py` — relative stats
- `ABC_dataset/download_ABC.sh:run_with_watchdog` — hang 복구
- `lerobot/src/lerobot/processor/relative_action_processor.py` — relative 변환 스텝(미배선)
- `lerobot/examples/libero/skill_divider.py` — SBD (구면 probe, §6-C 재설계 대상)
- `lerobot/examples/libero/FSQ.py:54` — `N_GRIPPER_DIMS` (§6-B)
- `lerobot/src/lerobot/policies/skillVLA/dataset_skillVLA.py:39` — `CAM_3RD`/`CAM_WRIST` (§6-A)
