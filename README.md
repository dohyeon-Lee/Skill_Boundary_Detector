# Skill Boundary Detector (SkillVLA)

학습 조건 변경·추후 점검 사항: [SkillVLA 구현·실험 주의사항](lerobot/examples/libero/configs/train_skillVLA/KNOWN_ISSUES.md).

Slurm job과 종료 후 Codex 지시 관리 UI: `bash slurm_monitor_ui.sh` → 포트 **8767** 전달.
실행·대기·종료 job, 출력 폴더명, job별 지시와 전송 이력을 확인할 수 있다.
[Slurm Desk 사용법](tools/slurm_monitor/README.md).

## Slurm 실행 방식

`lerobot/examples/libero/configs/servers/rllab.yaml`의 `slurm_submit_mode`에서
`sbatch`, `srun`, `srun_existing`을 선택한다. `scheduler: slurm`은 그대로 둔다.

- `sbatch`: 기존처럼 학습 종료와 함께 GPU 할당을 반납한다.
- `srun`: `./submit_train.sh` 실행 시 고유한 `srun-잡이름-시각-고유번호` tmux를 자동 생성한다.
  지정한 노드·GPU를 기다렸다가 확보하면 학습을 자동 시작한다. SSH 연결이 끊겨도 계속된다.
  두 모델 동시 학습도 같은 할당 안에서 실행한다. 실행할 때 출력되는 명령으로 tmux에 접속한다.
  학습이 끝나거나 실패해도 셸과 GPU 할당은 제한시간 내에 유지된다.
  `sbd_retry`는 제출 당시 설정으로 재시도하고, 설정 변경 후 `./submit_train.sh`를 실행하면
  현재 할당을 재사용한다. GPU를 반납할 때는 할당된 셸에서 `exit`한다.
- 한 번만 기존 방식으로 제출: `SBD_SLURM_SUBMIT_MODE=sbatch ./submit_train.sh`.
  배열·의존성·숫자 job ID가 필요한 제출은 sbatch를 유지한다.
- 이미 확보한 GPU에 추가 학습/평가를 붙이려면 `slurm_submit_mode: srun_existing`과
  `srun_job_id: "실행중인 잡 ID"`를 설정한다. `rllab.yaml`에 주석 처리된 예시가 있다.
  본인 소유의 RUNNING 상태인 단일 노드·단일 GPU 할당만 허용한다.
  새 tmux에서 `srun --jobid=<ID> --overlap` step을 시작하며 GPU를 새로 요청하지 않는다.
  잡이 대기 중·종료·다른 사용자 소유이면 오류로 종료하며 새 할당으로 전환하지 않는다.
  추가 step을 닫아도 원래 할당은 유지되고, 원래 할당이 종료되면 추가 step들도 종료된다.
  추가 작업마다 모델 output/평가 output_name을 구분한다. 로그와 staging 경로는 step별로 분리한다.

  ```bash
  # VSA 폴더: YAML 목록 전체가 아니라 추가할 단일 모델을 지정
  SBD_SLURM_SUBMIT_MODE=srun_existing SBD_SRUN_JOB_ID=68847 ./submit_train.sh <model>
  # eval 폴더: 기존 할당의 한 GPU로 모든 task/panel을 분배
  SBD_SLURM_SUBMIT_MODE=srun_existing SBD_SRUN_JOB_ID=68847 ./submit_eval.sh
  ```

  srun/srun_existing 평가는 한 GPU를 사용한다. `eval_max_workers_per_gpu`는 함께 실행할
  평가 프로세스 수이므로 남은 VRAM에 맞게 설정한다. sbatch 평가는 기존 다중 GPU 배열을 유지한다.
- srun의 Slurm 상태는 **할당의 상태**다. 학습이 실패해도 셸이 살아 있으면 Slurm Desk에는
  `RUNNING`으로 보인다. 학습별 로그는 기존 `logs/`, 실행 기록은 `.cache/srun_sessions/`에 저장된다.

## VSA gradient checkpointing

`PT/VSA/vsa_train_config.yaml`의 `training.gradient_checkpointing`에서 경로별로 설정한다.

```yaml
gradient_checkpointing:
  action_expert: true
  skill_planner: true
  vlm: false
  vision: false
```

각 값은 `false`/`0`이면 끄기, `true`이면 해당 경로 전체, 정수 N이면 마지막 N개 레이어,
`[1, 18]`처럼 목록이면 지정한 레이어만 켠다. 목록의 레이어 번호는 1부터 시작한다.
`action_expert`는 Action 생성 경로, `skill_planner`는 같은 Expert 전반층을 재사용하는 Fullskill 경로다.
별도 Planner 네트워크를 추가하는 설정은 아니다.
`vlm`은 Gemma 조건 경로와 bottleneck 갱신, `vision`은 DINO 또는 SigLIP 이미지 인코더다.
17+1 구조에서는 Action Expert 18층과 Planner 17층을 각각 선택한다.
실제 레이어 수를 넘는 값은 오류로 처리하며, 기존 bool 설정과 저장된 체크포인트도 읽을 수 있다.
새 YAML의 기본값은 네 경로 모두 `false`다. 변경은 다음 학습 프로세스부터 적용된다.
구형 joint-attention 모델은 Action/VLM을 하나의 연산으로 실행하므로 두 경로의 선택을 같게 둔다.

## Vanilla π0.5 VLM variants

VSA의 `architecture.name`에 `vanila_last_pi05vlm` 또는
`vanila_nouv_last_pi05vlm`을 지정한다. π0.5의 SigLIP + Gemma를 사용하고
Skill·목표 UV·현재 Proprio는 기존 VLM 토큰 프롬프트로 입력한다.

이 두 모델은 Fullskill/Planner 없이 Action Expert 입력을 1+17로 분리한다.
L1은 timestep + Skill + skill-start XYZ + 선택적 목표 UV, L2–L18은
timestep + 현재 Proprio를 직접 입력받는다. `_nouv`는 L1의 직접 UV만 제거한다.
최종 norm도 후반 조건을 사용하며, 현재 최종 bottleneck은 L18에만 연결한다.
`visual_bridge_last_n_layers`는 두 이름을 선택하면 17로 설정된다.
`vanila_last_pi05vlm_9_9`는 UV가 있는 모델의 9+9 변형이다. L1–L9에
timestep + Skill + skill-start XYZ + 목표 UV, L10–L18에 timestep + 현재 Proprio를
입력하며, vision bottleneck은 동일하게 L18에만 연결한다. 이 이름은
`visual_bridge_last_n_layers`를 9로 설정하고 별도 revision/출력 폴더를 사용한다.
Vanilla 전체의 모델 이미지 입력은 학습·추론 모두 현재 Top/Wrist이다.
시작 이미지 디코딩·인코딩·시작 auxiliary loss를 생략하고 현재 aux3만 계산한다.
Skill·시작 XYZ·선택적 목표 UV 등 기존 직접 입력과 Expert 연결 범위는 유지한다.
aux3·XYZ 공간 UV 랜덤화·clipping과 경로별 checkpointing 설정을 유지한다.
Auxiliary 가중치는 `architecture.spatial_loss_weight`를 따른다.

π0.5 VLM의 `_wovis` 두 모델은 현재 Top/Wrist 이미지만 인코딩하고 현재 경로의
aux3 loss만 계산한다. 시작 프레임 디코딩·시작 VLM 인코딩·시작 auxiliary loss는
생략하며, Skill·시작 XYZ·목표 UV 및 Fullskill 학습 조건은 유지한다.
시작 이미지를 별도로 필요로 하는 Predictor/Joint 경로는 해당 입력을 유지한다.
Vanilla와 해당 `_wovis`의 새 학습 revision에는 `current_aux_only_v2`가 붙는다. 기존 revision의 체크포인트는
기존 조건으로 읽을 수 있지만 새 학습에서 자동 resume하지 않는다.
수정은 새로 시작하는 학습에 적용된다.

## RunPod

**처음 한 번 (계정)**: Hugging Face **Write** 토큰 생성 → `google/paligemma-3b-pt-224`,
`facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-lvd1689m` 이용 동의 →
wandb API 키 생성 (https://wandb.ai/authorize, 만들 때 한 번만 보인다).

### 1. 설치 → 로그인 → 데이터
```bash
cd ~/workspace 2>/dev/null || cd /workspace                              # 볼륨이 있는 폴더로
git clone https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git && cd Skill_Boundary_Detector
bash setup_env.sh && source .venv/bin/activate                           # 시스템 의존성 + 기본/Molmo/GR00T 환경 설치/검증

hf auth login               # 토큰 붙여넣기 (화면에 안 보이는 게 정상), git credential 질문은 n
wandb login                 # wandb API 키

bash hf_sync.sh pull dataset_filtered --models --yes     # 데이터셋 전체 + 사전학습 모델

# 기존 skillVLA_stage1 경로로 업로드된 VSA run의 010000부터 이어서 학습할 때:
# (Hub의 기존 폴더명은 유지되며, FT 소스/평가는 구 경로도 자동으로 찾는다.)
# 실제 checkpoint를 받은 뒤 각 run에 checkpoints/last -> 010000을 자동 생성한다.
bash hf_sync.sh pull dataset_filtered \
  --models \
  --checkpoints 'outputs_filtered/skillVLA_stage1/VSA/*/checkpoints/010000' \
  --yes

# 새 서버에서 구 경로로 받은 PT run을 이어학습하려면 1회 이름 변경:
# 이미 skillVLA_PT가 있다면 덮어쓰지 말고 필요한 run만 따로 옮긴다.
if [ -d outputs_filtered/skillVLA_stage1 ] && [ ! -e outputs_filtered/skillVLA_PT ]; then
  mv -T outputs_filtered/skillVLA_stage1 outputs_filtered/skillVLA_PT
fi
```
- `setup_env.sh`는 빌드 도구(`gcc`, `g++`, `make`), `curl`, `git`, `rsync`, `tmux`, `tar`와
  선택된 Python의 개발 헤더(`Python.h`, `pyconfig.h`)를 확인한다. Ubuntu/Debian에서 부족한 패키지는
  `apt-get`으로 자동 설치하며, 일반 계정에서는 `sudo` 비밀번호를 요청할 수 있다. 스크립트 전체를
  `sudo`로 실행할 필요는 없다. 이미 준비된 환경에서는 시스템 패키지를 설치하지 않는다.
  다른 OS나 관리자 권한이 없는 서버에서는 부족한 항목을 표시하고 중단한다.
- VSA + Terminator만 필요하면 `SBD_SETUP_MOLMOACT2=0 SBD_SETUP_GROOT=0 bash setup_env.sh`.
- pod 환경변수에 `HF_TOKEN`, `WANDB_API_KEY`를 넣어두면 로그인 생략. 요구사항: CUDA 12.8+ 드라이버, `gcc`·`make`, EGL.
- 이어서 학습할 체크포인트: `bash hf_sync.sh pull --checkpoints outputs_filtered/<group>/<run>/checkpoints/<step> --yes`
  (`checkpoints/last`가 자동으로 잡혀 바로 이어서 학습된다)
- PT의 새 출력 경로는 `outputs_filtered/skillVLA_PT/{VSA,Predictor,Terminator}/<run>`.
  새 경로로 업로드된 체크포인트는 위 `--checkpoints`에서도 `skillVLA_PT`를 사용한다.
  기존 Hub 데이터와 저장된 체크포인트 메타데이터는 이름을 바꾸지 않는다.
- 일부만: `bash hf_sync.sh pull` (메뉴), `dataset_filtered/libero_90_full_full`, `--models pi05_base,dinov3-vits16`
- `setup_env.sh`는 Hugging Face LeRobot MolmoAct2 코드를 고정 커밋으로 준비하고, 별도 `.venv_molmoact2_lerobot`에
  고정 패키지를 설치한다. LeRobot 전용 venv는 재사용하며, 기본 `.venv`와 패키지를 섞지 않는다.
  **모델 가중치/데이터는 자동 다운로드하지 않는다.** Molmo 모델도 별도로 받는다:
  `.venv_molmoact2_lerobot/bin/hf download allenai/MolmoAct2 --local-dir models/molmoact2_base`
  ([Molmo PT/FT 사용법](lerobot/examples/libero/configs/train_molmoact2/README.md)).
- 기존 환경을 유지하고 Molmo만 추가/보완: `bash setup_env.sh --molmoact2-only`.
  설치 없이 검사만: `bash setup_env.sh --molmoact2-only --check`.
  Molmo가 필요 없으면 `SBD_SETUP_MOLMOACT2=0 bash setup_env.sh`.

### 2. 학습 실행
yonsei와 같은 명령. RunPod이면 Slurm 대신 백그라운드로 바로 시작된다.
```bash
cd lerobot/examples/libero/configs/train_skillVLA/PT/VSA && ./submit_train.sh

S=~/workspace/Skill_Boundary_Detector/lerobot/examples/libero/configs/src/submit_job.sh
bash $S list          # squeue 대신
bash $S stop <id>     # scancel 대신
```
- 로그는 제출한 폴더의 `logs/`. 터미널을 닫아도 계속된다. 빈 GPU를 자동 배정하고, 없으면 시작하지 않는다.
- 죽었으면 같은 명령을 다시 치면 마지막 체크포인트부터 이어서 한다.
- 되는 것: SkillVLA PT VSA, Predictor/Terminator, FT, pi05 PT/FT, MolmoAct2 PT/FT, DP, FSQ.
  Molmo는 전용 Python을 자동 사용한다 (직접 activate할 필요 없음; 첫 GPU 학습은 `--smoke`로 확인).
  eval, build_data, pi05 cycle은 아직 Slurm 전용.

### 3. 체크포인트 자동 업로드 + pod 자동 종료 (tmux에서)
```bash
tmux new -s hfwatch
bash hf_sync.sh watch --keep 3 --protect 050000,100000                    # 업로드만
bash hf_sync.sh watch --keep 3 --protect 050000,100000 --done-step 100000 --done-time 2d
# 첫 질문에 y → Ctrl-b d (다시 보기: tmux attach -t hfwatch)
```
- 5분마다 다 저장된 체크포인트를 공개 저장소 `Dohyeon-Lee-02/SkillVLA-checkpoints`에 올린다 (resume 가능).
- `--keep 3 --protect`: run마다 최근 3개 + 지정 step만 남기고 **Hugging Face와 pod 디스크 둘 다에서** 지운다.
  대상은 이 pod가 학습한 run뿐. 받아오기만 한 run·step은 건드리지 않는다.
- `--done-step` / `--done-time`(`48:00:00`, `12h`, `2d`)을 주면 pod를 Terminate한다. 둘 다 없으면 끄지 않는다.
  조건: 그 step 업로드 / 그 시간 경과 / 학습이 전부 멈춤(에러 포함). 업로드를 두 번 확인한 뒤 끈다.
- 끄기 직전 학습 로그와 `jobs.tsv`를 비공개 저장소의 `runpod_logs/<시각>_<pod ID>/`에 올린다:
  `hf download Dohyeon-Lee-02/SkillVLA --repo-type dataset --include "runpod_logs/*" --local-dir runpod_logs`
- 옵션을 바꾸려면 `Ctrl-C` 후 다시 켠다. 학습을 모두 멈추면 pod도 꺼지니, 다시 돌릴 거면 watch를 먼저 끈다.

### 폴더 / 문제 해결
```
<workspace>/Skill_Boundary_Detector   코드, .venv, .venv_molmoact2_lerobot, third_party/lerobot_molmoact2
<workspace>/outputs_filtered          새 학습 결과 + 받아온 체크포인트 (컨테이너 디스크: pod를 지우면 사라짐)
/workspace-global/{dataset_filtered, models, outputs_filtered}   Global volume (없으면 전부 저장소 안, 자동 감지)
```
| 증상 | 해결 |
|---|---|
| `Not logged in` / 401 | `hf auth login` 또는 `HF_TOKEN` 설정 |
| 403 "이용 동의가 필요합니다" | 위 모델 페이지에서 같은 계정으로 동의 |
| wandb `api_key not configured` | `wandb login` 후 다시 제출 (체크포인트 없이 생긴 출력 폴더는 삭제) |
| `Cannot pick a server` | `git pull`, 급하면 `export SBD_SERVER=runpod` |

그 밖의 옵션: `bash hf_sync.sh --help`

### Hugging Face 동기화 UI

파일이 있는 서버에서 `bash hf_sync_ui.sh` (또는 `bash hf_sync.sh --web`) 실행 후,
VSCode Ports에서 **8766**을 전달해 브라우저로 엽니다. 데이터셋 업로드·다운로드,
사전학습 모델·체크포인트 다운로드, 새 체크포인트 자동 업로드를 지원합니다.
HF 데이터셋·체크포인트 관리 메뉴에서는 원격 파일·폴더를 선택해 파일 수·용량을 확인한 뒤 삭제할 수도 있습니다.
폴더 선택 → 전송 준비 → 경로 확인 → 시작 순서이며, 작업별 로그·취소와 실제 전송 없는
미리보기 실행을 제공합니다. 인증은 해당 서버의 `hf auth login` / `HF_TOKEN`을 사용합니다.
[UI 사용법과 실행 옵션](tools/server_sync/HF_README.md)

## 다른 서버 (yonsei, rllab)
- 설치는 똑같이 `bash setup_env.sh` (서버는 clone 경로로 자동 감지: `lerobot/examples/libero/configs/servers/*.yaml`).
- 데이터 옮기기: `./sync_server.sh` (yonsei ↔ rllab), `bash hf_sync.sh push` (Hugging Face 비공개 저장소).
- 여러 모델/체크포인트를 브라우저에서 골라 보내기: **외부 서버에서** `bash sync_server.sh --web` 실행 → VSCode Ports에서 `8765` 전달 → 브라우저 접속. 선택한 폴더들의 상대 경로를 유지하며, 전송 전 목적지를 확인합니다. [사용법](tools/server_sync/README.md)
- yonsei에서는 `watch`를 켜지 않는다 (기존 체크포인트를 전부 올리려 한다).

rllab의 `node100`에 배정된 Slurm 잡은 공통 `runtime_env.sh`에서 자동으로
`node100 → SSH node01 → W&B` 터널을 열어 온라인 기록을 유지한다. 이 잡의 HTTPS 요청도
같은 프록시를 사용하며, 다른 노드와 W&B offline/disabled 모드는 기존 설정을 유지한다.
node100에서 node01로 비밀번호 없는 SSH와 등록된 호스트 키가 필요하다. 연결을 확인한 뒤
학습을 시작하고, SSH 연결이 끊기면 다시 연결하며, 잡 종료 시 터널을 정리한다.
시스템 DNS 변경이나 추가 패키지 설치는 필요 없다.
`SBD_WANDB_TUNNEL=off`로 끄고, `on`으로 강제 사용하며,
`SBD_WANDB_RELAY_HOST`로 중계 노드를 바꿀 수 있다 (기본값 `node01`).

## 로컬 Linux PC

노트북의 `/home/dohyeon/SRLLAB/Skill_Boundary_Detector` 경로에서는 `servers/local.yaml`을 자동으로
선택한다. 다른 로컬 경로에서는 `export SBD_SERVER=local`로 선택하거나 `local.yaml`의 `detect:`에
경로를 추가한다. Slurm 없이 실행하며 데이터/모델 경로는 프로젝트 폴더를 기준으로 한다.
SSHFS 마운트와 평가 실행 경로는 별도로 설정해야 한다.

```bash
SBD_SETUP_MOLMOACT2=0 SBD_SETUP_GROOT=0 bash setup_env.sh  # VSA + Terminator용 기본 환경
```

이미 `[5/8] lerobot editable` 설치까지 완료하고 서버 감지에서 중단됐다면 환경을 재생성하지 않고
`bash check_env.sh`로 기본 환경을 검증할 수 있다.

## 환경 규칙
- `lerobot/` 안에서 `uv sync`, `uv run`을 쓰지 않는다 (`lerobot/.venv`가 따로 생김).
- 패키지 추가: `uv pip install --python .venv/bin/python "패키지==버전"` → `requirements.txt`에 추가 → `bash check_env.sh`
- Molmo는 별도 Linux x86_64 / Python 3.12 / CUDA 12.8 환경과
  `train_molmoact2/requirements.lock.txt`를 사용한다. 기본 환경의 LeRobot이나 CMake를 설치하지 않는다.
