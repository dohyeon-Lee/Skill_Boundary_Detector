# Skill Boundary Detector (SkillVLA)

학습 조건 변경·추후 점검 사항: [SkillVLA 구현·실험 주의사항](lerobot/examples/libero/configs/train_skillVLA/KNOWN_ISSUES.md).

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

## 로컬 Linux PC

서버 자동 감지는 연구실 서버와 RunPod를 대상으로 한다. 노트북/워크스테이션에서는 각 터미널에서
`export SBD_SERVER=local`로 `servers/local.yaml`을 선택한다. Slurm 없이 실행하며 데이터/모델 경로는
프로젝트 폴더를 기준으로 한다. SSHFS 마운트와 평가 실행 경로는 별도로 설정해야 한다.

```bash
export SBD_SERVER=local
SBD_SETUP_MOLMOACT2=0 SBD_SETUP_GROOT=0 bash setup_env.sh  # VSA + Terminator용 기본 환경
```

이미 `[5/8] lerobot editable` 설치까지 완료하고 서버 감지에서 중단됐다면 환경을 재생성하지 않고
`export SBD_SERVER=local` 후 `bash check_env.sh`로 기본 환경을 검증할 수 있다.

## 환경 규칙
- `lerobot/` 안에서 `uv sync`, `uv run`을 쓰지 않는다 (`lerobot/.venv`가 따로 생김).
- 패키지 추가: `uv pip install --python .venv/bin/python "패키지==버전"` → `requirements.txt`에 추가 → `bash check_env.sh`
- Molmo는 별도 Linux x86_64 / Python 3.12 / CUDA 12.8 환경과
  `train_molmoact2/requirements.lock.txt`를 사용한다. 기본 환경의 LeRobot이나 CMake를 설치하지 않는다.
