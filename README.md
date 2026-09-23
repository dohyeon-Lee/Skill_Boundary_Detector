# Skill Boundary Detector (SkillVLA)

## RunPod

**처음 한 번 (계정)**: Hugging Face **Write** 토큰 생성 → `google/paligemma-3b-pt-224`,
`facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-lvd1689m` 이용 동의 →
wandb API 키 생성 (https://wandb.ai/authorize, 만들 때 한 번만 보인다).

### 1. 설치 → 로그인 → 데이터
```bash
apt-get install -y rsync tmux

cd ~/workspace 2>/dev/null || { mkdir -p /workspace && cd /workspace; }   # 볼륨이 있는 폴더로
git clone https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git
cd Skill_Boundary_Detector
bash setup_env.sh           # "환경 검증 통과"가 나오면 완료
source .venv/bin/activate

hf auth login               # 토큰 붙여넣기 (화면에 안 보이는 게 정상), git credential 질문은 n
wandb login                 # wandb API 키

bash hf_sync.sh pull dataset_filtered --models --yes     # 데이터셋 전체 + 사전학습 모델
```
- pod 환경변수에 `HF_TOKEN`, `WANDB_API_KEY`를 넣어두면 로그인 생략. 요구사항: CUDA 12.8+ 드라이버, `gcc`·`make`, EGL.
- 이어서 학습할 체크포인트: `bash hf_sync.sh pull --checkpoints outputs_filtered/<group>/<run>/checkpoints/<step> --yes`
  (`checkpoints/last`가 자동으로 잡혀 바로 이어서 학습된다)
- 일부만: `bash hf_sync.sh pull` (메뉴), `dataset_filtered/libero_90_full_full`, `--models pi05_base,dinov3-vits16`

### 2. 학습 실행
yonsei와 같은 명령. RunPod이면 Slurm 대신 백그라운드로 바로 시작된다.
```bash
cd lerobot/examples/libero/configs/train_skillVLA/stage1/VSA && ./submit_train.sh

S=~/workspace/Skill_Boundary_Detector/lerobot/examples/libero/configs/src/submit_job.sh
bash $S list          # squeue 대신
bash $S stop <id>     # scancel 대신
```
- 로그는 제출한 폴더의 `logs/`. 터미널을 닫아도 계속된다. 빈 GPU를 자동 배정하고, 없으면 시작하지 않는다.
- 죽었으면 같은 명령을 다시 치면 마지막 체크포인트부터 이어서 한다.
- 되는 것: stage1 VSA, Predictor/Terminator, NewTask_FT, FT, stage2, pi05 PT/FT, DP, FSQ.
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
<workspace>/Skill_Boundary_Detector   코드, .venv
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

## 다른 서버 (yonsei, rllab)
- 설치는 똑같이 `bash setup_env.sh` (서버는 clone 경로로 자동 감지: `lerobot/examples/libero/configs/servers/*.yaml`).
- 데이터 옮기기: `./sync_server.sh` (yonsei ↔ rllab), `bash hf_sync.sh push` (Hugging Face 비공개 저장소).
- yonsei에서는 `watch`를 켜지 않는다 (기존 체크포인트를 전부 올리려 한다).

## 환경 규칙
- `lerobot/` 안에서 `uv sync`, `uv run`을 쓰지 않는다 (`lerobot/.venv`가 따로 생김).
- 패키지 추가: `uv pip install --python .venv/bin/python "패키지==버전"` → `requirements.txt`에 추가 → `bash check_env.sh`
