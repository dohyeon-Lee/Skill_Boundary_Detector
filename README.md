# Skill Boundary Detector (SkillVLA)

## RunPod에서 실행하기

### 0. 처음 한 번만 (Hugging Face 계정)
1. huggingface.co → Settings → Access Tokens → **Write** 토큰 생성
2. RunPod pod 환경변수(Secret)에 `HF_TOKEN=hf_...` 등록 (토큰은 코드·git에 넣지 않기)
3. 같은 계정으로 아래 세 페이지에서 이용 동의 (계정당 한 번):
   `google/paligemma-3b-pt-224`, `facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-lvd1689m`

### 1. 코드 + 환경
```bash
cd /workspace
git clone https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git
cd Skill_Boundary_Detector
bash setup_env.sh           # 마지막에 "환경 검증 통과"가 나오면 완료
source .venv/bin/activate
```
pod에 CUDA 12.8 이상 드라이버, `gcc`·`make`, EGL이 있어야 하고, `rsync`·`tmux`도 필요하다:
`apt-get install -y rsync tmux`

### 2. 데이터·모델 받기
```bash
bash hf_sync.sh pull        # 메뉴에서 [1] 내 데이터셋 [2] 사전학습 모델 고르기
# 또는 한 줄로
bash hf_sync.sh pull dataset_filtered/libero_90_full_full --models --yes
```
- Global volume(`/workspace-global`)이 있으면 거기에 받고 저장소 안으로 링크를 만든다. 없으면 저장소 안에 바로 받는다.
- SkillVLA 학습은 `dataset_filtered/skillvla_dataset/...`의 해당 run 폴더도 받아야 한다 (메뉴에서 고르기).
- 체크포인트 받기 (시작점으로 쓰거나 이어서 학습할 때):
  `bash hf_sync.sh pull --checkpoints outputs_filtered/<group>/<run>/checkpoints/<step> --yes`
  - 새 학습 결과와 같은 `/workspace/outputs_filtered`에 받아진다.
  - 받은 run에는 `checkpoints/last`가 자동으로 잡혀서, 같은 제출 명령을 치면 그 step부터 이어서 학습한다.

### 3. 학습 실행
yonsei와 같은 명령을 쓴다. RunPod으로 감지되면 Slurm 대신 그 자리에서 백그라운드로 시작된다.
```bash
cd lerobot/examples/libero/configs/train_skillVLA/stage1/VSA
./submit_train.sh           # vsa_train_config.yaml 설정으로 시작

# 저장소 루트에서
bash lerobot/examples/libero/configs/src/submit_job.sh list        # 실행 중인 학습 (squeue 대신)
bash lerobot/examples/libero/configs/src/submit_job.sh stop <id>   # 멈추기 (scancel 대신)
```
- 로그는 제출한 폴더의 `logs/`에 쌓인다. 터미널을 닫아도 학습은 계속된다.
- 비어 있는 GPU를 자동으로 배정한다. 모자라면 시작하지 않는다 (대기열 없음).
- 학습이 멈췄거나 죽었으면 같은 명령을 다시 치면 마지막 체크포인트부터 이어서 학습한다.
- 되는 것: 학습 제출 스크립트 (stage1 VSA, Predictor/Terminator, NewTask_FT, FT, stage2, pi05 PT/FT, DP, FSQ).
  eval, 데이터 생성(build_data), pi05 cycle은 아직 Slurm 전용.

### 4. 체크포인트 자동 업로드 (학습과 따로 tmux에서)
```bash
tmux new -s hfwatch
bash hf_sync.sh watch --keep 3 --protect 050000,100000
# 첫 질문에 y → Ctrl-b d 로 나오기 (다시 보기: tmux attach -t hfwatch)
```
- 5분마다 다 저장된 체크포인트를 공개 저장소 `Dohyeon-Lee-02/SkillVLA-checkpoints`에 올린다 (resume 가능).
- run마다 최근 3개 + 050000, 100000만 남기고, 오래된 건 **Hugging Face와 pod 디스크 둘 다에서** 지운다.
- 지우는 대상은 **이 pod에서 학습한 run뿐**이다. 다른 pod가 올린 run, 받아오기만 한 run, 받아온 step은
  어디서도 지우지 않는다 (받은 run을 이어서 학습하면 새로 생긴 step만 관리한다).
- 첫 검사 때 올릴 것 / Hugging Face에서 지울 것 / 로컬에서 지울 것을 보여주고 한 번 묻는다.

**pod 자동 종료 (Terminate)**: `--done-step`이나 `--done-time`을 주면 켜진다. 둘 다 없으면 pod를 끄지 않는다.
```bash
bash hf_sync.sh watch --keep 3 --protect 050000,100000 --done-step 100000                    # step 100000에서
bash hf_sync.sh watch --keep 3 --protect 050000,100000 --done-time 48:00:00                  # 학습 48시간 뒤
bash hf_sync.sh watch --keep 3 --protect 050000,100000 --done-step 100000 --done-time 2d     # 둘 중 먼저
```
- 종료 조건 (하나라도 만족하면): 모든 run이 `--done-step`을 올림 / 첫 학습 잡 시작 뒤 `--done-time` 경과 /
  제출한 학습이 전부 멈춤 (정상 종료든 에러든). `--done-step 0`이면 학습이 멈출 때만 끈다.
- 최신 체크포인트가 Hugging Face에 올라간 것을 두 번 연속 확인한 뒤에만 끈다. 조건에 걸리면 아직 도는 학습도 끊긴다.
- `--done-time` 형식: `48:00:00`, `2-00:00:00`, `12h`, `90m`, `1d`.
- 끄기 직전에 학습 로그(.out/.err)와 잡 요약(`jobs.tsv`: 종료 코드)을 비공개 저장소 `Dohyeon-Lee-02/SkillVLA`의
  `runpod_logs/<시각>_<pod ID>/`에 올린다. 웹 페이지(Files)에서 보거나 받는다:
  `hf download Dohyeon-Lee-02/SkillVLA --repo-type dataset --include "runpod_logs/*" --local-dir runpod_logs`
- 학습을 모두 멈추면(`submit_job.sh stop`) 두 번째 검사 때 pod가 꺼진다. 설정을 고쳐 다시 돌릴 거면 watch를 먼저 끈다.
- 조건을 바꾸려면 watch를 `Ctrl-C`로 멈추고 새 옵션으로 다시 켠다 (이미 올린 체크포인트는 다시 올리지 않는다).

### 폴더 구조
```
/workspace/Skill_Boundary_Detector    코드, .venv
/workspace/outputs_filtered           새 학습 결과 + Hugging Face에서 받은 체크포인트 (컨테이너 디스크: pod를 지우면 사라짐)
/workspace-global/dataset_filtered    데이터셋 (Global volume)
/workspace-global/models              사전학습 모델
/workspace-global/outputs_filtered    직접 넣어 둔 시작 체크포인트 (run 단위로 링크됨)
```
Global volume 없이 컨테이너 디스크만 써도 된다 (설정 변경 없음, 자동 감지). 이때는 `dataset_filtered`, `models`가
저장소 안에 받아진다. pod를 지우면 전부 사라지므로 체크포인트는 `watch`로 올려 둔다.

### 문제 해결
| 증상 | 해결 |
|---|---|
| `Not logged in` / 401 | `HF_TOKEN`이 설정됐는지 확인, 또는 `hf auth login` |
| 403 "이용 동의가 필요합니다" | 0-3의 모델 페이지에서 같은 계정으로 동의 |
| `Cannot pick a server` | `export SBD_SERVER=runpod` |

그 밖의 옵션: `bash hf_sync.sh --help`

## 다른 서버 (yonsei, rllab)
- 설치는 똑같이 `bash setup_env.sh`. 서버는 clone 경로로 자동 감지된다
  (`lerobot/examples/libero/configs/servers/*.yaml`).
- yonsei ↔ rllab 데이터 옮기기: `./sync_server.sh`
- 데이터셋을 Hugging Face(비공개 `Dohyeon-Lee-02/SkillVLA`)에 올리기: `bash hf_sync.sh push` (메뉴에서 폴더 선택)
- yonsei에서는 `watch`를 켜지 않는다 (기존 체크포인트를 전부 올리려 한다).

## 환경 규칙
- `lerobot/` 안에서 `uv sync`, `uv run`을 쓰지 않는다 (`lerobot/.venv`가 따로 생김).
- 패키지 추가: `uv pip install --python .venv/bin/python "패키지==버전"` → `requirements.txt`에 추가 → `bash check_env.sh`
