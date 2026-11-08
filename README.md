# Skill Boundary Detector (SkillVLA) — RunPod

## 1. 계정 준비

- Hugging Face **Write 토큰**을 준비한다.
- 같은 HF 계정으로 [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224),
  [DINOv3-S](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
  [DINOv3-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 이용 약관에 동의한다.
- [W&B API 키](https://wandb.ai/authorize)를 준비한다.

## 2. 프로젝트 설치와 로그인

RunPod 터미널에서 실행한다.

```bash
cd ~/workspace 2>/dev/null || cd /workspace
git clone --branch skillVLA_VSA https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git
cd Skill_Boundary_Detector

bash setup_env.sh
source .venv/bin/activate

hf auth login
wandb login
```

`setup_env.sh`는 시스템 의존성과 기본·MolmoAct2·GR00T 환경을 준비한다.
VSA·Predictor·Terminator만 필요하면 설치 명령을 아래로 바꾼다.

```bash
SBD_SETUP_MOLMOACT2=0 SBD_SETUP_GROOT=0 bash setup_env.sh
```

Pod 환경변수에 `HF_TOKEN`, `WANDB_API_KEY`를 설정했다면 로그인 명령은 생략할 수 있다.
RunPod 환경은 자동 감지한다. 감지에 실패하면 `export SBD_SERVER=runpod`로 지정한다.

## 3. 데이터·모델·체크포인트 다운로드

프로젝트 루트에서 데이터셋과 사전학습 모델을 받는다.

```bash
bash hf_sync.sh pull dataset_filtered --models --yes
```

일부만 선택하려면 `bash hf_sync.sh pull`의 메뉴를 사용한다.
이어학습할 때는 `<run>`을 실제 모델 폴더명으로 바꾸고 체크포인트도 받는다.

```bash
bash hf_sync.sh pull \
  --checkpoints 'outputs_filtered/skillVLA_PT/VSA/<run>/checkpoints/010000' \
  --yes
```

다운로드 후 `checkpoints/last`가 자동 연결된다. 같은 모델 설정과 출력명을 유지하면
마지막 체크포인트에서 이어학습한다.

브라우저로 선택하려면 `bash sync_server_ui.sh`를 실행하고 **8765 포트**를 전달한 뒤
Hugging Face 탭을 연다. [UI 사용법](tools/server_sync/HF_README.md).

## 4. 학습 실행과 확인

VSA 모델과 학습 조건은
`lerobot/examples/libero/configs/train_skillVLA/PT/VSA/vsa_train_config.yaml`에서 설정한다.

```bash
cd lerobot/examples/libero/configs/train_skillVLA/PT/VSA
./submit_train.sh
```

RunPod에서는 Slurm 없이 백그라운드 학습을 시작한다. 로그는 실행 폴더의 `logs/`에 남는다.
학습 목록과 중단은 프로젝트 루트에서 실행한다. `<id>`는 목록에 표시된 실제 ID로 바꾼다.

```bash
bash lerobot/examples/libero/configs/src/submit_job.sh list
bash lerobot/examples/libero/configs/src/submit_job.sh stop <id>
```

## 5. 체크포인트 자동 업로드

프로젝트 루트에서 tmux로 실행한다. `Ctrl-b`, `d`로 나오면 SSH를 끊어도 유지된다.

```bash
tmux new -s hfwatch
bash hf_sync.sh watch --keep 3 --protect 050000,100000
```

완료된 체크포인트를 HF에 업로드한다. RunPod에서는 각 run의 최근 3개와 보호 step을
제외한 오래된 체크포인트를 **HF와 로컬에서 모두 삭제**한다. 다운로드한 시작 체크포인트는 보존한다.

업로드 후 Pod 자동 종료까지 원하면 위 watch 명령 대신 다음을 사용한다.

```bash
bash hf_sync.sh watch --keep 3 --protect 050000,100000 --done-step 100000 --done-time 2d
```

자동 종료 옵션은 학습 중단·목표 step·시간 조건을 감시하므로, 추가 학습을 시작할 때는
기존 watch를 먼저 종료한다. 옵션 전체는 `bash hf_sync.sh --help`에서 확인한다.

## 저장 위치

- `/workspace-global` 볼륨이 있으면 데이터·사전학습 모델은 해당 볼륨을 사용한다.
- 새 학습 결과는 보통 프로젝트 옆 `outputs_filtered`에 저장된다.
  볼륨이 없으면 프로젝트 내부에 저장된다.
- **컨테이너 디스크의 체크포인트는 Pod를 삭제하면 사라진다. 삭제 전에 HF 업로드를 확인한다.**
