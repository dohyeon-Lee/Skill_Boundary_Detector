# Skill Boundary Detector (SkillVLA) — RunPod

## 1. 계정 준비

- Hugging Face **Write 토큰**을 준비한다.
- 같은 HF 계정으로 [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224),
  [DINOv3-S](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
  [DINOv3-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 이용 약관에 동의한다.
- [W&B API 키](https://wandb.ai/authorize)를 준비한다.
- RunPod **Secrets**에 `hf_token`, `wandb_key`로 각각 저장하고, 개인 Pod 템플릿의
  환경변수에 아래 참조를 설정한다. 실제 키를 Git에 저장하지 않는다.

  | 환경변수 | 값 |
  | --- | --- |
  | `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` |
  | `WANDB_API_KEY` | `{{ RUNPOD_SECRET_wandb_key }}` |

- 템플릿은 `22/tcp` 포트를 열고, Pod 생성 시 public IP 지원과 SSH 공개키 등록을 확인한다.

### PC의 SSH 접속 정보 자동 갱신

**SSH/VS Code를 실행하는 PC의 터미널에서** 실행한다. Pod나 연구실 서버에서 실행하면
그 컴퓨터의 SSH 설정이 바뀐다. Python 3.9 이상만 필요하며 추가 패키지는 없다.
PC에 이 저장소를 준비한 뒤 프로젝트 루트에서 아래 명령 하나를 실행한다.
스크립트가 Python을 찾아 실행하므로 학습 환경을 먼저 설치할 필요는 없다.
Windows에서는 Git Bash를 사용한다.

RunPod 콘솔에서 **Pod 조회가 가능한 RunPod API 키**를 준비한다(HF·W&B 키와 별개).
첫 실행에는 키를 숨김 입력으로 받아 PC 사용자 폴더의
`~/.config/skill-boundary-detector/runpod_api_key`에 저장하고 바로 SSH 설정까지 갱신한다.
다음부터는 같은 명령이 저장된 키를 재사용한다(Unix: 키 파일 권한 600).

```bash
bash runpod_ssh.sh
```

새 Pod를 생성한 뒤 아래 명령을 실행하면 현재 IP와 외부 SSH 포트를 조회해
`~/.ssh/config`의 `Host runpod` 접속 설정을 갱신한다. 다른 서버 블록과 해당 블록의
포트 전달 등 추가 옵션은 보존하며, 변경 전 원본을 같은 폴더에 백업한다.
`Host *` 기본값보다 우선하도록 대상 블록을 첫 Host 블록 위치로 옮긴다.
실행 중인 Pod가 하나면 자동 선택하고, 여러 개면 번호로 선택한다.
여러 Pod가 있고 대화형 터미널이 아니면 `--pod`로 대상을 지정해야 한다.
아직 SSH 주소가 없으면 변경하지 않는다.

```bash
bash runpod_ssh.sh
ssh runpod
```

원하면 목록을 확인하고 ID나 정확한 이름으로 직접 지정할 수도 있다.
`--dry-run`은 SSH 설정과 키 파일 모두 변경하지 않으므로 먼저 키를 등록해야 한다.

```bash
bash runpod_ssh.sh --list
bash runpod_ssh.sh --pod <pod-id> --dry-run
bash runpod_ssh.sh --pod <pod-id>
```

키를 교체할 때만 `bash runpod_ssh.sh --save-api-key`로 새 키를 저장한다(갱신은 다음 실행).
Pod 조회는 RunPod API v2를 사용한다. 기본적으로 상속된 프록시를 우회해 직접 연결하고,
Python 기본 인증서 저장소에 Linux 시스템 CA 인증서를 추가해 TLS를 검증한다.
프록시가 필수인 네트워크에서만 `bash runpod_ssh.sh --use-proxy`로 환경변수의 프록시를 사용한다.
연결만 확인하려면 `bash runpod_ssh.sh --list`를 실행한다. Pod가 없으면 없다고 표시하며
SSH 설정은 변경하지 않는다. TLS/DNS/시간 초과 오류는 별도 메시지로 구분한다.
다른 SSH 키/설정 파일은 `--identity <경로>`, `--ssh-config <경로>`로 지정한다.
`RUNPOD_API_KEY` 환경변수가 있으면 저장된 키보다 우선한다. 이 도구는 Pod를 조회하며
Pod 생성·정지·설정 변경은 하지 않는다. `Host runpod`가 여러 별칭과 묶였거나 중복되면
오류로 중단하므로 먼저 독립된 단일 블록으로 정리한다.

## 2. 프로젝트 설치와 로그인

RunPod 터미널에서 실행한다.

```bash
cd ~/workspace 2>/dev/null || cd /workspace
git clone --branch skillVLA_VSA https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git
cd Skill_Boundary_Detector

bash setup_env.sh --runpod
source .venv/bin/activate
```

`bash setup_env.sh --runpod`는 RunPod 서버 설정을 선택하고, 시스템 의존성과 기본 환경을
설치한 다음 `HF_TOKEN`·`WANDB_API_KEY`로 HF·W&B에 자동 로그인한다. 키 값은 출력하지 않는다.
키가 없는 서비스는 안내 후 건너뛰고, 키가 있는데 인증이 실패하면 오류로 종료한다.
키만 다시 등록/검증하려면 기본 환경을 재설치하지 않는 아래 명령을 사용한다.

```bash
bash setup_env.sh --runpod --login-only
```

다른 서버·로컬에서는 `bash setup_env.sh`로 시스템 의존성과 기본 환경(`.venv`)만 준비한다.
RunPod가 자동 감지되더라도 **`--runpod`를 지정하지 않으면 자동 로그인하지 않는다.**
직접 로그인하려면 `source .venv/bin/activate` 후 `hf auth login`, `wandb login`을 실행한다.
VSA·Predictor·Terminator는 이 기본 환경을 사용한다.
MolmoAct2·GR00T가 필요하면 기본 설치 후 해당 전용 환경을 추가한다.

```bash
bash setup_env.sh --molmoact2-only  # MolmoAct2 추가 설치
bash setup_env.sh --groot-only     # GR00T 추가 설치
```

RunPod에서 처음부터 둘 다 함께 설치하려면
`SBD_SETUP_MOLMOACT2=1 SBD_SETUP_GROOT=1 bash setup_env.sh --runpod`를 사용한다.

Pod에 환경변수만 설정해도 SDK가 해당 키를 인증에 사용할 수 있지만, `--runpod`는 로그인과
키 검증까지 수행한다. 모델 약관 동의는 별도로 완료해야 한다.

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

## 4. 학습·평가·데이터 작업 실행

아래 실행 스크립트는 Slurm 서버와 RunPod에서 같은 명령을 사용한다.
`servers/<name>.yaml`의 `scheduler`에 따라 Slurm 제출 또는 로컬 백그라운드 실행을 선택한다.
RunPod는 자동 감지하며, 감지가 안 되면 `export SBD_SERVER=runpod`로 지정한다.

| 작업 | 실행 스크립트 (`lerobot/examples/libero/configs/` 기준) |
| --- | --- |
| VSA / Predictor / Terminator PT·FT, Joint FT 학습 | 각 모델 폴더의 `submit_train.sh` |
| π0.5 / cyclic / MolmoAct2 / GR00T 학습 | 각 모델의 기존 `submit_train.sh` |
| SkillVLA closed-loop PT·FT 평가 | `train_skillVLA/{PT,FT}/eval/submit_eval.sh` |
| π0.5 / cyclic / MolmoAct2 / GR00T 평가 | 각 `eval/submit_eval.sh` |
| XYZ-head / FT Predictor 오프라인 평가 | `submit_xyz_head_eval.sh` / `submit_pred_eval.sh` |
| DP·FSQ 학습과 skill 평가 | `train_skills/`의 기존 제출 스크립트 |
| 데이터 다운로드·변환·분할 | `generate_training_dataset/`의 기존 제출 스크립트 |
| Skillset 생성·SkillVLA build·oracle matching | 각 `build_data/`의 기존 제출 스크립트 |

RunPod에서도 배열 작업(`--array`, `%동시실행수`), `--wrap`, `--export`와
`afterok` / `afterany` 의존 작업을 처리한다. 배열 전체가 끝나야 다음 단계로 넘어가며,
앞 단계가 실패하면 `afterok` 후속 작업은 실행하지 않는다. GPU가 사용 중이면 대기하고,
대기 중인 작업이 GPU를 먼저 예약하지 않으므로 한 GPU에서도 연속 파이프라인을 실행할 수 있다.
GPU 선택 범위를 제한하려면 실행 전에 `export SBD_GPUS=0`처럼 지정한다.

각 로컬 작업은 SSH를 끊어도 유지되며, 배열 작업을 중단하면 실행 중인 자식 작업도 종료한다.
상태·PID·종료 코드는 `.cache/local_jobs/`에, 출력은 기존 `logs/`에 기록한다.
배열 로그의 `%A`는 전체 작업 ID, `%a`는 배열 인덱스다.
CPU 전용 배열은 별도 동시 실행 수가 없으면 한 번에 하나씩 실행한다.

Slurm의 partition·QoS·노드 지정·메모리·시간 제한은 기존 Slurm 경로에서 적용된다.
RunPod는 한 Pod의 실제 GPU만 배분하며, Slurm 메모리·시간 제한은 강제하지 않는다.
배치 크기·데이터 로더·평가 동시 실행 수는 Pod의 VRAM·RAM·CPU에 맞게 설정한다.
MolmoAct2·GR00T는 전용 환경, 모든 학습·평가는 해당 데이터·체크포인트가 별도로 필요하다.

VSA 모델과 학습 조건은
`lerobot/examples/libero/configs/train_skillVLA/PT/VSA/vsa_train_config.yaml`에서 설정한다.
프로젝트 루트에서 학습을 실행한다.

```bash
cd lerobot/examples/libero/configs/train_skillVLA/PT/VSA
./submit_train.sh
```

VSA closed-loop 평가도 해당 설정 파일을 준비한 뒤 프로젝트 루트에서 실행한다.

```bash
cd lerobot/examples/libero/configs/train_skillVLA/PT/eval
./submit_eval.sh
```

RunPod에서는 Slurm 없이 백그라운드 작업을 시작한다. 로그는 실행 폴더의 `logs/`에 남는다.
작업 목록과 중단은 프로젝트 루트에서 실행한다. `<id>`는 목록에 표시된 실제 ID로 바꾼다.

```bash
bash lerobot/examples/libero/configs/src/submit_job.sh list
bash lerobot/examples/libero/configs/src/submit_job.sh stop <id>
```

완료된 체크포인트에서 중단하는 감시 도구도 두 환경에서 사용할 수 있다.
`bash lerobot/examples/libero/configs/src/stop_at_checkpoint.sh 100000 <id>`를 실행하면
`checkpoints/last`와 모델 파일을 확인한 뒤 해당 환경의 방식으로 작업을 중단한다.

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
- **컨테이너 디스크의 파일은 Pod 정지·초기화·삭제 시 사라질 수 있다. 정지 전에 HF 업로드를 확인한다.**
