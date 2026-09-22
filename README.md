# Skill Boundary Detector (SkillVLA)

## 새 서버 세팅

### 1. 서버 요구사항
- Linux x86_64, NVIDIA GPU와 드라이버: `nvidia-smi`의 CUDA Version이 12.8 이상 (torch `2.10.0+cu128`)
- EGL: MuJoCo/LIBERO를 화면 없이 렌더링할 때 필요
- `gcc`, `make`: 설치 중 `egl-probe`를 소스에서 빌드함 (cmake는 venv 안에 설치됨)
- PyPI, astral.sh 인터넷 접속
- Slurm: 학습·eval 스크립트는 `sbatch` 기반
- (선택) `zstd`, `flock`: 없으면 노드 로컬 venv 복사를 건너뛰고 공유 `.venv`를 그대로 씀

### 2. 코드 + Python 환경
```bash
git clone https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git
cd Skill_Boundary_Detector
bash setup_env.sh
source .venv/bin/activate
```
`setup_env.sh`가 하는 일:
1. uv가 없으면 설치
2. Python 3.12.13으로 `.venv` 생성
3. `egl-probe`를 패치해서 빌드·설치
4. `requirements.txt` 설치 (전 패키지 `==` 고정) + `robomimic==0.2.0`(`--no-deps`)
5. `lerobot` editable 설치
6. 서버 감지 + (RunPod) 저장소 링크
7. `check_env.sh`로 결과 검증 (파이썬 버전, 고정 패키지 전부, lerobot 위치)

### 3. 서버별 설정 — `lerobot/examples/libero/configs/`
```
global_config.yaml      # 모든 서버 공통 (dataset_root, outputs_root, server: auto)
servers/yonsei.yaml     # detect: /scratch2/mdorazi, /scratch/mdorazi  + Slurm 설정
servers/rllab.yaml      # detect: /data1/dohyeon, /data2/dohyeon      + Slurm 설정
servers/runpod.yaml     # detect: /workspace + storage_volume/storage_outputs (Slurm 없음)
src/                    # 공통 코드 (global_config_loader.py, link_storage.py, sbatch 도우미 sh)
```
- 서버 선택: `SBD_SERVER` 환경변수 → `global_config.yaml`의 `server:` → clone 경로로 자동 감지.
  맞는 서버가 없거나 둘 이상이면 에러로 멈춘다.
- `project_root`는 clone 위치로 자동 설정된다. 서버를 바꿀 때 주석을 켜고 끌 필요가 없다.
- 확인: `python lerobot/examples/libero/configs/src/global_config_loader.py`
- 새 서버 추가: `servers/<이름>.yaml`에 `detect:` 경로와 Slurm 4개 키를 적으면 된다.

### 4. 데이터·모델·체크포인트 (git에 없음)
`dataset_*/`, `models/`의 가중치, `outputs_*/`는 `.gitignore` 대상이라 따로 옮긴다.
- yonsei ↔ rllab: `./sync_server.sh` (같은 상대 경로로 rsync)
- 그 밖의 서버(RunPod 포함): Hugging Face — 아래 [Hugging Face 사용법](#hugging-face-사용법-hf_syncsh)
- RunPod은 Global volume(`/workspace-global`)에 **이 서버와 같은 폴더 이름**으로 두고, `setup_env.sh`(또는 `src/link_storage.py`)가 링크를 만든다.
  ```
  /workspace-global/dataset_filtered/...                   -> <repo>/dataset_filtered
  /workspace-global/models/pi05_base/...                   -> <repo>/models/pi05_base
  /workspace-global/outputs_filtered/<group>/<run>         -> 시작 체크포인트 (run 단위 링크)
  /workspace/outputs_filtered  (컨테이너 디스크)             -> <repo>/outputs_filtered  (새 학습 결과)
  ```
  데이터를 추가한 뒤에는 `python lerobot/examples/libero/configs/src/link_storage.py`를 다시 실행하면 된다
  (`hf_sync.sh pull`은 받은 뒤 자동으로 실행한다).

## Hugging Face 사용법 (`hf_sync.sh`)

`bash hf_sync.sh` 하나로 데이터셋 올리기/받기, 사전학습 모델 받기, 체크포인트 자동 업로드를 한다.
본체는 `lerobot/examples/libero/configs/src/hf_sync.py`, `hf_checkpoints.py`.

### 저장소 구성
| 무엇 | 저장소 | 공개 | 안의 경로 |
|---|---|---|---|
| 데이터셋 | `hf_dataset_repo` (`Dohyeon-Lee-02/SkillVLA`) | 비공개 | `dataset_filtered/...` 처럼 데이터 루트 폴더 그대로 |
| 체크포인트 | `hf_checkpoint_repo` (`Dohyeon-Lee-02/SkillVLA-checkpoints`) | **공개** | `outputs_filtered/<group>/<run>/checkpoints/<step>/` |
| 사전학습 모델 | 올리지 않음 — 원본 저장소에서 받음 | – | `models/<폴더>` 로 받음 |

두 저장소 이름은 `lerobot/examples/libero/configs/global_config.yaml`에 있다. 저장소는 처음 올릴 때 자동으로 만들어진다
(데이터셋은 비공개, 체크포인트는 공개로).

### 처음 한 번
1. huggingface.co 가입 → Settings → Access Tokens → **Write** 토큰 생성 (Write는 읽기도 포함).
   다운로드만 하는 서버에는 Read 토큰을 써도 된다. 세부 권한(fine-grained) 토큰이면
   "Read access to contents of all public gated repos you can access"를 켠다.
2. 같은 계정으로 아래 페이지에서 이용 동의 (계정당 한 번, 토큰이 바뀌어도 다시 할 필요 없음):
   `google/paligemma-3b-pt-224`, `facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitl16-pretrain-lvd1689m`
3. 서버마다 로그인
   ```bash
   source .venv/bin/activate
   hf auth login        # 토큰 붙여넣기 (화면에 안 보이는 게 정상), git credential 질문은 n
   hf auth whoami       # 아이디가 나오면 완료
   ```
   RunPod은 로그인 대신 pod 환경변수(Secret)에 `HF_TOKEN=hf_...`를 넣어두면 된다. 토큰은 코드·git에 넣지 않는다.

### 데이터셋 올리기 (push) — 이 서버 → 비공개 저장소
```bash
bash hf_sync.sh push                     # 데이터 루트 선택 → 하위 폴더 선택 (sync_server.sh 와 같은 방식)
bash hf_sync.sh push dataset_filtered/libero_90_full_full \
                     dataset_filtered/skillvla_dataset/libero_90_full_full/eval_init_states.npz
bash hf_sync.sh push --dry-run ...       # 무엇을 올릴지만 확인
```
- 고르는 방법: 번호 입력 → `[1] 이 폴더 전체 / [2] 하위 항목 고르기` → "다른 폴더/파일도 추가할까요?"에서 `y`면 계속 추가.
- 경로는 데이터 루트(`dataset`, `dataset_*`)로 시작해야 한다. `outputs_*`, `models`는 올라가지 않는다.
- 고른 폴더만 훑어서 올린다. 끊기면 같은 명령을 다시 치면 이미 올라간 데이터는 다시 보내지 않는다.
- 학습에 필요한 것만 올리면 된다. 예: pi05는 `dataset_filtered/libero_90_full_full`,
  SkillVLA는 `dataset_filtered/skillvla_dataset/libero_90_full_full/<run>/`과 `.../eval_init_states.npz`.

### 받기 (pull) — 데이터셋 / 사전학습 모델 / 체크포인트
```bash
bash hf_sync.sh pull                     # [1] 내 데이터셋 [2] 사전학습 모델 [3] 체크포인트 — 여러 개면 1,2,3
bash hf_sync.sh pull dataset_filtered/libero_90_full_full --yes
bash hf_sync.sh pull --models                              # 모델 4개 전부
bash hf_sync.sh pull --models pi05_base,dinov3-vits16      # 일부만
bash hf_sync.sh pull --checkpoints outputs_filtered/pi05_PT/<run>/checkpoints/030000
bash hf_sync.sh pull dataset_filtered/libero_90_full_full --models --yes   # 한 번에
```
- 받는 위치: yonsei/rllab은 저장소 루트(`<repo>/dataset_filtered/...`, `<repo>/models/...`, `<repo>/outputs_filtered/...`),
  RunPod은 Global volume(`/workspace-global/...`). RunPod에서는 받은 뒤 링크까지 자동으로 만든다
  (받은 체크포인트는 시작 체크포인트로 연결된다).
- 사전학습 모델은 **원본 저장소**에서 받는다 (버전 고정):

  | `models/` 폴더 | 원본 저장소 |
  |---|---|
  | `pi05_base` | `lerobot/pi05_base` |
  | `paligemma-3b-pt-224-tokenizer` | `google/paligemma-3b-pt-224` (토크나이저 파일 5개만) |
  | `dinov3-vits16` | `facebook/dinov3-vits16-pretrain-lvd1689m` |
  | `dinov3-vitl16` | `facebook/dinov3-vitl16-pretrain-lvd1689m` |

### 체크포인트 자동 업로드 (watch) — 공개 저장소
RunPod에서 학습과 별도로 tmux 창 하나에 띄워 둔다.
```bash
tmux new -s hfwatch
bash hf_sync.sh watch --keep 3 --protect 050000,100000     # run마다 최근 3개 + 050000, 100000 유지
# Ctrl-b d 로 빠져나오고, tmux attach -t hfwatch 로 다시 확인
```
- 5분마다(`--interval`) `outputs`, `outputs_*` 폴더를 모두 살핀다. `outputs_root`를 바꿔도 체크포인트가 생긴 폴더 이름 그대로 올라간다.
- **다 저장된 step만** 올린다: 저장이 끝나야 `checkpoints/last`가 옮겨지므로 `last`가 가리키는 step까지만 올린다.
- 올리는 것: `pretrained_model/` + `training_state/` (resume 가능). 올리지 않는 것: `last` 링크, `wandb/`, 로그, 링크된 시작 체크포인트.
- 처음 올릴 때 모델 카드(`README.md`)와 라이선스 파일(`NOTICE`: Gemma 고지, `LICENSE-DINOv3.md`)을 자동으로 만든다.
- 첫 스캔에서 "N GB 올리기 + 로컬 체크포인트 M개 지우기: 진행할까요?"를 한 번 묻는다 (`--yes`면 생략).
- `--keep N`: run마다 최근 N개(+ `--protect` step)만 남긴다.
  - Hugging Face에서 오래된 step을 지운다. 지운 step은 다시 올리지 않는다 (재시작해도 같음).
  - **RunPod에서는 컨테이너 디스크의 오래된 step도 같은 규칙으로 지운다** (`runpod.yaml`의 `hf_watch_prune_local: true`).
    최신 step, 저장 중인 step, 보호 step, 링크된 시작 체크포인트는 지우지 않는다.
  - 허깅페이스는 지운 파일도 이력에 남아 용량이 줄지 않는다. 실제로 비우려면 `--squash`(이력 합치기, 되돌릴 수 없음).
- **기존 체크포인트가 많은 서버(yonsei)에서는 켜지 않는다** (첫 스캔에 전부 올리려 한다). 확인만 하려면 `--dry-run`.
- 다른 서버로 체크포인트를 따로 전송한다면, 링크로 들어온 run은 빼고 보낸다 (rsync `--no-links`).

### 옵션 요약
| 옵션 | 모드 | 설명 |
|---|---|---|
| `--dry-run` | 전부 | 계획만 출력, 아무것도 올리거나 받거나 지우지 않음 |
| `--yes` | 전부 | 확인 질문 생략 |
| `--models [a,b]` | pull | 사전학습 모델 받기 (값 없으면 전부) |
| `--checkpoints p1,p2` | pull | 체크포인트 경로 받기 |
| `--root outputs_x` | watch | 이 outputs 루트만 살피기 (기본: 전부) |
| `--interval 300` | watch | 검사 간격(초) |
| `--once` | watch | 한 번만 검사하고 종료 |
| `--keep N` | watch | run마다 최근 N개 유지 (0 = 전부 유지, 기본) |
| `--protect 050000,100000` | watch | 절대 지우지 않을 step |
| `--prune-local` / `--no-prune-local` | watch | 로컬 오래된 step 정리 켜기/끄기 (기본: 서버 설정, RunPod은 켜짐) |
| `--squash` | watch | 지운 뒤 Hub 이력을 합쳐 용량 비우기 (되돌릴 수 없음) |
| `--repo`, `--checkpoint-repo` | 전부 | 저장소 이름 덮어쓰기 (`SBD_HF_DATASET_REPO`, `SBD_HF_CHECKPOINT_REPO` 환경변수도 가능) |
| `--server` | 전부 | 서버 감지 덮어쓰기 (`SBD_SERVER`와 같음) |

### RunPod에서 처음부터
```bash
cd /workspace && git clone https://github.com/dohyeon-Lee/Skill_Boundary_Detector.git && cd Skill_Boundary_Detector
bash setup_env.sh                                  # runpod 자동 감지, 링크 생성, 환경 검증
export HF_TOKEN=hf_...                             # 또는 pod Secret / hf auth login
bash hf_sync.sh pull dataset_filtered/libero_90_full_full --models --yes
bash hf_sync.sh pull --checkpoints outputs_filtered/pi05_PT/<run>/checkpoints/030000 --yes   # 이어서 학습할 때
tmux new -s hfwatch                                # 새 tmux 창에서 아래 실행 → 첫 확인에 y → Ctrl-b d
bash hf_sync.sh watch --keep 3 --protect 050000,100000
```

### 공개 저장소 주의
- 공개 = 누구나 본다. 가중치와 run 이름(방법, 하이퍼파라미터)이 드러나므로 연구 공개 시점을 확인한다.
- 체크포인트는 원본의 파생물이라 조건이 따라간다: pi05_base 기반 → **Gemma 약관**, SkillVLA → **DINOv3 라이선스**도 포함,
  학습 데이터 LIBERO → CC BY 4.0 출처 표시. 카드와 라이선스 파일은 자동으로 붙지만, Gemma 고지 문구는 첫 업로드 후 한 번 확인한다.
- 체크포인트 설정 파일에는 학습한 서버의 절대 경로가 들어 있다 (비밀 정보는 없음). `wandb/`는 노드·계정 정보가 있어 올리지 않는다.

### 문제 해결
| 증상 | 해결 |
|---|---|
| `Not logged in` / 401 | `hf auth login` 또는 `HF_TOKEN` 설정. 이 서버는 `HF_HOME=/scratch/mdorazi/.cache/huggingface`에 토큰을 저장한다 |
| "이용 동의가 필요합니다" (403) | 안내된 모델 페이지에서 같은 계정으로 동의 후 다시 실행 |
| `hf_dataset_repo 가 비어 있습니다` | `global_config.yaml`에 `<아이디>/<저장소>` 적기 |
| `Cannot pick a server` | `SBD_SERVER=runpod`처럼 서버 지정 (yonsei / rllab / runpod), 또는 `global_config.yaml`의 `server:` |

## 환경 관리 규칙
- 환경은 항상 `setup_env.sh`로 만든다. `lerobot/` 안에서 `uv sync`나 `uv run`을 쓰지 않는다.
  쓰면 별도의 `lerobot/.venv`가 생긴다. (`lerobot/uv.lock`은 upstream lerobot 파일이라 이 환경의 기준이 아님)
- 패키지를 추가하거나 버전을 바꿀 때:
  ```bash
  uv pip install --python .venv/bin/python "패키지==버전"
  # requirements.txt 에 "패키지==버전" 을 알파벳 순서로 추가
  bash check_env.sh   # "환경 검증 통과"가 나와야 함
  ```
- 현재 환경 점검: `bash check_env.sh`
