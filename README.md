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
`dataset_*/`, `models/`의 가중치, `outputs_*/`는 `.gitignore` 대상이라 따로 옮겨야 함.
- yonsei ↔ rllab: `./sync_server.sh` (같은 상대 경로로 rsync)
- Hugging Face (비공개 저장소, 모든 서버 공용): `bash hf_sync.sh`
  - `global_config.yaml`의 `hf_dataset_repo`에 `<아이디>/<저장소>`를 한 번 적어두고, 서버마다 `hf auth login`(또는 `HF_TOKEN`)
  - 저장소 안에 데이터 루트 폴더가 그대로 들어간다 (`dataset_filtered/...`, `dataset_calvin/...`)
  - `bash hf_sync.sh` → push/pull 선택 → 데이터 루트 선택 → `sync_server.sh`처럼 하위 폴더를 골라서 전송 (여러 개 선택 가능)
  - 바로 지정: `bash hf_sync.sh pull dataset_filtered/libero_90_full_full --yes`, 확인만: `--dry-run`
  - pull 에서는 **사전학습 모델**도 고를 수 있다: 원본 저장소(`lerobot/pi05_base`, `google/paligemma-3b-pt-224`, `facebook/dinov3-*`)에서 `models/<폴더>`로 받는다. 바로 받기: `bash hf_sync.sh pull --models` (일부만: `--models pi05_base,dinov3-vits16`). PaliGemma·DINOv3는 허깅페이스 페이지에서 한 번 동의가 필요하다.
  - RunPod에서는 Global volume(`/workspace-global/dataset_filtered`)으로 받고 링크까지 자동으로 만든다
- RunPod: Global volume(`/workspace-global`)에 **지금 서버와 같은 폴더 이름**으로 둔다. `setup_env.sh`(또는 `src/link_storage.py`)가 링크를 만든다.
  ```
  /workspace-global/dataset_filtered/...                   -> <repo>/dataset_filtered
  /workspace-global/models/pi05_base/...                   -> <repo>/models/pi05_base
  /workspace-global/outputs_filtered/<group>/<run>         -> 시작 체크포인트 (run 단위 링크)
  /workspace/outputs_filtered  (컨테이너 디스크)             -> <repo>/outputs_filtered  (새 학습 결과)
  ```
  데이터를 추가한 뒤에는 `python lerobot/examples/libero/configs/src/link_storage.py`를 다시 실행하면 된다.

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
