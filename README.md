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
6. `global_config.yaml`의 `project_root`를 clone한 위치로 설정
7. `check_env.sh`로 결과 검증 (파이썬 버전, 고정 패키지 전부, lerobot 위치)

### 3. 데이터·모델·체크포인트 (git에 없음)
`dataset_*/`, `models/`의 가중치, `outputs_*/`는 `.gitignore` 대상이라 따로 옮겨야 함.
기존 서버에서 `./sync_server.sh`로 rsync (yonsei ↔ rllab 지원).

### 4. 서버마다 확인할 설정: `lerobot/examples/libero/configs/global_config.yaml`
| 키 | 내용 |
|---|---|
| `project_root` | `setup_env.sh`가 자동 설정. 나머지 경로는 모두 이 값 기준 상대 경로 |
| `dataset_root`, `outputs_root` | 옮겨온 데이터·출력 폴더 이름 |
| `train_partition`, `train_qos`, `train_nodelist`, `train_exclude_nodes` | 클러스터 전용 값. 새 서버의 Slurm에 맞게 수정 |

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
