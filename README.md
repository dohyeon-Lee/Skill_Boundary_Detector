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
pod에 CUDA 12.8 이상 드라이버, `gcc`·`make`, EGL이 있어야 한다.

### 2. 데이터·모델 받기
```bash
bash hf_sync.sh pull        # 메뉴에서 [1] 내 데이터셋 [2] 사전학습 모델 고르기
# 또는 한 줄로
bash hf_sync.sh pull dataset_filtered/libero_90_full_full --models --yes
```
- `/workspace-global`에 받고, 저장소 안으로 링크를 자동으로 만든다.
- 이어서 학습할 체크포인트가 있으면:
  `bash hf_sync.sh pull --checkpoints outputs_filtered/<group>/<run>/checkpoints/<step> --yes`

### 3. 체크포인트 자동 업로드 (학습과 따로 tmux에서)
```bash
tmux new -s hfwatch
bash hf_sync.sh watch --keep 3 --protect 050000,100000
# 첫 질문에 y → Ctrl-b d 로 나오기 (다시 보기: tmux attach -t hfwatch)
```
- 5분마다 다 저장된 체크포인트를 공개 저장소 `Dohyeon-Lee-02/SkillVLA-checkpoints`에 올린다 (resume 가능).
- run마다 최근 3개 + 050000, 100000만 남기고, 오래된 건 **Hugging Face와 pod 디스크 둘 다에서** 지운다.
- 모델 카드와 라이선스 파일(Gemma, DINOv3)은 자동으로 붙는다. 공개 저장소라 누구나 볼 수 있다.

### 폴더 구조
```
/workspace/Skill_Boundary_Detector    코드, .venv
/workspace/outputs_filtered           새 학습 결과 (컨테이너 디스크: pod를 지우면 사라짐)
/workspace-global/dataset_filtered    데이터셋 (Global volume)
/workspace-global/models              사전학습 모델
/workspace-global/outputs_filtered    받아온 시작 체크포인트
```

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
