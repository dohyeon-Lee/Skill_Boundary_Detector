# RunPod Desk

연구실 서버의 통합 Desk에 있는 **RunPod Desk** 탭이다.

```bash
bash sync_server_ui.sh
# VS Code Ports에서 8765 전달 → http://127.0.0.1:8765/runpod/
```

- Pod 이름·GPU·VRAM 사용량과 기존 `submit_*.sh`로 실행한 잡을 표시한다.
- 완료된 평가의 전체·task별 성공률 및 `task_success_rates.png`를 서버에 보관한다.
- 실행 도중 수집한 결과는 부분 결과로 표시한다. 완료 상태를 확인하지 못한 과거 결과도 구분한다.
- 브라우저를 닫아도 Desk가 실행 중이면 조회·수집을 계속한다. 수집 전에 Pod가 삭제되면 복구할 수 없다.

## 연결

Desk 서버에 `bash runpod_ssh.sh`로 등록한 RunPod API 키를 재사용한다.
키에는 Pod 조회·중단 및 billing 조회 권한이 필요하다. 키를 브라우저에 보내지 않는다.
Pod에는 이 서버의 SSH 공개키가 등록되어 있어야 하며 `22/tcp` 직접 접속을 사용한다.

화면의 **조회 설정**에서 Pod 프로젝트 경로, 이 서버의 SSH 개인키 경로, 조회 주기를 변경한다.
기본값은 `/workspace/Skill_Boundary_Detector`, `~/.ssh/id_ed25519`, 30초다.
기록·그래프는 `.cache/runpod_desk/`에 저장한다. 서버→Pod SSH 조회만 하므로 FUSE는 필요 없다.

## 비용·중단

- **누적 집계 비용:** RunPod billing API의 해당 Pod GPU·CPU·디스크 비용 합계. 생성 시각을 알 수 없으면 최근 30일로 표시한다. 5분마다 조회하며 RunPod 반영 지연이 있을 수 있다.
- **현재 실행분 추정:** API가 제공하는 시간당 요금 × 현재 실행 시간. 누적 집계 비용과 합산하지 않는다.
- **Pod 중단:** 확인 창 후 마지막 결과 수집을 시도하고 중단을 요청한다. 실제 중단 상태를 API로 다시 확인한다. 결과 수집 실패 시에도 중단 요청은 진행한다.
- 중단은 Pod 삭제가 아니다. 실행 작업·컨테이너 파일에 영향이 있으며 볼륨·네트워크 저장소 요금은 남을 수 있다. 별도 네트워크 볼륨 요금은 Pod 집계에 포함되지 않는다.

평가 완료 후 **결과 폴더 전체를 HF에 보관하고 Pod를 자동 중단**하려면 연구실 서버의 tmux에서 실행한다.

```bash
.venv/bin/python tools/runpod/finish_eval.py --pod <Pod_ID> --job <작업_ID> --repo Dohyeon-Lee-02/SkillVLA-checkpoints
```

HF Desk로 공개 저장소의 `lerobot/examples/libero/configs/train_skillVLA/{PT,FT}/eval/outputs/<평가명>/`에 원래 구조대로 업로드한다.
파일 목록·해시 검증 후 중단한다. 평가·업로드 실패 시 중단하지 않으며, 기록은 `.cache/runpod_finish/`에 남는다.

[RunPod 공식 API 매핑](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod-migrate/reference/graphql-to-v2.md) · [공식 billing SDK](https://pkg.go.dev/github.com/runpod/go-sdk/v2/lib/billing)

기존 Desk를 유지한 채 새 버전을 열려면:

```bash
bash sync_server_ui.sh --port 8768 --reuse-desk-port 8765
```

기존 전송·HF·Slurm 작업은 원래 서비스에서 계속 실행된다. 새 화면은 8768 포트를 전달한다.
