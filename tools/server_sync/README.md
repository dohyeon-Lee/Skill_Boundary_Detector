# Server Sync UI — Transfer Desk

외부 **rllab 서버에서 실행**하는 로컬 웹 앱입니다. 터미널 메뉴 대신 여러 폴더를
선택하고, 마지막 확인 화면에서 한 번에 전송합니다. 서버 파일을 다루는 도구이므로
공개 사이트에 배포하지 않고 `127.0.0.1`에만 바인딩합니다.

## 실행 / 접속

외부 서버의 프로젝트 루트에서:

```bash
bash sync_server.sh --web
# 또는 bash sync_server_ui.sh
# 다른 포트: bash sync_server.sh --web --port 8766
```

프로젝트 `.venv/bin/python`을 우선 사용합니다. Python 3.10+와 `ssh`, `rsync`가 필요하며
추가 Python/Node 패키지는 설치하지 않습니다. 원격 Yonsei에는 Python 3.6+와 rsync
3.1+가 필요합니다. 기존 `bash sync_server.sh` 터미널 메뉴도 그대로 사용할 수 있습니다.

- **VSCode Remote SSH:** Ports 탭에서 `8765`를 전달하고 브라우저로 엽니다.
- **일반 SSH:** 본인 컴퓨터 터미널에서 아래처럼 터널을 연 다음
  `http://127.0.0.1:8765`로 접속합니다. SSH 포트가 다르면 `-p <포트>`를 추가하세요.

```bash
ssh -N -L 8765:127.0.0.1:8765 <외부서버 SSH 사용자@주소>
```

SSH 세션을 끊어도 앱을 유지하려면 외부 서버에서 tmux 안에 실행하세요.
**브라우저를 닫는 것은 전송을 중단하지 않습니다.** 웹 서버를 Ctrl+C로 종료하면
실행·대기 항목도 취소됩니다. 대기열/로그는 메모리에만 있으며 서버를 재시작하면 사라집니다.

## 예: Residual 1~6의 50k만 보내기

1. 출발 `rllab /data1`, 도착 `yonsei /scratch2`를 선택합니다.
2. `outputs_filtered → skillVLA_PT → VSA`를 열고 검색창에 `050000`을 입력합니다.
   검색은 각 모델 폴더 안을 최대 2단계(`checkpoints/050000`)까지 확인하고,
   해당 체크포인트가 있는 모델 폴더만 현재 목록에 표시합니다.
3. 검색 결과의 체크박스를 누르면 모델 전체가 아니라 매칭된 **`checkpoints/050000`만**
   선택됩니다. 모델 이름을 클릭하면 기존처럼 해당 모델 폴더를 엽니다.
4. 필요한 모델을 모두 체크합니다. **검색이나 탐색 경로가 바뀌어도 선택 목록은 유지됩니다.**
5. 6개가 담긴 목록에서 **전송 준비 → 목적지 확인 → 전송 시작**을 누릅니다.

각 항목은 원래 상대 경로를 유지합니다. 예를 들어
`outputs_filtered/.../residual_patch_1.../checkpoints/050000`은 목적지에서도 같은 위치에
저장되므로 모델별 체크포인트가 섞이지 않습니다. 상위 폴더와 그 안의 항목을 함께
선택하면 중복 전송은 합칩니다. `last` 같은 symlink 자체를 선택하지 말고 숫자 폴더를
선택하세요. 상위 폴더 전체를 전송할 때 그 안의 symlink는 rsync의 기존 `-a` 동작대로 보존됩니다.

## 동작 / 안전 장치

- 기본 동시 전송 **2개**, 나머지는 큐에서 기다립니다. 항목별 취소·로그와 실패 후 다시 담기를 제공합니다.
- 로컬 폴더 복사는 기존 `sync_server.sh`의 `parallel_local_copy`를 재사용합니다(기본 8 streams).
- 원격 전송은 SSH + rsync, 같은 원격 호스트의 scratch↔scratch2는 그 호스트에서 rsync합니다.
- 진행률은 로컬 병렬 복사의 경우 파일/폴더 개수 기준, 원격의 경우 rsync 기준 **추정치**입니다.
  성공 종료 전에는 완료로 표시하지 않습니다.
- **`--delete`를 사용하지 않습니다.** 목적지의 추가 파일은 유지되지만 같은 이름의 변경 파일은
  rsync의 크기/수정시각 기준으로 갱신될 수 있으므로 최종 경로를 확인하세요.
- 선택된 경로 밖의 파일 탐색, 절대 경로/`..`, symlink를 통한 우회,
  출발·도착 경로 중첩과 동시에 같은 목적지에 쓰는 중복 작업을 차단합니다.
- 서버가 아닌 외부 웹사이트에서 실행 요청을 보내지 못하도록 Host/Origin 검증과 세션 토큰을 사용합니다.
- **아직 저장 중인 체크포인트를 복사하지 마세요.** 전송 시작 시 경로를 다시 확인하지만
  학습 프로세스의 쓰기를 잠그거나 원본을 스냅샷으로 만들지는 않습니다.

## 학습 중 체크포인트 자동 보관 (Predictor FT)

Predictor FT YAML의 `training`에 다음 두 옵션을 설정하면, 체크포인트 저장이
완료될 때마다 UI 없이 별도 CPU 프로세스가 Yonsei로 자동 전송합니다:

```yaml
training:
  checkpoint_archive: scratch2  # Yonsei 165.132.142.208, 기존 YONSEI_* 환경변수 재사용
  keep_checkpoints: 2          # 로컬 최신 2개. -1이면 전송하되 로컬도 전부 보존
  schedule:
    steps: 50000
    save_every: 10000
    save_steps: [0, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
```

`save_steps`는 `save_every`에 추가되는 것이 아니라 **대체**합니다. 위 설정은
모든 지정 시점을 원격에 보관하고, 로컬은 최종적으로 40000/50000만 남깁니다.
`checkpoint_archive: ""`, `keep_checkpoints: -1`이면 기존 방식으로 작동합니다.
현재 Predictor FT에 연결되어 있으며, 공유 auxiliary 제출 경로를 쓰는
Predictor/Terminator도 명시적으로 같은 옵션을 설정할 수 있습니다.
Molmo의 별도 원본 trainer에 이 기능이 자동으로 적용되는 것은 아닙니다.

- 모델뿐 아니라 optimizer, scheduler(존재 시), RNG 등 체크포인트 전체를 전송합니다.
- SSH/rsync 설정은 수동 UI와 공유하지만, **자동 전송 대기열은 UI 메모리 큐와 별개**로
  해당 run의 `.checkpoint_archive/queue`에 기록됩니다. 브라우저/UI 실행은 필요 없습니다.
- 저장 중인 폴더나 임의의 기존 폴더를 검색해서 지우지 않습니다. 저장 완료 후 등록된
  숫자 체크포인트만 관리합니다. 이전 실험 결과는 소급 등록/삭제하지 않습니다.
- 원격 숨김 staging 폴더로 전송하고 파일 목록/크기/SHA256 검증 후 정식 숫자 폴더로
  원자적으로 공개합니다. 다른 내용의 기존 원격 체크포인트는 덮어쓰지 않습니다.
- 로컬 삭제 직전에도 양쪽 파일 해시를 다시 검사합니다. 미전송·전송 실패·내용 변경·
  원격 사본 누락 시에는 삭제하지 않으며 최신 N개와 로컬 `last`는 항상 보존합니다.
- 원격 `checkpoints/last`도 완료된 체크포인트로 갱신합니다. 같은 상대 경로를 유지합니다.
- 전송/해시는 백그라운드에서 하지만 CPU·디스크·네트워크 I/O 비용은 발생합니다.
  전송보다 저장이 빠르거나 네트워크가 막히면 로컬에 2개 이상 쌓일 수 있습니다.
- **정상 학습 종료 시 마지막 전송/정리 완료까지 대기**한 후 job을 종료합니다.
  장시간 통신 장애 시 job이 전송 대기 상태로 남을 수 있습니다. Slurm 시간 제한이나
  강제 종료로 worker가 죽더라도 미완료본은 로컬에 남고 다음 저장/종료 때 재시도합니다.
  마지막 체크포인트 이후 전송만 남았다면 아래 `drain`으로 학습 없이 복구할 수 있습니다.
- 실행하는 rllab 계정/노드에서 비밀번호 없는 SSH, remote Python 3.6+, rsync가 필요합니다.
  목적지는 환경변수로 변경할 수 있지만, 이미 등록한 run의 목적지를 도중에 바꾸면
  오래된 검증 기록으로 삭제하지 않도록 차단합니다.

프로젝트 루트에서 (run 경로에는 실제 학습 결과 폴더 지정):

```bash
python tools/server_sync/checkpoint_archive.py status --run outputs_filtered/skillVLA_FT/Predictor/<run>
python tools/server_sync/checkpoint_archive.py drain --run outputs_filtered/skillVLA_FT/Predictor/<run>
```

진행/오류 로그는 `<run>/.checkpoint_archive/worker.log`에 남습니다.
자동 대기열과 확인용 해시 파일은 관리용 메타데이터이며 학습 가중치/손실에는 관여하지 않습니다.

## 기존 환경변수 / 옵션

기존 `sync_server.sh`와 같은 엔드포인트 설정을 읽습니다:
`YONSEI_HOST`, `YONSEI_SCRATCH_BASE`, `YONSEI_SCRATCH2_BASE`, `RLLAB_BASE`, `RLLAB2_BASE`,
`SYNC_MODE`, `SYNC_JOBS`. Yonsei 기본 접속 주소는 `mdorazi@165.132.142.208`입니다.
실행 호스트 이름이 `yonsei*`면 내부 scratch↔scratch2만 표시합니다.

```bash
YONSEI_HOST=mdorazi@165.132.142.208 bash sync_server.sh --web
SYNC_JOBS=4 bash sync_server.sh --web --concurrency 2
# Python 선택: SYNC_UI_PYTHON=/path/to/python bash sync_server.sh --web
```

## SSH 인증 오류가 뜰 때

웹 요청 안에서는 비밀번호를 대화식으로 입력할 수 없으므로 SSH는 `BatchMode=yes`를 사용합니다.
외부 서버 터미널에서 먼저 `ssh mdorazi@165.132.142.208`로 연결해 호스트 키를 확인하고,
그 **외부 서버 계정의 SSH 키/agent 또는 기존 SSH multiplex 세션**으로 비밀번호 없이 접속되게 설정하세요.
호스트 키 검증을 끄거나 비밀번호를 웹 앱에 저장하지 않습니다.

```bash
ssh -o BatchMode=yes mdorazi@165.132.142.208 true
```

위 명령이 성공하면 UI도 같은 인증 환경을 사용할 수 있습니다. 다른 로그인 서버를 사용하면
`YONSEI_HOST`를 변경하세요. `rsync` 3.1 미만이면 `--info=progress2`를 지원하지 않아 전송이 실패할 수 있습니다.

## 테스트

```bash
.venv/bin/python -m unittest discover -s tools/server_sync/tests -v
node tools/server_sync/tests/test_ui.mjs
```

실제 복사는 새 임시 디렉토리 안에서만 테스트합니다. SSH는 mock으로 검증하며 실제 서버 전송은 수행하지 않습니다.
