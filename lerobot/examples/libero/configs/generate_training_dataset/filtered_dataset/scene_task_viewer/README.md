# scene_task_viewer

YAML로 분석할 (filtered) LIBERO 데이터셋들을 받아, **scene별로 어떤 task가 있는지**를
탐색할 수 있는 **단일 HTML**을 만든다. 왼쪽에서 scene을 고르면 그 scene의 task들이
**첫 프레임 이미지 + language + 소속 데이터셋 + task_index** 카드로 표시된다.

## 실행

```bash
cd .../filtered_dataset/scene_task_viewer
/data2/dohyeon/SBD/.venv/bin/python build_scene_task_html.py --config config.yaml
# -> scene_tasks.html  (브라우저로 열기)
```

`--output other.html`로 출력 경로를 덮어쓸 수 있다.

## 동작 원리

- `meta/tasks.parquet` → `task_index → language`
- `meta/episodes/*` → 각 task를 담은 첫 에피소드의 정면 카메라 비디오 위치(`from_timestamp`)
  를 찾아 PyAV로 **첫 프레임**을 디코드, JPEG→base64로 HTML에 임베드(외부 파일 0개).
- scene은 LIBERO benchmark의 `libero_task_map`(bddl 이름, 예: `KITCHEN_SCENE1_open_the_...`)
  에서 유도. `libero_{spatial,object,goal}`처럼 scene 접두가 없는 suite는
  `"<suite> · single scene"`으로 묶는다.

## config.yaml

| 키 | 설명 |
|---|---|
| `project_root`, `dataset_root` | 데이터셋 루트(`{dataset_root}/{name}`) |
| `libero_pkg` | LIBERO benchmark 경로(scene↔language 매핑용) |
| `image_key` | 카드 이미지 카메라(`observation.images.image` 정면 / `...wrist_image` 손목) |
| `jpeg_quality` | 첫 프레임 JPEG 품질(1–95) — HTML 용량 조절 |
| `datasets` | `- name: libero_90_full_full` 목록. `suite`는 이름에서 자동 유도(필요 시 명시) |
| `output_html` | 출력 HTML 경로 |

의존성은 저장소 `.venv`에 이미 있음(pandas, pyarrow, PyAV, Pillow, PyYAML).
