#!/usr/bin/env python
"""scene_task_viewer — scene별 task 탐색용 단일 HTML 생성기.

YAML로 분석할 (filtered) LIBERO 데이터셋들을 받아, 각 데이터셋의
  meta/tasks.parquet  (task_index -> language)
  meta/episodes/*     (task -> 대표 에피소드의 첫 프레임 비디오 위치)
를 읽고, LIBERO benchmark의 libero_task_map(bddl 이름)으로 language를 scene에
매핑한다. 결과를 하나의 self-contained HTML로 렌더 — 왼쪽에서 scene을 고르면
그 scene에 속한 task들의 첫 프레임 + language + 소속 데이터셋 + task_index가 뜬다.

    python build_scene_task_html.py --config config.yaml
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
import sys
from pathlib import Path

import av
import pandas as pd
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent

# libero_task_map의 suite 키(긴 것부터 — 접두 매칭 우선순위).
SUITE_KEYS = ["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"]
SCENE_RE = re.compile(r"^([A-Z][A-Z_]*SCENE\d+)_(.+)$")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(root: Path, p: str) -> Path:
    """절대경로면 그대로, 아니면 root 기준."""
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def derive_suite(name: str) -> str:
    for key in SUITE_KEYS:
        if name == key or name.startswith(key + "_"):
            return key
    raise ValueError(
        f"'{name}'에서 suite를 유도할 수 없음. config에 suite를 명시하세요 "
        f"(가능한 값: {SUITE_KEYS})."
    )


def build_lang2scene(task_map: dict, suite: str) -> dict:
    """suite의 bddl 이름들 -> {정규화된 language: scene}."""
    out = {}
    single_scene = f"{suite} · single scene"
    for bddl in task_map[suite]:
        m = SCENE_RE.match(bddl)
        if m:
            scene, lang = m.group(1), m.group(2)
        else:
            scene, lang = single_scene, bddl
        out[lang.replace("_", " ").strip()] = scene
    return out


def first_frame_jpeg(video_path: Path, ts: float, quality: int) -> str | None:
    """비디오의 ts(초) 프레임을 디코드해 base64 data-URI(JPEG)로 반환."""
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        # 해당 에피소드 시작 지점 근처로 seek 후 첫 유효 프레임 취득.
        offset = int(ts / stream.time_base) if stream.time_base else 0
        container.seek(offset, stream=stream, any_frame=False, backward=True)
        picked = None
        for frame in container.decode(stream):
            if frame.time is None:
                picked = frame
                break
            if frame.time >= ts - 1e-3:
                picked = frame
                break
            picked = frame
        container.close()
        if picked is None:
            return None
        img = Image.fromarray(picked.to_ndarray(format="rgb24"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:  # 개별 프레임 실패는 전체를 막지 않음
        print(f"    ! frame decode 실패 ({video_path.name} @ {ts:.2f}s): {e}")
        return None


def process_dataset(ds_dir: Path, suite: str, task_map: dict, image_key: str,
                    jpeg_quality: int) -> list[dict]:
    """한 데이터셋 -> task 레코드 리스트."""
    meta = ds_dir / "meta"
    tasks = pd.read_parquet(meta / "tasks.parquet").reset_index()  # cols: task, task_index
    ep_files = sorted(glob.glob(str(meta / "episodes" / "**" / "*.parquet"), recursive=True))
    if not ep_files:
        raise FileNotFoundError(f"episodes 메타가 없음: {meta/'episodes'}")
    episodes = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)

    lang2scene = build_lang2scene(task_map, suite)

    ci_col = f"videos/{image_key}/chunk_index"
    fi_col = f"videos/{image_key}/file_index"
    ts_col = f"videos/{image_key}/from_timestamp"
    if ci_col not in episodes.columns:
        raise KeyError(
            f"image_key '{image_key}'가 episodes에 없음. "
            f"사용 가능: {[c.split('/')[1] for c in episodes.columns if c.startswith('videos/') and c.endswith('/chunk_index')]}"
        )

    records = []
    n = len(tasks)
    for i, row in tasks.iterrows():
        lang = str(row["task"])
        tidx = int(row["task_index"])
        scene = lang2scene.get(lang, f"{suite} · (scene 미상)")
        # 이 task를 담은 첫 에피소드.
        mask = episodes["tasks"].apply(lambda x: lang in list(x))
        ep_match = episodes[mask]
        img_uri = None
        if len(ep_match):
            cand = ep_match.iloc[0]
            ci, fi = int(cand[ci_col]), int(cand[fi_col])
            ts = float(cand[ts_col])
            vpath = ds_dir / "videos" / image_key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
            if vpath.exists():
                img_uri = first_frame_jpeg(vpath, ts, jpeg_quality)
        print(f"    [{i+1:>3}/{n}] task {tidx:>3} · {scene:<24} · {lang[:48]}"
              + ("" if img_uri else "  (이미지 없음)"))
        records.append({
            "dataset": ds_dir.name,
            "suite": suite,
            "scene": scene,
            "task_index": tidx,
            "language": lang,
            "image": img_uri,
        })
    return records


def scene_sort_key(scene: str):
    """KITCHEN_SCENE1 < KITCHEN_SCENE10 자연 정렬, single scene은 뒤로."""
    single = "single scene" in scene
    m = re.match(r"^([A-Z_]+?)_?SCENE(\d+)$", scene.split(" · ")[0])
    if m:
        return (1 if single else 0, m.group(1), int(m.group(2)), scene)
    return (1 if single else 0, scene, 0, scene)


def render_html(records: list[dict], datasets: list[str], image_key: str) -> str:
    # scene 정렬 순서(사이드바 그룹 순서)만 미리 계산하고, task 그룹핑은 JS에서.
    scene_order = sorted({r["scene"] for r in records}, key=scene_sort_key)
    payload = json.dumps({
        "tasks": records,
        "scene_order": scene_order,
        "dataset_order": sorted(datasets),
        "image_key": image_key,
        "total_tasks": len(records),
    }, ensure_ascii=False)

    return _HTML_TEMPLATE.replace("__PAYLOAD__", payload)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIBERO scene · task viewer</title>
<style>
  :root{
    --bg:#0f1115; --panel:#161a21; --panel2:#1c222c; --border:#2a3240;
    --text:#e6e9ef; --muted:#9aa4b2; --accent:#5b9dff; --accent2:#3b6fd4;
    --chip:#243044; --chiptext:#a9c4ff; --chip2:#2c2440; --chip2text:#d3b9ff;
  }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f4f6fa; --panel:#ffffff; --panel2:#f0f3f8; --border:#d8dee8;
      --text:#1a2230; --muted:#5a6676; --accent:#2f6fe0; --accent2:#2258c4;
      --chip:#e6eefc; --chiptext:#2258c4; --chip2:#efe8fc; --chip2text:#6a3fc0; }
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--text);
    font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
    display:flex;flex-direction:column;height:100vh;overflow:hidden}
  header{padding:14px 20px;border-bottom:1px solid var(--border);background:var(--panel);
    display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;font-weight:650}
  header .meta{color:var(--muted);font-size:12.5px}
  .toggle{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-left:auto}
  .toggle button{border:none;background:var(--panel2);color:var(--muted);padding:6px 14px;
    font-size:12.5px;cursor:pointer;font-weight:550}
  .toggle button.on{background:var(--accent);color:#fff}
  .chip.scene{background:var(--chip2);color:var(--chip2text)}
  .wrap{display:flex;flex:1;min-height:0}
  aside{width:270px;flex:none;border-right:1px solid var(--border);background:var(--panel);
    overflow-y:auto;padding:10px}
  aside .search{width:100%;padding:8px 10px;margin-bottom:8px;border-radius:8px;
    border:1px solid var(--border);background:var(--panel2);color:var(--text);font-size:13px}
  .scene-btn{display:flex;justify-content:space-between;align-items:center;gap:8px;
    width:100%;text-align:left;padding:9px 11px;margin:2px 0;border-radius:8px;cursor:pointer;
    border:1px solid transparent;background:transparent;color:var(--text);font-size:13px}
  .scene-btn:hover{background:var(--panel2)}
  .scene-btn.active{background:var(--accent);color:#fff;border-color:var(--accent2)}
  .scene-btn .cnt{font-size:11px;background:var(--chip);color:var(--chiptext);
    padding:1px 8px;border-radius:20px;flex:none}
  .scene-btn.active .cnt{background:rgba(255,255,255,.25);color:#fff}
  main{flex:1;overflow-y:auto;padding:18px 22px;min-width:0}
  main h2{margin:0 0 4px;font-size:18px}
  main .sub{color:var(--muted);font-size:12.5px;margin-bottom:16px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;overflow:hidden;
    display:flex;flex-direction:column}
  .card .imgwrap{aspect-ratio:1/1;background:var(--panel2);display:flex;align-items:center;
    justify-content:center;overflow:hidden}
  .card img{width:100%;height:100%;object-fit:cover;display:block}
  .card .noimg{color:var(--muted);font-size:12px}
  .card .body{padding:11px 12px;display:flex;flex-direction:column;gap:8px}
  .card .lang{font-size:13.5px;font-weight:550;line-height:1.35}
  .card .chips{display:flex;gap:6px;flex-wrap:wrap}
  .chip{font-size:11px;padding:2px 9px;border-radius:20px;background:var(--chip);color:var(--chiptext);
    white-space:nowrap}
  .chip.idx{background:var(--panel2);color:var(--muted)}
  .empty{color:var(--muted);padding:40px;text-align:center}
</style>
</head>
<body>
<header>
  <h1>LIBERO · task viewer</h1>
  <span class="meta" id="hdr-meta"></span>
  <span class="toggle" id="toggle">
    <button data-mode="scene" class="on">Scene 기준</button>
    <button data-mode="dataset">Dataset 기준</button>
  </span>
</header>
<div class="wrap">
  <aside>
    <input class="search" id="search" placeholder="검색…" autocomplete="off">
    <div id="group-list"></div>
  </aside>
  <main id="main"><div class="empty">왼쪽에서 항목을 선택하세요.</div></main>
</div>
<script>
const DATA = __PAYLOAD__;
const listEl = document.getElementById('group-list');
const mainEl = document.getElementById('main');
const searchEl = document.getElementById('search');
document.getElementById('hdr-meta').textContent =
  `${DATA.scene_order.length} scenes · ${DATA.dataset_order.length} datasets · ${DATA.total_tasks} tasks · cam: ${DATA.image_key}`;

let mode = 'scene';   // 'scene' | 'dataset'
let active = null;

function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}

// mode에 따라 [{name, tasks[]}] 그룹 목록을 만든다.
function buildGroups(){
  const key = mode;                        // task 객체의 필드명과 동일
  const order = mode==='scene' ? DATA.scene_order : DATA.dataset_order;
  const map = new Map(order.map(n=>[n,[]]));
  for(const t of DATA.tasks){
    if(!map.has(t[key])) map.set(t[key],[]);
    map.get(t[key]).push(t);
  }
  const sortTasks = mode==='scene'
    ? (a,b)=> a.dataset.localeCompare(b.dataset) || a.task_index-b.task_index
    : (a,b)=> a.scene.localeCompare(b.scene) || a.task_index-b.task_index;
  return [...map.entries()].map(([name,tasks])=>({name,tasks:tasks.sort(sortTasks)}));
}

let GROUPS = [];

function renderList(){
  const filter=(searchEl.value||'').toLowerCase();
  listEl.innerHTML='';
  GROUPS.filter(g=>g.name.toLowerCase().includes(filter)).forEach(g=>{
    const b=document.createElement('button');
    b.className='scene-btn'+(g.name===active?' active':'');
    b.innerHTML=`<span>${esc(g.name)}</span><span class="cnt">${g.tasks.length}</span>`;
    b.onclick=()=>{active=g.name;renderList();renderGroup(g);};
    listEl.appendChild(b);
  });
}

function renderGroup(g){
  const cards=g.tasks.map(t=>`
    <div class="card">
      <div class="imgwrap">${t.image
        ? `<img loading="lazy" src="${t.image}" alt="">`
        : `<span class="noimg">이미지 없음</span>`}</div>
      <div class="body">
        <div class="lang">${esc(t.language)}</div>
        <div class="chips">
          <span class="chip scene">${esc(t.scene)}</span>
          <span class="chip">${esc(t.dataset)}</span>
          <span class="chip idx">task ${t.task_index}</span>
        </div>
      </div>
    </div>`).join('');
  mainEl.innerHTML=`<h2>${esc(g.name)}</h2>
    <div class="sub">${g.tasks.length} tasks</div>
    <div class="grid">${cards}</div>`;
  mainEl.scrollTop=0;
}

function refresh(){
  GROUPS=buildGroups();
  active = GROUPS.length ? GROUPS[0].name : null;
  renderList();
  if(GROUPS.length) renderGroup(GROUPS[0]);
  else mainEl.innerHTML='<div class="empty">항목이 없습니다.</div>';
}

document.getElementById('toggle').addEventListener('click',e=>{
  const btn=e.target.closest('button'); if(!btn) return;
  mode=btn.dataset.mode;
  [...document.querySelectorAll('#toggle button')].forEach(b=>b.classList.toggle('on',b===btn));
  searchEl.placeholder = mode==='scene' ? 'scene 검색…' : 'dataset 검색…';
  refresh();
});
searchEl.addEventListener('input',renderList);
refresh();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="YAML 설정 경로 (기본: 스크립트 옆 config.yaml)")
    ap.add_argument("--output", default=None, help="출력 HTML 경로(설정 override)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    project_root = Path(cfg.get("project_root", ".")).resolve()
    dataset_root = resolve(project_root, cfg.get("dataset_root", "dataset_filtered"))
    libero_pkg = resolve(project_root, cfg.get("libero_pkg", "tools/lerobot-libero"))
    image_key = cfg.get("image_key", "observation.images.image")
    jpeg_quality = int(cfg.get("jpeg_quality", 82))

    sys.path.insert(0, str(libero_pkg))
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map as task_map

    all_records: list[dict] = []
    ds_names: list[str] = []
    for entry in cfg["datasets"]:
        if isinstance(entry, str):
            entry = {"name": entry}
        name = entry["name"]
        suite = entry.get("suite") or derive_suite(name)
        ds_dir = resolve(dataset_root, name)
        print(f"[{name}] suite={suite}  ({ds_dir})")
        if not ds_dir.exists():
            print(f"  ! 폴더 없음, 건너뜀: {ds_dir}")
            continue
        recs = process_dataset(ds_dir, suite, task_map, image_key, jpeg_quality)
        all_records.extend(recs)
        ds_names.append(name)

    if not all_records:
        print("처리된 task가 없습니다.")
        sys.exit(1)

    out = args.output or cfg.get("output_html", "scene_tasks.html")
    out_path = resolve(HERE, out)
    html = render_html(all_records, ds_names, image_key)
    out_path.write_text(html, encoding="utf-8")
    n_scenes = len({r["scene"] for r in all_records})
    print(f"\n완료: {len(all_records)} tasks · {n_scenes} scenes → {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
