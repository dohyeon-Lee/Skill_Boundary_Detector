# SkillVLA 구현·실험 주의사항

## 2026-10-19: transition jitter의 미래 시작 관측 수정

### 발견된 문제

`SkillVLADataset`은 현재 프레임 `t`에서 선택한 skill `k'`의 시작을
`GT_start[k'] + offset`으로 계산했다. 에피소드 범위만 제한했으므로,
양의 offset이 현재 skill 진행 시간보다 크면 시작 관측이 `t`보다 미래였다.
예: GT 시작 100, 현재 105, offset +10 → 시작 영상·상태는 110.

Both3/Both_LIT_7의 VSA 단독 경로는 시작 상태 전체를 쓰는 것이 아니라,
`skill_start_state[:3]`를 **skill-start XYZ**로 전달한다. 따라서 현재
proprio 자체가 미래 것으로 교체된 것은 아니지만 시작 XYZ에는 미래
상태가 들어갈 수 있었다. 기존 VSA는 시작 영상을 action 경로에 사용하지
않는다. **Skill-end XYZ는 의도한 목표 조건이며 이 문제와 구분한다.**

### 수정 계약

- 관측 시작 프레임: `clip(GT_start[k'] + offset, 0, t)`.
- 시작 top/wrist 영상과 ISS의 시작 상태는 반드시 같은 수정 프레임을 쓴다.
- ISS 조회 offset은 `observed_start - GT_start[k']`로 따로 계산한다.
- 원래 jitter draw와 skill 선택은 유지한다. 특히 delayed transition의
  `effective_de`에는 원래 offset을 사용해 종료 감독 신호를 바꾸지 않는다.
- 현재 영상·proprio, 선택된 skill의 종료 XYZ/focus, canonical full-skill
  정답 궤적은 바꾸지 않는다. 데이터셋 재빌드는 필요 없다.
- step/percent 및 half-normal/uniform의 **추출 분포**는 유지하지만,
  실제 적용 시작 offset은 현재 프레임에서 clipping되므로 시작 근처의
  적용 분포는 달라진다.
- 디버깅 필드 `skill_start_requested_frame` / `skill_start_frame`을 비교하면
  clipping 발생 여부를 확인할 수 있다.
- VSA 로그의 `regime/start_observation_future_clipped_fraction`은 현재 배치에서
  미래 시작 관측을 현재 프레임으로 제한한 비율이다.

공통 `SkillVLADataset`을 새로 import하는 프로세스에 적용된다. 이미 로딩된
코드나 기존 체크포인트의 학습 이력은 소급 변경되지 않는다. 실행 중 새로
생성되는 worker의 import 방식은 별도 확인이 필요하다. 새 조건으로
resume하면 그 지점부터 증강 계약이 달라지므로 과거 실험과 구분해 기록한다.
이 문제를 기존 성능 저하의 원인으로 확정한 것은 아니다.

### 후속 확인: skill predictor / Joint

- [ ] PT/FT skill predictor가 사용하는 시작 영상·정규화 시작 proprio와
  skill/XYZ 정답이 같은 jitter 선택·프레임을 따르는지 실제 배치로 확인.
- [ ] Predictor의 별도 occurrence/group sampler와 current-frame mode에서도
  새 clipping이 적용되고 그룹별 시작 관측 가정에 영향이 없는지 확인.
  현재 occurrence sampler 코드는 virtual start 이후에서만 프레임을 고르므로,
  단순 frame-level sampling과 미래 시작 문제의 발생 조건이 다를 수 있다.
  아직 실제 predictor batch를 이용한 end-to-end 확인은 하지 않았다.
- [ ] Joint Predictor의 시작 입력, VSA의 시작 XYZ와 현재 관측이 일관적인지
  확인. `effective_de`/action mask는 기존 transition draw와 유지돼야 한다.
- [ ] Joint의 별도 Terminator 시작 입력 및 post-end predictor 입력은 이
  start-observation 경로와 다르므로 독립적으로 causality를 검사.
- [ ] 과거/수정 계약을 통제한 재학습 비교 전에는 predictor/Joint 성능
  차이의 원인이라고 단정하지 않는다.

### Fullskill visual 모델 연결 (2026-10-19 노션 갱신 반영)

설계 출처: [Notion Fullskill](https://app.notion.com/p/6af5815289a846e290c3f43e7ce337ba).

시작 병목은 위의 causal 시작 영상·상태로 생성하고, 별도 Start bridge가
아래의 Expert layer에서 읽는다. 현재 병목은 기존 Current
bridge가 Expert L18에서 읽는다. 인코더·Cond·병목 생성기는 공유하지만
두 bridge의 가중치와 시작/현재 병목 상태는 분리한다. 시작 병목은
rollout에서 실제 skill activation 시 저장하며 episode/skill 전환 시 갱신한다.

| architecture.name | 기반 | Start bridge | 직접 current proprio |
| --- | --- | --- | --- |
| fullskill_vis_1_firstlayer | Both_LIT_7 | L1 | 없음 |
| fullskill_vis_1_lastlayer | Both_LIT_7 | L17 | 없음 |
| fullskill_vis_2_firstlayer | Both3 | L1 | L18만 |
| fullskill_vis_2_lastlayer | Both3 | L17 | L18만 |

기존 `fullskill_vis_1`은 최초 firstlayer V1 모델의 이전 이름으로
checkpoint 호환을 위해 원래 동작으로 계속 읽을 수 있다. 새 실험은 명시적인 새 이름을
사용한다. firstlayer/lastlayer는 **17-layer Expert 경로의 첫/마지막**을
의미하며, Cond의 첫/마지막 layer를 선택한다는 뜻이 아니다.

구현 선택: 시작 시점 Cond의 **최종 Z18**을 L1 또는 L17에 주입한다(Vision/Cond의
첫 layer 출력이라는 뜻이 아님). 현재 V2의 Start/Current bridge는 모두
action용 98-query reader + 별도 gated 2-query reader를 갖는다. 두 bridge의
reader/gate 가중치는 서로 분리된다. 주 bridge gate 초기값은 0.01,
2-query bridge gate 초기값은 양쪽 모두 0이다.

시작/현재 영상 모두 공유 top/wrist patch head로 skill-end patch loss를
계산한다. 최종 보조항은 기존 weight × 0.5 × (start top+wrist loss +
current top+wrist loss)이다. 시작 영상 인코딩 직후 시작 patch loss를
계산해 공용 readout cache가 current 영상에 의해 덮이기 전에 소비한다.
배치당 인코딩은 여전히 두 번이며 세 번째 인코딩은 없다.
시작 wrist target은 raw `skill_start_state`의 EEF pose로 재투영한다.
Top target은 runtime dataset이 저장한 `skill_start_focus_uv/valid`를 사용해
current-frame 증강 metadata와 분리한다. 화면 밖/카메라 뒤/자기 gripper
patch의 wrist target은 기존 validity 규칙대로 제외한다.
시작/current raw loss와 시작 top/wrist valid fraction을 별도로 기록한다.
Action/full-skill loss의 gradient는 시작 인코더/bridge까지 detach 없이 전달한다.

최초 V1은 Start bridge가 98개만 읽고 patch loss도 current만 계산했다.
4개 새 이름의 fresh training은 `*_symmetric_start_patch_v2` revision을 쓴다.
V1 revision checkpoint는 V1 구조/목표 그대로 읽으며, shape가 다른 V2로
조용히 재해석하지 않는다. 새 보조항을 위해 데이터셋 재빌드는 필요 없다.
같은 출력 폴더에 V1의 last checkpoint가 있으면 새 V2 학습의 자동 resume을
중단하고 `run.suffix`로 새 run을 선택하도록 안내한다. 기존 파일은 건드리지 않는다.

`PT/VSA/vsa_train_config.yaml`의 `architecture.name` 또는
`./submit_train.sh fullskill_vis_2_lastlayer`와 같이 모델명으로 선택한다.
출력 run 이름에도 각 모델명이 들어간다. 저장된 revision으로
PT/FT 평가가 구조를 구별한다. Standalone FT는 기존 동결 정책을 따르되
Start bridge는 외부 adapter이므로 학습 가능하다. Latent Best-of-N과 Joint
routing은 아직 새 시작-vision 경로를 검증하지 않았으므로 명시적으로 거부한다.
vis_2의 직접 proprio projection은 Both3와 동일한 zero-init adapter이며,
L18의 AdaRMS에만 더한다. checkpoint 재계산에는 이미 계산한 layer별
condition을 명시적으로 전달해, 이후 forward가 proprio cache를 바꿔도
gradient 경로가 달라지지 않는다.
추가 Cond/비전 인코더 파라미터는 없지만 PT에서는 시작/현재 관측을 각각
인코딩하므로 연산·activation 메모리는 증가한다. 추론의 시작 병목은
스킬별로 한 번 인코딩하여 재사용한다.

## 2026-10-19: 통합 Cond/Expert SDPA

`PT/VSA/vsa_train_config.yaml`의 `architecture.attention_backend`는
`eager`(기본·이전 체크포인트 동작) 또는 `sdpa`이다. Layerwise Cond 모델
(Both3/LIT, Direct/Residual 및 fullskill-vis)에 적용하며, 시작/현재 Cond와
Expert의 action/full-skill/4H self-attention을 함께 바꾼다. Bridge·DINO의
backend, 가중치 구조와 loss는 유지한다. 모두 **양방향 attention**이므로
SDPA에 `is_causal=False`를 명시한다. All-visible mask만 생략하며 canonical
full-skill의 padding attention mask는 그대로 전달한다. Direct/Residual은
기존처럼 전체 4H 토큰을 읽고 중앙 H를 마지막 레이어에 전달한다.

SDPA 선택 시 architecture 이름 뒤에 `_sdpa`를 붙여 기존 eager 결과의
자동 resume과 분리한다. backend는 policy config에 저장되어 평가/FT에서
다시 로드하며, 필드가 없는 기존 checkpoint는 eager이다. Attention map
진단의 별도 명시적 계산 경로는 바꾸지 않는다. SDPA의 실제 fused kernel
선택·VRAM/속도 효과는 GPU/PyTorch 환경에 달려 있으며 절약량은 실측해야 한다.
수학적 attention 연산은 같지만 수치 연산 순서가 달라 bitwise 일치는 보장하지 않는다.

이전 Cond-only checkpoint의 `cond_attention_backend`는 호환용으로 읽는다.
새 `attention_backend`가 없으면 Expert는 eager로 남아 과거 동작을 보존한다.
기존 YAML snapshot의 Cond-only 설정도 그대로 실행 가능하지만 새 YAML에는
통합 옵션 하나만 사용한다. 두 옵션을 함께 쓰면 에러로 안내한다. 같은
`_sdpa` 폴더의 checkpoint가 Cond-only이면 unified SDPA로 자동 resume하지
않고 새 `run.suffix`를 요구한다. 기존 모델/optimizer 파일은 수정하지 않는다.
