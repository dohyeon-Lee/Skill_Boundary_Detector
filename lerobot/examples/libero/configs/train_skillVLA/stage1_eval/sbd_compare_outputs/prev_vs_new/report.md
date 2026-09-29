# SBD dataset comparison

재학습 없이 parquet의 canonical skill 구간과 현재 jitter 계약을 분석한 결과입니다.

| 지표 | prev | new | new-prev | 선호 방향 |
|---|---:|---:|---:|---|
| 평균 skill 길이 | 50.730684 | 59.420564 | +8.689880 | higher |
| 평균 유효 action / 10 | 9.112963 | 9.242686 | +0.129724 | higher |
| 마스킹 비율 | 0.088704 | 0.075731 | -0.012972 | lower |
| 5-step 완전 유효 비율 | 0.921152 | 0.932683 | +0.011531 | higher |
| 10-step 완전 유효 비율 | 0.822593 | 0.848537 | +0.025945 | higher |
| Jitter 포함 평균 유효 action | 9.185622 | 9.296852 | +0.111229 | higher |
| Token 최대 점유율 | 0.183962 | 0.158555 | -0.025407 | lower |
| Action 분산: token / 전체 | 0.702642 | 0.741489 | +0.038847 | lower |
| Action 분산: task+token+phase / task+phase | 0.437876 | 0.503623 | +0.065747 | lower |
| End XYZ 분산: task+token / task | 0.188742 | 0.208573 | +0.019832 | lower |
| Boundary action-change percentile | 49.836445 | 44.996722 | -4.839723 | higher |
| Boundary ±window 내 top-10% 변화 비율 | 0.504658 | 0.377098 | -0.127559 | higher |
| Boundary 전후 action 평균 변화 | 1.041312 | 0.900162 | -0.141150 | higher |

## Boundary correspondence

- 동일 episode/frame/action: `True` / max action diff `0`
- new boundary와 가장 가까운 prev boundary 거리: 평균 `7.028` frame, median `3.000` frame
- ±2 frame 일치율: `42.322%`
- ±5 frame 일치율: `72.557%`

## 해석 기준

- 마스킹 가설은 유효 action 수와 5/10-step 완전 유효 비율 차이가 클 때 지지됩니다.
- skill 표현 일관성은 action/end-XYZ 분산 비율이 낮을수록 좋습니다.
- boundary-action 정렬은 percentile, local top-10%, 전후 action 변화가 높을수록 강합니다.
- 이 통계는 상관 근거이며 단독으로 성공률 차이의 인과를 확정하지 않습니다.
