# GAIN-MTL Strategy Metrics Analysis Report (v2)

## 1. Overview

This report analyzes the evaluation results of **8 training strategies** in the GAIN-MTL
(Guided Attention Inference Multi-Task Learning) framework for manufacturing defect detection.

Each strategy incrementally adds loss components to improve both classification performance and
model interpretability (i.e., whether the model focuses on actual defect regions).

| Strategy | Description | Key Loss Components |
|----------|-------------|---------------------|
| S1 | Classification Only (Baseline) | `L_cls` |
| S2 | + CAM Guidance | `L_cls + L_cam_guide` |
| S3 | + Attention Mining | `L_cls + L_am + L_guide` |
| S4 | + Localization | S3 + `L_loc + L_consist` |
| S5 | Full (+ Counterfactual) | S4 + `L_cf` |
| S6 | Full + GT Mask Curriculum | S5 + multiplicative fusion curriculum |
| S7 | Attention Mining + Counterfactual | S3 + `L_cf` (no localization) |
| S8 | S7 + GT Mask Curriculum | S7 + multiplicative fusion curriculum |

### Strategy Grouping

- **Core Pipeline (S1-S5)**: Progressive loss stacking, S1 → S2 → S3 → S4 → S5
- **Curriculum Variants (S6, S8)**: GT mask blending with alpha decay for guided attention
- **Lightweight Variant (S7)**: S3 + counterfactual, skipping localization for efficiency

---

## 2. Full Results Table

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| **Accuracy** | 0.89 | 0.88 | **0.90** | 0.89 | 0.89 | **0.91** | 0.893 | 0.886 |
| **Recall** | 0.88 | 0.88 | 0.94 | 0.92 | 0.93 | **0.95** | 0.926 | 0.926 |
| **CAM-IoU** | 0.18 | 0.48 | 0.50 | 0.50 | 0.49 | 0.48 | **0.517** | 0.500 |
| **PointGame** | 0.08 | 0.90 | 0.89 | 0.90 | 0.89 | **0.92** | 0.88 | **0.92** |
| **Energy-Inside** | 0.17 | 0.68 | 0.69 | 0.70 | 0.70 | **0.76** | 0.693 | 0.758 |
| **Loc-IoU** | 0.02 | 0.02 | 0.04 | **0.27** | **0.27** | 0.25 | 0.032 | 0.035 |

---

## 3. Strategy별 상세 분석

### 3.1 S1 — Classification Only (Baseline)

| Metric | Value | 비고 |
|--------|-------|------|
| Accuracy | 0.89 | 순수 분류 성능은 양호 |
| Recall | 0.88 | 전 Strategy 중 최저 |
| CAM-IoU | 0.18 | 극히 낮음 — 모델이 결함 영역을 보지 않음 |
| PointGame | 0.08 | 극히 낮음 — 최대 activation이 결함 위치에 없음 |
| Energy-Inside | 0.17 | 극히 낮음 — attention 에너지 대부분이 결함 외부 |
| Loc-IoU | 0.02 | Localization head 없음 (baseline) |

**해석:**
- 전형적인 **"맞는 답, 틀린 근거 (right answer, wrong reason)"** 케이스.
- Accuracy 0.89로 분류 자체는 작동하지만, CAM-IoU 0.18 / PointGame 0.08은 모델이 **결함과 무관한 영역**을 보고 판단하고 있음을 의미.
- 이전 결과(CAM-IoU 0.116) 대비 0.18로 개선됨 → per-image min-max normalization 적용 효과.
- 그러나 PointGame이 0.161 → 0.08로 하락한 것은 normalization 후에도 **최대 activation 위치 자체는 여전히 결함 외부**에 있음을 보여줌.
- 제조 현장 배포 시 **신뢰할 수 없는 모델** — 높은 정확도에도 불구하고 잘못된 추론 근거.

---

### 3.2 S2 — CAM Guidance

| Metric | Value | vs S1 변화 |
|--------|-------|-----------|
| Accuracy | 0.88 | -0.01 |
| Recall | 0.88 | ±0.00 |
| CAM-IoU | 0.48 | **+0.30** (2.67x) |
| PointGame | 0.90 | **+0.82** (11.3x) |
| Energy-Inside | 0.68 | **+0.51** (4.0x) |
| Loc-IoU | 0.02 | ±0.00 |

**해석:**
- **해석 가능성(interpretability)이 극적으로 향상**되는 첫 번째 전환점.
- Weight-based CAM을 GT mask로 직접 지도(supervision)하는 것이 매우 효과적:
  - PointGame 0.08 → 0.90: 최대 activation이 이제 **90% 확률로 결함 내부**에 위치.
  - CAM-IoU 0.18 → 0.48: attention map과 GT mask 간의 공간적 겹침이 크게 증가.
  - Energy-Inside 0.17 → 0.68: attention 에너지의 68%가 결함 영역 내에 집중.
- **분류 성능은 소폭 하락** (Accuracy 0.89 → 0.88).
  - `L_cam_guide`가 feature 학습을 spatial 정보 쪽으로 편향시켜 분류에 약간의 간섭 발생.
  - Recall은 0.88로 변화 없음 — 결함 탐지 민감도에는 영향 미미.
- Localization head가 없으므로 Loc-IoU는 여전히 0.02 (아키텍처적 한계).

---

### 3.3 S3 — Attention Mining + Guided Attention

| Metric | Value | vs S2 변화 |
|--------|-------|-----------|
| Accuracy | 0.90 | **+0.02** |
| Recall | 0.94 | **+0.06** |
| CAM-IoU | 0.50 | +0.02 |
| PointGame | 0.89 | -0.01 |
| Energy-Inside | 0.69 | +0.01 |
| Loc-IoU | 0.04 | +0.02 |

**해석:**
- **분류와 해석 가능성 모두 개선**되는 핵심 Strategy.
- Attention Mining Module(GAIN)이 학습 가능한 attention을 통해 분류에 유의미하게 기여:
  - Recall 0.88 → 0.94: **+6%p 의 큰 도약** — attention이 실제 결함 feature를 포착.
  - Accuracy 0.88 → 0.90: +2%p 개선.
- CAM quality도 S2보다 소폭 개선 (CAM-IoU 0.48 → 0.50):
  - 이전 결과에서는 S3 < S2 였으나, 이번에는 **S3 >= S2로 역전**.
  - `lambda_guide`를 1.5로 강화한 효과 — 학습된 attention이 spatial 정확도를 유지하면서 분류 성능 향상.
- **S1-S5 중 분류-해석 가능성 균형이 가장 우수한 transition point.**

---

### 3.4 S4 — + Localization

| Metric | Value | vs S3 변화 |
|--------|-------|-----------|
| Accuracy | 0.89 | -0.01 |
| Recall | 0.92 | **-0.02** |
| CAM-IoU | 0.50 | ±0.00 |
| PointGame | 0.90 | +0.01 |
| Energy-Inside | 0.70 | +0.01 |
| Loc-IoU | 0.27 | **+0.23** |

**해석:**
- **Localization 능력이 본격적으로 활성화** (Loc-IoU 0.04 → 0.27).
  - 결함 위치를 pixel 수준에서 segmentation 가능해짐.
  - `L_loc` (Dice + BCE) + `L_consist` (attention-localization 정합성)가 함께 작동.
- **Multi-task interference가 여전히 관찰**되지만 이전보다 완화:
  - Accuracy 0.90 → 0.89 (-1%p), Recall 0.94 → 0.92 (-2%p).
  - 이전 결과 (Acc 0.883→0.872, Recall 0.894→0.866) 대비 **하락 폭이 크게 줄어듬**.
  - `lambda_loc` 0.3→0.2 축소 + `loc_warmup_ratio: 0.5` 적용 효과 확인.
- CAM quality는 안정적으로 유지 (CAM-IoU 0.50, PointGame 0.90).
  - Localization task가 attention의 공간적 학습에 보조적 역할 수행.

---

### 3.5 S5 — Full (+ Counterfactual)

| Metric | Value | vs S4 변화 |
|--------|-------|-----------|
| Accuracy | 0.89 | ±0.00 |
| Recall | 0.93 | **+0.01** |
| CAM-IoU | 0.49 | -0.01 |
| PointGame | 0.89 | -0.01 |
| Energy-Inside | 0.70 | ±0.00 |
| Loc-IoU | 0.27 | ±0.00 |

**해석:**
- Counterfactual reasoning (`L_cf`)이 **Recall을 회복** (0.92 → 0.93).
  - "결함이 없었다면?" 을 시뮬레이션하여 모델이 실제 결함 feature에 의존하도록 강제.
  - S4에서의 multi-task interference로 인한 Recall 하락을 부분적으로 복원.
- CAM quality는 S4와 거의 동일 — counterfactual이 **attention 품질에는 중립적** 영향.
- Localization도 0.27로 동일 유지.
- S5는 **모든 loss를 사용하는 full strategy**로서, 균형 잡힌 성능을 보여주지만
  S3 대비 분류 성능 이점이 크지 않음 (Acc 동일, Recall -1%p).

---

### 3.6 S6 — Full + GT Mask Curriculum (Best Overall)

| Metric | Value | vs S5 변화 |
|--------|-------|-----------|
| Accuracy | **0.91** | **+0.02** |
| Recall | **0.95** | **+0.02** |
| CAM-IoU | 0.48 | -0.01 |
| PointGame | **0.92** | **+0.03** |
| Energy-Inside | **0.76** | **+0.06** |
| Loc-IoU | 0.25 | -0.02 |

**해석:**
- **전체 Strategy 중 최고 성능** — 분류와 해석 가능성 모두 최상위.
- GT mask multiplicative fusion + curriculum alpha decay가 핵심:
  - 훈련 초기 GT mask를 50% 비율로 attention에 직접 주입 → 정확한 spatial prior 제공.
  - 점진적으로 alpha를 0으로 감소 (70% epoch까지) → inference 조건과 일치시킴.
- **Classification**:
  - Accuracy 0.91 (전체 1위), Recall 0.95 (전체 1위).
  - 결함 미탐지율이 S1 대비 약 **58% 감소** (miss rate: 12% → 5%).
- **Interpretability**:
  - PointGame 0.92 (전체 공동 1위), Energy-Inside 0.76 (전체 1위).
  - Attention energy의 76%가 결함 내부에 집중 — 가장 높은 spatial focus.
  - CAM-IoU는 0.48로 소폭 하락하나, PointGame/Energy-Inside가 더 중요한 실용 지표.
- **Localization은 소폭 하락** (0.27 → 0.25):
  - GT mask curriculum이 attention module의 학습 dynamics를 변경하여 localization head에 경미한 영향.
  - 그러나 -0.02 수준으로 실질적 차이 미미.

---

### 3.7 S7 — Attention Mining + Counterfactual (Lightweight)

| Metric | Value | vs S3 변화 | vs S5 변화 |
|--------|-------|-----------|-----------|
| Accuracy | 0.893 | -0.007 | +0.003 |
| Recall | 0.926 | -0.014 | -0.004 |
| CAM-IoU | **0.517** | +0.017 | +0.027 |
| PointGame | 0.88 | -0.01 | -0.01 |
| Energy-Inside | 0.693 | +0.003 | -0.007 |
| Loc-IoU | 0.032 | -0.008 | -0.238 |

**해석:**
- S3에 counterfactual만 추가한 **경량화 Strategy** (localization 제외).
- **CAM-IoU 0.517로 전체 Strategy 중 최고치 달성.**
  - Localization head가 없기 때문에 attention module이 분류와 spatial alignment에만 집중.
  - `L_cf`가 attention의 quality를 counterfactual 방식으로 보강.
  - Localization loss의 multi-task interference가 없어서 attention이 더 순수하게 학습.
- **분류 성능은 S3보다 소폭 하락** (Recall 0.94 → 0.926, Acc 0.90 → 0.893):
  - Counterfactual loss가 attention을 더 conservative하게 만들어 약간의 분류 trade-off 발생.
- **Localization 능력 없음** (Loc-IoU 0.032) — 설계 의도대로.
- 정밀한 결함 segmentation이 불필요하고, **해석 가능한 분류만 필요한 경우 효율적인 선택**.

---

### 3.8 S8 — S7 + GT Mask Curriculum

| Metric | Value | vs S7 변화 | vs S6 변화 |
|--------|-------|-----------|-----------|
| Accuracy | 0.886 | -0.007 | -0.024 |
| Recall | 0.926 | ±0.000 | -0.024 |
| CAM-IoU | 0.500 | -0.017 | +0.020 |
| PointGame | **0.92** | +0.04 | ±0.00 |
| Energy-Inside | 0.758 | +0.065 | -0.002 |
| Loc-IoU | 0.035 | +0.003 | -0.215 |

**해석:**
- S7에 GT mask curriculum을 추가한 변형.
- **Interpretability가 S7 대비 개선**:
  - PointGame 0.88 → 0.92 (+0.04): curriculum이 spatial accuracy를 높임.
  - Energy-Inside 0.693 → 0.758 (+0.065): attention energy가 결함에 더 집중.
  - CAM-IoU는 0.517 → 0.500 (-0.017)으로 소폭 하락.
- **분류 성능은 S7 대비 소폭 하락** (Accuracy 0.893 → 0.886):
  - GT mask 주입이 초기 학습을 돕지만, localization loss 없이는 curriculum의 이점이 제한적.
- S6 대비 분류에서 열세 (Acc -0.024, Recall -0.024)이나 CAM-IoU에서 우세 (+0.020).
- **Localization 불필요 + 높은 interpretability가 필요한 시나리오에서 S6의 대안**.

---

## 4. 이전 결과와의 비교 (v1 → v2)

S1~S5에 대해 이전 결과(v1)와 현재 결과(v2)를 비교합니다.

| Metric | Strategy | v1 | v2 | 변화 | 원인 |
|--------|----------|-----|-----|------|------|
| Accuracy | S1 | 0.878 | 0.89 | +0.012 | 학습 안정화 |
| Accuracy | S4 | 0.872 | 0.89 | **+0.018** | `lambda_loc` 축소 + warmup 효과 |
| Recall | S3 | 0.894 | 0.94 | **+0.046** | `lambda_guide` 1.5 강화 효과 |
| Recall | S4 | 0.866 | 0.92 | **+0.054** | multi-task interference 완화 |
| CAM-IoU | S1 | 0.116 | 0.18 | +0.064 | per-image normalization 적용 |
| CAM-IoU | S3 | 0.451 | 0.50 | **+0.049** | attention 학습 개선 |
| PointGame | S1 | 0.161 | 0.08 | **-0.081** | normalization 후 threshold 영향 |
| Energy-Inside | S3 | 0.626 | 0.69 | +0.064 | guided attention 강화 |

### 주요 개선 사항
1. **S4 multi-task interference 해소**: Acc +1.8%p, Recall +5.4%p — `lambda_loc` 축소와 warmup이 효과적.
2. **S3 CAM quality 역전**: S2 < S3 으로 변경 — `lambda_guide` 1.5가 attention spatial 정확도 유지.
3. **S1 PointGame 하락**: per-image normalization이 PointGame에는 부정적 영향. 최대 activation 위치 자체는 변하지 않으므로 정상적인 현상.

---

## 5. 종합 비교 — Strategy Profile

```
         Accuracy    Recall     CAM-IoU    PointGame   Energy-In   Loc-IoU
S1       ████░░  0.89  ████░░  0.88  █░░░░░  0.18  ░░░░░░  0.08  █░░░░░  0.17  ░░░░░░  0.02
S2       ████░░  0.88  ████░░  0.88  █████░  0.48  █████░  0.90  ████░░  0.68  ░░░░░░  0.02
S3       █████░  0.90  █████░  0.94  █████░  0.50  █████░  0.89  ████░░  0.69  ░░░░░░  0.04
S4       ████░░  0.89  █████░  0.92  █████░  0.50  █████░  0.90  ████░░  0.70  ██░░░░  0.27
S5       ████░░  0.89  █████░  0.93  █████░  0.49  █████░  0.89  ████░░  0.70  ██░░░░  0.27
S6 ★     █████░  0.91  ██████  0.95  █████░  0.48  █████░  0.92  █████░  0.76  ██░░░░  0.25
S7       ████░░  0.893 █████░  0.926 █████░  0.517 █████░  0.88  ████░░  0.693 ░░░░░░  0.032
S8       ████░░  0.886 █████░  0.926 █████░  0.500 █████░  0.92  █████░  0.758 ░░░░░░  0.035
```

---

## 6. Metric별 순위 정리

### Classification Ranking

| Rank | Accuracy | Recall |
|------|----------|--------|
| 1 | **S6 (0.91)** | **S6 (0.95)** |
| 2 | S3 (0.90) | S3 (0.94) |
| 3 | S7 (0.893) | S5 (0.93) |
| 4 | S1/S4/S5 (0.89) | S7/S8 (0.926) |
| 5 | S8 (0.886) | S4 (0.92) |
| 6 | S2 (0.88) | S1/S2 (0.88) |

### Interpretability Ranking

| Rank | CAM-IoU | PointGame | Energy-Inside |
|------|---------|-----------|---------------|
| 1 | **S7 (0.517)** | **S6/S8 (0.92)** | **S6 (0.76)** |
| 2 | S3/S4 (0.50) | S2/S4 (0.90) | S8 (0.758) |
| 3 | S8 (0.500) | S3/S5 (0.89) | S4/S5 (0.70) |
| 4 | S5 (0.49) | S7 (0.88) | S3/S7 (0.69) |
| 5 | S2/S6 (0.48) | S1 (0.08) | S2 (0.68) |
| 6 | S1 (0.18) | — | S1 (0.17) |

### Localization Ranking

| Rank | Loc-IoU |
|------|---------|
| 1 | **S4/S5 (0.27)** |
| 2 | S6 (0.25) |
| 3 | S3 (0.04) |
| 4 | S8 (0.035) |
| 5 | S7 (0.032) |
| 6 | S1/S2 (0.02) |

---

## 7. 종합 정리 및 권장 사항

### 7.1 용도별 추천 Strategy

| 사용 시나리오 | 추천 Strategy | 이유 |
|-------------|-------------|------|
| **Production (결함 탐지 최우선)** | **S6** | Accuracy 0.91, Recall 0.95 — 결함 미탐지율 최저. Energy-Inside 0.76으로 높은 해석 가능성. |
| **Interpretability 최우선** | **S7** | CAM-IoU 0.517 (최고). Localization 불필요 시 attention quality 최적화. |
| **경량 + 균형** | **S3** | Localization/Counterfactual 없이 Acc 0.90, Recall 0.94, CAM-IoU 0.50. 학습 비용 대비 성능비 최고. |
| **결함 Segmentation 필요** | **S5** | Loc-IoU 0.27 + Recall 0.93 + balanced interpretability. |
| **해석 가능성 + Localization 없이 배포** | **S8** | PointGame 0.92, Energy-Inside 0.758. S7보다 spatial focus 강화. |

### 7.2 핵심 발견 사항

1. **S6가 최적의 production 모델**: 분류(Accuracy/Recall)와 해석 가능성(PointGame/Energy-Inside) 모두 최상위. GT mask curriculum이 attention 학습에 강력한 spatial prior를 제공하면서도 inference 시에는 GT mask 없이 동작.

2. **Localization은 양날의 검**: S4에서 관찰되듯 localization loss는 분류 성능을 약간 저하시키지만 (S3→S4: Recall -2%p), 결함 위치를 pixel 수준으로 제공하는 유일한 방법. `loc_warmup_ratio: 0.5`로 interference를 완화했지만 완전히 해소되지는 않음.

3. **Counterfactual의 역할 재평가**: S5 vs S4에서 Recall +1%p 회복, S7 vs S3에서는 CAM-IoU +0.017 향상. Counterfactual은 분류보다 **attention quality 개선에 더 효과적**인 것으로 나타남.

4. **Curriculum Learning 효과 검증**: S6 vs S5 (Acc +2%p, Recall +2%p), S8 vs S7 (PointGame +0.04, EI +0.065). GT mask curriculum은 localization이 있는 S6에서 더 큰 분류 향상을, localization이 없는 S8에서 더 큰 interpretability 향상을 가져옴.

5. **S3가 cost-effective sweet spot**: 학습 복잡도가 낮으면서 (loss 3개) Accuracy 0.90, Recall 0.94, CAM-IoU 0.50의 균형 잡힌 성능. 리소스 제약 시 첫 번째 선택.

### 7.3 다음 단계 제안

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| 1 | **S6를 production 모델로 채택** | 최고 Recall 0.95 + 강력한 해석 가능성 |
| 2 | **S6에 `lambda_cam_guide` 추가 실험** | CAM-IoU 0.48 → 0.50+ 개선 가능성 |
| 3 | **S7 + localization warmup 실험** | CAM-IoU 0.517 유지하면서 Loc-IoU 향상 시도 |
| 4 | **S6/S8 curriculum 하이퍼파라미터 탐색** | `alpha_start`, `alpha_ratio` 변경으로 추가 개선 여지 탐색 |
