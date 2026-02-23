# 연구 방법

## 1. 전체 아키텍처

### 1.1 시스템 개요

GAIN-MTL은 하나의 backbone에서 추출된 특성을 **4가지 경로**로 활용하는 multi-task 구조이다.

```
Input Image (B, 3, 512, 512)
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                  EfficientNetV2 Backbone                      │
│                     (mmpretrain)                              │
│  Stem → Block1 → Block2 → ... → Block6 → Conv_head(1280ch)  │
└──────────────────────────────────────────────────────────────┘
         │                                          │
    Block 출력 (멀티스케일)                    Conv_head 출력 (1280ch)
         │                                          │
         ▼                              ┌───────────┼───────────┐
┌─────────────────┐                     │           │           │
│       FPN       │                     ▼           ▼           ▼
│ (256-ch fusion) │              경로 1        경로 2        경로 3
└────────┬────────┘           Classification   GAIN        Counterfactual
         │                       Head        Attention       Module
         ▼                        │          Module            │
      경로 4                       │           │               │
   Localization                    ▼           ▼               ▼
      Head                   cls_logits  attended_cls     cf_logits
         │                   + CAM       + attention_map
         ▼
  localization_map
```

### 1.2 데이터 흐름 상세

| 경로 | 입력 | 모듈 | 출력 | 용도 |
|------|------|------|------|------|
| **경로 1** | Conv_head (1280ch) | Classification Head | cls_logits, Weight-based CAM | Baseline 분류, CAM 생성 |
| **경로 2** | Conv_head (1280ch) | CBAM → Attention Conv → Feature Adapter → Attended Cls Head | attended_cls_logits, attention_map | **최종 분류 출력**, attention 시각화 |
| **경로 3** | Conv_head + defect_mask | Counterfactual Module | cf_logits | 추론 검증 (학습 시만) |
| **경로 4** | FPN 융합 특성 | Localization Head | localization_map | 결함 위치 segmentation |

---

## 2. 모듈별 상세 설계

### 2.1 Backbone: EfficientNetV2

```
EfficientNetV2-S (기본 설정):
  out_indices = (3, 4, 5, 6, 7)
                 ↑  ↑  ↑  ↑  ↑
              Block stages   Conv_head
              (FPN 입력)     (1280ch)

  Block 3~6 출력 → FPN → Localization Head
  Conv_head 출력 → Classification / GAIN Attention / Counterfactual
```

- mmpretrain의 ImageNet 사전학습 가중치 활용
- Conv_head의 1280채널은 의미적으로 풍부한 고수준 특성
- Backbone stage별 freeze 옵션으로 미세 조정 수준 제어 가능

### 2.2 GAIN Attention Module

```
입력 (1280ch)
    │
    ▼
┌─────────┐
│  CBAM   │  → Channel Attention → Spatial Attention
└────┬────┘
     │ (정제된 특성)
     ▼
┌──────────────────────────────┐
│   Attention Conv Network     │
│   Conv(3×3) → BN → ReLU     │
│   Conv(3×3) → BN → ReLU     │
│   Conv(1×1)                  │
│   ÷ Temperature (학습 가능)  │
└──────────────┬───────────────┘
               │ (attention logits)
               ▼
          σ(logits) → attention_map (0~1)
               │
               ▼
    features × attention_map → attended_features
```

**Learnable Temperature**: attention의 sharpness를 학습 가능한 파라미터로 제어. 학습 초기에는 넓게, 후기에는 결함 영역에 집중되도록 자동 조절.

### 2.3 Attention Mining Head (S_am)

GAIN 논문의 S_am stream을 구현한 보조 attention 생성기이다.

```
입력 (1280ch) → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1)
                                                                      │
                                                              mined_attention_logits
```

**최종 attention**: GAIN Attention과 Mining의 평균으로 결합

```
combined_attention = (attention_logits + mined_attention_logits) / 2
```

### 2.4 Classification Head

```
입력 (1280ch)
    │
    ▼
AdaptiveAvgPool2d(1)  →  (1280, 1, 1)  →  Flatten  →  (1280,)
    │
    ▼
Linear(1280 → 512) → ReLU → Dropout(0.5) → Linear(512 → 2)
    │
    ▼
cls_logits (2,)

Weight-based CAM 생성:
  W = classifier.weight[class=1]     # (1280,)
  F = conv_head_features             # (1280, H, W)
  CAM(x,y) = Σ_c W_c × F_c(x,y)    # (1, H, W)
```

두 개의 Classification Head가 존재:
1. **Classification Head**: backbone 특성으로 직접 분류 (S1~S2에서 사용)
2. **Attended Classification Head**: attention-weighted 특성으로 분류 (S3~S6에서 사용)

### 2.5 Feature Adapter

Attended features를 classification head에 입력하기 전 변환하는 어댑터이다.

```
attended_features → Conv(1×1) → BN → ReLU → attended_adapted
```

- Attention module과 classification head 사이의 분포 차이를 완화
- 1×1 Conv로 채널 간 관계를 재조정

### 2.6 Localization Head

FPN 융합 특성에서 결함 위치를 pixel-level로 예측하는 FCN 구조이다.

```
FPN 융합 특성 (256ch)
    │
    ▼
Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1)
    │
    ▼
Upsample(×4) → localization_logits (1, H, W)
```

### 2.7 Counterfactual Module

```
입력: features (1280ch) + defect_mask (1ch) + attention_map

Step 1: Mask Processing
  defect_mask → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → σ
                                                                        │
                                                              suppression_mask (1280ch)

Step 2: Feature Suppression (learned 모드)
  replacement = feature_replacer(features)
  cf_features = features × (1 - suppression_mask) + replacement × suppression_mask

Step 3: Feature Refinement
  cf_features = feature_refiner(cf_features) + cf_features  ← Residual Connection

Step 4: Re-classification
  cf_features → AdaptiveAvgPool → Linear → Linear → cf_logits

  기대: cf_logits → "양품" (class 0)
```

---

## 3. Loss Function 체계

### 3.1 전체 Loss 구성

```
L_total = λ_cls × L_cls              # Classification (Focal Loss)
        + λ_am × L_am               # Attention Mining (Focal Loss)
        + λ_cam_guide × L_cam       # CAM Guidance (BCE + Dice)
        + λ_loc × L_loc             # Localization (Dice + BCE)
        + λ_guide × L_guide         # Guided Attention (BCE + Dice + IoU + Entropy)
        + λ_cf × L_cf               # Counterfactual (Cross-Entropy)
        + λ_consist × L_consist     # Consistency (MSE)
```

### 3.2 Loss별 상세

| Loss | 수식 | 적용 대상 | 역할 |
|------|------|----------|------|
| **L_cls** | Focal Loss(cls_logits, labels) | 전체 | 기본 분류 학습 |
| **L_am** | Focal Loss(attended_cls_logits, labels) | 전체 | Attention 적용 분류 학습 |
| **L_cam_guide** | BCE + Dice(CAM, GT_mask) | 불량만 | Weight-based CAM이 결함 위치와 일치 |
| **L_loc** | Dice + BCE(loc_map, GT_mask) | 불량만 | 결함 영역 segmentation |
| **L_guide** | BCE + Dice + IoU(attn_map, GT_mask) + Entropy(normal) | 불량: 마스크 정렬, 양품: 엔트로피 최대화 | Attention이 결함 위치와 일치 |
| **L_cf** | CE(cf_logits, class=0) | 불량만 | 결함 제거 시 양품 판정 강제 |
| **L_consist** | MSE(attn_map, loc_map) | 불량만 | Attention과 Localization 정합성 |

### 3.3 Loss Weight 설정

| 파라미터 | S1 | S2 | S3 | S4 | S5 |
|---------|----|----|----|----|-----|
| λ_cls | 1.0 | 1.0 | **0.3** | **0.3** | **0.3** |
| λ_am | 0 | 0 | **1.0** | **1.0** | **1.0** |
| λ_cam_guide | 0 | 0.3 | **1.0** | **1.0** | **1.0** |
| λ_loc | 0 | 0 | 0 | **0.2** | **0.2** |
| λ_guide | 0 | 0 | 0.5 | 0.5 | 0.5 |
| λ_cf | 0 | 0 | 0 | 0 | 0.3 |
| λ_consist | 0 | 0 | 0 | 0.2 | 0.2 |

---

## 4. Multi-Stage Training 전략

### 4.1 단계별 학습 스케줄

전체 에폭을 4단계로 나누어 점진적으로 loss를 추가한다.

```
에폭:  0%          25%         50%         75%        100%
       ├───────────┼───────────┼───────────┼──────────┤
       │  Stage 1  │  Stage 2  │  Stage 3  │  Stage 4 │
       │           │           │           │          │
       │  L_cls    │  + L_am   │  + L_loc  │  + L_cf  │
       │           │  + L_guide│  + L_consist│          │
       │           │  + L_cam  │  (warmup) │          │
       │           │           │           │          │
       │ Feature   │ Attention │ 공간 정보  │ 추론     │
       │ warm-up   │ 학습      │ 통합      │ 강건화   │
```

### 4.2 Localization Warmup

S4에서 localization loss가 갑자기 도입되면 multi-task interference가 발생한다.

```
해결: Gradual Warmup (loc_warmup_ratio = 0.5)

에폭:  Stage 3 시작                    Stage 3 끝
       ├────────────────────────────────┤
       │   λ_loc 선형 증가              │
       │   0.0 ────────→ 0.2 (목표)    │
       │   [---warmup 구간---][full]    │
       │   0%     25%     50%  100%    │
```

- Stage 3의 처음 50% 동안 λ_loc를 0에서 목표값까지 선형 증가
- 분류 학습이 안정화된 후 localization이 점진적으로 개입

### 4.3 전략별 비교 실험 설계

각 전략은 이전 전략에 하나의 요소를 추가하여, **각 구성 요소의 기여도**를 분리 분석한다.

```
S1: Classification Only          → Baseline
S2: S1 + CAM Guidance            → CAM supervision의 효과
S3: S2 + Attention Mining        → 학습 가능한 attention의 효과
S4: S3 + Localization            → MTL의 공간 정보 기여
S5: S4 + Counterfactual          → 추론 검증의 기여
S6: S5 + GT Mask Attention       → 직접적 mask 주입의 효과
```

---

## 5. Strategy 6: GT Mask Attention (확장)

### 5.1 핵심 아이디어

GT defect mask를 attention으로 직접 활용하되, **학습과 추론의 분포 차이(distribution mismatch)를 최소화**하는 설계이다.

### 5.2 학습 시: Multiplicative Fusion + Curriculum Blending

```
배치 내 샘플별 자동 분기:

  mask 있는 defect:
    internal_attn = σ(attention_logits)
    fused = GT_mask × internal_attn
    fused_norm = fused / max(fused)                ← per-sample normalization

    Curriculum blending (초기 학습):
      effective = α × GT_mask + (1-α) × fused_norm
      α: 0.5 → 0.0 (학습 진행에 따라 감소)

  mask 없는 샘플 / normal:
    effective = σ(attention_logits)                 ← 내부 attention 사용

  effective_attn × features → Feature Adapter → Attended Cls Head
```

### 5.3 추론 시: Pure Multiplicative Fusion

```
모든 샘플:
  fused = external_mask × σ(internal_logits)
  fused_norm = fused / max(fused)                  ← per-sample normalization

  Normal 안전성: internal이 낮으면 fused도 낮음 → false positive 억제
  Defect 정확성: 둘 다 동의하는 영역만 활성화
```

---

## 6. 평가 방법

### 6.1 분류 성능 평가

| 지표 | 수식 | 중점 |
|------|------|------|
| Accuracy | (TP+TN) / N | 전체 정확도 |
| Precision | TP / (TP+FP) | 오검출 비율 |
| **Recall** | TP / (TP+FN) | **미검출 비율 (제조에서 가장 중요)** |
| F1-Score | 2PR / (P+R) | Precision-Recall 균형 |

### 6.2 CAM 품질 평가 (해석 가능성)

| 지표 | 수식 | 의미 |
|------|------|------|
| **CAM-IoU** | \|CAM ∩ GT\| / \|CAM ∪ GT\| | Attention과 결함 영역의 겹침 |
| **PointGame** | 1(argmax(CAM) ∈ GT) | 최대 활성화가 결함 내부인지 |
| **Energy-Inside** | Σ(CAM·GT) / Σ(CAM) | 결함 내부의 attention 에너지 비율 |

**Per-image min-max normalization**: 전략 간 공정 비교를 위해 이미지별로 CAM을 [0, 1]로 정규화한 후 이진화하여 평가.

### 6.3 위치 추정 평가

| 지표 | 설명 |
|------|------|
| IoU | 예측 mask와 GT mask의 겹침 비율 |
| Dice | 영역 겹침 계수 (2×IoU / (1+IoU)) |
| Pixel Accuracy | 픽셀별 정확도 |

---

## 7. 실험 환경

### 7.1 학습 설정

| 항목 | 설정 |
|------|------|
| Backbone | EfficientNetV2-S (mmpretrain, ImageNet pretrained) |
| 입력 크기 | 512 × 512 |
| Batch Size | 16 |
| 에폭 수 | 100 |
| Optimizer | Adam (lr=1e-4) |
| LR Scheduler | Cosine Annealing |
| Mixed Precision | AMP (Automatic Mixed Precision) |

### 7.2 데이터 증강

| 증강 기법 | 확률 |
|----------|------|
| Horizontal Flip | 0.5 |
| Vertical Flip | 0.5 |
| Random Rotation (90°) | 0.5 |
| Color Jitter | 0.3 |
| Gaussian Noise | 0.2 |

### 7.3 데이터 구성

```
data/
├── images/
│   ├── train/
│   │   ├── normal/       # 양품 이미지
│   │   └── defective/    # 불량 이미지
│   └── test/
│       ├── normal/
│       └── defective/
└── masks/
    └── defective/        # 결함 위치 binary mask (0: 배경, 255: 결함)
```

- Train/Val 자동 분할 (val_ratio=0.2 권장)
- 이미지-마스크 쌍 로딩 (동일 파일명)
