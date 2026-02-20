# 관련 연구

## 1. 논문 요약

본 연구는 다음 핵심 논문들의 아이디어를 통합하여 설계되었다.

---

### 1.1 GAIN: "Tell Me Where to Look" (CVPR 2018)

**핵심 아이디어**: External guidance(외부 감독 신호)를 사용하여 CNN의 attention을 올바른 영역으로 유도

```
구조:
  S_cl (Classification Stream)  ──→  분류 + Weight-based CAM 생성
  S_am (Attention Mining Stream) ──→  판별적 영역 탐색
  External Supervision (GT mask) ──→  CAM이 올바른 영역을 가리키도록 loss 적용
```

**주요 기여**:
- 분류 모델의 CAM을 GT 위치 정보로 직접 supervision하는 프레임워크 제안
- Attention Mining을 통해 모델이 스스로 판별적 영역을 발견하도록 유도
- 분류 정확도를 유지하면서 CAM 품질을 대폭 향상

**본 연구와의 관계**: GAIN의 2-stream 구조(S_cl + S_am)와 외부 감독 방식을 핵심 프레임워크로 채택. 여기에 CBAM, FPN, Counterfactual을 추가하여 확장.

---

### 1.2 CBAM: Convolutional Block Attention Module (ECCV 2018)

**핵심 아이디어**: Channel Attention과 Spatial Attention을 순차 결합하여 특성 맵을 정제

```
Channel Attention: "어떤 채널이 중요한가?"
  AvgPool + MaxPool → Shared MLP → Sigmoid → Channel Weight

Spatial Attention: "어디가 중요한가?"
  Channel-wise AvgPool + MaxPool → Conv(7×7) → Sigmoid → Spatial Weight

결합: F → CA(F) → SA(CA(F))
```

**주요 기여**:
- 경량화된 attention 모듈로 기존 CNN에 플러그인 방식으로 추가 가능
- Channel과 Spatial 정보를 모두 활용하여 포괄적인 attention 생성
- ImageNet, COCO 등 다양한 벤치마크에서 일관된 성능 향상

**본 연구와의 관계**: GAIN Attention Module의 전처리 단계로 CBAM을 적용. Backbone의 특성을 CBAM으로 정제한 후 attention 생성 네트워크에 입력하여 더 정확한 spatial attention 학습.

---

### 1.3 EfficientNetV2: Smaller Models and Faster Training (ICML 2021)

**핵심 아이디어**: 학습 속도와 파라미터 효율성을 동시에 최적화한 CNN 아키텍처

```
핵심 기술:
  1. Fused-MBConv: 초기 레이어에 일반 Conv 사용 → 학습 속도 3~4배 향상
  2. Progressive Learning: 이미지 크기, 정규화를 점진적으로 증가
  3. NAS(Neural Architecture Search)로 최적 구조 탐색
```

**모델 변종**:

| 모델 | 파라미터 | ImageNet Top-1 |
|------|---------|---------------|
| EfficientNetV2-S | 21.5M | 83.9% |
| EfficientNetV2-M | 54.1M | 85.2% |
| EfficientNetV2-L | 118.5M | 85.7% |

**본 연구와의 관계**: Backbone으로 EfficientNetV2를 채택. mmpretrain(OpenMMLab)의 사전학습 가중치를 활용하여 ImageNet에서 학습된 풍부한 시각적 특성을 전이. Conv_head의 1280채널 출력을 분류와 CAM 생성에 직접 활용.

---

### 1.4 Focal Loss for Dense Object Detection (ICCV 2017)

**핵심 아이디어**: 클래스 불균형 상황에서 쉬운 샘플의 loss 기여를 줄여 어려운 샘플에 집중

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

γ = 0: 일반 Cross-Entropy와 동일
γ = 2: 쉬운 샘플(p_t > 0.9)의 loss를 ~100배 줄임
```

**본 연구와의 관계**: 제조 데이터의 양품/불량 비율 불균형 처리에 Focal Loss 적용. Classification Loss(L_cls)와 Attention Mining Loss(L_am) 모두에 사용.

---

### 1.5 Multi-Task Learning for Manufacturing Defect Detection (PMC 2019)

**핵심 아이디어**: 결함 분류와 결함 위치 추정을 동시에 학습하여 상호 보완적 표현 획득

```
공유 Backbone → Classification Head (분류)
             → Segmentation Head (위치 추정)

Joint Training: L_total = L_cls + λ × L_seg
```

**주요 기여**:
- 분류와 segmentation의 공동 학습이 각각의 단일 학습보다 우수한 성능
- 위치 정보가 분류 backbone에 공간적 인지 능력을 부여
- 제조 환경에서의 MTL 적용 가능성을 실증

**본 연구와의 관계**: Classification + Localization의 MTL 구조를 채택. FPN 기반 Localization Head로 결함 위치를 예측하며, Consistency Loss로 attention과 localization의 정합성을 유지.

---

## 2. 기존 연구의 한계와 본 연구의 차별점

### 2.1 기존 연구의 한계

| 연구 | 한계 |
|------|------|
| **GAIN** | 분류와 attention만 다룸, 결함 위치의 정밀한 segmentation 미지원 |
| **CBAM** | Attention을 외부 신호로 supervision하는 기능 없음 (자기 학습만) |
| **EfficientNetV2** | 분류 전용 backbone, 해석 가능성 메커니즘 미포함 |
| **MTL Defect Detection** | Attention guidance 미적용, 모델이 올바른 근거로 판단하는지 검증 없음 |
| **Focal Loss** | Loss 함수 관점의 기여, 모델 해석 가능성과 직접적 관련 없음 |

### 2.2 본 연구의 차별점

```
GAIN (Attention Supervision)
  +
CBAM (Channel + Spatial Attention)
  +
EfficientNetV2 (효율적 Backbone)
  +
Multi-Task Learning (분류 + 위치 추정)
  +
Counterfactual Reasoning (추론 검증)       ← 기존 연구에서 다루지 않은 결합
  +
Progressive Multi-Stage Training (안정적 학습)
  ═══════════════════════════════════════
  GAIN-MTL: 통합 프레임워크
```

| 차별점 | 설명 |
|--------|------|
| **통합 프레임워크** | Attention guidance + MTL + Counterfactual을 하나의 모델에 통합 |
| **Counterfactual 검증** | "결함 제거 시 양품 판정" 메커니즘으로 추론의 올바름을 강제 |
| **단계적 학습 전략** | S1→S5까지 점진적으로 loss를 추가하여 multi-task interference 최소화 |
| **정량적 비교** | 각 구성 요소의 기여도를 6단계에 걸쳐 ablation study 형태로 분석 |
| **Weight-based CAM의 직접 학습** | Grad-CAM과 달리 미분 가능한 CAM을 loss에 직접 활용 |

---

## 3. 관련 기술 비교

### 3.1 CAM 기법 비교

| 기법 | 미분 가능 | 학습 중 사용 | 추가 연산 | 본 연구 적용 |
|------|----------|------------|----------|------------|
| CAM (원본) | O | O | 없음 | **O (Weight-based CAM)** |
| Grad-CAM | X | X | Gradient 계산 | X |
| Grad-CAM++ | X | X | Gradient 계산 | X |
| Score-CAM | X | X | 다수 forward pass | X |

본 연구에서는 **미분 가능한 Weight-based CAM**을 사용하여 학습 중 직접 supervision 가능.

### 3.2 Attention Mechanism 비교

| 기법 | 외부 감독 | Channel + Spatial | 학습 가능 온도 |
|------|----------|-------------------|--------------|
| SE-Net | X | Channel만 | X |
| CBAM | X | O | X |
| GAIN | O | X | X |
| **본 연구** | **O** | **O (CBAM 통합)** | **O** |

### 3.3 결함 검출 접근 비교

| 접근 | 분류 | 위치 추정 | 해석 가능성 | 추론 검증 |
|------|------|----------|-----------|----------|
| 분류만 (ResNet 등) | O | X | X | X |
| Detection (YOLO 등) | O | O (bbox) | X | X |
| Segmentation (U-Net 등) | X | O (mask) | X | X |
| GAIN | O | X | O | X |
| MTL Defect Detection | O | O | X | X |
| **GAIN-MTL (본 연구)** | **O** | **O (mask)** | **O** | **O** |
