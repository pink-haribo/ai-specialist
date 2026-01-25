# GAIN-MTL: Guided Attention Inference Multi-Task Learning

> EfficientNetV2 기반 제조 결함 검출을 위한 해석 가능한 분류 모델

## 개요

GAIN-MTL은 제조 환경에서 양품/불량품 판정을 위한 딥러닝 프레임워크입니다. 단순히 분류 정확도만 높이는 것이 아니라, **모델이 올바른 근거로 판단하도록** 학습시키는 것이 핵심입니다.

### 핵심 특징

- **EfficientNetV2 Backbone**: 효율적인 특성 추출
- **GAIN (Guided Attention)**: Detection label로 attention을 직접 supervision
- **Multi-Task Learning**: Classification + Localization 동시 학습
- **Counterfactual Learning**: "결함이 없었다면?" 시뮬레이션으로 올바른 추론 강제
- **Multi-Stage Training**: 단계별 학습으로 안정적 수렴

## 설치

```bash
git clone https://github.com/yourusername/ai-specialist.git
cd ai-specialist
pip install -r requirements.txt
```

## 데이터 준비

### 데이터 구조

```
data/
├── images/
│   ├── train/
│   │   ├── normal/
│   │   │   ├── img_001.png
│   │   │   └── ...
│   │   └── defective/
│   │       ├── img_001.png
│   │       └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── masks/
    └── defective/
        ├── img_001.png  (결함 마스크)
        └── ...
```

### MVTec AD 데이터셋 (공개 벤치마크)

```bash
# MVTec AD 다운로드
wget https://www.mvtec.com/company/research/datasets/mvtec-ad
```

## 학습

### 기본 학습

```bash
python train.py --config configs/default.yaml --data_root ./data
```

### 옵션 설정

```bash
# 다른 백본 사용
python train.py --backbone efficientnetv2_m

# 배치 사이즈 및 에폭 조정
python train.py --batch_size 32 --epochs 200

# GPU 지정
python train.py --device cuda:0

# W&B 로깅 활성화
python train.py --wandb
```

### Multi-Stage Training

학습은 4단계로 진행됩니다:

| Stage | 에폭 비율 | 학습 내용 |
|-------|----------|----------|
| 1 | 25% | Classification만 학습 (warm-up) |
| 2 | 25% | + Attention Mining 추가 |
| 3 | 25% | + Localization 추가 |
| 4 | 25% | + Counterfactual (전체 학습) |

## 평가

```bash
# 테스트 셋 평가
python evaluate.py --checkpoint checkpoints/best_model.pth --data_root ./data

# 시각화 생성
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize --vis_dir ./vis

# 에러 분석
python evaluate.py --checkpoint checkpoints/best_model.pth --error_analysis

# 결과 내보내기
python evaluate.py --checkpoint checkpoints/best_model.pth --export_results results.json
```

## 모델 아키텍처

```
Input Image
     │
     ▼
┌─────────────────┐
│ EfficientNetV2  │ ──► Multi-scale features
│    Backbone     │
└─────────────────┘
          │
          ▼
┌─────────────────┐
│      FPN        │ ──► Unified multi-scale features
└─────────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐  ┌─────────────┐
│ GAIN  │  │ Localization│
│ Attn  │  │    Head     │
└───────┘  └─────────────┘
    │           │
    ▼           ▼
┌───────┐  ┌───────┐
│ Class │  │ Defect│
│ Head  │  │  Map  │
└───────┘  └───────┘
```

## Loss Functions

```python
Total Loss = λ_cls × L_cls           # Classification (Focal Loss)
           + λ_am × L_am             # Attention Mining
           + λ_loc × L_loc           # Localization (Dice + BCE)
           + λ_guide × L_guide       # Guided Attention
           + λ_cf × L_cf             # Counterfactual
           + λ_consist × L_consist   # Consistency
```

### 기본 가중치

| Loss | Weight | 설명 |
|------|--------|------|
| λ_cls | 1.0 | 분류 손실 |
| λ_am | 0.5 | Attention mining |
| λ_loc | 0.3 | 위치 예측 |
| λ_guide | 0.5 | Attention guidance (핵심!) |
| λ_cf | 0.3 | Counterfactual |
| λ_consist | 0.2 | Attention-Localization 일관성 |

## 평가 지표

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score, AUC-ROC

### CAM Quality Metrics (핵심!)
- **CAM-IoU**: Attention과 실제 결함 영역의 IoU
- **PointGame**: 최대 activation이 결함 영역 내부에 있는 비율
- **Energy-Inside**: 결함 영역 내부의 attention 에너지 비율

### Localization Metrics
- IoU, Dice, Pixel Accuracy

## 사용 예시

### 추론

```python
from src.models import GAINMTLModel
from src.evaluation import DefectExplainer
import torch

# 모델 로드
model = GAINMTLModel(backbone_name='efficientnetv2_s')
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

# 예측
image = ...  # (3, 512, 512) tensor
outputs = model(image.unsqueeze(0))

pred = outputs['cls_logits'].argmax(dim=1)  # 0: 양품, 1: 불량
attention = outputs['attention_map']  # 모델이 본 영역
localization = outputs['localization_map']  # 결함 위치 예측
```

### 설명 생성

```python
explainer = DefectExplainer(model)

# 종합 설명 생성
explanation = explainer.explain(image)
print(f"예측: {explanation['prediction_name']}")
print(f"신뢰도: {explanation['confidence']:.2%}")
print(f"CAM-IoU: {explanation.get('cam_iou', 'N/A')}")

# 시각화
explainer.visualize(image, save_path='explanation.png')
```

## 참고 논문

- **GAIN**: "Tell Me Where to Look: Guided Attention Inference Network" (CVPR 2018)
- **CBAM**: "Convolutional Block Attention Module" (ECCV 2018)
- **EfficientNetV2**: "EfficientNetV2: Smaller Models and Faster Training" (ICML 2021)
- **Focal Loss**: "Focal Loss for Dense Object Detection" (ICCV 2017)

## 프로젝트 구조

```
ai-specialist/
├── configs/
│   └── default.yaml          # 설정 파일
├── src/
│   ├── models/
│   │   ├── backbone.py       # EfficientNetV2
│   │   ├── attention.py      # GAIN, CBAM
│   │   ├── heads.py          # Classification, Localization
│   │   ├── counterfactual.py # Counterfactual module
│   │   └── gain_mtl.py       # 메인 모델
│   ├── losses/
│   │   └── gain_mtl_loss.py  # 통합 loss function
│   ├── training/
│   │   └── trainer.py        # Multi-stage trainer
│   ├── data/
│   │   └── dataset.py        # Dataset classes
│   ├── evaluation/
│   │   ├── metrics.py        # 평가 지표
│   │   └── explainer.py      # 설명 생성
│   └── utils/
│       └── helpers.py        # 유틸리티
├── train.py                  # 학습 스크립트
├── evaluate.py               # 평가 스크립트
└── requirements.txt          # 의존성
```

## License

MIT License

## Contact

질문이나 피드백은 GitHub Issues를 통해 남겨주세요.
