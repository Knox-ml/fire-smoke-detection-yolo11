# 🔥 Intelligent Vision-Based Fire & Smoke Detection System

Real-time fire and smoke detection using **YOLO11** fine-tuned on the [D-Fire dataset](https://github.com/gaiasd/DFireDataset) (21K+ annotated images). Features multi-variant model comparison, cross-domain generalization testing, temporal frame analysis for false positive suppression, and ONNX export for edge deployment.


## 🎯 Live Demo

> **[Try it on HuggingFace Spaces →](https://huggingface.co/spaces/KNOX10/fire-smoke-detection-yolo11)**

---

## Results

### Model Comparison — Speed vs Accuracy Tradeoff

Three YOLO11 variants were trained on the D-Fire dataset and evaluated on its held-out test set (4,291 images). YOLO11s emerged as the optimal variant — the medium model showed diminishing returns, suggesting the dataset complexity is well-matched to the small architecture's capacity.

| Model | Params (M) | Size (MB) | mAP@50 | mAP@50-95 | Precision | Recall | Inference (ms) |
|-------|-----------|-----------|--------|-----------|-----------|--------|---------------|
| YOLO11n | 2.6 | 5.4 | 0.742 | 0.419 | 0.733 | 0.679 | 1.6 |
| **YOLO11s** | **9.4** | **19.2** | **0.771** | **0.440** | **0.766** | **0.704** | **2.7** |
| YOLO11m | 20.1 | 40.5 | 0.763 | 0.432 | 0.756 | 0.693 | 5.9 |

> YOLO11s was selected as the best model and used for all subsequent evaluations.

### Per-Class Performance (YOLO11s — Test Set)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| Smoke | 0.824 | 0.769 | 0.828 | 0.505 |
| Fire | 0.707 | 0.639 | 0.715 | 0.374 |

Fire detection is inherently harder — fire patches are often small, vary in color (yellow, orange, white), and can be confused with sunlight or bright objects. Smoke, being larger and more diffuse, offers a bigger detection target.

### Cross-Domain Generalization

To validate that the model generalizes beyond D-Fire, it was evaluated on an entirely unseen dataset ([phylake1337/fire-dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)) containing 755 fire images and 244 non-fire images from different sources and camera types.

| Metric | Result |
|--------|--------|
| Fire detection rate | 96.8% (731/755) |
| False positive rate | 6.2% (15/243) |
| Avg confidence (fire) | 0.485 |

The model detected fire in 96.8% of unseen images while maintaining a low false positive rate. Cross-domain confidence dropped from 0.7+ (in-distribution) to 0.485, indicating domain shift, but detection capability held strong. The 6.2% false positive rate is further mitigated by temporal filtering in video mode.

### Failure Analysis

The model primarily struggles with:
- **Distant/small fires** — sub-pixel fire patches at long range
- **Smoke blending with haze** — overcast or foggy conditions reduce contrast
- **Low-confidence edge cases** — borderline detections around 0.3-0.4 confidence

<!-- Add failure_analysis.png here -->

---

## Architecture

```
Input (Image / Video Frame)
        │
        ▼
┌──────────────────────┐
│  YOLO11s Backbone     │ ── Feature extraction (C3k2 blocks)
│  (9.4M parameters)   │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│   Detection Head      │ ── Bounding boxes + class probabilities
│   (fire, smoke)       │    for fire and smoke
└─────────┬────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Temporal Filter              │ ── Sliding window (K of N frames)
│  (video inference only)       │    IoU-based tracking across frames
│                               │    Suppresses transient false positives
└─────────┬────────────────────┘
          │
          ▼
     Confirmed Detections
```

---

## Technical Details

### Training Configuration

- **Optimizer:** AdamW (lr=0.001, weight decay=0.0005) — chosen over SGD for faster convergence and better generalization via decoupled weight decay
- **Schedule:** Cosine LR decay (final LR = lr₀ × 0.01) with 3-epoch warmup
- **Augmentation:** HSV shifts, rotation (±10°), scale (0.5), horizontal flip, mosaic (4-image composition), mixup (α=0.1). Vertical flip disabled — fire/smoke orientation is semantically meaningful
- **Epochs:** 50 with early stopping (patience=10)
- **Image size:** 640×640
- **Hardware:** Kaggle P100/T4 GPU

### Why YOLO11s?

YOLO11m (20M params) scored lower than YOLO11s (9.4M params) despite being 2x larger. This is a classic case of diminishing returns — the dataset complexity (~21K images, 2 classes) is well-matched to the small variant's capacity. Adding more parameters introduces capacity the data can't fill, leading to marginal overfitting without accuracy gains.

### Temporal Filtering

Single-frame detection suffers from transient false positives — sunlight glare, orange objects, and steam can trigger false alarms. The temporal filter maintains a sliding window of N consecutive frames, promoting a detection to "confirmed" only when it appears in at least K frames at the same spatial location (IoU > 0.3). This mirrors commercial fire alarm systems that require sustained readings before triggering.

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/fire-smoke-detection-yolo11.git
cd fire-smoke-detection-yolo11
pip install -r requirements.txt
```

### 2. Dataset

The project uses the **D-Fire dataset** (21,527 images):

| Split | Images | Fire BBoxes | Smoke BBoxes | Negatives |
|-------|--------|-------------|-------------|-----------|
| Train | 14,122 | 7,794 | 9,638 | 6,458 |
| Val | 3,099 | 1,756 | 2,176 | 1,375 |
| Test | 4,306 | 2,315 | 2,878 | 2,005 |

Download from [Kaggle](https://www.kaggle.com/datasets) (search "D-Fire") or [Roboflow Universe](https://universe.roboflow.com/uweeaai/d-fire-ehbj2).

### 3. Training

```bash
# Full training pipeline (trains nano, small, medium variants)
# Run as a Kaggle notebook with GPU enabled
python notebooks/train.py
```

### 4. Video Inference

```bash
# On video file
python src/inference.py --source video.mp4 --weights best.pt

# On webcam
python src/inference.py --source 0 --weights best.pt

# Adjust temporal filter sensitivity
python src/inference.py --source video.mp4 --weights best.pt --window 7 --min-hits 4
```

### 5. Gradio Demo (local)

```bash
python app.py
```

### 6. ONNX Export

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

---

## Project Structure

```
fire-smoke-detection-yolo11/
├── notebooks/
│   └── train.py                  # Full training pipeline (Kaggle)
├── src/
│   ├── inference.py              # Video inference with temporal filtering
│   ├── temporal_filter.py        # Sliding-window false positive reduction
│   └── cross_domain_test.py      # Cross-domain evaluation script
├── app.py                        # Gradio demo (HuggingFace Spaces)
├── results/
│   └── plots/                    # Training curves, confusion matrices, etc.
├── requirements.txt
└── README.md
```

---

## Citation

If you use this project, please cite the D-Fire dataset:

```bibtex
@article{dfire2022,
  title={An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices},
  author={de Venâncio, Pedro Vinícius Almeida Borges and Lisboa, Adriano Chaves and Barbosa, Adriano Vilela},
  journal={Neural Computing and Applications},
  year={2022}
}
```

---

## License

MIT
