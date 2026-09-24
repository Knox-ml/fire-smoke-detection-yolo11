"""
Fire & Smoke Detection — YOLO11 Multi-Variant Training Pipeline
================================================================
Run as a Kaggle notebook with GPU (P100/T4).

Dataset: D-Fire (21K+ images, YOLO format)
Source: https://github.com/gaiasd/DFireDataset

Pipeline:
  1. Dataset path correction & validation
  2. EDA — class distribution, sample visualization
  3. Train YOLO11 variants (nano, small, medium)
  4. Evaluate all variants on test set
  5. Failure analysis on best model
  6. Cross-domain generalization test
  7. Export best model to ONNX
"""

# ============================================================
# 0. Setup
# ============================================================
# !pip install ultralytics albumentations -q

import os
import glob
import random
import yaml
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. Dataset Configuration
# ============================================================
INPUT_BASE = "/kaggle/input/smoke-fire-detection-yolo/data"
WORK_YAML = "/kaggle/working/data.yaml"

with open("/kaggle/input/smoke-fire-detection-yolo/data.yaml", "r") as f:
    data_cfg = yaml.safe_load(f)

data_cfg["path"] = INPUT_BASE
data_cfg["train"] = f"{INPUT_BASE}/train/images"
data_cfg["val"] = f"{INPUT_BASE}/val/images"
data_cfg["test"] = f"{INPUT_BASE}/test/images"

with open(WORK_YAML, "w") as f:
    yaml.safe_dump(data_cfg, f, default_flow_style=False)

print("Dataset config:")
for split in ["train", "val", "test"]:
    img_dir = os.path.join(INPUT_BASE, split, "images")
    n_imgs = len(glob.glob(os.path.join(img_dir, "*.*")))
    print(f"  {split}: {n_imgs} images")

# ============================================================
# 2. EDA — Class Distribution
# ============================================================
CLASS_NAMES = {0: "fire", 1: "smoke"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, split in enumerate(["train", "val", "test"]):
    label_dir = os.path.join(INPUT_BASE, split, "labels")
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    class_counts = Counter()
    bbox_counts = []
    empty = 0

    for lf in label_files:
        with open(lf, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        bbox_counts.append(len(lines))
        if len(lines) == 0:
            empty += 1
        for line in lines:
            cls_id = int(line.split()[0])
            class_counts[cls_id] += 1

    print(f"\n--- {split.upper()} ---")
    print(f"  Images: {len(label_files)} | Negatives: {empty}")
    for cid, name in CLASS_NAMES.items():
        print(f"  {name}: {class_counts.get(cid, 0)} bboxes")

    names = ["fire", "smoke", "negative"]
    counts = [class_counts.get(0, 0), class_counts.get(1, 0), empty]
    colors = ["#FF6B6B", "#A0A0A0", "#4ECDC4"]
    axes[idx].bar(names, counts, color=colors)
    axes[idx].set_title(f"{split.upper()}", fontsize=14, fontweight="bold")
    axes[idx].set_ylabel("Count")

plt.suptitle("D-Fire — Class Distribution per Split", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/eda_class_distribution.png", dpi=150)
plt.show()

# ============================================================
# 3. Train All Variants
# ============================================================
TRAIN_CONFIG = dict(
    data=WORK_YAML,
    epochs=50,
    imgsz=640,
    seed=SEED,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=10,
    save=True,
    plots=True,
)

variants = [
    ("yolo11n.pt", "yolo11n_fire", 16),
    ("yolo11s.pt", "yolo11s_fire", 16),
    ("yolo11m.pt", "yolo11m_fire", 8),
]

for model_name, run_name, batch_size in variants:
    print(f"\n{'='*60}")
    print(f"Training {model_name} — batch={batch_size}")
    print(f"{'='*60}")

    model = YOLO(model_name)
    model.train(
        **TRAIN_CONFIG,
        batch=batch_size,
        project="/kaggle/working/runs",
        name=run_name,
    )
    print(f"{run_name} complete.")

# ============================================================
# 4. Evaluate All Variants on Test Set
# ============================================================
variant_weights = {
    "YOLO11n": "/kaggle/working/runs/yolo11n_fire/weights/best.pt",
    "YOLO11s": "/kaggle/working/runs/yolo11s_fire/weights/best.pt",
    "YOLO11m": "/kaggle/working/runs/yolo11m_fire/weights/best.pt",
}

comparison = []

for name, weights in variant_weights.items():
    model = YOLO(weights)
    metrics = model.val(data=WORK_YAML, split="test", verbose=False)
    row = {
        "Model": name,
        "Params (M)": f"{sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}",
        "mAP@50": f"{metrics.box.map50:.4f}",
        "mAP@50-95": f"{metrics.box.map:.4f}",
        "Precision": f"{metrics.box.mp:.4f}",
        "Recall": f"{metrics.box.mr:.4f}",
    }
    comparison.append(row)

print("\n" + "=" * 85)
print("MODEL COMPARISON — Speed vs Accuracy Tradeoff")
print("=" * 85)
print(f"{'Model':<12} {'Params(M)':<12} {'mAP@50':<10} {'mAP@50-95':<12} {'Precision':<12} {'Recall':<10}")
print("-" * 85)
for r in comparison:
    print(f"{r['Model']:<12} {r['Params (M)']:<12} {r['mAP@50']:<10} {r['mAP@50-95']:<12} {r['Precision']:<12} {r['Recall']:<10}")
print("=" * 85)

# ============================================================
# 5. Failure Analysis (Best Model — YOLO11s)
# ============================================================
best_weights = "/kaggle/working/runs/yolo11s_fire/weights/best.pt"
best_model = YOLO(best_weights)

test_img_dir = os.path.join(INPUT_BASE, "test", "images")
test_lbl_dir = os.path.join(INPUT_BASE, "test", "labels")
test_images = glob.glob(os.path.join(test_img_dir, "*.*"))

failures = []
for img_path in random.sample(test_images, min(500, len(test_images))):
    results = best_model(img_path, verbose=False)
    result = results[0]

    lbl_path = os.path.join(test_lbl_dir, Path(img_path).stem + ".txt")
    has_gt = False
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            gt_lines = [l.strip() for l in f if l.strip()]
        has_gt = len(gt_lines) > 0

    n_preds = len(result.boxes)

    if has_gt and n_preds == 0:
        failures.append((img_path, "MISSED"))
    elif not has_gt and n_preds > 0:
        failures.append((img_path, "FALSE POSITIVE"))
    elif n_preds > 0:
        confs = result.boxes.conf.cpu().numpy()
        if any((confs > 0.15) & (confs < 0.45)):
            failures.append((img_path, f"LOW CONF (min={confs.min():.2f})"))

    if len(failures) >= 6:
        break

if failures:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, (img_path, reason) in enumerate(failures[:6]):
        results = best_model(img_path, verbose=False)
        annotated = results[0].plot(line_width=2)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[i].imshow(annotated)
        axes[i].set_title(f"{reason}\n{Path(img_path).name}", fontsize=9)
        axes[i].axis("off")
    for j in range(len(failures), 6):
        axes[j].axis("off")
    plt.suptitle("Failure Analysis — YOLO11s", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/kaggle/working/failure_analysis.png", dpi=150)
    plt.show()

# ============================================================
# 6. ONNX Export + Speed Benchmark
# ============================================================
import time

onnx_path = best_model.export(format="onnx", imgsz=640, simplify=True)

test_img = test_images[0]

pt_model = YOLO(best_weights)
start = time.time()
for _ in range(50):
    pt_model(test_img, verbose=False)
pt_time = (time.time() - start) / 50 * 1000

onnx_model = YOLO(onnx_path)
start = time.time()
for _ in range(50):
    onnx_model(test_img, verbose=False)
onnx_time = (time.time() - start) / 50 * 1000

print(f"\nInference Speed (avg over 50 runs):")
print(f"  PyTorch: {pt_time:.1f} ms/image")
print(f"  ONNX:    {onnx_time:.1f} ms/image")
print(f"  Speedup: {pt_time/onnx_time:.2f}x")
