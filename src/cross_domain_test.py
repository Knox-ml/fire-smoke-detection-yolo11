"""
Cross-Domain Generalization Test
=================================
Evaluates model on a completely unseen dataset to verify
the detector generalizes beyond its training distribution.

Tests fire detection rate (sensitivity) on fire images and
false positive rate (specificity) on non-fire images from
an external source the model was never trained on.

Usage:
    python cross_domain_test.py \
        --weights best.pt \
        --fire-dir /path/to/fire_images \
        --nofire-dir /path/to/non_fire_images
"""

import argparse
import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO


def run_cross_domain_test(
    weights: str,
    fire_dir: str,
    nofire_dir: str,
    save_path: str = "cross_domain_test.png",
):
    """Evaluate model on an unseen fire/non-fire dataset."""
    model = YOLO(weights)

    fire_imgs = glob.glob(os.path.join(fire_dir, "*.*"))
    nofire_imgs = glob.glob(os.path.join(nofire_dir, "*.*"))

    print(f"Cross-domain dataset: {len(fire_imgs)} fire, {len(nofire_imgs)} non-fire")

    # Fire images — model should detect something
    fire_detected = 0
    fire_total = 0
    fire_confs = []
    for img_path in fire_imgs:
        try:
            results = model(img_path, verbose=False)
            if not results or len(results) == 0:
                continue
            fire_total += 1
            if len(results[0].boxes) > 0:
                fire_detected += 1
                fire_confs.extend(results[0].boxes.conf.cpu().numpy().tolist())
        except:
            continue

    # Non-fire images — model should not detect anything
    false_positives = 0
    nofire_total = 0
    fp_confs = []
    for img_path in nofire_imgs:
        try:
            results = model(img_path, verbose=False)
            if not results or len(results) == 0:
                continue
            nofire_total += 1
            if len(results[0].boxes) > 0:
                false_positives += 1
                fp_confs.extend(results[0].boxes.conf.cpu().numpy().tolist())
        except:
            continue

    fire_rate = fire_detected / max(fire_total, 1) * 100
    fp_rate = false_positives / max(nofire_total, 1) * 100

    print(f"\n{'='*60}")
    print("CROSS-DOMAIN EVALUATION")
    print(f"{'='*60}")
    print(f"  Fire detection rate: {fire_detected}/{fire_total} ({fire_rate:.1f}%)")
    print(f"  False positive rate: {false_positives}/{nofire_total} ({fp_rate:.1f}%)")
    if fire_confs:
        print(f"  Avg confidence (fire): {np.mean(fire_confs):.3f}")
    if fp_confs:
        print(f"  Avg confidence (false pos): {np.mean(fp_confs):.3f}")
    print(f"{'='*60}")

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    fire_samples = random.sample(fire_imgs, min(4, len(fire_imgs)))
    for i, img_path in enumerate(fire_samples):
        try:
            results = model(img_path, verbose=False)
            annotated = results[0].plot(line_width=2)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            n = len(results[0].boxes)
        except:
            annotated = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            n = -1
        axes[i].imshow(annotated)
        axes[i].set_title(f"FIRE — {n} det(s)", fontsize=10,
                          color="green" if n > 0 else "red")
        axes[i].axis("off")

    nofire_samples = random.sample(nofire_imgs, min(4, len(nofire_imgs)))
    for i, img_path in enumerate(nofire_samples):
        try:
            results = model(img_path, verbose=False)
            annotated = results[0].plot(line_width=2)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            n = len(results[0].boxes)
        except:
            annotated = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            n = -1
        axes[i + 4].imshow(annotated)
        axes[i + 4].set_title(f"NON-FIRE — {n} det(s)", fontsize=10,
                               color="green" if n == 0 else "red")
        axes[i + 4].axis("off")

    plt.suptitle("Cross-Domain Test — Unseen Fire Dataset", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    return {
        "fire_detection_rate": fire_rate,
        "false_positive_rate": fp_rate,
        "avg_fire_confidence": np.mean(fire_confs) if fire_confs else 0,
    }


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Cross-Domain Fire Detection Test")
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--fire-dir", type=str, required=True)
    parser.add_argument("--nofire-dir", type=str, required=True)
    parser.add_argument("--save", type=str, default="cross_domain_test.png")
    args = parser.parse_args()

    run_cross_domain_test(args.weights, args.fire_dir, args.nofire_dir, args.save)
