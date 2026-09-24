"""
Video Inference Pipeline — Fire & Smoke Detection
===================================================
Runs YOLOv8 on video/webcam with temporal filtering.

Usage:
    python inference.py --source video.mp4 --weights best.pt
    python inference.py --source 0 --weights best.pt  # webcam
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from temporal_filter import Detection, TemporalFilter, DetectionLogger


# Colors for visualization
CLASS_COLORS = {
    0: (0, 0, 255),    # Fire — red (BGR)
    1: (180, 180, 180), # Smoke — gray
}
CLASS_NAMES = {0: "fire", 1: "smoke"}


def parse_yolo_results(results) -> list[Detection]:
    """Convert ultralytics Results to our Detection format."""
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detections.append(
            Detection(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=conf,
                class_id=cls_id,
                class_name=CLASS_NAMES.get(cls_id, "unknown"),
            )
        )
    return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    label_prefix: str = "",
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        color = CLASS_COLORS.get(det.class_id, (0, 255, 0))

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{label_prefix}{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def draw_status_bar(
    frame: np.ndarray,
    fps: float,
    raw_count: int,
    filtered_count: int,
    frame_idx: int,
) -> np.ndarray:
    """Draw info bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 40

    # Semi-transparent bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Status text
    status = (
        f"FPS: {fps:.1f} | "
        f"Frame: {frame_idx} | "
        f"Raw: {raw_count} | "
        f"Confirmed: {filtered_count}"
    )
    cv2.putText(
        frame, status, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )

    # Alert if fire confirmed
    if filtered_count > 0:
        alert = "FIRE/SMOKE DETECTED"
        cv2.putText(
            frame, alert, (w - 350, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return frame


def run_inference(
    source: str,
    weights: str,
    conf_threshold: float = 0.25,
    window_size: int = 5,
    min_hits: int = 3,
    save_output: bool = True,
    show: bool = True,
):
    """Run inference on video or webcam."""
    # Load model
    model = YOLO(weights)
    print(f"Loaded model: {weights}")

    # Open video
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source '{source}'")
        return

    # Video properties
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps_video:.0f} FPS, {total_frames} frames")

    # Output writer
    writer = None
    if save_output:
        output_path = f"output_{Path(source).stem if not is_webcam else 'webcam'}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_video, (w, h))
        print(f"Saving output to: {output_path}")

    # Initialize temporal filter and logger
    temporal_filter = TemporalFilter(
        window_size=window_size,
        min_hits=min_hits,
        iou_threshold=0.3,
        max_misses=3,
    )
    logger = DetectionLogger()

    frame_idx = 0
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0

    print("\nRunning inference... Press 'q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run model
        results = model(frame, conf=conf_threshold, verbose=False)
        raw_detections = parse_yolo_results(results)

        # Apply temporal filter
        stable_detections = temporal_filter.update(raw_detections)

        # Log
        logger.log(frame_idx, len(raw_detections), len(stable_detections))

        # Draw confirmed detections only
        annotated = draw_detections(frame, stable_detections, label_prefix="[confirmed] ")

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # Status bar
        annotated = draw_status_bar(
            annotated, current_fps,
            len(raw_detections), len(stable_detections),
            frame_idx,
        )

        # Save
        if writer:
            writer.write(annotated)

        # Display
        if show:
            cv2.imshow("Fire Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

        # Progress
        if frame_idx % 100 == 0 and total_frames > 0:
            pct = frame_idx / total_frames * 100
            print(f"  Frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary
    summary = logger.summary()
    print("\n" + "=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)
    print(f"  Total frames processed: {summary.get('total_frames', 0)}")
    print(f"  Raw detections: {summary.get('total_raw_detections', 0)}")
    print(f"  Confirmed detections: {summary.get('total_confirmed_detections', 0)}")
    print(f"  Suppressed (false positives): {summary.get('total_suppressed', 0)}")
    print(f"  Suppression rate: {summary.get('suppression_rate', 0):.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Detection — Video Inference")
    parser.add_argument("--source", type=str, required=True, help="Video path or '0' for webcam")
    parser.add_argument("--weights", type=str, default="best.pt", help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--window", type=int, default=5, help="Temporal filter window size")
    parser.add_argument("--min-hits", type=int, default=3, help="Min hits to confirm detection")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--no-save", action="store_true", help="Disable output saving")
    args = parser.parse_args()

    run_inference(
        source=args.source,
        weights=args.weights,
        conf_threshold=args.conf,
        window_size=args.window,
        min_hits=args.min_hits,
        save_output=not args.no_save,
        show=not args.no_show,
    )
