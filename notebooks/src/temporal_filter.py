"""
Temporal Frame Analysis for Fire/Smoke Detection
=================================================
Reduces false positives by requiring consistent detections
across consecutive video frames before triggering an alert.

Analogy: Think of it like a smoke alarm that only goes off
after detecting smoke for 3 seconds straight, not on a single
whiff that could be burnt toast.

Usage:
    filter = TemporalFilter(window_size=5, min_hits=3, iou_threshold=0.3)
    for frame in video:
        raw_detections = model(frame)
        stable_detections = filter.update(raw_detections)
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """Single bounding box detection."""
    bbox: np.ndarray        # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    bbox: np.ndarray
    class_id: int
    class_name: str
    hit_count: int = 0
    miss_count: int = 0
    max_confidence: float = 0.0
    confirmed: bool = False


class TemporalFilter:
    """
    Sliding-window temporal filter for video detections.

    Only promotes a detection to "confirmed" if it appears in
    at least `min_hits` out of the last `window_size` frames
    at roughly the same location (IoU > threshold).

    Parameters
    ----------
    window_size : int
        Number of recent frames to consider.
    min_hits : int
        Minimum detections in the window to confirm.
    iou_threshold : float
        IoU threshold to match detections across frames.
    max_misses : int
        Drop a tracked object after this many consecutive misses.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_misses: int = 3,
    ):
        self.window_size = window_size
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracked_objects: list[TrackedObject] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        Process one frame's detections. Returns only stable detections.

        Parameters
        ----------
        detections : list[Detection]
            Raw detections from the model for current frame.

        Returns
        -------
        list[Detection]
            Only detections that have been consistent across frames.
        """
        matched_tracked = set()
        matched_det = set()

        # Match current detections to existing tracked objects
        if self.tracked_objects and detections:
            iou_matrix = self._compute_iou_matrix(
                [t.bbox for t in self.tracked_objects],
                [d.bbox for d in detections],
            )

            # Greedy matching (highest IoU first)
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]

                if max_iou < self.iou_threshold:
                    break

                t_idx, d_idx = max_idx
                matched_tracked.add(t_idx)
                matched_det.add(d_idx)

                # Update tracked object
                tracked = self.tracked_objects[t_idx]
                det = detections[d_idx]
                tracked.bbox = det.bbox
                tracked.hit_count += 1
                tracked.miss_count = 0
                tracked.max_confidence = max(tracked.max_confidence, det.confidence)

                if tracked.hit_count >= self.min_hits:
                    tracked.confirmed = True

                # Mask out matched rows/cols
                iou_matrix[t_idx, :] = -1
                iou_matrix[:, d_idx] = -1

        # Handle unmatched tracked objects (missed in this frame)
        for i, tracked in enumerate(self.tracked_objects):
            if i not in matched_tracked:
                tracked.miss_count += 1

        # Remove tracked objects that have been missing too long
        self.tracked_objects = [
            t for t in self.tracked_objects
            if t.miss_count <= self.max_misses
        ]

        # Create new tracked objects for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det:
                self.tracked_objects.append(
                    TrackedObject(
                        bbox=det.bbox,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        hit_count=1,
                        miss_count=0,
                        max_confidence=det.confidence,
                        confirmed=False,
                    )
                )

        # Trim old objects beyond window (keep it bounded)
        if len(self.tracked_objects) > 50:
            # Keep only the most active ones
            self.tracked_objects.sort(key=lambda t: t.hit_count, reverse=True)
            self.tracked_objects = self.tracked_objects[:50]

        # Return only confirmed detections
        confirmed = []
        for tracked in self.tracked_objects:
            if tracked.confirmed:
                confirmed.append(
                    Detection(
                        bbox=tracked.bbox,
                        confidence=tracked.max_confidence,
                        class_id=tracked.class_id,
                        class_name=tracked.class_name,
                    )
                )

        return confirmed

    def reset(self):
        """Clear all tracked objects."""
        self.tracked_objects = []

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: list[np.ndarray],
        boxes_b: list[np.ndarray],
    ) -> np.ndarray:
        """Compute IoU between two sets of bounding boxes."""
        a = np.array(boxes_a)  # (N, 4)
        b = np.array(boxes_b)  # (M, 4)

        # Intersection
        x1 = np.maximum(a[:, 0:1], b[:, 0].T)  # (N, M)
        y1 = np.maximum(a[:, 1:2], b[:, 1].T)
        x2 = np.minimum(a[:, 2:3], b[:, 2].T)
        y2 = np.minimum(a[:, 3:4], b[:, 3].T)

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Union
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)
        union = area_a[:, None] + area_b[None, :] - intersection

        return intersection / (union + 1e-6)


# ============================================================
# Convenience: frame-level stats for logging
# ============================================================
class DetectionLogger:
    """Log detection counts per frame for later analysis."""

    def __init__(self):
        self.frame_log: list[dict] = []

    def log(self, frame_idx: int, raw_count: int, filtered_count: int):
        self.frame_log.append({
            "frame": frame_idx,
            "raw_detections": raw_count,
            "filtered_detections": filtered_count,
            "suppressed": raw_count - filtered_count,
        })

    def summary(self) -> dict:
        if not self.frame_log:
            return {}
        total_raw = sum(f["raw_detections"] for f in self.frame_log)
        total_filtered = sum(f["filtered_detections"] for f in self.frame_log)
        return {
            "total_frames": len(self.frame_log),
            "total_raw_detections": total_raw,
            "total_confirmed_detections": total_filtered,
            "total_suppressed": total_raw - total_filtered,
            "suppression_rate": (
                (total_raw - total_filtered) / total_raw * 100
                if total_raw > 0 else 0.0
            ),
        }
