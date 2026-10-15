"""
event_aggregator.py
-------------------
Converts a raw detection timeline (one entry per sampled frame) into a
ranked list of *temporal events* — continuous windows where something
interesting is happening.

Algorithm (in plain English):
  1. For each sampled frame that has relevant detections, mark it "active".
  2. Merge consecutive active frames that are < GAP_SEC apart into one event.
  3. Score each event based on:
       - Presence of priority_object
       - How many targets co-occur
       - Proximity of bounding boxes (if proximity_required)
       - Duration of the event
  4. Return events sorted by score descending.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.nlp.query_parser import QueryIntent
from src.scanner.video_scanner import FrameDetection, ScanResult

logger = logging.getLogger("OmniSight-Aggregator")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TemporalEvent:
    """A continuous time window where a relevant event was detected."""
    start_sec: float
    end_sec: float
    peak_sec: float             # Timestamp of the highest-scoring frame
    score: float                # 0.0 – 1.0 relevance score
    detected_labels: List[str]  # All unique labels seen in this window
    frame_count: int            # How many sampled frames are in this window
    explanation: str            # Human-readable description
    box_2d: Optional[List[int]] = None       # Normalized [ymin, xmin, ymax, xmax] in 0-1000
    trajectory_desc: Optional[str] = None    # Text summary of spatial motion path

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "peak_sec": round(self.peak_sec, 2),
            "score": round(self.score, 3),
            "detected_labels": self.detected_labels,
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 2),
            "explanation": self.explanation,
            "box_2d": self.box_2d,
            "trajectory_desc": self.trajectory_desc,
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

# Maximum gap (seconds) between two active frames before we split into
# separate events. E.g. 3.0 means: if there's a 3-second pause in
# detections, it's a new event.
_GAP_SEC = 3.0

# Clip padding added around each event when extracting video clips
CLIP_PADDING_SEC = 5.0


class EventAggregator:
    """
    Aggregates raw frame detections into scored temporal events.

    Args:
        gap_sec: Maximum gap (seconds) allowed inside a single event window.
        top_k:   Maximum number of events to return.
    """

    def __init__(self, gap_sec: float = _GAP_SEC, top_k: int = 5):
        self.gap_sec = gap_sec
        self.top_k = top_k

    def aggregate(self, scan_result: ScanResult, intent: QueryIntent) -> List[TemporalEvent]:
        """
        Main entry point.

        Args:
            scan_result: Output of VideoScanner.scan()
            intent:      Structured query from the NLP parser

        Returns:
            List of TemporalEvent sorted by score descending, up to top_k.
        """
        if not scan_result.detections:
            logger.info("No detections found in scan result.")
            return []

        targets = set(intent.detection_targets)
        priority = intent.priority_object

        # Step 1: Score each frame individually
        scored_frames = []
        for fd in scan_result.detections:
            frame_score = self._score_frame(fd, targets, priority, intent.proximity_required)
            if frame_score > 0:
                scored_frames.append((fd, frame_score))

        if not scored_frames:
            return []

        # Step 2: Merge consecutive frames into windows
        windows = self._merge_into_windows(scored_frames)

        # Step 3: Score each window and build TemporalEvent objects
        events = []
        for window in windows:
            event = self._build_event(window, intent)
            events.append(event)

        # Step 4: Sort by score descending, return top_k
        events.sort(key=lambda e: e.score, reverse=True)
        top_events = events[: self.top_k]

        logger.info(f"Aggregated {len(scan_result.detections)} detections → {len(events)} events → top {len(top_events)}")
        for e in top_events:
            logger.info(f"  [{e.start_sec:.1f}s–{e.end_sec:.1f}s] score={e.score:.2f} labels={e.detected_labels}")

        return top_events

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _score_frame(
        self,
        fd: FrameDetection,
        targets: set,
        priority: Optional[str],
        proximity_required: bool,
    ) -> float:
        """Score a single frame on a 0–1 scale."""
        if not fd.labels:
            return 0.0

        detected = set(fd.labels)

        # Base score: fraction of targets present
        overlap = len(detected & targets) / max(len(targets), 1)

        # Bonus: priority object is present
        priority_bonus = 0.3 if priority and priority in detected else 0.0

        # Bonus: multiple targets present simultaneously
        multi_bonus = 0.2 if len(detected & targets) >= 2 else 0.0

        # Bonus: proximity (bounding box overlap/closeness) when required
        proximity_bonus = 0.0
        if proximity_required and len(fd.boxes) >= 2:
            proximity_bonus = self._compute_proximity_bonus(fd.boxes, fd.labels, priority)

        # Average confidence of relevant detections
        relevant_confs = [
            c for c, l in zip(fd.confidences, fd.labels) if l in targets
        ]
        avg_conf = sum(relevant_confs) / len(relevant_confs) if relevant_confs else 0.0

        raw = overlap * 0.4 + avg_conf * 0.2 + priority_bonus + multi_bonus + proximity_bonus
        return min(raw, 1.0)

    def _compute_proximity_bonus(
        self, boxes: List[List[int]], labels: List[str], priority: Optional[str]
    ) -> float:
        """
        Bonus score when a priority object's bounding box is close to another
        object's bounding box (indicating interaction).
        """
        if not priority:
            return 0.0

        priority_boxes = [b for b, l in zip(boxes, labels) if l == priority]
        other_boxes = [b for b, l in zip(boxes, labels) if l != priority]

        if not priority_boxes or not other_boxes:
            return 0.0

        min_dist = float("inf")
        for pb in priority_boxes:
            px, py = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            for ob in other_boxes:
                ox, oy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
                dist = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
                min_dist = min(min_dist, dist)

        # Proximity bonus: up to 0.2 when distance < 100px
        return max(0.0, 0.2 * (1 - min_dist / 200))

    def _merge_into_windows(self, scored_frames):
        """Group scored frames into continuous windows separated by gap_sec."""
        windows = []
        current_window = [scored_frames[0]]

        for item in scored_frames[1:]:
            fd, _ = item
            prev_fd, _ = current_window[-1]
            if fd.timestamp_sec - prev_fd.timestamp_sec <= self.gap_sec:
                current_window.append(item)
            else:
                windows.append(current_window)
                current_window = [item]

        windows.append(current_window)
        return windows

    def _build_event(self, window: list, intent: QueryIntent) -> TemporalEvent:
        """Build a TemporalEvent from a window of (FrameDetection, score) pairs."""
        timestamps = [fd.timestamp_sec for fd, _ in window]
        scores = [s for _, s in window]
        all_labels = list({l for fd, _ in window for l in fd.labels})

        start_sec = timestamps[0]
        end_sec = timestamps[-1]
        peak_idx = scores.index(max(scores))
        peak_sec = timestamps[peak_idx]
        window_score = max(scores)

        # Duration bonus: longer sustained events are more credible
        duration_bonus = min(0.1, (end_sec - start_sec) / 30.0 * 0.1)
        final_score = min(1.0, window_score + duration_bonus)

        explanation = (
            f"Detected {', '.join(all_labels)} "
            f"from {self._fmt(start_sec)} to {self._fmt(end_sec)} "
            f"({end_sec - start_sec:.1f}s). "
            f"Confidence: {final_score:.0%}."
        )

        return TemporalEvent(
            start_sec=start_sec,
            end_sec=end_sec,
            peak_sec=peak_sec,
            score=final_score,
            detected_labels=all_labels,
            frame_count=len(window),
            explanation=explanation,
        )

    @staticmethod
    def _fmt(sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"
