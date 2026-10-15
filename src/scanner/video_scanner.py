"""
video_scanner.py
----------------
Scans a video file at a reduced frame rate, running object detection
on each sampled frame. Reports progress via a callback.

KEY PERFORMANCE NOTE:
  We read frames SEQUENTIALLY and skip unwanted ones — NOT by seeking.
  Random seeking (cap.set CAP_PROP_POS_FRAMES) is extremely slow for
  MP4/H264 because the decoder must decode all intermediate keyframes.
  Sequential read + skip is 10-50x faster for typical videos.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2

from src.nlp.query_parser import QueryIntent

logger = logging.getLogger("OmniSight-Scanner")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameDetection:
    """A single detection result for one sampled frame."""
    frame_idx: int
    timestamp_sec: float
    labels: List[str]
    confidences: List[float]
    boxes: List[List[int]]  # [[x1,y1,x2,y2], ...]


@dataclass
class ScanResult:
    """Full scan output: a timeline of all detections."""
    video_path: str
    scan_fps: float
    total_frames: int
    video_duration_sec: float
    detections: List[FrameDetection] = field(default_factory=list)
    scan_duration_sec: float = 0.0


# Callback type: receives (percent 0-100, message_string)
# IMPORTANT: must be a plain synchronous callable (not async)
# because it's called from a background thread.
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class VideoScanner:
    """
    Scans a video at `scan_fps` by reading frames sequentially
    (much faster than random seeking for compressed video).

    Args:
        detector:  Any detector with .set_vocabulary() and .detect() methods.
        scan_fps:  Frames per second to sample. Default 4.
    """

    def __init__(self, detector, scan_fps: float = 4.0):
        self.detector = detector
        self.scan_fps = scan_fps

    def scan(
        self,
        video_path: str,
        intent: QueryIntent,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ScanResult:
        """
        Synchronous video scan — runs sequentially, skips frames.
        Call scan_async() from async code to avoid blocking the event loop.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.detector.set_vocabulary(intent.detection_targets)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec: float = total_frames / video_fps if video_fps > 0 else 0

        # How many original frames to skip between each sample
        frame_step = max(1, round(video_fps / self.scan_fps))
        # Total number of frames we will actually run detection on
        n_samples = max(1, total_frames // frame_step)

        logger.info(
            f"Scanning '{path.name}': {duration_sec:.1f}s @ {video_fps:.1f}fps "
            f"-> 1 sample every {frame_step} frames ({n_samples} detections)"
        )

        result = ScanResult(
            video_path=str(path),
            scan_fps=self.scan_fps,
            total_frames=total_frames,
            video_duration_sec=duration_sec,
        )

        t_start = time.time()
        frame_idx = 0
        samples_processed = 0

        # Report start
        if on_progress:
            on_progress(0.0, f"Starting scan of {duration_sec:.0f}s video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only run detection on sampled frames
            if frame_idx % frame_step == 0:
                timestamp_sec = frame_idx / video_fps
                detections = self.detector.detect(frame)

                if detections:
                    fd = FrameDetection(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        labels=[d.class_name for d in detections],
                        confidences=[d.confidence for d in detections],
                        boxes=[
                            [int(d.bbox.x1), int(d.bbox.y1),
                             int(d.bbox.x2), int(d.bbox.y2)]
                            for d in detections
                            if d.bbox is not None
                        ],
                    )
                    result.detections.append(fd)

                samples_processed += 1

                # Progress update every ~5% (not too chatty, not too sparse)
                report_every = max(1, n_samples // 20)
                if on_progress and (samples_processed % report_every == 0 or samples_processed == n_samples):
                    pct = min(99.0, (samples_processed / n_samples) * 100)
                    elapsed = time.time() - t_start
                    rate = samples_processed / elapsed if elapsed > 0 else 1
                    remaining = (n_samples - samples_processed) / rate
                    on_progress(
                        pct,
                        f"Scanning {pct:.0f}% — {timestamp_sec:.0f}s / {duration_sec:.0f}s "
                        f"(~{remaining:.0f}s remaining)"
                    )

            frame_idx += 1

        cap.release()
        result.scan_duration_sec = time.time() - t_start

        logger.info(
            f"Scan complete: {len(result.detections)} frames with detections "
            f"in {result.scan_duration_sec:.1f}s "
            f"({samples_processed / result.scan_duration_sec:.1f} fps throughput)"
        )
        return result

    async def scan_async(
        self,
        video_path: str,
        intent: QueryIntent,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ScanResult:
        """
        Async wrapper that runs the blocking scan in a thread pool executor
        so it doesn't block the FastAPI event loop.

        The on_progress callback must be a plain sync function here;
        use a thread-safe bridge if you need to call async code from it.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.scan(video_path, intent, on_progress)
        )
