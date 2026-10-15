from dataclasses import dataclass
import time
from typing import Any, Callable, Generator, List, Optional, Tuple
import cv2
import numpy as np

from src.core.schemas import IncidentReport, TrackedObject
from src.tracker.byte_tracker import MultiObjectTracker
from src.analytics.zone import SpatialZoneMonitor, SpatialEvent
from src.reasoning.vlm_engine import BaseReasoningEngine, MockReasoningEngine
from src.visualization.annotator import SceneVisualizer


@dataclass
class FrameResult:
    """Encapsulates all analytics output for a single processed video frame."""
    frame_index: int
    timestamp: float
    fps: float
    annotated_frame: np.ndarray
    tracks: List[TrackedObject]
    events: List[SpatialEvent]
    incidents: List[IncidentReport]


class OmniSightEngine:
    """
    End-to-End Orchestrator for OmniSight-VLM.
    
    Coordinates:
    - Open-Vocabulary Detector (YOLO-World)
    - Multi-Object Tracker (ByteTrack)
    - Spatial Zone & Dwell-Time Monitors
    - Vision-Language Reasoning Agent
    - Scene Visualizer & Annotator
    """

    def __init__(
        self,
        detector: Any,
        tracker: MultiObjectTracker,
        zones: List[SpatialZoneMonitor],
        reasoning_engine: Optional[BaseReasoningEngine] = None,
        visualizer: Optional[SceneVisualizer] = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.zones = zones
        self.reasoning = reasoning_engine or MockReasoningEngine()
        self.visualizer = visualizer or SceneVisualizer()

        # Cooldown management to prevent alerting spam for the same track
        # { (track_id, event_type): last_alert_time }
        self.alert_cooldowns: dict = {}
        self.cooldown_seconds: float = 10.0

        # Performance telemetry
        self.last_frame_time: float = time.time()
        self.smoothed_fps: float = 0.0

    def set_vocabulary(self, classes: List[str]) -> None:
        """Dynamically update zero-shot detection vocabulary."""
        self.detector.set_vocabulary(classes)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: Optional[float] = None,
    ) -> FrameResult:
        """
        Process a single video frame through the complete pipeline.
        
        Args:
            frame: Input BGR image (H x W x 3).
            frame_index: Sequential integer index of the frame.
            timestamp: Video Presentation Time Stamp (PTS) in seconds.
            
        Returns:
            FrameResult containing annotations, active tracks, and any triggered incidents.
        """
        now = timestamp if timestamp is not None else (frame_index / 30.0)
        dt = time.time() - self.last_frame_time
        self.last_frame_time = time.time()
        instant_fps = 1.0 / dt if dt > 0 else 30.0
        self.smoothed_fps = 0.9 * self.smoothed_fps + 0.1 * instant_fps if self.smoothed_fps > 0 else instant_fps

        # 1. Open-Vocabulary Detection
        detections = self.detector.detect(frame)

        # 2. Temporal Association & Tracking
        tracks = self.tracker.update(detections, current_timestamp=now)

        # 3. Spatial Zone & Dwell Analytics
        frame_events: List[SpatialEvent] = []
        for zone in self.zones:
            events = zone.update(tracks, current_timestamp=now)
            frame_events.extend(events)

        # 4. Reasoning Layer (VLM Incident Synthesis)
        frame_incidents: List[IncidentReport] = []
        for event in frame_events:
            cd_key = (event.track.track_id, event.event_type)
            if cd_key in self.alert_cooldowns:
                if (now - self.alert_cooldowns[cd_key]) < self.cooldown_seconds:
                    continue

            self.alert_cooldowns[cd_key] = now

            # Extract target object crop with margin
            bbox = event.track.current_bbox
            if bbox is None:
                continue
            h, w = frame.shape[:2]
            margin = 20
            x1 = max(0, int(bbox.x1) - margin)
            y1 = max(0, int(bbox.y1) - margin)
            x2 = min(w, int(bbox.x2) + margin)
            y2 = min(h, int(bbox.y2) + margin)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                incident = self.reasoning.analyze_event(event, crop, full_frame=frame)
                frame_incidents.append(incident)

        # 5. Scene Visualization
        annotated = frame.copy()
        annotated = self.visualizer.draw_zones(annotated, self.zones)
        annotated = self.visualizer.draw_tracks(annotated, tracks, draw_trails=True)
        latest_incident = frame_incidents[-1] if frame_incidents else None
        annotated = self.visualizer.draw_hud(
            annotated,
            fps=self.smoothed_fps,
            active_tracks_count=len(tracks),
            latest_incident=latest_incident,
        )

        return FrameResult(
            frame_index=frame_index,
            timestamp=now,
            fps=self.smoothed_fps,
            annotated_frame=annotated,
            tracks=tracks,
            events=frame_events,
            incidents=frame_incidents,
        )

    def stream_video(
        self,
        video_source: str,
        display: bool = True,
        loop: bool = False,
        on_result: Optional[Callable[[FrameResult], None]] = None,
    ) -> Generator[FrameResult, None, None]:
        """
        Stream and process video from a local file, webcam, or RTSP stream.
        """
        cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: '{video_source}'")

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                result = self.process_frame(frame, frame_index=frame_idx, timestamp=frame_idx / 30.0)
                frame_idx += 1

                if on_result:
                    on_result(result)

                if display:
                    cv2.imshow("OmniSight-VLM Engine", result.annotated_frame)
                    # Press 'q' to break early
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield result
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
