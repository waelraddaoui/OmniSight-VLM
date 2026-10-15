from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.schemas import AlertSeverity, BoundingBox, IncidentReport, TrackedObject
from src.analytics.zone import SpatialZoneMonitor


class SceneVisualizer:
    """
    Renders high-fps visual overlays on video frames:
    - Bounding boxes with track ID and class labels
    - Trajectory motion trails
    - Semi-transparent polygon zones (color changes on breach/loiter)
    - Live HUD (FPS, track statistics, and active alerts)
    """

    # Color palette (BGR format for OpenCV)
    COLOR_ZONE_NORMAL = (0, 200, 0)      # Green
    COLOR_ZONE_BREACH = (0, 0, 255)      # Red
    COLOR_TEXT = (255, 255, 255)         # White
    COLOR_BOX_DEFAULT = (255, 140, 0)    # Deep Sky Blue / Orange
    COLOR_TRAIL = (0, 255, 255)          # Yellow

    def __init__(self, line_thickness: int = 2):
        self.line_thickness = line_thickness
        # Map track ID to a distinct, consistent color
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self.track_colors:
            # Deterministic pseudo-random color based on track_id
            np.random.seed(track_id * 101)
            color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
            self.track_colors[track_id] = color  # type: ignore
        return self.track_colors[track_id]

    def draw_zones(
        self,
        frame: np.ndarray,
        zones: List[SpatialZoneMonitor],
        alpha: float = 0.25,
    ) -> np.ndarray:
        """Draw semi-transparent polygon zones on frame."""
        overlay = frame.copy()
        for zone in zones:
            is_breached = len(zone.occupants) > 0
            color = self.COLOR_ZONE_BREACH if is_breached else self.COLOR_ZONE_NORMAL

            # Draw filled semi-transparent polygon
            cv2.fillPoly(overlay, [zone.polygon], color)
            # Draw crisp boundary line
            cv2.polylines(frame, [zone.polygon], isClosed=True, color=color, thickness=2)

            # Zone label at the top-left vertex
            label_pos = (int(zone.polygon[0][0]), max(20, int(zone.polygon[0][1]) - 10))
            status_text = f"{zone.zone_name} ({len(zone.occupants)} occupants)"
            cv2.putText(
                frame,
                status_text,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Blend overlay for translucency
        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        draw_trails: bool = True,
    ) -> np.ndarray:
        """Render bounding boxes, track IDs, labels, and motion trails."""
        for track in tracks:
            color = self._get_track_color(track.track_id)
            bbox = track.current_bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

            # Draw trajectory motion trail
            if draw_trails and len(track.trajectory) > 1:
                points = [np.array(pt, dtype=np.int32) for pt in track.trajectory]
                for j in range(1, len(points)):
                    # Fade older points
                    thickness = max(1, int(self.line_thickness * (j / len(points))))
                    cv2.line(frame, tuple(points[j - 1]), tuple(points[j]), color, thickness)

            # Draw label banner
            label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_h - 6)),
                (x1 + text_w + 4, max(text_h + 6, y1)),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, max(text_h + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        active_tracks_count: int,
        latest_incident: Optional[IncidentReport] = None,
    ) -> np.ndarray:
        """Render top-level telemetry HUD."""
        h, w = frame.shape[:2]

        # Top banner background
        cv2.rectangle(frame, (10, 10), (320, 80), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (320, 80), (100, 100, 100), 1)

        # FPS and stats
        cv2.putText(frame, f"OmniSight-VLM | Real-Time Engine", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f} | Active Tracks: {active_tracks_count}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Status: ACTIVE MONITORING", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Alert banner if an incident was reported
        if latest_incident:
            severity_colors = {
                AlertSeverity.INFO: (255, 200, 0),
                AlertSeverity.LOW: (0, 255, 0),
                AlertSeverity.MEDIUM: (0, 165, 255),
                AlertSeverity.HIGH: (0, 69, 255),
                AlertSeverity.CRITICAL: (0, 0, 255),
            }
            alert_color = severity_colors.get(latest_incident.severity, (0, 0, 255))
            alert_box_y = h - 60
            cv2.rectangle(frame, (10, alert_box_y), (w - 10, h - 10), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, alert_box_y), (w - 10, h - 10), alert_color, 2)
            alert_text = f"[{latest_incident.severity}] {latest_incident.event_type}: {latest_incident.description}"
            cv2.putText(frame, alert_text, (20, alert_box_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
