from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.core.schemas import TrackedObject


class EventType(str, Enum):
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_LOITERING = "ZONE_LOITERING"


@dataclass
class SpatialEvent:
    """An event triggered when an object violates a spatial or temporal condition."""
    event_type: EventType
    zone_name: str
    track: TrackedObject
    timestamp: float
    dwell_duration: float = 0.0
    message: str = ""


class SpatialZoneMonitor:
    """
    Monitors arbitrary 2D polygon zones, tripwires, and object dwell times.
    
    Mathematical Core:
    - Uses Point-in-Polygon (Ray Casting / winding number algorithm via OpenCV cv2.pointPolygonTest).
    - Checks the bottom-center contact point of the tracked object's bounding box:
        P = ((x1 + x2)/2, y2)
    - If pointPolygonTest(polygon, P, measureDist=False) >= 0, the object is inside the zone.
    - Maintains a temporal state map:
        state[(track_id, zone_name)] = {entry_time, last_active_time, alert_emitted}
    """

    def __init__(
        self,
        zone_name: str,
        polygon_points: List[Tuple[int, int]],
        target_classes: Optional[List[str]] = None,
        dwell_time_threshold_sec: float = 5.0,
    ):
        """
        Args:
            zone_name: Descriptive name, e.g. "Restricted Robot Area"
            polygon_points: Ordered list of (x, y) vertices defining the closed polygon.
            target_classes: Optional list of class names to monitor (None = monitor all classes).
            dwell_time_threshold_sec: Time in seconds before triggering a LOITERING alert.
        """
        self.zone_name = zone_name
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.target_classes = set(target_classes) if target_classes else None
        self.dwell_time_threshold_sec = dwell_time_threshold_sec

        # Internal tracking state: {track_id: {"entry_time": float, "last_seen": float, "loiter_alerted": bool}}
        self.occupants: Dict[int, Dict[str, float]] = {}

    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        """Check if 2D point (x, y) is inside or on the boundary of the polygon."""
        pt = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(self.polygon, pt, False) >= 0

    def update(
        self,
        tracks: List[TrackedObject],
        current_timestamp: Optional[float] = None,
    ) -> List[SpatialEvent]:
        """
        Evaluate current frame tracks against the zone boundary and dwell times.
        
        Args:
            tracks: Current active tracks from the tracker.
            current_timestamp: Current video timestamp in seconds.
            
        Returns:
            List of generated SpatialEvent triggers for this frame.
        """
        now = current_timestamp if current_timestamp is not None else time.time()
        events: List[SpatialEvent] = []
        current_frame_occupant_ids: Set[int] = set()

        for track in tracks:
            # Filter by target class if specified
            if self.target_classes and track.class_name not in self.target_classes:
                continue

            # Contact point: ground position (bottom center of bbox)
            ground_pt = track.current_bbox.bottom_center
            inside = self.is_point_inside(ground_pt)

            if inside:
                current_frame_occupant_ids.add(track.track_id)

                if track.track_id not in self.occupants:
                    # ZONE_ENTER Event
                    self.occupants[track.track_id] = {
                        "entry_time": now,
                        "last_seen": now,
                        "loiter_alerted": False,
                    }
                    track.current_zone = self.zone_name
                    track.time_in_current_zone = 0.0

                    events.append(
                        SpatialEvent(
                            event_type=EventType.ZONE_ENTER,
                            zone_name=self.zone_name,
                            track=track,
                            timestamp=now,
                            dwell_duration=0.0,
                            message=f"Track #{track.track_id} ({track.class_name}) entered '{self.zone_name}'.",
                        )
                    )
                else:
                    # Still inside zone: compute dwell time
                    entry_data = self.occupants[track.track_id]
                    entry_data["last_seen"] = now
                    dwell_time = now - entry_data["entry_time"]
                    track.time_in_current_zone = dwell_time

                    # ZONE_LOITERING Event
                    if dwell_time >= self.dwell_time_threshold_sec and not entry_data["loiter_alerted"]:
                        entry_data["loiter_alerted"] = True
                        events.append(
                            SpatialEvent(
                                event_type=EventType.ZONE_LOITERING,
                                zone_name=self.zone_name,
                                track=track,
                                timestamp=now,
                                dwell_duration=dwell_time,
                                message=(
                                    f"Track #{track.track_id} ({track.class_name}) has loitered in '{self.zone_name}' "
                                    f"for {dwell_time:.1f}s (threshold: {self.dwell_time_threshold_sec}s)."
                                ),
                            )
                        )

        # Check for occupants that have exited the zone
        exited_ids = set(self.occupants.keys()) - current_frame_occupant_ids
        for track_id in exited_ids:
            entry_data = self.occupants.pop(track_id)
            total_dwell = now - entry_data["entry_time"]

            # Find the track object if available in current frame
            track_match = next((t for t in tracks if t.track_id == track_id), None)
            if track_match:
                track_match.current_zone = None
                track_match.time_in_current_zone = 0.0

            events.append(
                SpatialEvent(
                    event_type=EventType.ZONE_EXIT,
                    zone_name=self.zone_name,
                    track=track_match or TrackedObject(
                        track_id=track_id,
                        class_name="unknown",
                        current_bbox=None,  # type: ignore
                        confidence=0.0,
                    ),
                    timestamp=now,
                    dwell_duration=total_dwell,
                    message=f"Track #{track_id} exited '{self.zone_name}' after {total_dwell:.1f}s.",
                )
            )

        return events
