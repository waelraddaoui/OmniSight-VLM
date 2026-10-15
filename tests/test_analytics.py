import pytest
from src.core.schemas import BoundingBox, TrackedObject
from src.analytics.zone import SpatialZoneMonitor, EventType


def test_zone_point_in_polygon():
    # Square zone from (100, 100) to (200, 200)
    polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]
    zone = SpatialZoneMonitor(zone_name="TestZone", polygon_points=polygon)

    # Center is inside
    assert zone.is_point_inside((150, 150)) is True
    # Outside points
    assert zone.is_point_inside((50, 50)) is False
    assert zone.is_point_inside((250, 150)) is False


def test_zone_enter_and_loitering_lifecycle():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    zone = SpatialZoneMonitor(
        zone_name="HazardArea",
        polygon_points=polygon,
        dwell_time_threshold_sec=3.0,
    )

    # Track enters the zone at t = 0.0
    # Bottom center at (50, 50) is inside (0, 0)-(100, 100)
    bbox_inside = BoundingBox(x1=40, y1=10, x2=60, y2=50)
    track = TrackedObject(
        track_id=10,
        class_name="person",
        current_bbox=bbox_inside,
        confidence=0.9,
    )

    # Frame 1: t = 0.0 -> Should trigger ZONE_ENTER
    events = zone.update([track], current_timestamp=0.0)
    assert len(events) == 1
    assert events[0].event_type == EventType.ZONE_ENTER
    assert events[0].zone_name == "HazardArea"

    # Frame 2: t = 1.0 -> Still inside, dwell = 1.0s, no new alert
    events = zone.update([track], current_timestamp=1.0)
    assert len(events) == 0

    # Frame 3: t = 3.5s -> Dwell exceeds 3.0s threshold -> Should trigger ZONE_LOITERING
    events = zone.update([track], current_timestamp=3.5)
    assert len(events) == 1
    assert events[0].event_type == EventType.ZONE_LOITERING
    assert events[0].dwell_duration >= 3.0

    # Frame 4: Track moves outside to (200, 200) at t = 4.0s -> Should trigger ZONE_EXIT
    bbox_outside = BoundingBox(x1=190, y1=150, x2=210, y2=200)
    track.current_bbox = bbox_outside
    events = zone.update([track], current_timestamp=4.0)
    assert len(events) == 1
    assert events[0].event_type == EventType.ZONE_EXIT
    assert events[0].dwell_duration == 4.0
