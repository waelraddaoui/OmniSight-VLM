import pytest
from src.core.schemas import BoundingBox, Detection, TrackedObject, AlertSeverity, IncidentReport


def test_bounding_box_geometry():
    bbox = BoundingBox(x1=100.0, y1=200.0, x2=300.0, y2=400.0)
    
    assert bbox.width == 200.0
    assert bbox.height == 200.0
    assert bbox.area == 40000.0
    assert bbox.center == (200.0, 300.0)
    assert bbox.bottom_center == (200.0, 400.0)
    assert bbox.to_xyxy() == (100.0, 200.0, 300.0, 400.0)
    assert bbox.to_xywh() == (100.0, 200.0, 200.0, 200.0)


def test_detection_creation():
    bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
    det = Detection(bbox=bbox, class_name="safety_helmet", confidence=0.88, class_id=1)
    
    assert det.class_name == "safety_helmet"
    assert det.confidence == 0.88
    assert det.bbox.width == 100.0


def test_tracked_object_lifecycle():
    bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    track = TrackedObject(
        track_id=1,
        class_name="forklift",
        current_bbox=bbox,
        confidence=0.92,
        first_seen_timestamp=100.0,
        last_seen_timestamp=102.5,
        total_frames_active=75,
    )
    
    assert track.track_id == 1
    assert track.total_frames_active == 75
    assert track.last_seen_timestamp - track.first_seen_timestamp == 2.5


def test_incident_report_schema():
    report = IncidentReport(
        event_type="ZONE_LOITERING: Loading Bay",
        severity=AlertSeverity.HIGH,
        target_object_id=42,
        description="Forklift parked in fire escape lane for over 30 seconds.",
        recommended_action="Dispatch ground safety crew.",
    )
    
    assert report.severity == AlertSeverity.HIGH
    assert report.target_object_id == 42
    assert "fire escape" in report.description
