from .mock_detector import SyntheticDetector

try:
    from .yolo_world import YOLOWorldDetector
except ImportError:
    # Graceful fallback if torch is not yet initialized
    YOLOWorldDetector = None  # type: ignore

__all__ = ["YOLOWorldDetector", "SyntheticDetector"]
