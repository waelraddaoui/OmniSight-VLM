import time
import warnings
from typing import Dict, List, Optional
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")
import supervision as sv

from src.core.schemas import BoundingBox, Detection, TrackedObject


class MultiObjectTracker:
    """
    Temporal Object Tracker leveraging ByteTrack algorithm.
    
    Why ByteTrack is State-of-the-Art:
    - Traditional trackers (e.g. DeepSORT) discard low-confidence detections, causing severe ID switches
      during partial occlusions or motion blur.
    - ByteTrack retains low-confidence detections and performs a two-stage association:
      1. First stage: Match high-confidence boxes to existing tracks using Kalman Filter predictions.
      2. Second stage: Match remaining unmatched tracks to low-confidence boxes to preserve tracks
         through heavy occlusions without introducing false positive noise.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        max_trajectory_length: int = 60,
    ):
        """
        Initialize the multi-object tracker.
        
        Args:
            track_activation_threshold: Detection confidence threshold to initialize a track.
            lost_track_buffer: Number of frames to keep a lost track in memory before terminating it.
            minimum_matching_threshold: Minimum IoU similarity for track association.
            frame_rate: Expected video frame rate (helps tune Kalman filter motion covariance).
            max_trajectory_length: Maximum number of historical (x, y) center points to retain per track.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self.max_trajectory_length = max_trajectory_length
        # Internal state to store track history across frames: {track_id: TrackedObject}
        self.active_tracks: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Detection], current_timestamp: Optional[float] = None) -> List[TrackedObject]:
        """
        Update tracker state with new frame detections.
        
        Args:
            detections: List of Detection objects from the detector.
            current_timestamp: Optional Unix timestamp of current video frame.
            
        Returns:
            List of active TrackedObject instances with updated IDs and trajectories.
        """
        ts = current_timestamp if current_timestamp is not None else time.time()

        if not detections:
            # Empty frame - run tracker update with empty sv.Detections
            empty_sv_detections = sv.Detections.empty()
            self.tracker.update_with_detections(empty_sv_detections)
            return []

        # Convert our custom Detection objects to supervision Detections format
        xyxy_boxes = np.array([d.bbox.to_xyxy() for d in detections], dtype=np.float32)
        confidences = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)
        class_names = [d.class_name for d in detections]

        sv_detections = sv.Detections(
            xyxy=xyxy_boxes,
            confidence=confidences,
            class_id=class_ids,
            data={"class_name": np.array(class_names)},
        )

        # Run ByteTrack association
        tracked_sv = self.tracker.update_with_detections(sv_detections)

        updated_tracked_objects: List[TrackedObject] = []

        if tracked_sv.tracker_id is None or len(tracked_sv.tracker_id) == 0:
            return updated_tracked_objects

        for i in range(len(tracked_sv.tracker_id)):
            track_id = int(tracked_sv.tracker_id[i])
            box_coords = tracked_sv.xyxy[i]
            conf = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 1.0
            cls_name = str(tracked_sv.data["class_name"][i])

            bbox = BoundingBox(
                x1=float(box_coords[0]),
                y1=float(box_coords[1]),
                x2=float(box_coords[2]),
                y2=float(box_coords[3]),
            )
            center_point = bbox.bottom_center

            if track_id in self.active_tracks:
                # Update existing track
                tracked_obj = self.active_tracks[track_id]
                tracked_obj.current_bbox = bbox
                tracked_obj.confidence = conf
                tracked_obj.last_seen_timestamp = ts
                tracked_obj.total_frames_active += 1
                tracked_obj.trajectory.append(center_point)
                
                # Trim trajectory history to max length
                if len(tracked_obj.trajectory) > self.max_trajectory_length:
                    tracked_obj.trajectory.pop(0)
            else:
                # Initialize new track
                tracked_obj = TrackedObject(
                    track_id=track_id,
                    class_name=cls_name,
                    current_bbox=bbox,
                    confidence=conf,
                    trajectory=[center_point],
                    first_seen_timestamp=ts,
                    last_seen_timestamp=ts,
                    total_frames_active=1,
                )
                self.active_tracks[track_id] = tracked_obj

            updated_tracked_objects.append(tracked_obj)

        return updated_tracked_objects

    def reset(self) -> None:
        """Reset tracker state and active tracks."""
        self.tracker.reset()
        self.active_tracks.clear()
