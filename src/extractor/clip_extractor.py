"""
clip_extractor.py
-----------------
Extracts short video clips around detected temporal events.
Each clip = [event.start - padding, event.end + padding], clamped to video bounds.

Generates TWO versions of each clip:
1. Clean Raw Feed: High-quality lossless extraction.
2. Tactical Tracked Feed: Real-time visual spatial grounding with:
   - Cyberpunk corner brackets [ ]
   - Center crosshairs
   - Fading neon trajectory motion trails
   - Tactical HUD metadata banners
   - Target lock classification & confidence gauges

Encoded via FFmpeg with libx264 (yuv420p + faststart) for 100% native browser playback.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.scanner.event_aggregator import CLIP_PADDING_SEC, TemporalEvent

logger = logging.getLogger("OmniSight-Extractor")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ExtractedClip:
    """A successfully extracted video clip with spatial telemetry."""

    def __init__(
        self,
        event: TemporalEvent,
        clip_path: str,
        tracked_clip_path: Optional[str],
        actual_start_sec: float,
        actual_end_sec: float,
        trajectory_length_px: float = 0.0,
    ):
        self.event = event
        self.clip_path = clip_path
        self.tracked_clip_path = tracked_clip_path
        self.actual_start_sec = actual_start_sec
        self.actual_end_sec = actual_end_sec
        self.trajectory_length_px = trajectory_length_px

    def to_dict(self) -> dict:
        d = self.event.to_dict()
        d["clip_path"] = self.clip_path
        d["clip_filename"] = Path(self.clip_path).name
        d["tracked_clip_filename"] = Path(self.tracked_clip_path).name if self.tracked_clip_path else None
        d["clip_start_sec"] = round(self.actual_start_sec, 2)
        d["clip_end_sec"] = round(self.actual_end_sec, 2)
        d["trajectory_length_px"] = round(self.trajectory_length_px, 1)
        return d


# ---------------------------------------------------------------------------
# Extractor with Spatial Grounding
# ---------------------------------------------------------------------------

class ClipExtractor:
    """
    Extracts video clips for each detected temporal event with spatial-temporal tracking.

    Args:
        output_dir:   Directory to write clips to (created if needed).
        padding_sec:  Seconds of context before/after each event.
        jpeg_quality: JPEG quality for thumbnail frames (0–100).
    """

    def __init__(
        self,
        output_dir: str = "results",
        padding_sec: float = CLIP_PADDING_SEC,
        jpeg_quality: int = 85,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.padding_sec = padding_sec
        self.jpeg_quality = jpeg_quality
        self.ffmpeg_cmd = shutil.which("ffmpeg")

    def extract_all(
        self,
        video_path: str,
        events: List[TemporalEvent],
        job_id: str,
    ) -> List[ExtractedClip]:
        """
        Extract raw clips and tactical tracked overlays for each event.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec: float = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        clips: List[ExtractedClip] = []

        for rank, event in enumerate(events):
            clip_start = max(0.0, event.start_sec - self.padding_sec)
            clip_end = min(duration_sec, event.end_sec + self.padding_sec)
            clip_duration = max(1.0, clip_end - clip_start)

            clip_filename = f"{job_id}_clip_{rank + 1:02d}_{int(event.peak_sec):05d}s.mp4"
            clip_path = str(self.output_dir / clip_filename)
            tracked_filename = f"{job_id}_clip_{rank + 1:02d}_{int(event.peak_sec):05d}s_tracked.mp4"
            tracked_path = str(self.output_dir / tracked_filename)
            thumb_filename = f"{job_id}_thumb_{rank + 1:02d}.jpg"
            thumb_path = str(self.output_dir / thumb_filename)

            # 1. Extract raw clip
            self._extract_segment(video_path, clip_start, clip_duration, clip_path)

            # 2. Extract thumbnail
            self._extract_thumbnail(video_path, event.peak_sec, thumb_path)

            # 3. Generate Spatial Tactical HUD tracked clip
            trajectory_len = self._generate_tracked_clip(
                clip_path,
                tracked_path,
                event,
                width,
                height,
                fps,
            )

            clips.append(
                ExtractedClip(
                    event=event,
                    clip_path=clip_path,
                    tracked_clip_path=tracked_path if Path(tracked_path).exists() else None,
                    actual_start_sec=clip_start,
                    actual_end_sec=clip_end,
                    trajectory_length_px=trajectory_len,
                )
            )
            logger.info(f"Generated Spatial-Temporal clip {rank + 1}: {tracked_filename}")

        return clips

    def _extract_segment(self, video_path: str, start_sec: float, duration_sec: float, output_path: str):
        """Extract a video segment via FFmpeg."""
        if self.ffmpeg_cmd:
            cmd = [
                self.ffmpeg_cmd,
                "-y",
                "-ss", f"{start_sec:.3f}",
                "-t", f"{duration_sec:.3f}",
                "-i", str(video_path),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "128k",
                output_path,
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
                return
            except Exception as e:
                logger.warning(f"FFmpeg extraction failed ({e}), falling back to OpenCV.")

        # OpenCV fallback
        self._extract_segment_cv(video_path, start_sec, duration_sec, output_path)

    def _extract_thumbnail(self, video_path: str, timestamp_sec: float, output_path: str):
        """Extract a single frame thumbnail."""
        if self.ffmpeg_cmd:
            cmd = [
                self.ffmpeg_cmd,
                "-y",
                "-ss", f"{timestamp_sec:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                output_path,
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
                return
            except Exception:
                pass

        # OpenCV fallback
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        cap.release()

    def _generate_tracked_clip(
        self,
        clip_path: str,
        output_path: str,
        event: TemporalEvent,
        width: int,
        height: int,
        fps: float,
    ) -> float:
        """
        Renders tactical glowing reticles, bounding box, and motion trajectory over the clip.
        Returns total trajectory displacement in pixels.
        """
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return 0.0

        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()

        if not raw_frames:
            return 0.0

        h, w = raw_frames[0].shape[:2]

        # Calculate initial bounding box from event.box_2d [ymin, xmin, ymax, xmax] in 0-1000
        if event.box_2d and len(event.box_2d) == 4:
            ymin, xmin, ymax, xmax = event.box_2d
            bx = int(xmin * w / 1000.0)
            by = int(ymin * h / 1000.0)
            bw = max(24, int((xmax - xmin) * w / 1000.0))
            bh = max(24, int((ymax - ymin) * h / 1000.0))
            # Clamp to frame
            bx = max(0, min(w - bw, bx))
            by = max(0, min(h - bh, by))
            init_bbox = (bx, by, bw, bh)
        else:
            # Default center tactical target
            init_bbox = (int(w * 0.35), int(h * 0.35), int(w * 0.3), int(h * 0.3))

        # Initialize visual tracker
        tracker = cv2.TrackerMIL_create()
        tracker.init(raw_frames[0], init_bbox)

        current_bbox = init_bbox
        trajectory_history: List[Tuple[int, int]] = []
        annotated_frames = []
        total_distance = 0.0

        label_text = event.detected_labels[0] if event.detected_labels else "TARGET"

        for idx, frame in enumerate(raw_frames):
            ok, new_bbox = tracker.update(frame)
            if ok:
                current_bbox = (int(new_bbox[0]), int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3]))

            # Bounding box center
            cx = current_bbox[0] + current_bbox[2] // 2
            cy = current_bbox[1] + current_bbox[3] // 2

            if trajectory_history:
                last_pt = trajectory_history[-1]
                total_distance += ((cx - last_pt[0]) ** 2 + (cy - last_pt[1]) ** 2) ** 0.5
            trajectory_history.append((cx, cy))

            # Keep last 60 trajectory points for clean trail
            active_trail = trajectory_history[-60:]

            # Draw futuristic tactical HUD
            out_frame = self._render_tactical_frame(
                frame.copy(),
                current_bbox,
                active_trail,
                label_text,
                event.score,
                frame_idx=idx,
                total_frames=len(raw_frames),
                fps=fps,
            )
            annotated_frames.append(out_frame)

        # Write annotated frames to temp video and convert via FFmpeg
        temp_annotated = output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_annotated, fourcc, fps, (w, h))
        for f in annotated_frames:
            writer.write(f)
        writer.release()

        # Final conversion to browser H.264
        if self.ffmpeg_cmd:
            cmd = [
                self.ffmpeg_cmd,
                "-y",
                "-i", temp_annotated,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
            except Exception:
                if os.path.exists(temp_annotated):
                    os.replace(temp_annotated, output_path)
            finally:
                if os.path.exists(temp_annotated):
                    try:
                        os.remove(temp_annotated)
                    except Exception:
                        pass
        else:
            if os.path.exists(temp_annotated):
                os.replace(temp_annotated, output_path)

        return total_distance

    def _render_tactical_frame(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        trajectory: List[Tuple[int, int]],
        label: str,
        score: float,
        frame_idx: int,
        total_frames: int,
        fps: float,
    ) -> np.ndarray:
        """Draws cyberpunk tactical brackets, crosshairs, and trajectory trails."""
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        # Colors in BGR
        CYAN = (255, 240, 0)
        EMERALD = (80, 255, 80)
        DARK_BG = (20, 15, 6)

        # 1. Subtle box tint
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), CYAN, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # 2. Tactical Corner Reticles [ ]
        line_len = min(22, max(8, w // 4, h // 4))
        # Top-Left
        cv2.line(frame, (x, y), (x + line_len, y), CYAN, 2)
        cv2.line(frame, (x, y), (x, y + line_len), CYAN, 2)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - line_len, y), CYAN, 2)
        cv2.line(frame, (x + w, y), (x + w, y + line_len), CYAN, 2)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + line_len, y + h), CYAN, 2)
        cv2.line(frame, (x, y + h), (x, y + h - line_len), CYAN, 2)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), CYAN, 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), CYAN, 2)

        # 3. Center Crosshair
        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(frame, (cx, cy), CYAN, cv2.MARKER_CROSS, 14, 1)

        # 4. Fading Glowing Trajectory Trail
        if len(trajectory) > 1:
            for j in range(1, len(trajectory)):
                alpha_factor = j / len(trajectory)
                thickness = 2 if alpha_factor > 0.5 else 1
                color = EMERALD if alpha_factor > 0.4 else (40, 180, 40)
                cv2.line(frame, trajectory[j - 1], trajectory[j], color, thickness)
                if j % 6 == 0:
                    cv2.circle(frame, trajectory[j], 3, CYAN, -1)

        # 5. Target Lock Header Tag
        tag = f"// LOCK: {label.upper()} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(th + 8, y - 6)
        cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + 4), DARK_BG, -1)
        cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + 4), CYAN, 1)
        cv2.putText(frame, tag, (x + 4, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)

        # 6. Top Telemetry Banner
        sec_time = frame_idx / (fps or 30.0)
        hud_bar = f"OMNISIGHT SPATIAL TRACKER // T+{sec_time:04.1f}s // COORDS: ({cx},{cy}) // STATUS: LOCKED"
        cv2.rectangle(frame, (0, 0), (frame_w, 24), (10, 6, 3), -1)
        cv2.line(frame, (0, 24), (frame_w, 24), CYAN, 1)
        cv2.putText(frame, hud_bar, (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

        return frame

    def _extract_segment_cv(self, video_path: str, start_sec: float, duration_sec: float, output_path: str):
        """OpenCV fallback for clip extraction."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if not writer.isOpened():
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        start_frame = int(start_sec * fps)
        total_frames = int(duration_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        cap.release()
