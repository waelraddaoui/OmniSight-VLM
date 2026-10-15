"""
gemini_grounder.py
------------------
Uses Google Gemini 2.0 Flash's multimodal video understanding to directly ground
natural language queries in video without needing frame-by-frame object detection.

How it works:
  1. Uploads video to Gemini File API.
  2. Waits for video processing (ACTIVE state).
  3. Prompts Gemini 2.0 Flash to locate the exact timestamps and intervals matching the query.
  4. Returns structured TemporalEvent objects for the ClipExtractor to cut.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

from src.scanner.event_aggregator import TemporalEvent

logger = logging.getLogger("OmniSight-GeminiGrounder")

_GROUNDING_PROMPT = """\
You are an expert video intelligence and surveillance analyst.
Your task is to analyze the provided video and locate the EXACT temporal moment(s) where the following request occurs:

User Request: "{query}"

Instructions:
1. Locate every occurrence or relevant moment matching the query.
2. For each moment, determine:
   - "start_sec": float, timestamp in seconds where the moment begins.
   - "end_sec": float, timestamp in seconds where the moment ends (ensure at least 2 to 5 seconds duration).
   - "score": float between 0.50 and 0.99 indicating confidence.
   - "explanation": a concise, factual explanation of what visually appears in the video during this interval.
   - "detected_labels": list of key objects/entities visible (e.g. ["smartphone", "person", "hand"]).
   - "box_2d": [ymin, xmin, ymax, xmax] coordinates normalized from 0 to 1000 pinpointing the primary target object on the keyframe (e.g. [350, 420, 680, 590]).
   - "trajectory": string describing how the object/person moves across the scene during this interval (e.g. "enters from bottom-left, moves to center-right").
3. If no matching moments exist in the video, return an empty list [].
4. Return ONLY valid JSON matching this schema:
[
  {{
    "start_sec": 1.5,
    "end_sec": 6.0,
    "score": 0.95,
    "explanation": "Target object appears centered in the frame.",
    "detected_labels": ["target", "person"],
    "box_2d": [320, 410, 650, 580],
    "trajectory": "moves horizontally toward the right edge"
  }}
]
Do not wrap in markdown or add conversational text. Only valid JSON array.
"""


class GeminiVideoGrounder:
    """Grounds natural language queries into temporal video segments using Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key
        self.model = model
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def ground(
        self,
        video_path: str,
        query: str,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> List[TemporalEvent]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if on_progress:
            on_progress(15.0, "Uploading video to Gemini Multimodal AI...")

        logger.info(f"Uploading {path.name} to Gemini...")
        video_file = self.client.files.upload(file=str(path))
        file_name = video_file.name

        try:
            # Poll until video processing is completed
            if on_progress:
                on_progress(35.0, "Gemini is processing video frames...")

            logger.info(f"Waiting for Gemini file processing ({file_name})...")
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = self.client.files.get(name=file_name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"Gemini video processing failed: {video_file.error}")

            if on_progress:
                on_progress(60.0, "Gemini Vision AI is analyzing scenes & grounding query...")

            logger.info("Generating grounding content from Gemini 2.0 Flash...")
            prompt = _GROUNDING_PROMPT.format(query=query)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[video_file, prompt],
            )

            raw = response.text.strip()
            # Clean markdown formatting if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            if on_progress:
                on_progress(85.0, "Parsing detected temporal intervals...")

            data = json.loads(raw)
            if not isinstance(data, list):
                logger.warning(f"Unexpected Gemini response structure: {raw}")
                data = []

            events: List[TemporalEvent] = []
            for item in data:
                start_sec = max(0.0, float(item.get("start_sec", 0.0)))
                end_sec = max(start_sec + 1.0, float(item.get("end_sec", start_sec + 3.0)))
                score = min(1.0, max(0.0, float(item.get("score", 0.85))))
                explanation = item.get("explanation", f"Moment matching '{query}'")
                labels = item.get("detected_labels", [query])

                raw_box = item.get("box_2d")
                box_2d = None
                if isinstance(raw_box, list) and len(raw_box) == 4:
                    try:
                        box_2d = [int(v) for v in raw_box]
                    except Exception:
                        box_2d = None

                trajectory_desc = item.get("trajectory", "Target active in interval")

                events.append(
                    TemporalEvent(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        peak_sec=(start_sec + end_sec) / 2.0,
                        score=score,
                        detected_labels=labels,
                        frame_count=int((end_sec - start_sec) * 4),
                        explanation=explanation,
                        box_2d=box_2d,
                        trajectory_desc=trajectory_desc,
                    )
                )

            events.sort(key=lambda e: e.score, reverse=True)
            logger.info(f"Gemini grounded {len(events)} events for query '{query}'.")
            return events

        finally:
            try:
                logger.info(f"Cleaning up Gemini file {file_name}...")
                self.client.files.delete(name=file_name)
            except Exception as e:
                logger.warning(f"Could not delete Gemini file {file_name}: {e}")
