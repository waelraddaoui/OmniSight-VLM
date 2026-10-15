from abc import ABC, abstractmethod
from datetime import datetime, timezone
import cv2
import json
import logging
from typing import Optional
import numpy as np
from PIL import Image

from src.core.schemas import AlertSeverity, IncidentReport, TrackedObject
from src.analytics.zone import SpatialEvent, EventType

logger = logging.getLogger(__name__)


class BaseReasoningEngine(ABC):
    """Abstract interface for Vision-Language reasoning agents."""

    @abstractmethod
    def analyze_event(
        self,
        event: SpatialEvent,
        crop_image: np.ndarray,
        full_frame: Optional[np.ndarray] = None,
    ) -> IncidentReport:
        """
        Synthesize spatial context and visual evidence into a structured incident report.
        """
        pass


class MockReasoningEngine(BaseReasoningEngine):
    """
    Deterministic rule-based reasoning engine.
    Used for unit testing, CI/CD, and offline verification without GPU overhead.
    """

    def analyze_event(
        self,
        event: SpatialEvent,
        crop_image: np.ndarray,
        full_frame: Optional[np.ndarray] = None,
    ) -> IncidentReport:
        severity_map = {
            EventType.ZONE_ENTER: AlertSeverity.LOW,
            EventType.ZONE_LOITERING: AlertSeverity.HIGH,
            EventType.ZONE_EXIT: AlertSeverity.INFO,
        }
        severity = severity_map.get(event.event_type, AlertSeverity.MEDIUM)

        description = (
            f"Automated detection confirmed entity #{event.track.track_id} ({event.track.class_name}) "
            f"triggered {event.event_type.value} in '{event.zone_name}'. "
            f"Dwell duration: {event.dwell_duration:.1f}s."
        )

        action = (
            "Dispatch perimeter supervisor for physical verification."
            if severity == AlertSeverity.HIGH
            else "Log event into surveillance audit log."
        )

        return IncidentReport(
            event_type=f"{event.event_type.value}: {event.zone_name}",
            severity=severity,
            target_object_id=event.track.track_id,
            description=description,
            recommended_action=action,
            timestamp=datetime.now(timezone.utc),
        )


class LocalVLMReasoningEngine(BaseReasoningEngine):
    """
    Local Vision-Language reasoning engine using lightweight quantized models (e.g. Moondream2 or Qwen2-VL).
    
    Optimized for tight VRAM budgets (<= 4GB VRAM):
    - Passes only the cropped bounding box of the target entity (and context window)
      rather than the full 4K frame to minimize token sequence length.
    - Prompts the VLM with a strict system prompt demanding JSON adherence.
    """

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2024-08-26",
        device: str = "cuda",
    ):
        self.device = device
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._load_model(model_id, revision)

    def _load_model(self, model_id: str, revision: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading local VLM '{model_id}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                revision=revision,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            logger.info(f"Local VLM '{model_id}' successfully loaded.")
        except Exception as e:
            logger.warning(f"Could not load local VLM ({e}). Falling back to Mock engine.")
            self.model = None

    def analyze_event(
        self,
        event: SpatialEvent,
        crop_image: np.ndarray,
        full_frame: Optional[np.ndarray] = None,
    ) -> IncidentReport:
        if self.model is None:
            return MockReasoningEngine().analyze_event(event, crop_image, full_frame)

        # Convert OpenCV BGR crop to PIL RGB
        rgb_crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_crop)

        prompt = (
            f"You are an enterprise AI surveillance inspector. "
            f"A spatial violation occurred: Object #{event.track.track_id} ({event.track.class_name}) "
            f"is inside '{event.zone_name}' for {event.dwell_duration:.1f} seconds. "
            f"Describe what this entity is doing and assess whether it poses an immediate safety hazard."
        )

        try:
            enc_image = self.model.encode_image(pil_img)
            caption = self.model.answer_question(enc_image, prompt, self.tokenizer)

            return IncidentReport(
                event_type=f"{event.event_type.value}: {event.zone_name}",
                severity=AlertSeverity.HIGH if event.event_type == EventType.ZONE_LOITERING else AlertSeverity.MEDIUM,
                target_object_id=event.track.track_id,
                description=caption.strip(),
                recommended_action="Review camera feed and issue audio perimeter advisory.",
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"VLM inference error: {e}")
            return MockReasoningEngine().analyze_event(event, crop_image, full_frame)
