"""
query_parser.py
---------------
Converts a natural-language user query into structured detection targets
using the Gemini API. Falls back to rule-based extraction if the API is
unavailable or the key is not set.

Engineering note:
  We use a strict JSON-output prompt so the LLM always returns machine-
  readable data, not free text.  The pydantic model validates the response
  and raises an informative error when the LLM goes off-script.
"""

import json
import logging
import os
import re
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("OmniSight-NLP")


# ---------------------------------------------------------------------------
# Output schema — what we want the LLM to produce
# ---------------------------------------------------------------------------

class QueryIntent(BaseModel):
    """Structured representation of what the user wants to find in the video."""

    detection_targets: List[str] = Field(
        description="List of object classes to detect with YOLO-World (e.g. ['person', 'phone', 'hand'])"
    )
    event_description: str = Field(
        description="One-sentence description of the event to find"
    )
    interaction_type: str = Field(
        description="Type of interaction: object_transfer | person_action | object_presence | crowd_event | vehicle_event | other"
    )
    priority_object: Optional[str] = Field(
        default=None,
        description="The single most important object to track (e.g. 'phone')"
    )
    proximity_required: bool = Field(
        default=False,
        description="True if the targets must be spatially close to each other"
    )
    context_keywords: List[str] = Field(
        default_factory=list,
        description="Extra context words from the query (e.g. ['stolen', 'grab', 'theft'])"
    )
    human_summary: str = Field(
        description="One sentence explaining what the system will search for, written for the user"
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a video intelligence assistant. Your job is to convert a user's natural-language
question about a video into a structured JSON object that a computer vision system can act on.

The detection_targets must be concrete, visible objects that an object-detection model can find
in a single video frame. Examples: "person", "phone", "car", "bag", "knife", "helmet", "fire".

Rules:
- detection_targets: 2 to 6 items, all lowercase, singular nouns
- interaction_type: exactly one of [object_transfer, person_action, object_presence, crowd_event, vehicle_event, other]
- proximity_required: true only when the event requires multiple objects to be near each other
- human_summary: friendly one sentence starting with "I will search for..."
- Output ONLY valid JSON matching the schema. No markdown, no explanation.

Schema:
{
  "detection_targets": ["<str>", ...],
  "event_description": "<str>",
  "interaction_type": "<str>",
  "priority_object": "<str or null>",
  "proximity_required": <bool>,
  "context_keywords": ["<str>", ...],
  "human_summary": "<str>"
}
"""


# ---------------------------------------------------------------------------
# Gemini-powered parser
# ---------------------------------------------------------------------------

class GeminiQueryParser:
    """Uses Gemini API to parse user queries into structured detection intents."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def parse(self, user_query: str) -> QueryIntent:
        """
        Convert a natural-language query into a QueryIntent.

        Args:
            user_query: The user's plain-English description, e.g.
                        "find when someone stole my phone"

        Returns:
            QueryIntent with detection targets, interaction type, etc.
        """
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY set — falling back to rule-based parser.")
            return RuleBasedQueryParser().parse(user_query)

        try:
            client = self._get_client()
            prompt = (
                f"{_SYSTEM_PROMPT}\n\n"
                f"User query: {user_query}\n\n"
                f"JSON:"
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            raw = response.text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            intent = QueryIntent(**data)
            logger.info(f"Gemini parsed query: targets={intent.detection_targets}")
            return intent
        except Exception as e:
            logger.warning(f"Gemini parser failed ({e}), falling back to rule-based.")
            return RuleBasedQueryParser().parse(user_query)


# ---------------------------------------------------------------------------
# Rule-based fallback parser (no API key required)
# ---------------------------------------------------------------------------

_KEYWORD_RULES = [
    # (pattern, targets, interaction_type, priority_object)
    (r"stolen|theft|steal|rob|grab.*phone|take.*phone",
     ["person", "phone", "hand"], "object_transfer", "phone"),
    (r"fight|punch|hit|assault|attack",
     ["person"], "person_action", "person"),
    (r"fire|smoke|flame|burn",
     ["fire", "smoke", "person"], "object_presence", "fire"),
    (r"car|vehicle|crash|accident",
     ["car", "person", "truck"], "vehicle_event", "car"),
    (r"bag|luggage|suitcase.*stolen|steal.*bag",
     ["person", "bag", "hand"], "object_transfer", "bag"),
    (r"gun|weapon|knife|armed",
     ["person", "gun", "knife"], "person_action", "gun"),
    (r"crowd|gather|protest|group",
     ["person"], "crowd_event", "person"),
    (r"fall|slip|trip|collapse",
     ["person"], "person_action", "person"),
]


class RuleBasedQueryParser:
    """Simple keyword-based fallback parser — no API key required."""

    def parse(self, user_query: str) -> QueryIntent:
        q = user_query.lower()

        for pattern, targets, interaction_type, priority in _KEYWORD_RULES:
            if re.search(pattern, q):
                return QueryIntent(
                    detection_targets=targets,
                    event_description=user_query,
                    interaction_type=interaction_type,
                    priority_object=priority,
                    proximity_required=(interaction_type == "object_transfer"),
                    context_keywords=q.split()[:5],
                    human_summary=f"I will search for {priority} interactions in the video.",
                )

        # Generic fallback — clean conversational fillers and extract meaningful targets
        clean_q = re.sub(
            r"\b(can|could|would|please|will|you|we|i|me|us|find|show|look\s+for|search\s+for|detect|tell\s+me|is\s+there|are\s+there|check\s+for|spot)\b",
            " ",
            q
        )
        words = re.findall(r"\b[a-z]{3,}\b", clean_q)
        stopwords = {
            "the", "and", "find", "when", "where", "show", "was", "were",
            "that", "this", "with", "from", "video", "clip", "moment",
            "can", "you", "for", "are", "any", "all", "out", "about",
            "have", "has", "had", "not", "into", "some", "what", "which",
            "there", "here", "just", "more", "most", "also", "then", "than"
        }
        filtered_words = [w for w in words if w not in stopwords]

        # Check for meaningful multi-word phrases like 'microsoft logo'
        targets = []
        if "logo" in filtered_words:
            idx = filtered_words.index("logo")
            if idx > 0:
                targets.append(f"{filtered_words[idx-1]} logo")
        
        for w in filtered_words:
            if w not in targets and not any(w in t for t in targets):
                targets.append(w)

        targets = targets[:4] or ["person"]

        return QueryIntent(
            detection_targets=targets,
            event_description=user_query,
            interaction_type="other",
            priority_object=targets[0] if targets else "person",
            proximity_required=False,
            context_keywords=filtered_words[:5],
            human_summary=f"I will search for {', '.join(targets)} in the video.",
        )


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def create_query_parser(api_key: Optional[str] = None) -> GeminiQueryParser:
    """Create a query parser, preferring Gemini API if key is available."""
    return GeminiQueryParser(api_key=api_key)
