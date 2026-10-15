from typing import List, Union
import cv2
import numpy as np

from src.core.schemas import BoundingBox, Detection


class SyntheticDetector:
    """
    Color-segmented dynamic detector for synthetic test video.
    
    Detects simulated objects directly from frame pixel contours:
    - Worker entity: blue body / cyan head
    - Cart entity: orange cargo rectangle
    """

    def __init__(self, confidence: float = 0.92):
        self.confidence = confidence
        self.current_classes: List[str] = ["worker", "cart"]

    def set_vocabulary(self, classes: List[str]) -> None:
        self.current_classes = [c.strip().lower() for c in classes if c.strip()]

    def detect(self, image: Union[np.ndarray, str]) -> List[Detection]:
        if isinstance(image, str):
            frame = cv2.imread(image)
        else:
            frame = image

        if frame is None:
            return []

        h, w = frame.shape[:2]
        detections: List[Detection] = []

        # 1. Detect Worker (Blue body: BGR (180, 100, 50))
        if any(c in self.current_classes for c in ["worker", "person", "human"]):
            lower_blue = np.array([150, 70, 30], dtype=np.uint8)
            upper_blue = np.array([210, 130, 80], dtype=np.uint8)
            mask_worker = cv2.inRange(frame, lower_blue, upper_blue)
            contours, _ = cv2.findContours(mask_worker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > 10 and ch > 15:
                    # Enclose both head and body
                    bbox = BoundingBox(
                        x1=float(x - 5),
                        y1=float(max(0, y - 30)), # Include head
                        x2=float(x + cw + 5),
                        y2=float(y + ch + 5),
                    )
                    detections.append(
                        Detection(
                            bbox=bbox,
                            class_name="worker",
                            confidence=self.confidence,
                            class_id=0,
                        )
                    )

        # 2. Detect Cart (Orange: BGR around (70, 140, 220))
        if any(c in self.current_classes for c in ["cart", "vehicle", "forklift"]):
            lower_orange = np.array([50, 120, 190], dtype=np.uint8)
            upper_orange = np.array([90, 160, 240], dtype=np.uint8)
            mask_cart = cv2.inRange(frame, lower_orange, upper_orange)
            contours, _ = cv2.findContours(mask_cart, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > 20 and ch > 10:
                    bbox = BoundingBox(
                        x1=float(x - 2),
                        y1=float(y - 2),
                        x2=float(x + cw + 2),
                        y2=float(y + ch + 2),
                    )
                    detections.append(
                        Detection(
                            bbox=bbox,
                            class_name="cart",
                            confidence=self.confidence - 0.04,
                            class_id=1,
                        )
                    )

        return detections
