from typing import List, Optional, Union
import numpy as np

from src.core.schemas import BoundingBox, Detection


class YOLOWorldDetector:
    """
    Open-Vocabulary Zero-Shot Object Detector using YOLO-World.
    
    Architecture & Theory:
    - Traditional object detectors have a fixed classification head (e.g., 80 COCO classes).
    - YOLO-World decouples the classification head by using text embeddings from a vision-language
      backbone (e.g., CLIP / SigLIP).
    - For any user-specified prompt list P = [p_1, p_2, ..., p_K], text embeddings T in R^{K x D}
      are computed once and cached.
    - Image regions V in R^{N x D} are matched against text tokens via cross-modal attention
      and cosine similarity: S = V * T^T.
    - This allows detecting *any arbitrary vocabulary* at real-time speeds (>30-60 FPS) without retraining.
    """

    def __init__(
        self,
        model_id: str = "yolov8s-worldv2.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        fp16: bool = True,
    ):
        """
        Initialize the YOLO-World detector.
        
        Args:
            model_id: Model checkpoint ('yolov8s-worldv2.pt' for low VRAM, 'yolov8m-worldv2.pt' for higher accuracy).
            confidence_threshold: Minimum detection confidence score.
            iou_threshold: Non-Maximum Suppression (NMS) IoU cutoff.
            device: 'cuda' or 'cpu'. Defaults to auto-detecting CUDA.
            fp16: Enable half-precision inference to halve VRAM footprint and boost throughput.
        """
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"PyTorch and Ultralytics are required for YOLOWorldDetector: {e}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.fp16 = fp16 and (self.device == "cuda")
        self.current_classes: List[str] = []

        # Load model weights
        self.model = YOLO(model_id)
        if self.device == "cuda" and self.fp16:
            self.model.to(self.device)

    def set_vocabulary(self, classes: List[str]) -> None:
        """
        Dynamically update the detection vocabulary at runtime.
        This recomputes the offline text embeddings in the model head.
        
        Args:
            classes: List of natural language phrases, e.g. ["safety helmet", "forklift", "laptop"]
        """
        if not classes:
            raise ValueError("Vocabulary cannot be empty.")
        
        self.current_classes = [c.strip() for c in classes if c.strip()]
        self.model.set_classes(self.current_classes)

    def detect(self, image: Union[np.ndarray, str]) -> List[Detection]:
        """
        Perform zero-shot inference on an input frame.
        
        Args:
            image: BGR image as numpy array (from OpenCV) or path to an image file.
            
        Returns:
            List of structured Detection objects.
        """
        if not self.current_classes:
            raise RuntimeError("Vocabulary has not been set. Call `set_vocabulary(...)` first.")

        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.fp16,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract coordinates, confidences, and class IDs
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            box_coords = xyxy[i]
            cls_id = cls_ids[i]
            conf = float(confs[i])
            
            # Map class ID back to vocabulary string
            class_name = self.current_classes[cls_id] if cls_id < len(self.current_classes) else f"class_{cls_id}"

            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x1=float(box_coords[0]),
                        y1=float(box_coords[1]),
                        x2=float(box_coords[2]),
                        y2=float(box_coords[3]),
                    ),
                    class_name=class_name,
                    confidence=conf,
                    class_id=cls_id,
                )
            )

        return detections
