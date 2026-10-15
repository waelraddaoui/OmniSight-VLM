import argparse
import logging
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import yaml

from src.core.schemas import IncidentReport
from src.tracker.byte_tracker import MultiObjectTracker
from src.analytics.zone import SpatialZoneMonitor
from src.reasoning.vlm_engine import MockReasoningEngine, LocalVLMReasoningEngine
from src.visualization.annotator import SceneVisualizer
from src.pipeline.engine import OmniSightEngine, FrameResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OmniSight")


def parse_args():
    parser = argparse.ArgumentParser(description="OmniSight-VLM: Real-Time Open-Vocabulary Video Intelligence")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--source", type=str, default=None, help="Video source (path, webcam index, or RTSP)")
    parser.add_argument("--classes", nargs="+", default=None, help="Custom zero-shot classes (e.g. --classes worker helmet forklift)")
    parser.add_argument("--detector", type=str, choices=["auto", "yolo-world", "synthetic"], default="auto", help="Detector engine")
    parser.add_argument("--reasoning", type=str, choices=["mock", "local"], default="mock", help="Reasoning engine type")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI window display")
    parser.add_argument("--loop", action="store_true", help="Loop video continuously")
    parser.add_argument("--save-output", type=str, default=None, help="Path to save processed video (.mp4)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    video_source = args.source or config["video"]["source"]
    classes = args.classes or config["detector"]["vocabulary"]
    device = config["system"]["device"]
    fp16 = config["system"]["fp16"]

    logger.info("=" * 60)
    logger.info("Initializing OmniSight-VLM Pipeline...")
    logger.info(f"Video Source: {video_source}")
    logger.info(f"Target Vocabulary ({len(classes)} classes): {classes}")
    logger.info(f"Device: {device} | FP16: {fp16}")
    logger.info("=" * 60)

    from src.detector import YOLOWorldDetector, SyntheticDetector

    # 1. Initialize Detector
    if args.detector == "synthetic" or YOLOWorldDetector is None:
        logger.info("Using SyntheticDetector (simulated open-vocabulary mode).")
        detector = SyntheticDetector()
    else:
        try:
            logger.info("Using YOLOWorldDetector (neural zero-shot model).")
            detector = YOLOWorldDetector(
                model_id=config["detector"]["model_id"],
                confidence_threshold=config["detector"]["confidence_threshold"],
                iou_threshold=config["detector"]["iou_threshold"],
                device=device,
                fp16=fp16,
            )
        except Exception as e:
            logger.warning(f"Could not load YOLOWorldDetector ({e}). Falling back to SyntheticDetector.")
            detector = SyntheticDetector()

    detector.set_vocabulary(classes)

    # 2. Initialize Tracker
    tracker = MultiObjectTracker(
        track_activation_threshold=config["tracker"]["track_activation_threshold"],
        lost_track_buffer=config["tracker"]["lost_track_buffer"],
        minimum_matching_threshold=config["tracker"]["minimum_matching_threshold"],
        frame_rate=config["tracker"]["frame_rate"],
    )

    # 3. Initialize Spatial Zones
    zones = []
    for z_conf in config["analytics"]["zones"]:
        zone = SpatialZoneMonitor(
            zone_name=z_conf["name"],
            polygon_points=z_conf["polygon"],
            target_classes=z_conf.get("target_classes"),
            dwell_time_threshold_sec=config["analytics"]["dwell_time_threshold_sec"],
        )
        zones.append(zone)

    # 4. Initialize Reasoning Engine
    if args.reasoning == "local":
        reasoning_engine = LocalVLMReasoningEngine(
            model_id=config["reasoning"]["vlm_model_id"],
            device=device,
        )
    else:
        reasoning_engine = MockReasoningEngine()

    # 5. Initialize Visualizer & Pipeline Engine
    visualizer = SceneVisualizer()
    engine = OmniSightEngine(
        detector=detector,
        tracker=tracker,
        zones=zones,
        reasoning_engine=reasoning_engine,
        visualizer=visualizer,
    )

    # Setup video writer if output path is requested
    writer = None
    if args.save_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Writer initialized on first frame

    def on_frame_processed(result: FrameResult):
        nonlocal writer
        if args.save_output:
            if writer is None:
                h, w = result.annotated_frame.shape[:2]
                writer = cv2.VideoWriter(args.save_output, fourcc, 30.0, (w, h))
            writer.write(result.annotated_frame)

        for incident in result.incidents:
            logger.warning(f"[ALERT] INCIDENT REPORT [{incident.severity.value}]: {incident.event_type}")
            logger.warning(f"   Details: {incident.description}")
            logger.warning(f"   Action:  {incident.recommended_action}")

    # Run stream
    try:
        for _ in engine.stream_video(
            video_source=video_source,
            display=not args.no_display,
            loop=args.loop,
            on_result=on_frame_processed,
        ):
            pass
    finally:
        if writer:
            writer.release()
            logger.info(f"Saved processed output video to: {args.save_output}")


if __name__ == "__main__":
    main()
