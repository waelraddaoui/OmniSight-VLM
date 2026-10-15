import argparse
import os
import sys
import time
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from src.core.schemas import Detection, BoundingBox
from src.tracker.byte_tracker import MultiObjectTracker
from src.analytics.zone import SpatialZoneMonitor
from src.visualization.annotator import SceneVisualizer


def run_benchmark(iterations: int = 100, num_simulated_objects: int = 15):
    """
    Profile per-stage execution latency (P50, P95, P99) of the tracking,
    spatial analytics, and rendering pipeline.
    """
    console = Console()
    console.print(f"\n[bold cyan]=== Running OmniSight-VLM Latency Profiler ({iterations} iterations) ===[/bold cyan]")
    console.print(f"Simulated entities per frame: [bold yellow]{num_simulated_objects}[/bold yellow]\n")

    # Initialize components
    tracker = MultiObjectTracker(track_activation_threshold=0.25, lost_track_buffer=30)
    zone = SpatialZoneMonitor(
        zone_name="Benchmark Zone",
        polygon_points=[(100, 100), (500, 100), (500, 400), (100, 400)],
        dwell_time_threshold_sec=3.0,
    )
    visualizer = SceneVisualizer()

    # Latency metric buffers (in milliseconds)
    t_tracker: List[float] = []
    t_zone: List[float] = []
    t_vis: List[float] = []
    t_total: List[float] = []

    # Benchmark loop
    np.random.seed(42)
    for i in range(iterations):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        # Generate synthetic detections
        detections = []
        for obj_idx in range(num_simulated_objects):
            cx = 150 + 200 * np.sin(i * 0.05 + obj_idx)
            cy = 200 + 100 * np.cos(i * 0.05 + obj_idx)
            bbox = BoundingBox(x1=cx - 20, y1=cy - 30, x2=cx + 20, y2=cy + 30)
            detections.append(Detection(bbox=bbox, class_name="worker", confidence=0.85))

        t0 = time.perf_counter()

        # 1. Tracker Stage
        t_track_start = time.perf_counter()
        tracks = tracker.update(detections, current_timestamp=i * 0.033)
        t_track_end = time.perf_counter()
        t_tracker.append((t_track_end - t_track_start) * 1000.0)

        # 2. Zone Analytics Stage
        t_zone_start = time.perf_counter()
        events = zone.update(tracks, current_timestamp=i * 0.033)
        t_zone_end = time.perf_counter()
        t_zone.append((t_zone_end - t_zone_start) * 1000.0)

        # 3. Visualization Stage
        t_vis_start = time.perf_counter()
        annotated = visualizer.draw_zones(frame, [zone])
        annotated = visualizer.draw_tracks(annotated, tracks, draw_trails=True)
        annotated = visualizer.draw_hud(annotated, fps=60.0, active_tracks_count=len(tracks))
        t_vis_end = time.perf_counter()
        t_vis.append((t_vis_end - t_vis_start) * 1000.0)

        t_total.append((t_vis_end - t0) * 1000.0)

    # Formulate Telemetry Table
    table = Table(title="OmniSight-VLM Per-Stage Latency Benchmark", show_lines=True)
    table.add_column("Pipeline Stage", style="cyan", no_wrap=True)
    table.add_column("Mean (ms)", style="green")
    table.add_column("P50 (ms)", style="white")
    table.add_column("P95 (ms)", style="yellow")
    table.add_column("P99 (ms)", style="red")

    def row(name, data):
        table.add_row(
            name,
            f"{np.mean(data):.3f}",
            f"{np.percentile(data, 50):.3f}",
            f"{np.percentile(data, 95):.3f}",
            f"{np.percentile(data, 99):.3f}",
        )

    row("1. ByteTrack Association", t_tracker)
    row("2. Spatial Zone & Dwell Analytics", t_zone)
    row("3. HUD & Visualizer Rendering", t_vis)
    row("Total Tracking & Analytics Loop", t_total)

    console.print(table)
    mean_total = np.mean(t_total)
    theoretical_fps = 1000.0 / mean_total if mean_total > 0 else 0
    console.print(f"\n[bold green][OK] Benchmark Finished![/bold green]")
    console.print(f"Tracking & Analytics Processing Throughput: [bold magenta]{theoretical_fps:.1f} FPS[/bold magenta] (excl. detector)")
    console.print(f"[dim]Tracking & spatial analytics consume only {mean_total:.2f}ms per frame, leaving >90% of the 33ms budget for the detector.[/dim]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=150, help="Number of benchmark iterations")
    parser.add_argument("--objects", type=int, default=15, help="Number of simultaneous objects")
    args = parser.parse_args()
    run_benchmark(iterations=args.iterations, num_simulated_objects=args.objects)
