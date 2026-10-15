import os
import cv2
import numpy as np


def generate_synthetic_traffic_video(
    output_path: str = "data/synthetic_sample.mp4",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    duration_sec: int = 8,
) -> str:
    """
    Generate a deterministic synthetic test video:
    - Moving objects simulating a worker/person and a forklift/vehicle
    - Moving across and dwelling inside a defined polygon zone
    - Useful for CI testing, benchmarking, and immediate visualization
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Zone geometry: Center area
    zone_pts = np.array([[180, 140], [460, 140], [460, 360], [180, 360]], np.int32)

    # Object 1: Starts outside, enters zone, loiters, then exits
    # Object 2: Crosses horizontally without entering the zone
    for f in range(total_frames):
        # Create dark industrial workshop floor background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw floor markings / grid
        for y in range(0, height, 60):
            cv2.line(frame, (0, y), (width, y), (55, 55, 55), 1)
        for x in range(0, width, 60):
            cv2.line(frame, (x, 0), (x, height), (55, 55, 55), 1)

        # Draw faint physical zone markings on floor
        cv2.polylines(frame, [zone_pts], isClosed=True, color=(80, 80, 80), thickness=2)
        cv2.putText(frame, "RESTRICTED ROBOTICS BAY", (200, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)

        # Entity 1: Simulating Person
        # Moves from (50, 250) into (300, 250), dwells for 4 seconds, then moves to (580, 250)
        t = f / fps
        if t < 2.0:
            p1_x = int(50 + (250 / 2.0) * t)
        elif t < 6.0:
            p1_x = 300  # Loitering in center of zone
        else:
            p1_x = int(300 + (280 / 2.0) * (t - 6.0))
        p1_y = 250

        # Draw simulated person (circle head + rect body)
        cv2.circle(frame, (p1_x, p1_y - 25), 12, (200, 180, 100), -1)  # Head / helmet
        cv2.rectangle(frame, (p1_x - 14, p1_y - 12), (p1_x + 14, p1_y + 35), (180, 100, 50), -1)  # Torso / legs
        cv2.putText(frame, "Worker", (p1_x - 20, p1_y - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        # Entity 2: Simulating Cart / Vehicle moving along bottom edge
        cart_x = int((width + 100) * (f / total_frames)) - 50
        cart_y = 420
        cv2.rectangle(frame, (cart_x - 30, cart_y - 15), (cart_x + 30, cart_y + 15), (70, 140, 220), -1)
        cv2.putText(frame, "Cart", (cart_x - 15, cart_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print(f"Synthetic test video generated: {output_path} ({total_frames} frames, {duration_sec}s)")
    return output_path


if __name__ == "__main__":
    generate_synthetic_traffic_video()
