"""
app.py — OmniSight v2 Command Center
--------------------------------------
FastAPI server with:
  - /api/upload          : receive a video file and return a job_id
  - /api/query           : start an async scan job with a natural-language query
  - /api/status/{job_id} : poll scan progress
  - /api/results/{job_id}: get ranked clips and explanations
  - /api/clips/{filename}: serve extracted clip files
  - /ws/progress/{job_id}: real-time WebSocket progress stream
  - /ws/stream           : original live-stream WebSocket (kept for demo tab)
  - /                    : serve the new v2 HTML UI
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import cv2
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import yaml

from src.core.schemas import IncidentReport
from src.detector.yolo_world import YOLOWorldDetector
from src.tracker.byte_tracker import MultiObjectTracker
from src.analytics.zone import SpatialZoneMonitor
from src.reasoning.vlm_engine import MockReasoningEngine
from src.visualization.annotator import SceneVisualizer
from src.pipeline.engine import OmniSightEngine, FrameResult
from src.nlp.query_parser import create_query_parser, QueryIntent
from src.scanner.video_scanner import VideoScanner
from src.scanner.event_aggregator import EventAggregator
from src.extractor.clip_extractor import ClipExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniSight-API")

app = FastAPI(title="OmniSight-VLM v2 Command Center", version="2.0.0")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
engine: Optional[OmniSightEngine] = None
incident_history: List[dict] = []
current_vocabulary: List[str] = ["worker", "helmet", "cart", "person"]

# Job state: job_id -> {status, progress, message, results, error}
jobs: Dict[str, dict] = {}

# WebSocket connections per job: job_id -> list of WebSocket
job_ws_connections: Dict[str, List[WebSocket]] = {}


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    job_id: str
    query: str
    gemini_api_key: Optional[str] = None

class VocabularyUpdate(BaseModel):
    vocabulary: List[str]


# ---------------------------------------------------------------------------
# Engine initialization (for live stream tab)
# ---------------------------------------------------------------------------

def init_engine():
    global engine
    from src.detector.mock_detector import SyntheticDetector
    detector = SyntheticDetector()
    detector.set_vocabulary(current_vocabulary)
    logger.info("Live Demo Stream initialized with SyntheticDetector.")
    tracker = MultiObjectTracker(track_activation_threshold=0.25, lost_track_buffer=30, frame_rate=30)
    zones = [SpatialZoneMonitor(
        zone_name="Restricted Robotics Bay",
        polygon_points=[(180, 140), (460, 140), (460, 360), (180, 360)],
        dwell_time_threshold_sec=2.0,
    )]
    engine = OmniSightEngine(
        detector=detector, tracker=tracker, zones=zones,
        reasoning_engine=MockReasoningEngine(), visualizer=SceneVisualizer(),
    )


def make_scan_detector():
    """Create a fresh detector instance for video scanning jobs."""
    try:
        from src.detector import YOLOWorldDetector as _YW
        if _YW is None:
            raise RuntimeError("not available")
        return _YW(model_id="yolov8s-worldv2.pt", confidence_threshold=0.05, device="cpu", fp16=False)
    except Exception:
        from src.detector.mock_detector import SyntheticDetector
        return SyntheticDetector()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    try:
        init_engine()
        logger.info("OmniSight Engine initialized.")
    except Exception as e:
        logger.warning(f"Engine deferred: {e}")


# ---------------------------------------------------------------------------
# Serve HTML
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = Path("src/api/static/index.html")
    async with aiofiles.open(str(html_path), "r", encoding="utf-8") as f:
        return await f.read()


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept a video file upload and return a job_id.
    The client then calls /api/query with this job_id.
    """
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix or ".mp4"
    dest = UPLOAD_DIR / f"{job_id}{suffix}"

    async with aiofiles.open(str(dest), "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await out.write(chunk)

    # Get video info
    cap = cv2.VideoCapture(str(dest))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps
    cap.release()

    jobs[job_id] = {
        "status": "uploaded",
        "video_path": str(dest),
        "filename": file.filename,
        "duration_sec": round(duration, 1),
        "progress": 0,
        "message": "Video uploaded. Ready for query.",
        "results": None,
        "error": None,
    }

    logger.info(f"[{job_id}] Uploaded '{file.filename}' ({duration:.1f}s)")
    return {"job_id": job_id, "duration_sec": round(duration, 1), "filename": file.filename}


# ---------------------------------------------------------------------------
# Query endpoint — starts background scan
# ---------------------------------------------------------------------------

@app.post("/api/query")
async def start_query(req: QueryRequest, background_tasks: BackgroundTasks):
    """
    Parse the user's natural-language query and start an async video scan.
    Returns immediately; poll /api/status/{job_id} for progress.
    """
    if req.job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found. Upload a video first."})

    job = jobs[req.job_id]
    if job["status"] == "scanning":
        return JSONResponse(status_code=409, content={"error": "Scan already in progress."})

    # Parse query (fast — uses Gemini API)
    api_key = req.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    parser = create_query_parser(api_key=api_key)
    try:
        intent: QueryIntent = parser.parse(req.query)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Query parsing failed: {e}"})

    job["status"] = "scanning"
    job["intent"] = intent.model_dump()
    job["query"] = req.query
    job["progress"] = 0
    job["message"] = intent.human_summary

    background_tasks.add_task(_run_scan, req.job_id, job["video_path"], intent, api_key)

    return {
        "job_id": req.job_id,
        "status": "scanning",
        "human_summary": intent.human_summary,
        "detection_targets": intent.detection_targets,
    }


# ---------------------------------------------------------------------------
# Background scan task
# ---------------------------------------------------------------------------

async def _run_scan(job_id: str, video_path: str, intent: QueryIntent, api_key: Optional[str] = None):
    """
    The actual scan pipeline, run as a FastAPI background task.
    Streams progress updates to any connected WebSocket clients.
    """

    loop = asyncio.get_event_loop()

    def on_progress(pct: float, message: str):
        job["progress"] = round(pct, 1)
        job["message"] = message
        # Schedule the async broadcast on the main event loop from this thread
        asyncio.run_coroutine_threadsafe(
            _broadcast_job_progress(job_id, pct, message), loop
        )

    job = jobs[job_id]
    t_start = asyncio.get_event_loop().time()

    try:
        events = None
        extractor = ClipExtractor(output_dir=str(RESULTS_DIR), padding_sec=5.0)

        # Mode 1: If Gemini API key is provided, use multimodal video grounding (handles abstract concepts, logos, actions)
        if api_key:
            try:
                from src.scanner.gemini_grounder import GeminiVideoGrounder
                logger.info(f"[{job_id}] Running Gemini Multimodal Video Grounding...")
                grounder = GeminiVideoGrounder(api_key=api_key)
                events = await loop.run_in_executor(
                    None,
                    lambda: grounder.ground(video_path, job.get("query", ""), on_progress=on_progress)
                )
            except Exception as e:
                logger.warning(f"[{job_id}] Gemini grounding failed ({e}). Falling back to local scanner.")
                events = None

        # Mode 2: Local scanner (YOLO-World or detector)
        if events is None:
            detector = make_scan_detector()
            scanner = VideoScanner(detector=detector, scan_fps=4.0)
            aggregator = EventAggregator(gap_sec=3.0, top_k=5)

            # Run blocking scan in thread pool (passes sync on_progress)
            scan_result = await scanner.scan_async(video_path, intent, on_progress=on_progress)

            on_progress(95, "Aggregating events...")
            events = aggregator.aggregate(scan_result, intent)
            duration_sec = scan_result.video_duration_sec
        else:
            duration_sec = job.get("duration_sec", 0.0)

        on_progress(97, "Extracting clips...")
        clips = extractor.extract_all(video_path, events, job_id)

        scan_duration = round(asyncio.get_event_loop().time() - t_start, 1)
        results = {
            "query": job.get("query", ""),
            "human_summary": intent.human_summary,
            "detection_targets": intent.detection_targets,
            "video_duration_sec": duration_sec,
            "scan_duration_sec": scan_duration,
            "total_events_found": len(events),
            "clips": [c.to_dict() for c in clips],
        }

        job["status"] = "done"
        job["progress"] = 100
        job["message"] = f"Done! Found {len(clips)} relevant moment(s)."
        job["results"] = results

        await _broadcast_job_progress(job_id, 100, job["message"], results=results)
        logger.info(f"[{job_id}] Scan complete: {len(clips)} clips extracted.")

    except Exception as e:
        logger.exception(f"[{job_id}] Scan failed: {e}")
        job["status"] = "error"
        job["error"] = str(e)
        job["message"] = f"Error: {e}"
        await _broadcast_job_progress(job_id, -1, job["message"])


async def _broadcast_job_progress(job_id: str, pct: float, message: str, results=None):
    """Send a progress update to all WebSocket clients watching this job."""
    payload = json.dumps({
        "type": "progress",
        "job_id": job_id,
        "progress": pct,
        "message": message,
        "results": results,
    })
    for ws in list(job_ws_connections.get(job_id, [])):
        try:
            await ws.send_text(payload)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Status / Results endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = jobs[job_id]
    if job["status"] != "done":
        return JSONResponse(status_code=202, content={"status": job["status"], "message": job["message"]})
    return job["results"]


@app.api_route("/api/clips/{filename}", methods=["GET", "HEAD"])
async def serve_clip(filename: str):
    """Serve extracted clip video files for playback in the browser."""
    clip_path = RESULTS_DIR / filename
    if not clip_path.exists():
        return JSONResponse(status_code=404, content={"error": "Clip not found"})
    return FileResponse(
        str(clip_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@app.get("/api/thumbs/{filename}")
async def serve_thumb(filename: str):
    thumb_path = RESULTS_DIR / filename
    if not thumb_path.exists():
        return JSONResponse(status_code=404, content={"error": "Thumbnail not found"})
    return FileResponse(str(thumb_path), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Forensic Dossier Export (PDF & SIEM JSON)
# ---------------------------------------------------------------------------

@app.get("/api/dossier/{job_id}/pdf")
async def export_dossier_pdf(job_id: str):
    """Generate and return a court-ready cryptographic forensic PDF dossier."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = jobs[job_id]
    if job.get("status") != "done":
        return JSONResponse(status_code=400, content={"error": "Scan is still processing or incomplete."})

    from src.forensics.dossier_generator import ForensicDossierGenerator
    pdf_filename = f"OmniSight_Dossier_{job_id}.pdf"
    pdf_path = RESULTS_DIR / pdf_filename

    generator = ForensicDossierGenerator(output_dir=str(RESULTS_DIR))
    generator.generate_pdf(job, str(pdf_path))

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=pdf_filename,
        headers={"Content-Disposition": f"attachment; filename={pdf_filename}"}
    )


@app.get("/api/dossier/{job_id}/json")
async def export_dossier_json(job_id: str):
    """Export structured incident intelligence JSON for SIEM/Splunk ingestion."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    job = jobs[job_id]
    from src.forensics.dossier_generator import compute_file_sha256
    video_p = job.get("video_path", "")
    sha256 = compute_file_sha256(video_p) if Path(video_p).exists() else None

    dossier = {
        "case_id": job_id,
        "classification": "RESTRICTED // EVIDENTIARY",
        "sha256_hash": sha256,
        "ingestion_metadata": {
            "source_file": job.get("filename"),
            "duration_sec": job.get("duration_sec"),
            "query": job.get("query"),
        },
        "results": job.get("results"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(content=dossier)


# ---------------------------------------------------------------------------
# WebSocket: real-time progress for a specific job
# ---------------------------------------------------------------------------

@app.websocket("/ws/progress/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in job_ws_connections:
        job_ws_connections[job_id] = []
    job_ws_connections[job_id].append(websocket)
    logger.info(f"[{job_id}] WebSocket progress client connected.")

    try:
        # Send current state immediately on connect
        if job_id in jobs:
            job = jobs[job_id]
            await websocket.send_text(json.dumps({
                "type": "progress",
                "job_id": job_id,
                "progress": job["progress"],
                "message": job["message"],
                "results": job.get("results"),
            }))
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in job_ws_connections:
            job_ws_connections[job_id] = [
                ws for ws in job_ws_connections[job_id] if ws != websocket
            ]


# ---------------------------------------------------------------------------
# Original live-stream WebSocket (demo tab, unchanged)
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to live stream WebSocket.")
    cap = cv2.VideoCapture("data/synthetic_sample.mp4")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            if engine:
                result: FrameResult = engine.process_frame(frame, frame_index=frame_idx, timestamp=frame_idx / 30.0)
                frame_idx += 1
                annotated = result.annotated_frame
                for inc in result.incidents:
                    inc_dict = {
                        "event_type": inc.event_type,
                        "severity": inc.severity.value,
                        "description": inc.description,
                        "action": inc.recommended_action,
                        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    }
                    incident_history.append(inc_dict)
                    await websocket.send_text(json.dumps({"type": "incident", "data": inc_dict}))
            else:
                annotated = frame

            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        logger.info("Live stream client disconnected.")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Vocabulary API (for live stream tab)
# ---------------------------------------------------------------------------

@app.get("/api/vocabulary")
async def get_vocabulary():
    return {"vocabulary": current_vocabulary}

@app.post("/api/vocabulary")
async def update_vocabulary(update: VocabularyUpdate):
    global current_vocabulary, engine
    current_vocabulary = update.vocabulary
    if engine:
        engine.set_vocabulary(current_vocabulary)
    return {"status": "success", "vocabulary": current_vocabulary}

@app.get("/api/incidents")
async def get_incidents():
    return {"incidents": incident_history[-50:]}


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
