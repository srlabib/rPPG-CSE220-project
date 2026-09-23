"""
FastAPI Server for rPPG Localhost Frontend.

Provides:
- Video streaming via MJPEG (/api/video_feed)
- High-frequency live telemetry via WebSocket (/ws/telemetry)
- REST endpoints for source switching, playback control, recording, and CSV export
- Static asset serving for the modern dashboard UI
"""

import asyncio
import os
import shutil
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import cv2 as cv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web_pipeline import WebRPPGPipeline


# Resolve directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Global pipeline instance initialized in standby mode (idle until user selects a source)
pipeline = WebRPPGPipeline(initial_source=None)

_standby_jpeg_cache = None

def get_standby_jpeg() -> bytes:
    """Generates an aesthetic dark-theme standby frame when no video source is active."""
    global _standby_jpeg_cache
    if _standby_jpeg_cache is None:
        img = np.full((360, 640, 3), (16, 10, 6), dtype=np.uint8)
        # Background subtle grid pattern
        for x in range(0, 640, 40):
            cv.line(img, (x, 0), (x, 360), (28, 18, 12), 1)
        for y in range(0, 360, 40):
            cv.line(img, (0, y), (640, y), (28, 18, 12), 1)
        # Standby text
        cv.putText(img, "STANDBY MODE", (210, 160), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 240, 255), 2, cv.LINE_AA)
        cv.putText(img, "Select a video from data/ or choose a webcam to begin", (90, 205), cv.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv.LINE_AA)
        _, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 80])
        _standby_jpeg_cache = buf.tobytes()
    return _standby_jpeg_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for clean shutdown. Does NOT auto-start processing without user selection."""
    print("[INFO] PulseVision server started in STANDBY mode. Awaiting user video/camera selection.")
    yield
    print("[INFO] Shutting down rPPG Pipeline...")
    pipeline.stop()


app = FastAPI(
    title="PulseVision rPPG Studio",
    description="Real-time remote photoplethysmography heart rate extraction server",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SettingsPayload(BaseModel):
    show_overlay: Optional[bool] = None
    show_hud: Optional[bool] = None
    roi_alpha: Optional[float] = None
    jpeg_quality: Optional[int] = None
    loop_video: Optional[bool] = None


class StartSessionPayload(BaseModel):
    source: Optional[str] = None


# --- Static and Web UI Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main application dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>rPPG Studio UI is loading...</h1>")


# Mount /static directory
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Streaming & WebSocket Endpoints ---

def mjpeg_frame_generator():
    """Generates continuous MJPEG multipart frame boundaries."""
    while True:
        if pipeline.is_running:
            jpeg_bytes = pipeline.get_latest_jpeg()
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" +
                    jpeg_bytes +
                    b"\r\n"
                )
                time.sleep(0.025)  # Cap at ~40 FPS stream rate to conserve bandwidth
            else:
                time.sleep(0.05)
        else:
            # When idle/standby, emit standby frame at low frame rate
            standby_bytes = get_standby_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(standby_bytes)).encode() + b"\r\n\r\n" +
                standby_bytes +
                b"\r\n"
            )
            time.sleep(0.4)


@app.get("/api/video_feed")
async def video_feed():
    """MJPEG streaming endpoint for HTML <img> tag."""
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """High-frequency bi-directional telemetry and command channel."""
    await websocket.accept()
    print("[INFO] WebSocket client connected.")
    try:
        while True:
            # Send latest telemetry
            telemetry = pipeline.get_telemetry()
            await websocket.send_json(telemetry)

            # Check if client sent any command without blocking
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.033)
                action = msg.get("action")
                if action == "pause":
                    pipeline.pause()
                elif action == "reset":
                    pipeline.reset()
                elif action == "record_toggle":
                    if pipeline.is_recording:
                        pipeline.stop_recording()
                    else:
                        pipeline.start_recording()
                elif action == "set_source":
                    if "source" in msg:
                        pipeline.set_source(msg["source"])
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        print("[INFO] WebSocket client disconnected.")
    except Exception as e:
        print(f"[WARN] WebSocket error: {e}")


# --- REST API Endpoints ---

def scan_data_folder() -> List[Dict[str, Any]]:
    """
    Recursively scans the data/ folder and all its subfolders for supported video files.
    Returns metadata for each video including relative path, subfolder, size, and date.
    """
    valid_extensions = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    video_files = []

    if os.path.exists(DATA_DIR):
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(root, f)
                    rel_to_base = os.path.relpath(full_path, BASE_DIR).replace("\\", "/")
                    rel_to_data = os.path.relpath(full_path, DATA_DIR).replace("\\", "/")
                    subfolder = os.path.dirname(rel_to_data).replace("\\", "/")
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    mtime = os.path.getmtime(full_path)
                    
                    video_files.append({
                        "name": f,
                        "path": rel_to_base,
                        "data_path": rel_to_data,
                        "subfolder": subfolder if subfolder else "Root",
                        "size_mb": round(size_mb, 1),
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
                        "is_upload": "uploads" in rel_to_base.split("/"),
                    })

    # Sort so regular data/ videos come first, then uploads, sorted alphabetically
    video_files.sort(key=lambda x: (x["is_upload"], x["subfolder"], x["name"]))
    return video_files


@app.get("/api/sources")
async def get_available_sources():
    """Discovers available local webcam devices and all videos inside the data/ folder."""
    video_files = scan_data_folder()

    cameras = [
        {"id": "0", "name": "Camera 0 (Default Integrated / USB)"},
        {"id": "1", "name": "Camera 1 (Secondary / DroidCam / Virtual)"},
        {"id": "2", "name": "Camera 2 (External / Capture Card)"},
    ]

    return {
        "cameras": cameras,
        "files": video_files,
        "current_source": pipeline.source,
    }


@app.get("/api/data/browse")
async def browse_data_folder():
    """Returns a full recursive list of video files discovered in data/ and subfolders."""
    files = scan_data_folder()
    subfolders = sorted(list({f["subfolder"] for f in files}))
    return {
        "total_files": len(files),
        "subfolders": subfolders,
        "files": files,
        "current_source": pipeline.source,
    }


class ValidatePathPayload(BaseModel):
    path: str


@app.post("/api/data/validate")
async def validate_video_path(payload: ValidatePathPayload):
    """Validates if a specified video path exists and can be opened by OpenCV."""
    test_path = payload.path.strip()

    # If relative path without 'data/', check if it exists in data/
    candidates = [
        test_path,
        os.path.join(DATA_DIR, test_path),
        os.path.join(BASE_DIR, test_path),
    ]

    resolved_path = None
    for cand in candidates:
        if os.path.isfile(cand):
            resolved_path = os.path.relpath(cand, BASE_DIR).replace("\\", "/")
            break

    if not resolved_path:
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found at '{test_path}'. Please check filename or path inside data/."
        )

    # Test open with OpenCV
    cap = cv.VideoCapture(resolved_path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail=f"OpenCV could not decode video file: '{resolved_path}'")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "valid": True,
        "path": resolved_path,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 1),
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 1),
    }



@app.post("/api/session/start")
async def start_session(payload: Optional[StartSessionPayload] = None):
    """Starts or switches the rPPG processing session."""
    if payload and payload.source:
        pipeline.set_source(payload.source)
    else:
        if not pipeline.is_running:
            pipeline.start()
        pipeline.pause(False)
    return {"status": "ok", "source": pipeline.source, "is_running": pipeline.is_running}


@app.post("/api/session/pause")
async def pause_session():
    """Toggles or sets the pause state."""
    pipeline.pause()
    return {"status": "ok", "is_paused": pipeline.is_paused}


@app.post("/api/session/stop")
async def stop_session():
    """Stops the rPPG processing pipeline and enters standby mode."""
    pipeline.stop()
    return {"status": "ok", "message": "Pipeline stopped, standby mode active"}


@app.post("/api/session/reset")
async def reset_session():
    """Resets the rPPG buffer and vitals."""
    pipeline.reset()
    return {"status": "ok", "message": "Pipeline buffers reset"}


@app.post("/api/session/record/toggle")
async def toggle_recording():
    """Toggles timestamped data recording."""
    if pipeline.is_recording:
        count = pipeline.stop_recording()
        return {"is_recording": False, "recorded_samples": count}
    else:
        pipeline.start_recording()
        return {"is_recording": True, "recorded_samples": 0}


@app.get("/api/session/export")
async def export_session_csv():
    """Exports recorded rPPG time-series as a CSV file."""
    csv_content = pipeline.export_csv()
    filename = f"rppg_recording_{int(time.time())}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video file and makes it available for immediate processing."""
    valid_exts = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .avi, .mov, .mkv, .webm")

    safe_name = f"upload_{int(time.time())}_{file.filename}"
    save_path = os.path.join(UPLOADS_DIR, safe_name)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rel_path = os.path.relpath(save_path, BASE_DIR).replace("\\", "/")
    return {
        "status": "ok",
        "filename": file.filename,
        "path": rel_path,
        "size_mb": round(os.path.getsize(save_path) / (1024 * 1024), 2)
    }


@app.post("/api/settings")
async def update_settings(settings: SettingsPayload):
    """Updates runtime parameters."""
    pipeline.update_settings(settings.model_dump(exclude_unset=True))
    return {
        "status": "ok",
        "settings": {
            "show_overlay": pipeline.show_overlay,
            "show_hud": pipeline.show_hud,
            "roi_alpha": pipeline.roi_alpha,
            "jpeg_quality": pipeline.jpeg_quality,
            "loop_video": pipeline.loop_video,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
