from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import shutil
from typing import Optional
import os
import json
import subprocess
from datetime import datetime

from src.player_detection.api.router import router as player_detection_router
from src.ball_detection.api.router import router as ball_detection_router
from src.pitch_detection.api.router import router as pitch_detection_router

app = FastAPI(title="Football Video Analysis API")

app.include_router(player_detection_router)
app.include_router(ball_detection_router)
app.include_router(pitch_detection_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
FRONTEND_DIR = Path("src/final_app/frontend")
DEMO_DIR = Path("demo")
JOBS_FILE = OUTPUT_DIR / "jobs.json"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR)), name="demo")

# Store job status (loaded from disk on startup)
job_status = {}


def load_jobs():
    """Load persisted jobs from disk"""
    if JOBS_FILE.exists():
        try:
            with JOBS_FILE.open("r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_jobs():
    """Persist job metadata to disk"""
    try:
        with JOBS_FILE.open("w") as f:
            json.dump(job_status, f, indent=2)
    except Exception:
        pass


# Load existing jobs on startup
job_status.update(load_jobs())


@app.get("/")
async def serve_frontend():
    """Serve the main HTML front-end"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"error": "Front-end not found"}, status_code=404)


@app.get("/videos")
async def list_videos():
    """List all processed videos (from known jobs + orphaned output files)"""
    videos = []
    seen_job_ids = set()

    for job_id, info in job_status.items():
        if info.get("status") == "completed":
            output_path = OUTPUT_DIR / f"{job_id}_output.mp4"
            if output_path.exists():
                seen_job_ids.add(job_id)
                videos.append({
                    "job_id": job_id,
                    "original_filename": info.get("original_filename", "unknown"),
                    "created_at": info.get("created_at"),
                    "completed_at": info.get("completed_at"),
                    "download_url": f"/download/{job_id}",
                    "stream_url": f"/outputs/{job_id}_output.mp4",
                })

    # Discover output files with no matching job entry
    for output_path in OUTPUT_DIR.glob("*_output.mp4"):
        stem = output_path.stem  # e.g. "abc123_output"
        job_id = stem[: -len("_output")]
        if job_id not in seen_job_ids:
            stat = output_path.stat()
            completed_at = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
            videos.append({
                "job_id": job_id,
                "original_filename": "unknown",
                "created_at": None,
                "completed_at": completed_at,
                "download_url": f"/download/{job_id}",
                "stream_url": f"/outputs/{job_id}_output.mp4",
            })
            # Register in job_status so /download works
            job_status[job_id] = {"status": "completed"}

    # Sort by completed_at descending (most recent first)
    videos.sort(key=lambda v: v.get("completed_at") or "", reverse=True)
    return videos


@app.post("/demo")
async def process_demo_video(background_tasks: BackgroundTasks):
    """Start processing the demo video"""
    demo_video = DEMO_DIR / "commentary.mp4"
    if not DEMO_DIR.exists() or not demo_video.exists():
        return JSONResponse(status_code=404, content={"error": "Demo video not found"})

    job_id = str(uuid.uuid4())
    demo_copy = UPLOAD_DIR / f"{job_id}_demo.mp4"
    shutil.copy(demo_video, demo_copy)

    job_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Demo video queued",
        "original_filename": "demo_commentary.mp4",
        "created_at": datetime.utcnow().isoformat(),
    }
    save_jobs()

    background_tasks.add_task(
        process_video,
        job_id=job_id,
        video_path=str(demo_copy),
        enable_radar=True,
        enable_commentary=True,
        enable_tracking=True,
        enable_team_clustering=True,
        end_frame=None,
        cluster_train_frames=50,
        enable_tts=False,
    )

    return {
        "job_id": job_id,
        "message": "Demo video processing started",
        "status_url": f"/status/{job_id}",
    }


@app.post("/upload")
async def upload_video(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        enable_radar: bool = Form(True),
        enable_commentary: bool = Form(True),
        enable_tracking: bool = Form(True),
        enable_team_clustering: bool = Form(True),
        end_frame: Optional[int] = Form(None),
        cluster_train_frames: int = Form(50),
        enable_tts: bool = Form(False),
):
    """Upload video and start processing"""

    job_id = str(uuid.uuid4())

    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    job_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Video uploaded successfully",
        "original_filename": video.filename,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_jobs()

    background_tasks.add_task(
        process_video,
        job_id=job_id,
        video_path=str(video_path),
        enable_radar=enable_radar,
        enable_commentary=enable_commentary,
        enable_tracking=enable_tracking,
        enable_team_clustering=enable_team_clustering,
        end_frame=end_frame,
        cluster_train_frames=cluster_train_frames,
        enable_tts=enable_tts,
    )

    return {
        "job_id": job_id,
        "message": "Video processing started",
        "status_url": f"/status/{job_id}"
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status"""
    if job_id not in job_status:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )

    return job_status[job_id]


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download processed video"""
    if job_id not in job_status:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )

    if job_status[job_id]["status"] != "completed":
        return JSONResponse(
            status_code=400,
            content={"error": "Video processing not completed"}
        )

    output_path = OUTPUT_DIR / f"{job_id}_output.mp4"
    if not output_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Output video not found"}
        )

    original_name = job_status[job_id].get("original_filename", "video.mp4")
    stem = Path(original_name).stem
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"processed_{stem}.mp4"
    )


def convert_to_h264(input_path: str, output_path: str) -> bool:
    """Convert video to H.264 for browser compatibility using ffmpeg"""
    tmp_path = input_path + ".tmp.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-c:a", "copy",
                tmp_path,
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode == 0:
            shutil.move(tmp_path, output_path)
            return True
        return False
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_video(
        job_id: str,
        video_path: str,
        enable_radar: bool,
        enable_commentary: bool,
        enable_tracking: bool,
        enable_team_clustering: bool,
        end_frame: Optional[int],
        cluster_train_frames: int,
        enable_tts: bool = False,
):
    """Background task to process video"""
    from src.final_app.video_processor import VideoProcessor

    raw_output = str(OUTPUT_DIR / f"{job_id}_raw.mp4")
    final_output = str(OUTPUT_DIR / f"{job_id}_output.mp4")

    try:
        job_status[job_id]["status"] = "processing"
        job_status[job_id]["message"] = "Processing video..."
        save_jobs()

        processor = VideoProcessor(
            video_path=video_path,
            output_path=raw_output,
            enable_radar=enable_radar,
            enable_commentary=enable_commentary,
            enable_tracking=enable_tracking,
            enable_team_clustering=enable_team_clustering,
            end_frame=end_frame,
            cluster_train_frames=cluster_train_frames,
            enable_tts=enable_tts,
            progress_callback=lambda progress, message: update_progress(job_id, progress, message)
        )

        processor.process()

        # Convert to H.264 for browser compatibility
        job_status[job_id]["message"] = "Converting video for browser playback..."
        save_jobs()
        converted = convert_to_h264(raw_output, final_output)
        if not converted:
            # Fallback: use raw output as-is
            shutil.move(raw_output, final_output)
        elif os.path.exists(raw_output):
            os.remove(raw_output)

        job_status[job_id]["status"] = "completed"
        job_status[job_id]["progress"] = 100
        job_status[job_id]["message"] = "Processing completed"
        job_status[job_id]["download_url"] = f"/download/{job_id}"
        job_status[job_id]["completed_at"] = datetime.utcnow().isoformat()
        save_jobs()

    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["message"] = f"Error: {str(e)}"
        save_jobs()

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(raw_output):
            os.remove(raw_output)


def update_progress(job_id: str, progress: int, message: str):
    """Update job progress"""
    if job_id in job_status:
        job_status[job_id]["progress"] = progress
        job_status[job_id]["message"] = message