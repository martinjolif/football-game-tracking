from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import shutil
from typing import Optional
import os
from datetime import datetime, timezone

app = FastAPI(title="Football Video Analysis API")

# Detection routers
from src.player_detection.api.router import router as player_detection_router
from src.ball_detection.api.router import router as ball_detection_router
from src.pitch_detection.api.router import router as pitch_detection_router

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
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# Store job status
job_status = {}
cancelled_jobs: set = set()

EXAMPLE_VIDEO = FRONTEND_DIR / "08fd33_4.mp4"


@app.get("/")
async def serve_frontend():
    """Serve the main HTML front-end"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"error": "Front-end not found"}, status_code=404)


@app.get("/example-video")
async def get_example_video():
    """Serve the example video file"""
    if EXAMPLE_VIDEO.exists():
        return FileResponse(EXAMPLE_VIDEO, media_type="video/mp4", filename="08fd33_4.mp4")
    return JSONResponse({"error": "Example video not found"}, status_code=404)


@app.post("/upload-example")
async def upload_example(
        background_tasks: BackgroundTasks,
        enable_radar: bool = Form(True),
        enable_commentary: bool = Form(True),
        enable_tracking: bool = Form(True),
        enable_team_clustering: bool = Form(True),
        enable_tts: bool = Form(True),
        end_frame: Optional[int] = Form(None),
        cluster_train_frames: int = Form(50),
):
    """Start processing with the built-in example video"""
    if not EXAMPLE_VIDEO.exists():
        return JSONResponse({"error": "Example video not found"}, status_code=404)

    job_id = str(uuid.uuid4())

    # Copy example video to uploads
    video_path = UPLOAD_DIR / f"{job_id}_08fd33_4.mp4"
    shutil.copy2(EXAMPLE_VIDEO, video_path)

    job_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Example video loaded, starting processing...",
        "original_filename": "08fd33_4.mp4",
    }

    background_tasks.add_task(
        process_video,
        job_id=job_id,
        video_path=str(video_path),
        enable_radar=enable_radar,
        enable_commentary=enable_commentary,
        enable_tracking=enable_tracking,
        enable_team_clustering=enable_team_clustering,
        enable_tts=enable_tts,
        end_frame=end_frame,
        cluster_train_frames=cluster_train_frames,
    )

    return {
        "job_id": job_id,
        "message": "Example video processing started",
        "status_url": f"/status/{job_id}"
    }


@app.post("/upload")
async def upload_video(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        enable_radar: bool = Form(True),
        enable_commentary: bool = Form(True),
        enable_tracking: bool = Form(True),
        enable_team_clustering: bool = Form(True),
        enable_tts: bool = Form(True),
        end_frame: Optional[int] = Form(None),
        cluster_train_frames: int = Form(50),
):
    """Upload video and start processing"""

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded video
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Initialize job status
    job_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Video uploaded successfully",
        "original_filename": video.filename,
    }

    # Add processing task to background
    background_tasks.add_task(
        process_video,
        job_id=job_id,
        video_path=str(video_path),
        enable_radar=enable_radar,
        enable_commentary=enable_commentary,
        enable_tracking=enable_tracking,
        enable_team_clustering=enable_team_clustering,
        enable_tts=enable_tts,
        end_frame=end_frame,
        cluster_train_frames=cluster_train_frames,
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

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4"
    )


@app.post("/demo")
async def demo_video(background_tasks: BackgroundTasks):
    """Start processing with the built-in demo video using default options"""
    if not EXAMPLE_VIDEO.exists():
        return JSONResponse({"error": "Demo video not found"}, status_code=404)

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_08fd33_4.mp4"
    shutil.copy2(EXAMPLE_VIDEO, video_path)

    job_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Demo video loaded, starting processing...",
        "original_filename": "08fd33_4.mp4",
    }

    background_tasks.add_task(
        process_video,
        job_id=job_id,
        video_path=str(video_path),
        enable_radar=True,
        enable_commentary=True,
        enable_tracking=True,
        enable_team_clustering=True,
        enable_tts=True,
        end_frame=None,
        cluster_train_frames=50,
    )

    return {"job_id": job_id, "status_url": f"/status/{job_id}"}


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running processing job"""
    if job_id not in job_status:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job_status[job_id]["status"] not in ("queued", "processing"):
        return JSONResponse({"error": "Job is not running"}, status_code=400)
    cancelled_jobs.add(job_id)
    job_status[job_id]["status"] = "cancelled"
    job_status[job_id]["message"] = "Processing cancelled by user"
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/videos")
async def list_videos():
    """Return list of completed processed videos"""
    completed = [
        {"job_id": job_id, **{k: v for k, v in info.items() if k != "status" or True}}
        for job_id, info in job_status.items()
        if info.get("status") == "completed"
    ]
    completed.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
    return completed


def process_video(
        job_id: str,
        video_path: str,
        enable_radar: bool,
        enable_commentary: bool,
        enable_tracking: bool,
        enable_team_clustering: bool,
        enable_tts: bool,
        end_frame: Optional[int],
        cluster_train_frames: int,
):
    """Background task to process video"""
    from src.final_app.video_processor import VideoProcessor

    try:
        job_status[job_id]["status"] = "processing"
        job_status[job_id]["message"] = "Processing video..."

        processor = VideoProcessor(
            video_path=video_path,
            output_path=str(OUTPUT_DIR / f"{job_id}_output.mp4"),
            enable_radar=enable_radar,
            enable_commentary=enable_commentary,
            enable_tracking=enable_tracking,
            enable_team_clustering=enable_team_clustering,
            enable_tts=enable_tts,
            end_frame=end_frame,
            cluster_train_frames=cluster_train_frames,
            progress_callback=lambda progress, message: update_progress(job_id, progress, message),
            cancel_check=lambda: job_id in cancelled_jobs,
        )

        processor.process()

        if job_id in cancelled_jobs:
            cancelled_jobs.discard(job_id)
            job_status[job_id]["status"] = "cancelled"
            job_status[job_id]["message"] = "Processing cancelled by user"
        else:
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["progress"] = 100
            job_status[job_id]["message"] = "Processing completed"
            job_status[job_id]["download_url"] = f"/download/{job_id}"
            job_status[job_id]["stream_url"] = f"/outputs/{job_id}_output.mp4"
            job_status[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "")

    except InterruptedError:
        cancelled_jobs.discard(job_id)
        job_status[job_id]["status"] = "cancelled"
        job_status[job_id]["message"] = "Processing cancelled by user"

    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["message"] = f"Error: {str(e)}"

    finally:
        cancelled_jobs.discard(job_id)
        # Clean up uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)


def update_progress(job_id: str, progress: int, message: str):
    """Update job progress"""
    if job_id in job_status:
        job_status[job_id]["progress"] = progress
        job_status[job_id]["message"] = message