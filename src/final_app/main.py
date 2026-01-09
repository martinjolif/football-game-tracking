from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import shutil
from typing import Optional
import os

app = FastAPI(title="Football Video Analysis API")

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
FRONTEND_DIR = Path("src/final_app/frontend")  # <-- folder for your front-end
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)  # create if it doesn't exist

# Mount static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# Store job status
job_status = {}


@app.get("/")
async def serve_frontend():
    """Serve the main HTML front-end"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"error": "Front-end not found"}, status_code=404)


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
        "message": "Video uploaded successfully"
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


def process_video(
        job_id: str,
        video_path: str,
        enable_radar: bool,
        enable_commentary: bool,
        enable_tracking: bool,
        enable_team_clustering: bool,
        end_frame: Optional[int],
        cluster_train_frames: int,
):
    """Background task to process video"""
    from video_processor import VideoProcessor

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
            end_frame=end_frame,
            cluster_train_frames=cluster_train_frames,
            progress_callback=lambda progress, message: update_progress(job_id, progress, message)
        )

        processor.process()

        job_status[job_id]["status"] = "completed"
        job_status[job_id]["progress"] = 100
        job_status[job_id]["message"] = "Processing completed"
        job_status[job_id]["download_url"] = f"/download/{job_id}"

    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["message"] = f"Error: {str(e)}"

    finally:
        # Clean up uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)


def update_progress(job_id: str, progress: int, message: str):
    """Update job progress"""
    if job_id in job_status:
        job_status[job_id]["progress"] = progress
        job_status[job_id]["message"] = message