import asyncio

from fastapi import APIRouter, UploadFile, File, HTTPException

from player_detection.api.config import ALLOWED_IMG_TYPES, MAX_FILE_SIZE
from player_detection.api.inference import detect_players_in_image
from utils.schemas import DetectionInferenceResponse

router = APIRouter(prefix="/player-detection")

@router.post("/image", response_model=DetectionInferenceResponse)
async def player_detection_endpoint(file: UploadFile = File(...)):
    """Endpoint to perform player detection on an uploaded image."""
    if file.content_type not in ALLOWED_IMG_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG are allowed.")
    # Check file size without reading into memory
    file.file.seek(0, 2)  # Move pointer to end of file
    size = file.file.tell()  # Get current position = file size
    file.file.seek(0)  # Reset pointer to start
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max size is 5MB.")
    image_bytes = await file.read()
    loop = asyncio.get_running_loop()
    try:
        # Run the synchronous detection function in a separate thread to avoid blocking the event loop
        results = await loop.run_in_executor(None, detect_players_in_image, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during player detection: {e}")
    return results


#TODO
##@app.post("/player-detection/video")
