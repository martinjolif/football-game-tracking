from fastapi import APIRouter, UploadFile, File

from inference import detect_players_in_image
from schemas import InferenceResponse

router = APIRouter(prefix="/player-detection")

@router.post("/image", response_model=InferenceResponse)
async def player_detection_endpoint(file: UploadFile = File(...)):
    """Endpoint to perform player detection on an uploaded image."""
    image_bytes = await file.read()
    results = detect_players_in_image(image_bytes)
    return results


#TODO
##@app.post("/player-detection/video")
