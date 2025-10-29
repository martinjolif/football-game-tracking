import io
import time

from PIL import Image
from ultralytics import YOLO

from config import INFERENCE_MODEL_PATH
from schemas import InferenceResponse

model = YOLO(INFERENCE_MODEL_PATH)

def detect_players_in_image(image_bytes: bytes) -> InferenceResponse:
    """Perform player detection on a single image."""
    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("The input data is not a valid image") from e

    start_time = time.time()
    results = model(image_pil)
    end_time = time.time()
    inference_time = end_time - start_time
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "detected_class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3]),
                }
            })

    return {
        "detections": detections,
        "mapping_class_dict": model.model.names,
        "inference_time": inference_time
    }
