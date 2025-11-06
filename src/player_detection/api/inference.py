import io
import time

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from src.player_detection.api.config import INFERENCE_MODEL_PATH
from src.utils.schemas import DetectionInferenceResponse, BoundingBox, Detection

model = YOLO(INFERENCE_MODEL_PATH)
#model.export(format="onnx")  # Export the model to ONNX format
#TODO: Load the ONNX model for inference instead of the original YOLO model

def detect_players_in_image(image_bytes: bytes) -> DetectionInferenceResponse:
    """Perform player detection on a single image."""
    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("The input data is not a valid image") from e

    start_time = time.perf_counter()
    results = model(image_pil)
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append(Detection(
                detected_class_id=int(box.cls[0]),
                confidence=float(box.conf[0]),
                bbox=BoundingBox(
                    x0=float(box.xyxy[0][0]),
                    y0=float(box.xyxy[0][1]),
                    x1=float(box.xyxy[0][2]),
                    y1=float(box.xyxy[0][3]),
                )
            ))

    return DetectionInferenceResponse(
        detections=detections,
        mapping_class=model.model.names,
        inference_time=inference_time
    )
