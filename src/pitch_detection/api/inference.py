import io
import time

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from src.pitch_detection.api.config import INFERENCE_MODEL_PATH, DEVICE
from src.utils.schemas import PoseInferenceResponse, Pose, BoundingBox, Keypoint

model = YOLO(INFERENCE_MODEL_PATH)
#model.export(format="onnx")  # Export the model to ONNX format
#TODO: Load the ONNX model for inference instead of the original YOLO model

def detect_pitch_in_image(image_bytes: bytes) -> PoseInferenceResponse:
    """Perform pitch detection on a single image."""
    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("The input data is not a valid image") from e

    start_time = time.perf_counter()
    results = model(image_pil, device=DEVICE)
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    poses = []
    for result in results:
        for box in result.boxes:
            poses.append(Pose(
                detected_class_id=int(box.cls[0]),
                confidence=float(box.conf[0]),
                keypoints=[
                    Keypoint(
                        x=float(xy[0]),
                        y=float(xy[1]),
                        score=float(conf.item())
                    ) for xy, conf in zip(result.keypoints.xy[0], result.keypoints.conf[0])
                ],
                bbox=BoundingBox(
                    x0=float(box.xyxy[0][0]),
                    y0=float(box.xyxy[0][1]),
                    x1=float(box.xyxy[0][2]),
                    y1=float(box.xyxy[0][3]),
                )
            ))

    return PoseInferenceResponse(
        poses=poses,
        mapping_class=model.model.names,
        inference_time=inference_time
    )