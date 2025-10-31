from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x0: float = Field(..., description="Horizontal position of the bottom left corner of the box")
    y0: float = Field(..., description="Vertical position of the bottom left corner of the box")
    x1: float = Field(..., description="Horizontal position of the top right corner of the box")
    y1: float = Field(..., description="Vertical position of the top right corner of the box")

class Keypoint(BaseModel):
    x: float = Field(..., description="X coordinate of the keypoint (pixels or normalized)")
    y: float = Field(..., description="Y coordinate of the keypoint (pixels or normalized)")
    score: float = Field(..., description="Confidence score of the keypoint", ge=0, le=1, example=0.98)

# Schemas for object detection
class Detection(BaseModel):
    detected_class_id: int = Field(..., description="Detected class number", example=0)
    # Confidence between 0 and 1
    confidence: float = Field(..., description="Confidence score of the detection", example=0.92, ge=0, le=1)
    bbox: BoundingBox = Field(..., description="Bounding box coordinates of the detection")

class DetectionInferenceResponse(BaseModel):
    detections: list[Detection]
    mapping_class: dict[int, str]
    inference_time: float = Field(..., description="Time taken for inference in seconds", ge=0, example=0.123)

# Schemas for pose estimation
class Pose(BaseModel):
    detected_class_id: int = Field(..., description="Detected class id (e.g. person)", example=0)
    confidence: float = Field(..., description="Overall confidence for the pose detection", ge=0, le=1, example=0.95)
    keypoints: list[Keypoint] = Field(..., description="Ordered list of keypoints for the pose")
    bbox: BoundingBox = Field(..., description="Bounding box surrounding the detected instance")

class PoseInferenceResponse(BaseModel):
    poses: list[Pose] = Field(..., description="List of detected poses")
    mapping_class: dict[int, str] = Field(..., description="Mapping from keypoint index to name (e.g. 0 -> nose)")
    inference_time: float = Field(..., description="Inference time in seconds", ge=0, example=0.123)
