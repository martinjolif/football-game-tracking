from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    detected_class_id: int = Field(..., description="Detected class number", example=0)
    # Confidence between 0 and 1
    confidence: float = Field(..., description="Confidence score of the detection", example=0.92, ge=0, le=1)
    bbox: BoundingBox = Field(..., description="Bounding box coordinates of the detection")

class InferenceResponse(BaseModel):
    detections: list[Detection]
    mapping_class_dict: dict[int, str]
    inference_time: float = Field(..., description="Time taken for inference in seconds", example=0.123)