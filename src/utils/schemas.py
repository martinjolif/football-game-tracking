from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x0: float = Field(..., description="Horizontal position of the bottom left corner of the box")
    y0: float = Field(..., description="Vertical position of the bottom left corner of the box")
    x1: float = Field(..., description="Horizontal position of the top right corner of the box")
    y1: float = Field(..., description="Vertical position of the top right corner of the box")

class Detection(BaseModel):
    detected_class_id: int = Field(..., description="Detected class number", example=0)
    # Confidence between 0 and 1
    confidence: float = Field(..., description="Confidence score of the detection", example=0.92, ge=0, le=1)
    bbox: BoundingBox = Field(..., description="Bounding box coordinates of the detection")

class InferenceResponse(BaseModel):
    detections: list[Detection]
    mapping_class: dict[int, str]
    inference_time: float = Field(..., description="Time taken for inference in seconds", ge=0, example=0.123)