import torch

INFERENCE_MODEL_PATH = "weights/ball_detection/football-ball-detection-yolov8n2/weights/best.pt"  # Path to the trained model for inference
ALLOWED_IMG_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Mo
if torch.cuda.is_available():
    DEVICE = "0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"