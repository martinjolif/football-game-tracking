import torch

INFERENCE_MODEL_PATH = "weights/player_detection/football-player-detection-yolo11m/weights/best.pt"  # Path to the trained model for inference
ALLOWED_IMG_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Mo
if torch.cuda.is_available():
    DEVICE = "0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CONFIDENCE_THRESHOLD = 0.5 #default parameter from YOLO 0.25
IOU_THRESHOLD = 0.7 #default parameter from YOLO 0.7
INFERENCE_IMG_SIZE = 1280