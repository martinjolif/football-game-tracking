import torch

# Global configuration settings for training and evaluating a YOLO object detection model
IMG_SIZE = 1280
if torch.cuda.is_available():
    DEVICE = "0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
MODEL_NAME = "yolo11m" # Pretrained YOLO model name
DATA_ROOT = "training/player_detection/data" # Root directory for dataset
MODEL_DIR = "training/player_detection/models" # Directory to save trained models

# Training configuration parameters for YOLO object tracking model
TRAIN_BATCH_SIZE = 4
EPOCHS = 50

# Evaluation configuration
EVAL_BATCH_SIZE = 16
MODEL_EVAL_PATH = "football-player-detection-yolo11m/weights/best.pt" # Path to the trained model for evaluation
PLOTS = True # Whether to generate plots during evaluation