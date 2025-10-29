# Global configuration settings for training and evaluating a YOLO object detection model
IMG_SIZE = 640
DEVICE = "mps"  # or "cpu", "0" for GPU
MODEL_NAME = "yolov8n"
DATA_ROOT = "../data/yolov8-format"
MODEL_DIR = "../models"

# Training configuration parameters for YOLO object tracking model
TRAIN_BATCH_SIZE = 16
EPOCHS = 50

# Evaluation configuration
MODEL_EVAL_DIR = "/yolov8-football-yolov8n2/weights/best.pt"
