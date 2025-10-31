# Global configuration settings for training and evaluating a YOLO object detection model
IMG_SIZE = 640
DEVICE = "mps"  # or "cpu", "0" for GPU
MODEL_NAME = "yolov8n" # Pretrained YOLO model name
DATA_ROOT = "src/player_detection/data/yolov8-format" # Root directory for dataset
MODEL_DIR = "src/player_detection/models" # Directory to save trained models

# Training configuration parameters for YOLO object tracking model
TRAIN_BATCH_SIZE = 16
EPOCHS = 50

# Evaluation configuration
EVAL_BATCH_SIZE = 16
MODEL_EVAL_PATH = "football-player-detection-yolov8n/weights/best.pt" # Path to the trained model for evaluation
PLOTS = True # Whether to generate plots during evaluation