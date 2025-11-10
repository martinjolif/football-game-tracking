from config import DATA_ROOT, MODEL_DIR, MODEL_EVAL_PATH, DEVICE, EVAL_BATCH_SIZE, IMG_SIZE
from src.utils.yolo_evaluation import evaluate_trained_model
import os

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_EVAL_PATH)  # path to trained model
evaluate_trained_model(MODEL_PATH, DATA_ROOT, IMG_SIZE, EVAL_BATCH_SIZE, DEVICE)
