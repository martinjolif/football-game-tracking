import os

from ultralytics import YOLO

from config import DATA_ROOT

def evaluate_trained_model(model_path: str) -> None:
    """Evaluate a trained YOLO model on a test dataset."""
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Evaluate on specific images
    for filename in os.listdir(DATA_ROOT + "/test/images/"):
        if filename.endswith(".jpg"):
            img_path = os.path.join(DATA_ROOT + "/test/images/", filename)
            result = model(img_path)
            result[0].show(line_width=1)

    # Evaluate on the test set (must be defined in data.yaml)
    model.val(data=DATA_ROOT + "/data.yaml", split="test")


if __name__ == "__main__":
    MODEL_PATH = "runs/train/yolov8-football-yolov8n2/weights/best.pt"  # path to trained model
    evaluate_trained_model(MODEL_PATH)