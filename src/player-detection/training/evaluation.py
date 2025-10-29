import os

from ultralytics import YOLO

def evaluate_trained_model(model_path: str, data_root: str) -> None:
    """Evaluate a trained YOLO model on a test dataset."""
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Evaluate on specific images
    for filename in os.listdir(data_root + "/test/images/"):
        if filename.endswith(".jpg"):
            img_path = os.path.join(data_root + "/test/images/", filename)
            result = model(img_path)
            result[0].show(line_width=1)

    # Evaluate on the test set (must be defined in data.yaml)
    model.val(data=data_root + "/data.yaml", split="test")


if __name__ == "__main__":
    from config import DATA_ROOT, MODEL_DIR, MODEL_EVAL_DIR

    MODEL_PATH = MODEL_DIR + MODEL_EVAL_DIR  # path to trained model
    evaluate_trained_model(MODEL_PATH, DATA_ROOT)