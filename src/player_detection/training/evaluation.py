import os

from ultralytics import YOLO

def evaluate_trained_model(model_path: str, data_root: str, imgsz: int, eval_batch_size: int, device: str) -> None:
    """Evaluate a trained YOLO model on a test dataset."""
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Evaluate on specific images
    test_images_dir = os.path.join(data_root, "test", "images")
    for filename in os.listdir(test_images_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(test_images_dir, filename)
            detection_results = model(img_path)
            detection_results[0].show(line_width=1)

    # Evaluate on the test set (must be defined in data.yaml)
    model.val(data=os.path.join(data_root, "data.yaml"), split="test", device=device, batch=eval_batch_size, imgsz=imgsz)


if __name__ == "__main__":
    from config import DATA_ROOT, MODEL_DIR, MODEL_EVAL_PATH, DEVICE, EVAL_BATCH_SIZE, IMG_SIZE

    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_EVAL_PATH)  # path to trained model
    evaluate_trained_model(MODEL_PATH, DATA_ROOT, IMG_SIZE, EVAL_BATCH_SIZE, DEVICE)