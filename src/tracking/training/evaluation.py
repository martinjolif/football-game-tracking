from ultralytics import YOLO

def evaluate_trained_model(model_path: str):
    """Evaluate a trained YOLOv8 model on a test dataset."""
    # Load the trained YOLOv8 model
    model = YOLO(model_path)

    # Evaluate on specific images
    result = model("../data/yolov8-format/test/images/4b770a_3_9_png.rf.ae7cd2e9e19d140d6f3727b4d69c5fd1.jpg")
    result[0].show(line_width=1)

    result = model("../data/yolov8-format/test/images/573e61_9_6_png.rf.ec2f81dccbea93400877c92862b0a597.jpg")
    result[0].show(line_width=1)

    # Evaluate on the test set (must be defined in data.yaml)
    model.val(data="../data/yolov8-format/data.yaml", split="test")


if __name__ == "__main__":
    MODEL_PATH = "runs/train/yolov8-football-yolov8n2/weights/best.pt"  # path to your trained model
    evaluate_trained_model(MODEL_PATH)