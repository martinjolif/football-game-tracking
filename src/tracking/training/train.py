from pathlib import Path

from ultralytics import YOLO

def train_yolov8(data_yaml: str, model_checkpoint="yolov8n.pt", epochs=50, imgsz=640):
    # Load the pretrained YOLOv8 model
    model = YOLO(model_checkpoint)

    # Finetune the model on my custom dataset
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project="runs/train",
        name=f"yolov8-football-{Path(model_checkpoint).stem}",
        device='mps'  # use GPU if available by setting device='0', use 'cpu' for CPU
    )

    # Evaluate on the test set (must be defined in data.yaml)
    results = model.val(data=data_yaml, split="test")
    print("Test set evaluation results:", results)

    # Log model weights
    best_model_path = Path(model.trainer.best)
    print(f"Training completed. Best model saved at {best_model_path}")

if __name__ == "__main__":
    DATA_YAML = "../data/yolov8-format/data.yaml"  # path to your dataset.yaml
    train_yolov8(DATA_YAML, model_checkpoint="yolov8n.pt", epochs=50, imgsz=640)
