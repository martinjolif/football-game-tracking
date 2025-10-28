from pathlib import Path

from ultralytics import YOLO

def train_yolo(data_yaml: str, model_name: str, epochs:int, batch_size: int, imgsz: int, device: str) ->  None:
    # Load the pretrained YOLO model
    model_checkpoint = f"{model_name}.pt"
    model = YOLO(model_checkpoint)

    # Finetune the model on my custom dataset
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project="runs/train",
        name=f"football-player-detection-{Path(model_checkpoint).stem}",
        device=device  # use GPU if available by setting device='0', use 'cpu' for CPU
    )

    # Evaluate on the test set (must be defined in data.yaml)
    results = model.val(data=data_yaml, split="test")
    print("Test set evaluation results:", results)

    # Log model weights
    best_model_path = Path(model.trainer.best)
    print(f"Training completed. Best model saved at {best_model_path}")

if __name__ == "__main__":
    from config import BATCH_SIZE, EPOCHS, IMG_SIZE, DEVICE, MODEL_NAME, DATA_ROOT

    train_yolo(DATA_ROOT + "/data.yaml", MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE)
