from pathlib import Path
import logging
import os

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_yolo(
        data_yaml: str,
        model_name: str,
        model_dir: str,
        epochs: int,
        train_batch_size: int,
        eval_batch_size: int,
        imgsz: int,
        device: str,
        plots: bool
) ->  None:
    # Load the pretrained YOLO model
    model_checkpoint = f"{model_name}.pt"
    model = YOLO(model_checkpoint)

    # Finetune the model on my custom dataset
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=train_batch_size,
        imgsz=imgsz,
        project=model_dir,
        name=f"football-player-detection-{Path(model_checkpoint).stem}",
        device=device
    )

    # Evaluate on the test set (must be defined in data.yaml)
    results = model.val(data=data_yaml, imgsz=imgsz, batch=eval_batch_size, device=device, plots=plots, split="test")
    logger.info(f"Test set evaluation results: {results}")

    # Log model weights
    best_model_path = Path(model.trainer.best)
    logger.info(f"Training completed. Best model saved at {best_model_path}")

if __name__ == "__main__":
    from config import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, EPOCHS, IMG_SIZE, DEVICE, MODEL_NAME, DATA_ROOT, MODEL_DIR, PLOTS

    train_yolo(os.path.join(DATA_ROOT, "data.yaml"), MODEL_NAME, MODEL_DIR, EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, IMG_SIZE, DEVICE, PLOTS)
