import logging
import os
from pathlib import Path

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
        plots: bool,
        detection_type: str  # "ball", "player", "field_lines"
) ->  None:
    # Load the pretrained YOLO model
    model_checkpoint_path = f"{model_name}.pt"
    model = YOLO(model_checkpoint_path)

    # Finetune the model on my custom dataset
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=train_batch_size,
        imgsz=imgsz,
        project=model_dir,
        name=f"football-{detection_type}-detection-{Path(model_checkpoint_path).stem}",
        device=device
    )

    # Evaluate on the test set (must be defined in data.yaml)
    results = model.val(data=data_yaml, imgsz=imgsz, batch=eval_batch_size, device=device, plots=plots, split="test")
    logger.info(f"Test set evaluation results: {results}")

    # Log model weights
    if hasattr(model, "trainer") and hasattr(model.trainer, "best"):
        best_model_path = Path(model.trainer.best)
        logger.info(f"Training completed. Best model saved at {best_model_path}")
    else:
        logger.warning("Could not find the best model path.")