import logging
import os
from pathlib import Path

import mlflow
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

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
        detection_type: str  # "ball", "player", "pitch"
) ->  None:
    # Load the pretrained YOLO model
    model_checkpoint_path = f"{model_name}.pt"
    model = YOLO(model_checkpoint_path)

    if detection_type == "pitch":
        task="pose"
    elif detection_type in ["ball", "player"]:
        task="detect"
    else:
        raise ValueError(f"Unsupported detection type: {detection_type}, must be one of 'ball', 'player', or 'pitch'.")
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": train_batch_size,
        "imgsz": imgsz,
        "task": task,
        "project": model_dir,
        "name": f"football-{detection_type}-detection-{Path(model_checkpoint_path).stem}",
        "device": device
    }
    if task == "pose":
        train_kwargs["mosaic"] = 0.0

    # Finetune the model on custom dataset
    model.train(**train_kwargs)

    # Evaluate on the test set (must be defined in data.yaml)
    results = model.val(data=data_yaml, imgsz=imgsz, batch=eval_batch_size, device=device, plots=plots, split="test")
    # Get id from mlflow and log metrics on the test set
    experiment_id = mlflow.last_active_run().info.experiment_id
    run_id = mlflow.last_active_run().info.run_id
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id):
        log_summary_to_mlflow(results.summary(), task=task)


    # Log model weights
    if hasattr(model, "trainer") and hasattr(model.trainer, "best"):
        best_model_path = Path(model.trainer.best)
        logger.info(f"Training completed. Best model saved at {best_model_path}")
    else:
        logger.warning("Could not find the best model path.")


def log_summary_to_mlflow(results_summary: DetMetrics, task: str = "detect") -> None:
    """
    Logs the summary of results to MLflow and saves a formatted table as an artifact.
    """
    # Define column widths
    col_widths = {
        "Class": 15,
        "Images": 8,
        "Instances": 10,
        "Box-P": 9,
        "Box-R": 9,
        "Box-F1": 9,
        "Box-mAP50": 9,
        "Box-mAP50-95": 11
    }

    if task == "pose":
        # Extend column widths for pose metrics
        col_widths.update({
            "Pose-P": 9,
            "Pose-R": 9,
            "Pose-F1": 9,
            "Pose-mAP50": 9,
            "Pose-mAP50-95": 11
        })

    # Header
    header = (
        f"{'Class':<{col_widths['Class']}}"
        f"{'Images':>{col_widths['Images']}}"
        f"{'Instances':>{col_widths['Instances']}}"
        f"{'Box-P':>{col_widths['Box-P']}}"
        f"{'R':>{col_widths['Box-R']}}"
        f"{'F1':>{col_widths['Box-F1']}}"
        f"{'mAP50':>{col_widths['Box-mAP50']}}"
        f"{'mAP50-95':>{col_widths['Box-mAP50-95']}}"
    )

    if task == "pose":
        header += (
            f"{'Pose-P':>{col_widths['Pose-P']}}"
            f"{'Pose-R':>{col_widths['Pose-R']}}"
            f"{'Pose-F1':>{col_widths['Pose-F1']}}"
            f"{'Pose-mAP50':>{col_widths['Pose-mAP50']}}"
            f"{'Pose-mAP50-95':>{col_widths['Pose-mAP50-95']}}"
        )

    separator = "-" * len(header)
    lines = [header, separator]

    # Add each row
    for cls in results_summary:
        # Log box/detection metrics
        mlflow.log_metric(f"test_metrics/box_p/{cls['Class']}", float(cls['Box-P']))
        mlflow.log_metric(f"test_metrics/box_r/{cls['Class']}", float(cls["Box-R"]))
        mlflow.log_metric(f"test_metrics/box_f1/{cls['Class']}", float(cls["Box-F1"]))
        mlflow.log_metric(f"test_metrics/box_map50/{cls['Class']}", float(cls["mAP50"]))
        mlflow.log_metric(f"test_metrics/box_map50_95/{cls['Class']}", float(cls["mAP50-95"]))

        row = (
            f"{cls['Class']:<{col_widths['Class']}}"
            f"{int(cls['Images']):>{col_widths['Images']}}"
            f"{int(cls['Instances']):>{col_widths['Instances']}}"
            f"{cls['Box-P']:>{col_widths['Box-P']}.4f}"
            f"{cls['Box-R']:>{col_widths['Box-R']}.4f}"
            f"{cls['Box-F1']:>{col_widths['Box-F1']}.4f}"
            f"{cls['mAP50']:>{col_widths['Box-mAP50']}.4f}"
            f"{cls['mAP50-95']:>{col_widths['Box-mAP50-95']}.4f}"
        )

        # Add pose metrics if task=="pose"
        if task == "pose":
            mlflow.log_metric(f"test_metrics/pose_p/{cls['Class']}", float(cls.get("Pose-P", 0)))
            mlflow.log_metric(f"test_metrics/pose_r/{cls['Class']}", float(cls.get("Pose-R", 0)))
            mlflow.log_metric(f"test_metrics/pose_f1/{cls['Class']}", float(cls.get("Pose-F1", 0)))
            mlflow.log_metric(f"test_metrics/pose_map50/{cls['Class']}", float(cls.get("Pose-mAP50", 0)))
            mlflow.log_metric(f"test_metrics/pose_map50_95/{cls['Class']}", float(cls.get("Pose-mAP50-95", 0)))

            row += (
                f"{cls.get('Pose-P', 0):>{col_widths['Pose-P']}.4f}"
                f"{cls.get('Pose-R', 0):>{col_widths['Pose-R']}.4f}"
                f"{cls.get('Pose-F1', 0):>{col_widths['Pose-F1']}.4f}"
                f"{cls.get('Pose-mAP50', 0):>{col_widths['Pose-mAP50']}.4f}"
                f"{cls.get('Pose-mAP50-95', 0):>{col_widths['Pose-mAP50-95']}.4f}"
            )

        lines.append(row)

    # Join lines
    table_text = "\n".join(lines)

    # Save and log as artifact
    artifact_path = "class_metrics_table.txt"
    with open(artifact_path, "w") as f:
        f.write(table_text)
    mlflow.log_artifact(artifact_path)

    # Delete file after logging
    try:
        os.remove(artifact_path)
    except FileNotFoundError:
        logger.warning(f"Could not remove artifact at {artifact_path}")

    logger.info("Logged metrics as a nicely formatted table artifact")