if __name__ == "__main__":
    from config import (TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, EPOCHS, IMG_SIZE, DEVICE, MODEL_NAME, DATA_ROOT, MODEL_DIR,
                        PLOTS)
    from utils.yolo_training import train_yolo
    import os

    train_yolo(
        os.path.join(DATA_ROOT, "data.yaml"),
        MODEL_NAME,
        MODEL_DIR,
        EPOCHS,
        TRAIN_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        IMG_SIZE,
        DEVICE,
        PLOTS,
        detection_type="ball"
    )
