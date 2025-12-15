from pathlib import Path
import supervision as sv
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
import cv2
from tqdm import tqdm
import numpy as np
import argparse

from src.app.api_to_supervision import detections_from_results
from src.app.image_api import call_image_apis
from src.app.utils import collect_class_ids
from src.team_clustering.utils import load_model
from src.team_clustering.clustering_model import ClusteringModel

from training.logger import LOGGER

# ============================================================
#               PROCESS VIDEO WITH TRAIN + INFERENCE
# ============================================================

def process_video_clusters(
        video_path: str | Path,
        model_path: str | Path,
        img_size: int,
        device: torch.device,
        cluster_train_start: int = 0,
        cluster_train_end: int = 100,  # frames used to TRAIN clustering
        end_frame: int | None = None  # max frame to process
):
    """
    Trains the clustering model on frames [cluster_train_start → cluster_train_end]
    Then predicts cluster labels on the rest of the video, up to `end_frame`.

    Returns:
        tuple:
            - list of (frame_idx, xyxy, labels)
            - last_processed_frame_idx
    """

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    LOGGER.info(f"Total video frames: {total_frames}")
    LOGGER.info(f"Processing up to frame: {end_frame}")
    LOGGER.info(f"Training clustering on frames {cluster_train_start} → {cluster_train_end}")

    # load model (feature extractor)
    feature_model, _ = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    # storage
    train_crops, train_dets, train_frame_ids = [], [], []
    infer_crops, infer_dets, infer_frame_ids = [], [], []

    frame_idx = 0
    pbar = tqdm(total=min(total_frames, end_frame))

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        pil_img = Image.fromarray(frame)

        # ---------- RUN DETECTION ----------
        _, encoded = cv2.imencode(".jpg", frame)
        results = call_image_apis(
            endpoints=["http://localhost:8000/player-detection/image"],
            image_bytes=encoded.tobytes()
        )

        det = detections_from_results(
            results["http://localhost:8000/player-detection/image"]["detections"],
            detected_class_ids=collect_class_ids(
                results,
                endpoint="http://localhost:8000/player-detection/image",
                mapping_key="mapping_class",
                roles=["player"]
            )
        )

        xyxy = det.xyxy

        if xyxy.size > 0:
            crops = [transform(pil_img.crop((x1, y1, x2, y2))) for (x1, y1, x2, y2) in xyxy]
            crops = torch.stack(crops).to(device)

            if cluster_train_start <= frame_idx <= cluster_train_end:
                train_crops.append(crops if xyxy.size > 0 else torch.empty((0,3,img_size,img_size)))
                train_dets.append(xyxy if xyxy.size > 0 else np.zeros((0,4)))
                train_frame_ids.append(frame_idx)
            else:
                infer_crops.append(crops if xyxy.size > 0 else torch.empty((0,3,img_size,img_size)))
                infer_dets.append(xyxy if xyxy.size > 0 else np.zeros((0,4)))
                infer_frame_ids.append(frame_idx)

        frame_idx += 1

    cap.release()
    pbar.close()

    # ============================================================
    #                  TRAIN CLUSTERING MODEL
    # ============================================================

    if len(train_crops) == 0:
        LOGGER.warning("❌ No detections during training segment!")
        return [], frame_idx

    train_tensor = torch.cat(train_crops, dim=0)

    cluster_model = ClusteringModel(
        feature_extraction_model=feature_model,
        dimension_reducer=umap.UMAP(n_neighbors=15, min_dist=0.1),
        clustering_model=KMeans(n_clusters=2)
    )

    LOGGER.info("Training clustering...")
    train_labels = cluster_model.fit_predict(train_tensor)

    # ============================================================
    #                PREDICT REMAINING FRAMES
    # ============================================================

    if len(infer_crops) > 0:
        infer_tensor = torch.cat(infer_crops, dim=0)
        LOGGER.info("Predicting remaining frames...")
        infer_labels = cluster_model.predict(infer_tensor)
    else:
        infer_labels = np.array([])

    # ============================================================
    #        REBUILD FINAL LIST OF (frame_idx, detections, labels)
    # ============================================================

    result = []

    idx = 0
    for fidx, dets in zip(train_frame_ids, train_dets):
        n = len(dets)
        result.append((fidx, dets, train_labels[idx:idx + n]))
        idx += n

    idx = 0
    for fidx, dets in zip(infer_frame_ids, infer_dets):
        n = len(dets)
        result.append((fidx, dets, infer_labels[idx:idx + n]))
        idx += n

    result.sort(key=lambda x: x[0])

    LOGGER.info("Clustering complete.")
    return result


# ============================================================
#                 RENDER VIDEO WITH ANNOTATIONS
# ============================================================
def draw_inference_message(frame, text="Inference started"):
    overlay = frame.copy()

    h, w = frame.shape[:2]

    box_h = 60
    y_offset = 40

    cv2.rectangle(
        overlay,
        (0, y_offset),
        (w, y_offset + box_h),
        (0, 0, 0),
        -1
    )

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(
        frame,
        text,
        (20, y_offset + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame

def write_cluster_video(
    video_path: str | Path,
    clustering_output,
    output_path="clustered_output.mp4",
    train_end_frame: int | None = None,
    end_frame: int | None = None
):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    LOGGER.info("Writing video...")

    frame_dict = {frame_idx: (xyxy, labels) for frame_idx, xyxy, labels in clustering_output}

    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if end_frame is not None and frame_idx >= end_frame:
            break

        if frame_idx in frame_dict:
            xyxy, labels = frame_dict[frame_idx]
            dets = sv.Detections(xyxy=xyxy, class_id=labels)
            label_texts = [f"{c}" for c in labels]

            annotated = box_annotator.annotate(scene=frame, detections=dets)
            annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=label_texts)

            if train_end_frame is not None and frame_idx > train_end_frame:
                annotated = draw_inference_message(
                    annotated,
                    "Clustering training finished: inference started"
                )

            writer.write(annotated)
        else:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()
    LOGGER.info(f"Saved: {output_path}")

    return output_path

parser = argparse.ArgumentParser(description="Team clustering visualization.")

parser.add_argument(
    "--video-path",
    type=Path,
    default="england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool/1_720p.mkv",
    help="Path to the video file."
)
parser.add_argument(
    "--model-path",
    type=Path,
    default="runs/mlflow/750198089413804961/1385b27186ae46c19ddfc49afea0a75e/artifacts/best_mobilenetv3_small.pth",
    help="Path to the feature extraction model."
)
parser.add_argument(
    "--start-training-frame",
    type=int,
    default=0,
    help="Frame index to start training clustering."
)
parser.add_argument(
    "--end-training-frame",
    type=int,
    default=75,
    help="Frame index to end training clustering."
)
parser.add_argument(
    "--end-frame",
    type=int,
    default=150,
    help="Maximum frame index to process."
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


outputs = process_video_clusters(
    args.video_path,
    args.model_path,
    img_size=224,
    device=device,
    cluster_train_start=args.start_training_frame,
    cluster_train_end=args.end_training_frame,
    end_frame=args.end_frame,
)

output_video_path = write_cluster_video(
    args.video_path,
    outputs,
    output_path="training/team_clustering/visualization/team_clustering.mp4",
    train_end_frame=args.end_training_frame,
    end_frame=args.end_frame
)