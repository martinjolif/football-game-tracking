from pathlib import Path

import numpy as np
import supervision as sv
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms

from src.app.api_to_supervision import detections_from_results
from src.app.image_api import call_image_apis
from src.app.utils import collect_class_ids
from src.team_clustering.clustering_model import ClusteringModel
from src.team_clustering.utils import load_model
from src.app.functions import process_image

def visualize_clusters(
    img_path: str | Path,
    model_path: str | Path,
    img_size: int,
    device: torch.device
) -> str:
    """
    Run player detection, extract player crops, embed them using a feature model,
    cluster them (UMAP + KMeans), and visualize cluster labels on the original image.

    Args:
        img_path (str | Path):
            Path to the image file to process.
        model_path (str | Path):
            Path to the feature extraction model used for embeddings.
        device (torch.device):
            Torch device to run inference and clustering computations on
            (e.g., torch.device("cuda") or torch.device("cpu")).

    Returns:
        str:
            Path to the saved annotated image containing bounding boxes and
            cluster labels for each detected player.
    """
    # ---- LOAD IMAGE ----
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # ---- RUN DETECTION ----
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    results = call_image_apis(
        endpoints=["http://localhost:8000/player-detection/image"],
        image_bytes=image_bytes
    )

    player_detection = detections_from_results(
        results["http://localhost:8000/player-detection/image"]["detections"],
        detected_class_ids=collect_class_ids(
            results,
            endpoint="http://localhost:8000/player-detection/image",
            mapping_key="mapping_class",
            roles=["player"]
        )
    )

    detections = player_detection.xyxy  # Nx4 numpy array

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    # ---- CROP PLAYERS ----
    crops = []
    for (x1, y1, x2, y2) in detections:
        crop = img.crop((x1, y1, x2, y2))
        crop_tensor = transform(crop)
        crops.append(crop_tensor)

    crops_tensor = torch.stack(crops).to(device)

    # ---- LOAD FEATURE MODEL ----
    feature_model, _ = load_model(model_path, device)

    model = ClusteringModel(
        feature_extraction_model=feature_model,
        dimension_reducer=umap.UMAP(n_neighbors=15, min_dist=0.1),
        clustering_model=KMeans(n_clusters=2),
        n_clusters=2,
    )

    cluster_labels = model.fit_predict(crops_tensor)

    # ---- BUILD SUPERVISION DETECTIONS ----
    sv_dets = sv.Detections(
        xyxy=detections,
        class_id=cluster_labels,
    )

    # ---- VISUALIZATION ----
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
    label_annotator = sv.LabelAnnotator()

    labels = [f"Cluster {c}" for c in cluster_labels]

    annotated = box_annotator.annotate(scene=img_np, detections=sv_dets)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=sv_dets,
        labels=labels
    )

    # ---- SAVE OUTPUT ----
    out_path = "clustered_players.jpg"
    Image.fromarray(annotated).save(out_path)

    return out_path


img_path = "training/team_clustering/data/extracted_frames/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal/1_720p/frame_015000.jpg"
model_path = "runs/mlflow/750198089413804961/1385b27186ae46c19ddfc49afea0a75e/artifacts/best_mobilenetv3_small.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

annotated_image = process_image(
    image_path=img_path,
    debug_player_detection=True
)

# Display the result
out_path = "detected_players.jpg"
Image.fromarray(annotated_image).save(out_path)
out_image_path = visualize_clusters(img_path, model_path, 224, device)