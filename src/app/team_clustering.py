from pathlib import Path
import supervision as sv
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
import cv2

from src.app.api_to_supervision import detections_from_results
from src.app.image_api import call_image_apis
from src.app.utils import collect_class_ids
from src.team_clustering.utils import load_model
from src.team_clustering.clustering_model import ClusteringModel

# ============================================================
#                 LIVE STREAMING WITH CLUSTERING
# ============================================================

def live_preview_cluster_video(
    video_path: str | Path,
    model_path: str | Path,
    img_size: int,
    device: torch.device,
    cluster_train_frames: int = 100,
    end_frame: int | None = None,
    output_path: str = "live_clustered_preview.mp4"
):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Load feature extraction model
    feature_model, _ = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    # Initialize clustering model (trained later)
    cluster_model = ClusteringModel(
        feature_extraction_model=feature_model,
        dimension_reducer=umap.UMAP(n_neighbors=15, min_dist=0.1),
        clustering_model=KMeans(n_clusters=2)
    )

    train_crops = []
    train_labels_ready = False
    frame_idx = 0

    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    print("Starting live clustering preview. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end_frame is not None and frame_idx >= end_frame:
            break

        pil_img = Image.fromarray(frame)

        # ---------------- RUN DETECTION ----------------
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

        annotated_frame = frame.copy()

        if xyxy.size > 0:
            crops = [transform(pil_img.crop((x1, y1, x2, y2))) for (x1, y1, x2, y2) in xyxy]
            crops = torch.stack(crops).to(device)

            # ---------------- TRAIN CLUSTERING ON INITIAL FRAMES ----------------
            if frame_idx < cluster_train_frames:
                train_crops.append(crops)
            else:
                if not train_labels_ready:
                    train_tensor = torch.cat(train_crops, dim=0)
                    cluster_model.fit_predict(train_tensor)
                    train_labels_ready = True
                    print("✅ Clustering model trained, starting live predictions.")

                # Predict cluster labels for current frame
                infer_labels = cluster_model.predict(crops)
                dets = sv.Detections(xyxy=xyxy, class_id=infer_labels)
                label_texts = [f"{c}" for c in infer_labels]
                annotated_frame = box_annotator.annotate(scene=frame, detections=dets)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=dets, labels=label_texts)

        # ---------------- SHOW LIVE PREVIEW ----------------
        cv2.imshow("Live Cluster Preview", annotated_frame)
        writer.write(annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview stopped by user.")
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Live clustered video saved to: {output_path}")

# ======================= USAGE =======================

video_path = "../../england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United/1_720p.mkv"
model_path = "../../runs/mlflow/750198089413804961/1385b27186ae46c19ddfc49afea0a75e/artifacts/best_mobilenetv3_small.pth"
device = torch.device("mps")

live_preview_cluster_video(
    video_path,
    model_path,
    img_size=224,
    device=device,
    cluster_train_frames=75,
    end_frame=300,
    output_path="live_cluster_output.mp4"
)
