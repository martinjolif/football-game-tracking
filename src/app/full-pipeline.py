import logging
import sys
import traceback
from enum import Enum, auto
from pathlib import Path

import cv2
import supervision as sv
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms

from src.app.api_to_supervision import detections_from_results, keypoints_from_pose_results
from src.app.debug_visualization import render_detection_results
from src.app.image_api import call_image_apis
from src.app.player_tracking import visualize_frame
from src.app.utils import collect_class_ids
from src.radar.pitch_radar_visualization import render_pitch_radar
from src.team_clustering.clustering_model import ClusteringModel
from src.team_clustering.utils import load_model

# ====================== FEATURE & VISUALIZATION MODES ======================
class FeatureMode(Enum):
    PLAYER_DETECTION = auto()
    BALL_DETECTION = auto()
    PITCH_DETECTION = auto()
    RADAR = auto()        # always activates player + ball + pitch
    TRACKING = auto()     # can track player or ball or both
    TEAM = auto()         # needs player, optionally ball/pitch/tracking

class VizMode(Enum):
    PLAYER = auto()
    BALL = auto()
    PITCH = auto()
    RADAR = auto()        # can include team
    TRACKING = auto()

# ----------------- CONFIGURE MODES -----------------
FEATURE_MODES = {FeatureMode.RADAR, FeatureMode.TEAM}
VIZ_MODES = {VizMode.RADAR}

PROCESSED_FRAME_INTERVAL = 50
cluster_train_frames = 50
img_size = 224

logger = logging.getLogger(__name__)
video_path = "../videos/08fd33_4.mp4"

# ================= TEAM CLUSTERING ===================
model_path = Path("../../runs/mlflow/750198089413804961/1385b27186ae46c19ddfc49afea0a75e/artifacts/best_mobilenetv3_small.pth")
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

feature_model, _ = load_model(model_path, device)

transform = transforms.Compose([
    transforms.Resize(int(img_size * 256 / 224)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

cluster_model = ClusteringModel(
    feature_extraction_model=feature_model,
    dimension_reducer=umap.UMAP(n_neighbors=15, min_dist=0.1),
    clustering_model=KMeans(n_clusters=2)
)

train_crops = []
train_labels_ready = False

# ================= RENDERERS ===================
def render_teams(frame, detections, cluster_labels, team_box_annotator, label_annotator):
    detections = sv.Detections(
        xyxy=detections.xyxy,
        class_id=cluster_labels,
        tracker_id=None
    )
    labels = [f"Team {c}" for c in cluster_labels]
    frame = team_box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels)
    return frame

def render_tracker(frame, detections, box_annotator, tracker):
    frame = box_annotator.annotate(frame, detections)
    frame = visualize_frame(frame, detections, tracker=tracker, show_trace=False)
    return frame

# ================= VIDEO PROCESSING ===================
video_capture = None
tracker = sv.ByteTrack(minimum_consecutive_frames=5) if FeatureMode.TRACKING in FEATURE_MODES else None

try:
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        sys.exit(1)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    seconds_per_frame = 1 / fps if fps > 0 else 0.033
    frame_count = 0

    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
    team_box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    cluster_labels = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("✅ End of video stream.")
            break

        frame_count += 1
        annotated_frame = frame.copy()
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

        # ---------------- DETECTION ENDPOINTS ----------------
        endpoints = []
        if FeatureMode.PLAYER_DETECTION in FEATURE_MODES or FeatureMode.RADAR in FEATURE_MODES or FeatureMode.TRACKING in FEATURE_MODES or FeatureMode.TEAM in FEATURE_MODES:
            endpoints.append("http://localhost:8000/player-detection/image")
        if FeatureMode.BALL_DETECTION in FEATURE_MODES or FeatureMode.RADAR in FEATURE_MODES:
            endpoints.append("http://localhost:8001/ball-detection/image")
        if FeatureMode.PITCH_DETECTION in FEATURE_MODES or FeatureMode.RADAR in FEATURE_MODES or FeatureMode.TEAM in FEATURE_MODES:
            endpoints.append("http://localhost:8002/pitch-detection/image")

        results = call_image_apis(endpoints=endpoints, image_bytes=frame_bytes)

        # ---------------- FEATURE EXTRACTION ----------------
        player_detection = None
        ball_detection = None
        pitch_detection, keypoint_mask = None, None

        if "http://localhost:8000/player-detection/image" in results:
            player_detection = detections_from_results(
                results["http://localhost:8000/player-detection/image"]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint="http://localhost:8000/player-detection/image",
                    mapping_key="mapping_class",
                    roles=("player", "goalkeeper", "referee"),
                ),
            )

        if "http://localhost:8001/ball-detection/image" in results:
            ball_detection = detections_from_results(
                results["http://localhost:8001/ball-detection/image"]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint="http://localhost:8001/ball-detection/image",
                    mapping_key="mapping_class",
                    roles=["ball"],
                ),
            )

        if "http://localhost:8002/pitch-detection/image" in results:
            pitch_detection, keypoint_mask = keypoints_from_pose_results(
                results["http://localhost:8002/pitch-detection/image"],
                confidence_threshold=0.7,
            )
            keypoint_mask = keypoint_mask[0] if keypoint_mask else None

        # ---------------- TRACKER ----------------
        if tracker and player_detection:
            player_detection = tracker.update_with_detections(player_detection)

        # ---------------- TEAM CLUSTERING ----------------
        if FeatureMode.TEAM in FEATURE_MODES and player_detection and len(player_detection.xyxy) > 0:
            pil_img = Image.fromarray(frame)
            crops = [
                transform(pil_img.crop((x1, y1, x2, y2)))
                for (x1, y1, x2, y2) in player_detection.xyxy
            ]
            if crops:
                crops_tensor = torch.stack(crops).to(device)
                if frame_count <= cluster_train_frames:
                    train_crops.append(crops_tensor)
                elif not train_labels_ready:
                    cluster_model.fit_predict(torch.cat(train_crops, dim=0))
                    train_labels_ready = True
                    print("✅ Team clustering trained")

                if train_labels_ready:
                    cluster_labels = cluster_model.predict(crops_tensor).astype(int)

        # ---------------- VISUALIZATION ----------------
        if VizMode.PLAYER in VIZ_MODES and player_detection:
            annotated_frame = render_detection_results(annotated_frame, player_detections=player_detection)

        if VizMode.BALL in VIZ_MODES and ball_detection:
            annotated_frame = render_detection_results(annotated_frame, ball_detection=ball_detection)

        if VizMode.PITCH in VIZ_MODES and pitch_detection:
            annotated_frame = render_detection_results(
                annotated_frame,
                pitch_detection=pitch_detection,
                keypoint_mask=keypoint_mask
            )

        if VizMode.TRACKING in VIZ_MODES and player_detection:
            annotated_frame = render_tracker(frame, player_detection, box_annotator, tracker)

        if VizMode.RADAR in VIZ_MODES:
            radar = render_pitch_radar(
                pitch_detection,
                keypoint_mask,
                player_detection,
                ball_detection,
                player_teams=cluster_labels if FeatureMode.TEAM in FEATURE_MODES else None
            )
            h, w, _ = frame.shape
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(x=w // 2 - radar_w // 2, y=h - radar_h, width=radar_w, height=radar_h)
            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

        cv2.imshow("Visualization", annotated_frame)
        if cv2.waitKey(int(seconds_per_frame * 1000)) & 0xFF == ord('q'):
            break

except Exception:
    traceback.print_exc()
finally:
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()
