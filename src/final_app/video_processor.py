import os
from collections import defaultdict, deque
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
from src.commentary_generation.events3 import get_left_team, assign_teams, get_ball_possessor
from src.commentary_generation.main import generate_commentary_ollama
from src.commentary_generation.plot import draw_commentary
from src.radar.pitch_dimensions import PitchDimensions
from src.radar.pitch_radar_visualization import render_pitch_radar
from src.team_clustering.clustering_model import ClusteringModel
from src.team_clustering.utils import load_model
from src.utils.logger import LOGGER

class VideoProcessor:
    def __init__(
            self,
            video_path: str,
            output_path: str,
            enable_radar: bool = True,
            enable_commentary: bool = True,
            enable_tracking: bool = True,
            enable_team_clustering: bool = True,
            end_frame: int = None,
            cluster_train_frames: int = 50,
            model_path: str = "weights/team_clustering/hf_weights/mobilenetv3-football-jersey-classification.pth",
            img_size: int = 224,
            cluster_history_length: int = 20,
            ball_movement_threshold: int = 200,
            progress_callback=None
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.enable_radar = enable_radar
        self.enable_commentary = enable_commentary
        self.enable_tracking = enable_tracking
        self.enable_team_clustering = enable_team_clustering
        self.end_frame = end_frame
        self.cluster_train_frames = cluster_train_frames
        self.img_size = img_size
        self.cluster_history_length = cluster_history_length
        self.ball_movement_threshold = ball_movement_threshold
        self.progress_callback = progress_callback

        # Initialize device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Initialize team clustering if enabled
        if self.enable_team_clustering:
            self._init_team_clustering(model_path)

        # Initialize trackers
        self.player_tracker = sv.ByteTrack(minimum_consecutive_frames=5) if self.enable_tracking else None
        self.recent_clusters = defaultdict(lambda: deque(maxlen=self.cluster_history_length))

        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
        self.team_box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5)

        # State variables
        self.cluster_labels = None
        self.last_commentary = None
        self.last_ball_xy = None
        self.last_possession_team = None
        self.left_team = None
        self.right_team = None
        self.teams_barycenter = None

    def _init_team_clustering(self, model_path: str):
        """Initialize team clustering model"""
        model_path = Path(model_path)
        if not model_path.exists():
            LOGGER.warning(f"Model not found at {model_path}, team clustering disabled")
            self.enable_team_clustering = False
            return

        feature_model, _ = load_model(model_path, self.device)

        self.transform = transforms.Compose([
            transforms.Resize(int(self.img_size * 256 / 224)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])

        self.cluster_model = ClusteringModel(
            feature_extraction_model=feature_model,
            dimension_reducer=umap.UMAP(n_neighbors=15, min_dist=0.1),
            clustering_model=KMeans(n_clusters=2)
        )

        self.train_crops = []
        self.train_labels_ready = False

    def _update_progress(self, progress: int, message: str):
        """Update progress via callback"""
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _call_detection_apis(self, frame_bytes):
        """Call detection APIs based on enabled features"""
        endpoints = []

        # Player detection
        if self.enable_tracking or self.enable_team_clustering or self.enable_radar or self.enable_commentary:
            endpoints.append(os.getenv("PLAYER_DETECTION_URL", "http://localhost:8000/player-detection/image"))

        # Ball detection
        if self.enable_radar or self.enable_commentary:
            endpoints.append(os.getenv("BALL_DETECTION_URL", "http://localhost:8001/ball-detection/image"))

        # Pitch detection
        if self.enable_radar or self.enable_commentary or self.enable_team_clustering:
            endpoints.append(os.getenv("PITCH_DETECTION_URL", "http://localhost:8002/pitch-detection/image"))

        return call_image_apis(endpoints=endpoints, image_bytes=frame_bytes)

    def _extract_detections(self, results):
        """Extract detections from API results"""
        player_detection = None
        ball_detection = None
        pitch_detection = None
        keypoint_mask = None

        player_url = os.getenv("PLAYER_DETECTION_URL", "http://localhost:8000/player-detection/image")
        ball_url = os.getenv("BALL_DETECTION_URL", "http://localhost:8001/ball-detection/image")
        pitch_url = os.getenv("PITCH_DETECTION_URL", "http://localhost:8002/pitch-detection/image")

        # Player detection
        if player_url in results:
            player_detection = detections_from_results(
                results[player_url]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint=player_url,
                    mapping_key="mapping_class",
                    roles=["player", "goalkeeper"],
                ),
            )

        # Ball detection
        if ball_url in results:
            ball_detection = detections_from_results(
                results[ball_url]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint=ball_url,
                    mapping_key="mapping_class",
                    roles=["ball"],
                ),
            )

        # Pitch detection
        if pitch_url in results:
            pitch_detection, keypoint_mask = keypoints_from_pose_results(
                results[pitch_url],
                confidence_threshold=0.7,
            )
            keypoint_mask = keypoint_mask[0] if keypoint_mask else None

        return player_detection, ball_detection, pitch_detection, keypoint_mask

    def _process_team_clustering(self, frame, player_detection, frame_count):
        """Process team clustering"""
        if not self.enable_team_clustering or not player_detection or len(player_detection.xyxy) == 0:
            return

        pil_img = Image.fromarray(frame)
        crops = [
            self.transform(pil_img.crop((x1, y1, x2, y2)))
            for (x1, y1, x2, y2) in player_detection.xyxy
        ]

        if not crops:
            return

        crops_tensor = torch.stack(crops).to(self.device)

        # Training phase
        if frame_count <= self.cluster_train_frames:
            self.train_crops.append(crops_tensor)
        elif not self.train_labels_ready:
            self.cluster_model.fit_predict(torch.cat(self.train_crops, dim=0))
            self.train_labels_ready = True
            LOGGER.info("✅ Team clustering trained")

        # Prediction phase
        if self.train_labels_ready:
            self.cluster_labels = self.cluster_model.predict(crops_tensor).astype(int)

            # Correct clusters based on tracker_id history
            if player_detection.tracker_id is not None:
                for i, tracker_id in enumerate(player_detection.tracker_id):
                    self.recent_clusters[tracker_id].append(self.cluster_labels[i])
                    self.cluster_labels[i] = max(
                        set(self.recent_clusters[tracker_id]),
                        key=self.recent_clusters[tracker_id].count
                    )

    def _render_frame(self, frame, player_detection, ball_detection, pitch_detection, keypoint_mask, frame_count):
        """Render all visualizations on frame"""
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        # Render tracking
        if False and self.enable_tracking and player_detection:
            annotated_frame = self.box_annotator.annotate(annotated_frame, player_detection)
            annotated_frame = visualize_frame(annotated_frame, player_detection, tracker=self.player_tracker,
                                              show_trace=False)

        # Render team clustering
        if False and self.enable_team_clustering and player_detection and self.cluster_labels is not None:
            team_detections = sv.Detections(
                xyxy=player_detection.xyxy,
                class_id=self.cluster_labels,
                tracker_id=None
            )
            labels = [f"Team {c}" for c in self.cluster_labels]
            annotated_frame = self.team_box_annotator.annotate(annotated_frame, team_detections)
            annotated_frame = self.label_annotator.annotate(annotated_frame, team_detections, labels)

        # Render radar and commentary
        if self.enable_radar or self.enable_commentary:
            radar, players_xy, ball_xy = render_pitch_radar(
                pitch_detection,
                keypoint_mask,
                player_detection,
                ball_detection,
                player_teams=self.cluster_labels if self.enable_team_clustering else None,
                return_pitch_positions=True if self.enable_commentary else False,
                team_colors_legend={self.left_team: sv.Color.BLUE, self.right_team: sv.Color.RED}
                if frame_count > self.cluster_train_frames + 1 else None
            )

            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(x=w // 2 - radar_w // 2, y=h - radar_h, width=radar_w, height=radar_h)
            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

            # Generate commentary
            if self.enable_commentary and self.train_labels_ready:
                if frame_count == self.cluster_train_frames + 1:
                    players = assign_teams(players_xy, self.cluster_labels)
                    self.left_team, self.right_team, self.teams_barycenter = get_left_team(players)

                if ball_xy is not None and len(ball_xy) > 0 and self.left_team is not None:
                    players = assign_teams(players_xy, self.cluster_labels)
                    possession_team = players[get_ball_possessor(ball_xy, players_xy)]['team'] \
                        if get_ball_possessor(ball_xy, players_xy) is not None else None

                    generate_new = False
                    if self.last_ball_xy is None or self.last_possession_team is None:
                        generate_new = True
                    else:
                        ball_movement = ((ball_xy[0][0] - self.last_ball_xy[0][0]) ** 2 +
                                         (ball_xy[0][1] - self.last_ball_xy[0][1]) ** 2) ** 0.5
                        if ball_movement > self.ball_movement_threshold or possession_team != self.last_possession_team:
                            generate_new = True

                    if generate_new:
                        commentary = generate_commentary_ollama(
                            previous_ball_xy=self.last_ball_xy,
                            ball_xy=ball_xy,
                            players_xy=players_xy,
                            cluster_labels=self.cluster_labels,
                            left_team=self.left_team,
                            right_team=self.right_team,
                            teams_barycenter=self.teams_barycenter,
                            pitch=PitchDimensions()
                        )
                        if commentary is not None:
                            self.last_commentary = commentary
                        self.last_ball_xy = ball_xy
                        self.last_possession_team = possession_team

                if self.last_commentary is not None:
                    annotated_frame = draw_commentary(
                        annotated_frame,
                        self.last_commentary,
                        start_xy=(w // 2, int(0.05 * h))
                    )

        return annotated_frame

    def process(self):
        """Main processing loop"""
        video_capture = None
        video_writer = None

        try:
            video_capture = cv2.VideoCapture(self.video_path)
            if not video_capture.isOpened():
                raise ValueError("Failed to open video file")

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if self.end_frame:
                total_frames = min(total_frames, self.end_frame)

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = video_capture.read()
                if not ret or (self.end_frame and frame_count >= self.end_frame):
                    break

                frame_count += 1

                # Update progress
                progress = int((frame_count / total_frames) * 100)
                self._update_progress(progress, f"Processing frame {frame_count}/{total_frames}")

                # Encode frame
                frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

                # Call detection APIs
                results = self._call_detection_apis(frame_bytes)

                # Extract detections
                player_detection, ball_detection, pitch_detection, keypoint_mask = \
                    self._extract_detections(results)

                # Update tracker
                if self.player_tracker and player_detection:
                    player_detection = self.player_tracker.update_with_detections(player_detection)

                # Process team clustering
                self._process_team_clustering(frame, player_detection, frame_count)

                # Render frame
                annotated_frame = self._render_frame(
                    frame, player_detection, ball_detection,
                    pitch_detection, keypoint_mask, frame_count
                )

                # Write frame
                video_writer.write(annotated_frame)

            self._update_progress(100, "Processing completed")

        finally:
            if video_capture:
                video_capture.release()
            if video_writer:
                video_writer.release()