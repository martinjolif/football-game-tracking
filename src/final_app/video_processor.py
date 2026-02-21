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
from src.app.player_tracking import visualize_frame
from src.app.utils import collect_class_ids
from src.player_detection.api.inference import detect_players_in_image
from src.ball_detection.api.inference import detect_ball_in_image
from src.pitch_detection.api.inference import detect_pitch_in_image
from src.commentary_generation.events3 import get_left_team, assign_teams, get_ball_possessor, get_field_zone_3x3
from src.commentary_generation.main import generate_commentary_ollama
from src.commentary_generation.plot import draw_commentary
from src.commentary_generation.tts import TTSGenerator
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
            min_commentary_interval_frames: int = 300,
            enable_tts: bool = False,
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
        self.min_commentary_interval_frames = min_commentary_interval_frames
        self.enable_tts = enable_tts and enable_commentary
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

        # TTS state
        self._fps = None
        self.tts_generator = TTSGenerator() if self.enable_tts else None
        self.tts_clips: list[tuple[float, str]] = []  # (timestamp_sec, wav_path)

        # Commentary temporal tracking — negative init ensures first commentary fires immediately after clustering
        self._last_commentary_frame: int = -self.min_commentary_interval_frames
        self._ball_distance_since_last: float = 0.0
        self._possession_changes_since_last: int = 0
        self._prev_frame_ball_xy = None
        self._prev_frame_possession_team = None
        self._prev_commentary_ball_zone: str | None = None

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
        """Run detection models directly based on enabled features."""
        results = {}

        # Player detection
        if self.enable_tracking or self.enable_team_clustering or self.enable_radar or self.enable_commentary:
            results["player"] = detect_players_in_image(frame_bytes).model_dump()

        # Ball detection
        if self.enable_radar or self.enable_commentary:
            results["ball"] = detect_ball_in_image(frame_bytes).model_dump()

        # Pitch detection
        if self.enable_radar or self.enable_commentary or self.enable_team_clustering:
            results["pitch"] = detect_pitch_in_image(frame_bytes).model_dump()

        return results

    def _extract_detections(self, results):
        """Extract detections from inference results."""
        player_detection = None
        ball_detection = None
        pitch_detection = None
        keypoint_mask = None

        # Player detection
        if "player" in results:
            player_detection = detections_from_results(
                results["player"]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint="player",
                    mapping_key="mapping_class",
                    roles=["player", "goalkeeper"],
                ),
            )

        # Ball detection
        if "ball" in results:
            ball_detection = detections_from_results(
                results["ball"]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint="ball",
                    mapping_key="mapping_class",
                    roles=["ball"],
                ),
            )

        # Pitch detection
        if "pitch" in results:
            pitch_detection, keypoint_mask = keypoints_from_pose_results(
                results["pitch"],
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
                    possessor_idx = get_ball_possessor(ball_xy, players_xy)
                    possession_team = players[possessor_idx]['team'] if possessor_idx is not None else None

                    # Accumulate per-frame stats for temporal context
                    if self._prev_frame_ball_xy is not None:
                        dist = ((ball_xy[0][0] - self._prev_frame_ball_xy[0][0]) ** 2 +
                                (ball_xy[0][1] - self._prev_frame_ball_xy[0][1]) ** 2) ** 0.5
                        self._ball_distance_since_last += dist
                    if possession_team is not None and self._prev_frame_possession_team is not None:
                        if possession_team != self._prev_frame_possession_team:
                            self._possession_changes_since_last += 1
                    self._prev_frame_ball_xy = ball_xy
                    self._prev_frame_possession_team = possession_team

                    # Interval-based trigger
                    if frame_count - self._last_commentary_frame >= self.min_commentary_interval_frames:
                        seconds_since_last = (
                            (frame_count - self._last_commentary_frame) / self._fps
                            if self._last_commentary_frame > 0 and self._fps
                            else None
                        )
                        commentary = generate_commentary_ollama(
                            previous_ball_xy=self.last_ball_xy,
                            ball_xy=ball_xy,
                            players_xy=players_xy,
                            cluster_labels=self.cluster_labels,
                            left_team=self.left_team,
                            right_team=self.right_team,
                            teams_barycenter=self.teams_barycenter,
                            pitch=PitchDimensions(),
                            seconds_since_last=seconds_since_last,
                            possession_changes=self._possession_changes_since_last,
                            ball_distance_traveled=self._ball_distance_since_last,
                            prev_ball_zone=self._prev_commentary_ball_zone,
                        )
                        if commentary is not None:
                            self.last_commentary = commentary
                            if self.enable_tts and self._fps:
                                timestamp_sec = frame_count / self._fps
                                wav_path = self.tts_generator.generate_wav(commentary)
                                self.tts_clips.append((timestamp_sec, wav_path))
                        self.last_ball_xy = ball_xy
                        self.last_possession_team = possession_team
                        self._last_commentary_frame = frame_count
                        self._prev_commentary_ball_zone = get_field_zone_3x3(
                            [ball_xy[0][0], ball_xy[0][1]], PitchDimensions()
                        )
                        self._ball_distance_since_last = 0.0
                        self._possession_changes_since_last = 0

                if self.last_commentary is not None:
                    annotated_frame = draw_commentary(
                        annotated_frame,
                        self.last_commentary,
                        start_xy=(w // 2, int(0.05 * h))
                    )

        return annotated_frame

    def _merge_audio(self):
        """Use ffmpeg to embed TTS audio clips at their correct timestamps into the output video."""
        import subprocess

        tmp_output = self.output_path + ".audio_tmp.mp4"
        inputs = ["-i", self.output_path]
        filter_parts = []

        for i, (ts, wav_path) in enumerate(self.tts_clips):
            inputs += ["-i", wav_path]
            delay_ms = int(ts * 1000)
            filter_parts.append(f"[{i + 1}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

        n = len(self.tts_clips)
        if n == 1:
            filter_parts.append("[a0]anull[aout]")
        else:
            mix = "".join(f"[a{i}]" for i in range(n))
            filter_parts.append(f"{mix}amix=inputs={n}:normalize=0[aout]")

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex", ";".join(filter_parts),
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                tmp_output,
            ]
        )
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp_output, self.output_path)

        for _, wav_path in self.tts_clips:
            os.unlink(wav_path)

    def process(self):
        """Main processing loop"""
        video_capture = None
        video_writer = None

        try:
            video_capture = cv2.VideoCapture(self.video_path)
            if not video_capture.isOpened():
                raise ValueError("Failed to open video file")

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            self._fps = fps
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

            # Release writer before audio merge so the file is fully flushed
            if video_writer:
                video_writer.release()
                video_writer = None

            if self.enable_tts and self.tts_clips:
                self._update_progress(100, "Embedding audio commentary...")
                self._merge_audio()

        finally:
            if video_capture:
                video_capture.release()
            if video_writer:
                video_writer.release()