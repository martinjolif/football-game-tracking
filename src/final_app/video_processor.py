import os
import subprocess
import tempfile
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import supervision as sv
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
            enable_tts: bool = True,
            end_frame: int = None,
            frame_step: int = 1,
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
        self.frame_step = max(1, frame_step)
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

        # Commentary pacing
        self._tts_gap_sec = 3  # seconds to wait AFTER last audio clip ends
        self._last_commentary_frame = -999  # frame when last commentary was dispatched
        self._previous_event_summary = None  # prior event context for temporal narration
        self._last_tts_end_sec = 0.0  # video-time when last TTS audio clip finishes playing

        # Background commentary thread pool
        self._commentary_executor = ThreadPoolExecutor(max_workers=1)
        self._commentary_future: Future | None = None

        # TTS state
        self.enable_tts = enable_tts and enable_commentary
        self._tts_executor: ThreadPoolExecutor | None = None
        self._tts_futures: list[Future] = []
        self._tts_segments: list[dict] = []
        self._tts_sample_rate: int | None = None
        self._current_fps: float = 30.0

        if self.enable_tts:
            try:
                from src.commentary_generation.tts import get_tts_model
                model, sr = get_tts_model()
                if model is None:
                    LOGGER.warning("TTS model failed to load — disabling TTS")
                    self.enable_tts = False
                else:
                    self._tts_sample_rate = sr
                    self._tts_executor = ThreadPoolExecutor(max_workers=1)
                    LOGGER.info("TTS enabled (sample_rate=%d)", sr)
            except Exception:
                LOGGER.exception("TTS init failed — disabling TTS")
                self.enable_tts = False

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
            dimension_reducer=PCA(n_components=10),
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

    def _submit_tts_job(self, text: str, frame_number: int, fps: float):
        """Submit a TTS generation job to the background executor."""
        if not self.enable_tts or self._tts_executor is None:
            return

        from src.commentary_generation.tts import generate_tts_audio

        timestamp_sec = frame_number / fps
        processor = self  # capture reference for the closure

        def _run():
            waveform, sr = generate_tts_audio(text)
            if waveform is not None:
                duration = len(waveform) / sr
                processor._last_tts_end_sec = timestamp_sec + duration
                return {"timestamp_sec": timestamp_sec, "waveform": waveform}
            return None

        future = self._tts_executor.submit(_run)
        self._tts_futures.append(future)

    def _collect_tts_results(self, timeout: float | None = None):
        """Drain completed TTS futures into _tts_segments."""
        remaining = []
        for future in self._tts_futures:
            if future.done():
                try:
                    result = future.result(timeout=0)
                    if result is not None:
                        self._tts_segments.append(result)
                except Exception:
                    pass
            elif timeout is not None:
                try:
                    result = future.result(timeout=timeout)
                    if result is not None:
                        self._tts_segments.append(result)
                except Exception:
                    pass
            else:
                remaining.append(future)
        self._tts_futures = remaining

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

            # Generate commentary (non-blocking, with cooldown)
            if self.enable_commentary and self.train_labels_ready:
                if frame_count == self.cluster_train_frames + 1:
                    players = assign_teams(players_xy, self.cluster_labels)
                    self.left_team, self.right_team, self.teams_barycenter = get_left_team(players)

                # Check if background commentary is ready
                if self._commentary_future is not None and self._commentary_future.done():
                    try:
                        commentary = self._commentary_future.result()
                        if commentary is not None:
                            self.last_commentary = commentary
                            # Submit TTS job for the new commentary
                            self._submit_tts_job(commentary, frame_count, self._current_fps)
                    except Exception:
                        pass
                    self._commentary_future = None

                current_video_sec = frame_count / self._current_fps

                # Drain completed TTS futures so _last_tts_end_sec is up to date
                self._collect_tts_results()
                # Block until: all TTS jobs finished AND audio done playing + gap
                tts_all_done = all(f.done() for f in self._tts_futures)
                audio_ready = tts_all_done and (current_video_sec >= self._last_tts_end_sec + self._tts_gap_sec)

                if ball_xy is not None and len(ball_xy) > 0 and self.left_team is not None:
                    players = assign_teams(players_xy, self.cluster_labels)
                    possessor_idx = get_ball_possessor(ball_xy, players_xy)
                    possession_team = players[possessor_idx]['team'] if possessor_idx is not None else None

                    # First commentary fires right after clustering
                    is_first = self.last_ball_xy is None
                    situation_changed = False
                    if not is_first and self.last_ball_xy is not None:
                        ball_movement = ((ball_xy[0][0] - self.last_ball_xy[0][0]) ** 2 +
                                         (ball_xy[0][1] - self.last_ball_xy[0][1]) ** 2) ** 0.5
                        if ball_movement > self.ball_movement_threshold or possession_team != self.last_possession_team:
                            situation_changed = True

                    should_generate = is_first or (situation_changed and audio_ready)

                    if should_generate and self._commentary_future is None:
                        # Launch commentary generation in background
                        self._commentary_future = self._commentary_executor.submit(
                            generate_commentary_ollama,
                            previous_ball_xy=self.last_ball_xy,
                            ball_xy=ball_xy,
                            players_xy=players_xy,
                            cluster_labels=self.cluster_labels,
                            left_team=self.left_team,
                            right_team=self.right_team,
                            teams_barycenter=self.teams_barycenter,
                            pitch=PitchDimensions(),
                            previous_event_summary=self._previous_event_summary,
                        )
                        self._last_commentary_frame = frame_count
                        # Save current state as previous context for next commentary
                        from src.commentary_generation.events3 import get_field_zone_3x3
                        ball_zone = get_field_zone_3x3([ball_xy[0][0], ball_xy[0][1]], PitchDimensions())
                        self._previous_event_summary = (
                            f"Team {possession_team} had possession in the {ball_zone}"
                        )
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
            self._current_fps = fps
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if self.end_frame:
                total_frames = min(total_frames, self.end_frame)

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            frame_count = 0
            # Cached detection results for frame skipping
            last_player_detection = None
            last_ball_detection = None
            last_pitch_detection = None
            last_keypoint_mask = None

            while True:
                ret, frame = video_capture.read()
                if not ret or (self.end_frame and frame_count >= self.end_frame):
                    break

                frame_count += 1

                # Update progress
                progress = int((frame_count / total_frames) * 100)
                self._update_progress(progress, f"Processing frame {frame_count}/{total_frames}")

                # Only run detection on every Nth frame (or first frame)
                is_detection_frame = (frame_count == 1 or frame_count % self.frame_step == 0)

                if is_detection_frame:
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

                    # Cache FINAL results (post-tracker)
                    last_player_detection = player_detection
                    last_ball_detection = ball_detection
                    last_pitch_detection = pitch_detection
                    last_keypoint_mask = keypoint_mask
                else:
                    # Reuse cached detections
                    player_detection = last_player_detection
                    ball_detection = last_ball_detection
                    pitch_detection = last_pitch_detection
                    keypoint_mask = last_keypoint_mask

                # Run clustering every frame using current frame crops
                self._process_team_clustering(frame, player_detection, frame_count)

                # Render frame
                annotated_frame = self._render_frame(
                    frame, player_detection, ball_detection,
                    pitch_detection, keypoint_mask, frame_count
                )

                # Write frame
                video_writer.write(annotated_frame)

            self._update_progress(95, "Re-encoding video for browser playback...")

        finally:
            if video_capture:
                video_capture.release()
            if video_writer:
                video_writer.release()

        # Drain pending TTS futures and assemble audio track
        audio_path = None
        if self.enable_tts and self._tts_futures:
            self._update_progress(96, "Waiting for TTS generation to finish...")
            self._collect_tts_results(timeout=120)

        if self.enable_tts and self._tts_segments and self._tts_sample_rate:
            from src.commentary_generation.audio_assembler import assemble_audio_track

            total_duration = frame_count / fps if fps > 0 else 0
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            audio_path = tmp_wav.name

            if not assemble_audio_track(
                self._tts_segments, self._tts_sample_rate, total_duration, audio_path
            ):
                LOGGER.warning("Audio assembly failed — producing silent video")
                os.unlink(audio_path)
                audio_path = None

        # Re-encode from mp4v to H.264 so browsers can play it
        try:
            self._reencode_h264(audio_path=audio_path)
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

        self._update_progress(100, "Processing completed")

    def _reencode_h264(self, audio_path: str | None = None):
        """Re-encode the output video to H.264 for browser compatibility."""
        tmp_path = self.output_path + ".tmp.mp4"
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", self.output_path,
            ]

            if audio_path:
                cmd += ["-i", audio_path]

            cmd += [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
            ]

            if audio_path:
                cmd += ["-c:a", "aac", "-b:a", "128k", "-ac", "1"]
            else:
                cmd += ["-an"]

            cmd.append(tmp_path)

            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(tmp_path, self.output_path)
            LOGGER.info(
                "Re-encoded output to H.264%s", " with audio" if audio_path else ""
            )
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"ffmpeg re-encode failed: {e.stderr.decode()}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError("Failed to re-encode video for browser playback")