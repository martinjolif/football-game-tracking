import logging

import cv2
import supervision as sv

from src.app.api_to_supervision import detections_from_results, keypoints_from_pose_results
from src.app.debug_visualization import render_detection_results
from src.app.image_api import call_image_apis
from src.app.player_tracking import visualize_frame
from src.app.utils import collect_class_ids
from src.radar.pitch_radar_visualization import render_pitch_radar

logger = logging.getLogger(__name__)

def process_image(
    image_path: str,
    player_tracking_viz=False,
    pitch_radar_viz=False,
    debug_all=False,
    debug_player_detection=True,
    debug_ball_detection=False,
    debug_pitch_detection=False
):
    """
    Process a single image and return an annotated frame.
    Only one visualization/debug mode should be active at a time.
    """
    # Ensure only one flag is active
    flags = [player_tracking_viz, pitch_radar_viz, debug_all,
             debug_player_detection, debug_ball_detection, debug_pitch_detection]
    if sum(flags) > 1:
        raise ValueError("Only one debug or visualization flag can be True at a time.")

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Logging the active mode
    if debug_all:
        logger.info("Debug All")
    elif debug_player_detection:
        logger.info("Debug Player Detection")
    elif debug_ball_detection:
        logger.info("Debug Ball Detection")
    elif debug_pitch_detection:
        logger.info("Debug Pitch Detection")
    elif player_tracking_viz:
        logger.info("Player Tracking")
    elif pitch_radar_viz:
        logger.info("Pitch Radar")

    # Initialize tracker if needed
    tracker = sv.ByteTrack(minimum_consecutive_frames=5) if player_tracking_viz else None

    # Encode image to bytes for API call
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

    # Call APIs
    results = call_image_apis(
        endpoints=[
            "http://localhost:8000/player-detection/image",
            "http://localhost:8001/ball-detection/image",
            "http://localhost:8002/pitch-detection/image",
        ],
        image_bytes=frame_bytes
    )

    # Process detections
    keypoint_mask = None
    player_detection = detections_from_results(
        results["http://localhost:8000/player-detection/image"]["detections"],
        detected_class_ids=collect_class_ids(
            results,
            endpoint="http://localhost:8000/player-detection/image",
            mapping_key="mapping_class",
            roles=("player", "referee", "goalkeeper")
        ),
    ) if debug_all or debug_player_detection or player_tracking_viz or pitch_radar_viz else None

    ball_detection = detections_from_results(
        results["http://localhost:8001/ball-detection/image"]["detections"],
        detected_class_ids=collect_class_ids(
            results,
            endpoint="http://localhost:8001/ball-detection/image",
            mapping_key="mapping_class",
            roles=["ball"]
        )
    ) if debug_all or debug_ball_detection or pitch_radar_viz else None

    pitch_detection = None
    if debug_all or debug_pitch_detection or pitch_radar_viz:
        pitch_detection, keypoint_mask = keypoints_from_pose_results(
            results["http://localhost:8002/pitch-detection/image"],
            confidence_threshold=0.5
        )
        keypoint_mask = keypoint_mask[0] if keypoint_mask else None

    # Visualization
    if debug_all or debug_player_detection or debug_ball_detection or debug_pitch_detection:
        annotated_frame = render_detection_results(
            frame,
            player_detection,
            ball_detection,
            pitch_detection,
            keypoint_mask=keypoint_mask
        )

    elif player_tracking_viz:
        tracker.update_with_detections(player_detection)
        annotated_frame = visualize_frame(
            frame,
            player_detection,
            tracker=tracker,
            show_trace=False
        )

    elif pitch_radar_viz:
        h, w, _ = frame.shape
        annotated_frame = frame.copy()
        radar, players_xy, ball_xy = render_pitch_radar(pitch_detection, keypoint_mask, player_detection, ball_detection)
        radar = sv.resize_image(radar, (w//2, h//2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w//2 - radar_w//2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h,
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

    return annotated_frame
