import sys
import traceback
import logging

import cv2
import supervision as sv

from src.app.image_api import call_image_apis
from src.app.api_to_supervision import detections_from_results, keypoints_from_pose_results
from src.app.player_tracking import visualize_frame
from src.app.pitch_radar_visualization import render_pitch_radar
from src.app.debug_visualization import render_detection_results
from src.app.utils import collect_class_ids

PROCESSED_FRAME_INTERVAL = 50
PLAYER_TRACKING_VIZ = False
PITCH_RADAR_VIZ = True
DEBUG_ALL = False
DEBUG_PLAYER_DETECTION = False
DEBUG_BALL_DETECTION = False
DEBUG_PITCH_DETECTION = False

logger = logging.getLogger(__name__)

# List all flags
flags = [
    PLAYER_TRACKING_VIZ,
    PITCH_RADAR_VIZ,
    DEBUG_ALL,
    DEBUG_PLAYER_DETECTION,
    DEBUG_BALL_DETECTION,
    DEBUG_PITCH_DETECTION
]
# Check that at most one is True
if sum(flags) > 1:
    raise ValueError("Only one debug or visualization flag can be True at a time.")

if DEBUG_ALL:
    logger.info("Debug All")
elif DEBUG_PLAYER_DETECTION:
    logger.info("Debug Player Detection")
elif DEBUG_BALL_DETECTION:
    logger.info("Debug Ball Detection")
elif DEBUG_PITCH_DETECTION:
    logger.info("Debug Pitch Detection")
elif PLAYER_TRACKING_VIZ:
    logger.info("Player Tracking")
elif PITCH_RADAR_VIZ:
    logger.info("Pitch Radar")

# Path to your video (it could be an RTSP/HTTP URL or a local file path)
#video_path = "../../england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv"
video_path = "../videos/08fd33_4.mp4"

if PLAYER_TRACKING_VIZ:
    tracker = sv.ByteTrack(minimum_consecutive_frames=5)
video_capture = None
try:
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        sys.exit(1)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        sys.exit(1)
    seconds_per_frame = 1 / fps
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("✅ End of video stream.")
            break

        frame_count += 1

        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        results = call_image_apis(
            endpoints=[
                "http://localhost:8000/player-detection/image",
                "http://localhost:8001/ball-detection/image",
                "http://localhost:8002/pitch-detection/image",
            ],
            image_bytes=frame_bytes)

        keypoint_mask = None
        if not (DEBUG_ALL or DEBUG_PLAYER_DETECTION or PLAYER_TRACKING_VIZ or PITCH_RADAR_VIZ):
            player_detection = None
        else:
            player_detection = detections_from_results(
                results["http://localhost:8000/player-detection/image"]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint="http://localhost:8000/player-detection/image",
                    mapping_key="mapping_class",
                    roles=("player", "referee", "goalkeeper")
                ),
            )
        if not (DEBUG_ALL or DEBUG_BALL_DETECTION or PITCH_RADAR_VIZ):
            ball_detection = None
        else:
            ball_detection = detections_from_results(
                results["http://localhost:8001/ball-detection/image"]["detections"],
                detected_class_ids=collect_class_ids(
                    results,
                    endpoint="http://localhost:8001/ball-detection/image",
                    mapping_key="mapping_class",
                    roles=["ball"]
                )
            )
        if not (DEBUG_ALL or DEBUG_PITCH_DETECTION or PITCH_RADAR_VIZ):
            pitch_detection = None
        else:
            pitch_detection, keypoint_mask = keypoints_from_pose_results(
                results["http://localhost:8002/pitch-detection/image"],
                confidence_threshold=0.5
            )
            # There is one object in the image: football pitch
            keypoint_mask = keypoint_mask[0]

        if DEBUG_ALL or DEBUG_PLAYER_DETECTION or DEBUG_BALL_DETECTION or DEBUG_PITCH_DETECTION:
            annotated_frame = render_detection_results(
                frame,
                player_detection,
                ball_detection,
                pitch_detection,
                keypoint_mask=keypoint_mask
            )

        elif PLAYER_TRACKING_VIZ:
            tracker.update_with_detections(player_detection)
            annotated_frame = visualize_frame(
                frame,
                player_detection,
                tracker=tracker,
                show_trace=False)
            cv2.imshow("Tracked Players", annotated_frame)

        elif PITCH_RADAR_VIZ:
            h, w, _ = frame.shape
            annotated_frame = frame.copy()
            radar = render_pitch_radar(player_detection, ball_detection, pitch_detection, keypoint_mask)
            radar = sv.resize_image(radar, (w//2, h//2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(
                x=w//2 - radar_w//2,
                y=h - radar_h,
                width=radar_w,
                height=radar_h,
            )
            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

        cv2.imshow("Simulated Live Stream", annotated_frame)

        #if frame_count % PROCESSED_FRAME_INTERVAL == 0:
        #    cv2.imshow("Processed", frame)

        if cv2.waitKey(int(seconds_per_frame * 1000)) & 0xFF == ord('q'):
            break

except cv2.error as e:
    traceback.print_exc()
    sys.exit(2)
except Exception as e:
    traceback.print_exc()
    sys.exit(3)
finally:
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()