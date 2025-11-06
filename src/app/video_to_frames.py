import cv2
import sys
import traceback
from image_api import call_image_apis

PROCESSED_FRAME_INTERVAL = 50

# Path to your video (it could be an RTSP/HTTP URL or a local file path)
video_path = "../../england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv"

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
        cv2.imshow("Simulated Live Stream", frame)

        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        frame = frame.tolist()
        results = call_image_apis(
            endpoints=[
                "http://localhost:8000/player-detection/image",
                "http://localhost:8001/ball-detection/image",
                "http://localhost:8002/pitch-detection/image",
            ],
            image_bytes=frame_bytes)
        print(results)

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