import sys
from pathlib import Path
import cv2
import argparse

from utils import videos_by_match
from training.logger import LOGGER

parser = argparse.ArgumentParser()
parser.add_argument("--videos-dir", default="england_epl", help="folder with match videos")
parser.add_argument("--out-dir", default="training/team_clustering/data/extracted_frames", help="folder to save extracted frames")
parser.add_argument("--sample-interval", default=120.0, type=float, help="sampling interval in seconds")
args = parser.parse_args()

SRC_DIR = Path(args.videos_dir)
OUT_DIR = Path(args.out_dir)

if not SRC_DIR.exists():
    LOGGER.error(f"The folder `{SRC_DIR}` does not exist.")
    sys.exit(1)

save_all_global = False

grouped = videos_by_match(SRC_DIR)
for match, files in sorted(grouped.items()):
    LOGGER.info(match)
    for video in files:
        LOGGER.info(f"\n=== {video.name} ===")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            LOGGER.error(f"Cannot open `{video}`. Skipped.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(fps * args.sample_interval))
        frame_idx = 0
        saved_count = 0
        save_all_video = False
        window = f"{video.name} - frame preview"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                cv2.imshow(window, frame)
                cv2.waitKey(1)  # allows the image to be displayed

                if save_all_global or save_all_video:
                    save = True
                else:
                    ans = input(
                        f"Video `{video.name}` frame {frame_idx} — save? [y/n/a=all_video/A=all_videos/q=next/Q=quit] "
                    ).strip()

                    if ans == "Q":
                        cap.release()
                        cv2.destroyAllWindows()
                        LOGGER.info("Stop requested.")
                        sys.exit(0)
                    elif ans == "q":
                        break
                    elif ans == "a":
                        save_all_video = True
                        save = True
                    elif ans == "A":
                        save_all_global = True
                        save = True
                    elif ans.lower() == "y":
                        save = True
                    else:
                        save = False

                if save:
                    target_dir = OUT_DIR / Path(*video.parts[1:-1]) / video.stem
                    target_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"frame_{frame_idx:06d}.jpg"
                    target = target_dir / fname
                    cv2.imwrite(str(target), frame)
                    saved_count += 1
                    LOGGER.info(f"Saved -> `{target}`")

            frame_idx += 1

        cap.release()
        cv2.destroyWindow(window)
        LOGGER.info(f"Finished `{video.name}` — frames saved: {saved_count}")

LOGGER.info("\nDone for all files.")