import sys
from pathlib import Path
import cv2

from typing import Generator, Dict, List

SRC_DIR = Path("england_epl")
OUT_DIR = Path("training/team_clustering/extracted_frames")
SAMPLE_INTERVAL_SEC = 120.0  # sampling in seconds
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"}

def iter_video_files(src_dir: Path = SRC_DIR, pattern: str = "*.mkv") -> Generator[Path, None, None]:
    """
    Recursively traverse `src_dir` and yield each file matching `pattern`.
    """
    src = Path(src_dir)
    if not src.exists():
        return
    for p in src.rglob(pattern):
        if p.is_file():
            yield p

def videos_by_match(src_dir: Path = SRC_DIR) -> Dict[str, List[Path]]:
    """
    Returns a dictionary {match_folder_name: [Path(...), ...]}.
    """
    result: Dict[str, List[Path]] = {}
    for p in iter_video_files(src_dir):
        match_name = p.parent.name
        result.setdefault(match_name, []).append(p)
    return result

if not SRC_DIR.exists():
    print(f"The folder `{SRC_DIR}` does not exist.")
    sys.exit(1)


save_all_global = False

grouped = videos_by_match(SRC_DIR)
for match, files in sorted(grouped.items()):
    print(match)
    for f in files:
        video = f
        print(f"  - {f}")
        print(f"\n=== {video.name} ===")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"Cannot open `{video}`. Skipped.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(fps * SAMPLE_INTERVAL_SEC))
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
                        print("Stop requested.")
                        sys.exit(0)
                    if ans == "q":
                        break
                    if ans == "a":
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
                    print(f"Saved -> `{target}`")

            frame_idx += 1

        cap.release()
        cv2.destroyWindow(window)
        print(f"Finished `{video.name}` — frames saved: {saved_count}")

print("\nDone for all files.")