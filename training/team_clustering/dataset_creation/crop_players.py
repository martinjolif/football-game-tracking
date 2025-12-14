import argparse
import os

import numpy as np
from PIL import Image

from src.app.api_to_supervision import detections_from_results
from src.app.image_api import call_image_apis
from src.app.utils import collect_class_ids
from training.logger import LOGGER
from training.team_clustering.dataset_creation.utils import (load_progress, _is_processed, yolo_to_bbox,
                                                             save_progress, append_to_crops_index)

parser = argparse.ArgumentParser(description="Extract player crops from images using existing labels or YOLO detection.")
parser.add_argument(
    "--images-dir",
    default="training/team_clustering/data/extracted_frames",
    help="Directory containing the source images from which player crops will be extracted."
)
parser.add_argument(
    "--labels-dir",
    default=None,
    help="Optional path to the labels folder. If not provided, a YOLO model will automatically detect and crop players from the images."
)
parser.add_argument(
    "--out-dir",
    default="training/team_clustering/data/crops",
    help="Directory where the extracted player crops and CSV index will be saved."
)
parser.add_argument(
    "--class-indices",
    default="2", #2 corresponds to player
    help="Comma-separated list of class indices to crop (e.g., '0,1,2')."
)
parser.add_argument("--padding", type=float, default=0.0)
parser.add_argument("--ext", default="jpg", help ="File extension for the saved crops (e.g., jpg, png).")
parser.add_argument(
    "--ask-color",
    action="store_true",
    help="If set, shows each crop and asks the user for a color category to save in the CSV."
)
parser.add_argument(
    "--yolo-detection",
    action="store_true",
    help="Use YOLO model to detect players if no label files are provided."
)
args = parser.parse_args()

images_dir = args.images_dir
os.makedirs(args.out_dir, exist_ok=True)
progress_file = os.path.join(args.out_dir, "crop_progress.json")
processed = load_progress(progress_file)
processed_order = []

class_indices = set(int(x) for x in args.class_indices.split(",") if x.strip() != "")
csv_rows = []

# Recursively browse images_dir and store relative paths
img_files = []
for root, _, files in os.walk(images_dir):
    for fn in files:
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, images_dir)
            img_files.append(rel_path)

# Keep only those not yet processed (basename compatibility)
img_files = [f for f in img_files if not _is_processed(f, processed)]

quit_requested = False
try:
    for img_name in img_files:
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        if args.labels_dir is not None:
            label_path = args.labels_dir
            with open(label_path, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf if l.strip()]
            detections = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                if cls not in class_indices:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_bbox(xc, yc, w, h, W, H, padding=args.padding)
                detections.append([cls, x1, y1, x2, y2])
        else:
            if args.yolo_detection:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                results = call_image_apis(
                    endpoints=[
                        "http://localhost:8000/player-detection/image",
                    ],
                    image_bytes=image_bytes)
                player_detection = detections_from_results(
                    results["http://localhost:8000/player-detection/image"]["detections"],
                    detected_class_ids=collect_class_ids(
                        results,
                        endpoint="http://localhost:8000/player-detection/image",
                        mapping_key="mapping_class",
                        roles=["player"]
                    ),
                )
                # player_detection.xyxy shape (N,4), class_id shape (N,)
                detections = np.hstack((player_detection.class_id.reshape(-1, 1), player_detection.xyxy))
            else:
                continue

        box_idx = 0
        for (cls, x1, y1, x2, y2) in detections:
            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{os.path.splitext(img_name)[0]}_cls{cls}_{box_idx}.{args.ext}"
            out_path = os.path.join(args.out_dir, out_name)
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            crop.save(out_path, quality=95)

            # Mark the image as processed (relative path) as soon as a crop is saved.
            if img_name not in processed:
                processed.add(img_name)
                processed_order.append(img_name)

            color_label = ""
            if args.ask_color:
                try:
                    crop.show()
                except Exception:
                    pass
                try:
                    color_label = input(f"Crop {out_name} - color category (leave empty if unknown, 'q' to quit): ").strip()
                except EOFError:
                    color_label = ""
                if color_label.lower() == "q":
                    quit_requested = True
                    color_label = ""
            csv_rows.append([img_name, out_name, cls, x1, y1, x2, y2, color_label])
            box_idx += 1

            if quit_requested:
                break

        if len(processed_order) % 50 == 0 and processed_order:
            save_progress(list(processed), progress_file)

        if quit_requested:
            break

except KeyboardInterrupt:
    LOGGER.info("Keyboard interrupt received, saving in progress...")
    save_progress(list(processed), progress_file)
finally:
    save_progress(list(processed), progress_file)

csv_path = os.path.join(args.out_dir, "crops_index.csv")
fieldnames = ["source_image", "crop_image", "class", "x1", "y1", "x2", "y2", "color"]
if csv_rows:
    # Convert rows to list of dicts and append to CSV using the helper.
    records = [dict(zip(fieldnames, row)) for row in csv_rows]
    try:
        append_to_crops_index(records, csv_path, fieldnames=fieldnames)
    except ValueError as e:
        LOGGER.error(f"ERROR: {e}")

LOGGER.info(f"Done: {len(csv_rows)} crops -> {args.out_dir}")
