"""
Usage:
  python scripts/crop_players.py --images-dir train/images --labels-dir train/labels --out-dir crops
Options:
  --class-indices 2,1    # indices YOLO des classes à extraire (par défaut '2' -> player)
  --padding 0.1          # padding relatif autour de la bbox (10%)
  --ext jpg              # extension de sortie
"""

import argparse
import os
import csv
import json
import time

import numpy as np
from PIL import Image

from src.app.api_to_supervision import detections_from_results
from src.app.image_api import call_image_apis
from src.app.utils import collect_class_ids

def yolo_to_bbox(xc, yc, w, h, img_w, img_h, padding=0.0):
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    if padding:
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        x1 -= pw
        y1 -= ph
        x2 += pw
        y2 += ph
    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(img_w, round(x2)))
    y2 = int(min(img_h, round(y2)))
    return x1, y1, x2, y2


def find_label_path(rel_img_path, labels_dir, images_dir):
    """
    Returns the path of the corresponding label file, or None.
    First searches for the relative file in labels_dir (same directory structure), then falls back to basename.txt
    """
    if not labels_dir:
        return None
    # candidate with same relative path but .txt extension
    rel_txt = os.path.splitext(rel_img_path)[0] + ".txt"
    cand1 = os.path.join(labels_dir, rel_txt)
    if os.path.exists(cand1):
        return cand1
    # fallback: flat labels dir by basename
    base = os.path.splitext(os.path.basename(rel_img_path))[0]
    cand2 = os.path.join(labels_dir, base + ".txt")
    if os.path.exists(cand2):
        return cand2
    return None


def save_progress(processed_list, progress_file):
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({"processed": processed_list, "timestamp": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed", []))
        except Exception:
            return set()
    return set()


def _is_processed(rel_path, processed_set):
    """
    Compatibility: a processed element may be either the relative path (rel_path) or simply the basename.
    """
    if rel_path in processed_set:
        return True
    if os.path.basename(rel_path) in processed_set:
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="training/team_clustering/extracted_frames")
    p.add_argument("--labels-dir", default=None, help="Labels folder (optional). If missing, a sibling ‘labels’ directory is attempted")
    p.add_argument("--out-dir", default="training/team_clustering/crops")
    p.add_argument("--class-indices", default="2", help="comma-separated class indices to crop")
    p.add_argument("--padding", type=float, default=0.0)
    p.add_argument("--ext", default="jpg")
    p.add_argument("--ask-color", action="store_false", help="show each crop and ask for color category to save in CSV")
    p.add_argument("--yolo-detection", action="store_false", help="use YOLO model to detect players if no label file")
    args = p.parse_args()

    images_dir = args.images_dir
    # fallback de labels_dir : argument ou sibling 'labels' du dossier images
    labels_dir = args.labels_dir or os.path.join(os.path.dirname(images_dir), "labels")
    if labels_dir and not os.path.exists(labels_dir):
        labels_dir = None

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
            label_path = find_label_path(img_name, labels_dir, images_dir)
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

            if label_path:
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
                    image_bytes = open(img_path, "rb").read()
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
                base_name = os.path.splitext(os.path.basename(img_name))[0]
                out_name = f"{base_name}_cls{cls}_{box_idx}.{args.ext}"
                out_path = os.path.join(args.out_dir, out_name)
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
        print("Keyboard interrupt received, saving in progress...")
        save_progress(list(processed), progress_file)
    finally:
        save_progress(list(processed), progress_file)

    csv_path = os.path.join(args.out_dir, "crops_index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["source_image", "crop_image", "class", "x1", "y1", "x2", "y2", "color"])
        writer.writerows(csv_rows)

    print(f"Done: {len(csv_rows)} crops -> {args.out_dir}")


if __name__ == "__main__":
    main()
