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
from PIL import Image


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


def find_label_path(img_path, labels_dir):
    base = os.path.splitext(os.path.basename(img_path))[0]
    candidate = os.path.join(labels_dir, base + ".txt")
    return candidate if os.path.exists(candidate) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="../player_detection/data/yolov8-format/test/images")
    p.add_argument("--labels-dir", default="../player_detection/data/yolov8-format/test/labels")
    p.add_argument("--out-dir", default="crops")
    p.add_argument("--class-indices", default="2", help="comma-separated class indices to crop")
    p.add_argument("--padding", type=float, default=0.0)
    p.add_argument("--ext", default="jpg")
    args = p.parse_args()

    images_dir = args.images_dir
    labels_dir = args.labels_dir or os.path.join(os.path.dirname(images_dir), "labels")
    os.makedirs(args.out_dir, exist_ok=True)

    class_indices = set(int(x) for x in args.class_indices.split(",") if x.strip() != "")
    csv_rows = []
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in img_files:
        img_path = os.path.join(images_dir, img_name)
        label_path = find_label_path(img_path, labels_dir)
        if not label_path:
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        with open(label_path, "r") as lf:
            lines = [l.strip() for l in lf if l.strip()]
        box_idx = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls not in class_indices:
                continue
            xc, yc, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_bbox(xc, yc, w, h, W, H, padding=args.padding)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{os.path.splitext(img_name)[0]}_cls{cls}_{box_idx}.{args.ext}"
            out_path = os.path.join(args.out_dir, out_name)
            crop.save(out_path, quality=95)
            csv_rows.append([img_name, out_name, cls, x1, y1, x2, y2])
            box_idx += 1

    csv_path = os.path.join(args.out_dir, "crops_index.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["source_image", "crop_image", "class", "x1", "y1", "x2", "y2"])
        writer.writerows(csv_rows)

    print(f"Done: {len(csv_rows)} crops -> {args.out_dir}")


if __name__ == "__main__":
    main()
