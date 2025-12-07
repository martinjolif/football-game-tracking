import csv
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Generator

from training.logger import LOGGER

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

def append_to_crops_index(
    records: List[Dict[str, Any]],
    path: str = "crops_index.csv",
    fieldnames: Optional[List[str]] = None
) -> None:
    """
    Append records to `crops_index.csv` without overwriting existing data.
    - If the file exists and is not empty: use the existing header.
      If `fieldnames` is provided and does not match the existing header, raise an error.
    - If the file does not exist or is empty: determine header from `fieldnames`
      or from the first record and write the header.
    - records: list of dictionaries representing rows to append.
    - path: destination CSV path.
    - fieldnames: optional list of column names to enforce.
    """
    if not records:
        # Nothing to append
        return

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    if file_exists:
        # Read existing header to ensure compatibility
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
        if fieldnames is None:
            fieldnames = existing_fieldnames
        else:
            if existing_fieldnames and fieldnames != existing_fieldnames:
                raise ValueError(
                    f"Provided fieldnames {fieldnames} do not match existing header {existing_fieldnames}"
                )
    else:
        # File does not exist or is empty: derive fieldnames if not provided
        if fieldnames is None:
            # Use keys of the first record as header order
            fieldnames = list(records[0].keys())

    # Open in append or write mode depending on whether the file exists
    mode = "a" if file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            # Write header only when creating the file
            writer.writeheader()
        # Write rows in the order of fieldnames; missing keys produce empty values
        for rec in records:
            row = {k: rec.get(k, "") for k in fieldnames}
            writer.writerow(row)

def find_label_path(rel_img_path: str, labels_dir: Optional[str]) -> Optional[str]:
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

def save_progress(processed_list: List[str], progress_file: str) -> None:
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({"processed": processed_list, "timestamp": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOGGER.exception("Unexpected error during save_progress", progress_file, e)

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

def iter_video_files(src_dir: Path, pattern: str = "*.mkv") -> Generator[Path, None, None]:
    """
    Recursively traverse `src_dir` and yield each file matching `pattern`.
    """
    src = Path(src_dir)
    if not src.exists():
        return
    for p in src.rglob(pattern):
        if p.is_file():
            yield p

def videos_by_match(src_dir: Path) -> Dict[str, List[Path]]:
    """
    Returns a dictionary {match_folder_name: [Path(...), ...]}.
    """
    result: Dict[str, List[Path]] = {}
    for p in iter_video_files(src_dir):
        match_name = p.parent.name
        result.setdefault(match_name, []).append(p)
    return result