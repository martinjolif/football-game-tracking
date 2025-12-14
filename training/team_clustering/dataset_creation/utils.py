import csv
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Generator

from training.logger import LOGGER

def yolo_to_bbox(xc, yc, w, h, img_w, img_h, padding=0.0):
    """
    Convert YOLO-format normalized bbox to integer pixel coordinates (x1, y1, x2, y2).

    Parameters
    ----------
    xc, yc, w, h : float
        YOLO-format center x/y and width/height (normalized to [0, 1]).
    img_w, img_h : int
        Image width and height in pixels.
    padding : float, optional
        Fractional padding to expand the box on each side (default 0.0).

    Returns
    -------
    tuple[int, int, int, int]
        Clipped integer pixel coordinates (x1, y1, x2, y2).
    """
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
    Append records to a CSV file, preserving header order.
    Raises ValueError if provided fieldnames conflict with an existing header.

    Parameters
    ----------
    records : list of dict
        Records to append.
    path : str
        CSV file path.
    fieldnames : list of str, optional
        Header names; inferred from first record if not provided.
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

def save_progress(processed_list: List[str], progress_file: str) -> None:
    """
    Save the list of processed images to a JSON file with a timestamp.

    Parameters
    ----------
    processed_list : list of str
        List of processed image relative paths.
    progress_file : str
        Path to save progress JSON.
    """
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({"processed": processed_list, "timestamp": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOGGER.exception("Unexpected error during save_progress", progress_file, e)

def load_progress(progress_file):
    """
    Load the set of processed images from a progress JSON file.

    Parameters
    ----------
    progress_file : str
        Path to progress JSON.

    Returns
    -------
    set
        Set of processed image paths; empty set if file missing or invalid.
    """
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
    Check if an image has been processed (by relative path or basename).

    Parameters
    ----------
    rel_path : str
        Relative path of the image.
    processed_set : set
        Set of already processed images.

    Returns
    -------
    bool
        True if processed.
    """
    if rel_path in processed_set:
        return True
    if os.path.basename(rel_path) in processed_set:
        return True
    return False

def iter_video_files(src_dir: Path, pattern: str = "*.mkv") -> Generator[Path, None, None]:
    """
    Recursively yield all video files matching a pattern under a directory.

    Parameters
    ----------
    src_dir : Path
        Directory to search.
    pattern : str
        Glob pattern (default '*.mkv').

    Yields
    ------
    Path
        Matching video file paths.
    """
    src = Path(src_dir)
    if not src.exists():
        return
    for p in src.rglob(pattern):
        if p.is_file():
            yield p

def videos_by_match(src_dir: Path) -> Dict[str, List[Path]]:
    """
    Map match folder names to lists of video files found under a directory.

    Parameters
    ----------
    src_dir : Path
        Root directory to search.

    Returns
    -------
    dict
        Keys are parent folder names (matches), values are lists of video Paths.
    """
    result: Dict[str, List[Path]] = {}
    for p in iter_video_files(src_dir):
        match_name = p.parent.name
        result.setdefault(match_name, []).append(p)
    return result