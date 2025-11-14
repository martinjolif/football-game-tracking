import numpy as np
import supervision as sv

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

box_annotator = sv.BoxAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK, thickness=2, trace_length=100)
label_annotator = sv.LabelAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.BLACK,
    text_scale=0.8
)

def visualize_frame(
        frame: np.ndarray,
        detections: sv.Detections,
        tracker: sv.ByteTrack = None,
        show_trace: bool = False
):
    """
    Annotates a frame with bounding boxes, labels, and optionally traces.

    Parameters:
        frame (np.ndarray): The original frame from the video.
        detections (sv.Detections): Detections object with xyxy, confidence, class_id, tracker_id.
        tracker (optional): Tracker object
        show_trace (bool): Whether to draw trace lines for tracked objects.

    Returns:
        annotated_frame (np.ndarray): Annotated frame ready for display.
    """
    annotated_frame = frame.copy()

    # Annotate boxes
    annotated_frame = box_annotator.annotate(annotated_frame, detections)

    # Annotate labels (tracker IDs)
    labels = [str(tracker_id) for tracker_id in detections.tracker_id]
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    # Optionally annotate traces
    if show_trace and tracker is not None:
        annotated_frame = trace_annotator.annotate(annotated_frame, detections)

    return annotated_frame
