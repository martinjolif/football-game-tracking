import numpy as np
import supervision as sv

# Initialize annotators (reuse your existing setup)
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

def visualize_frame(frame, detections, tracker=None, show_trace=False):
    """
    Annotates a frame with bounding boxes, labels, and optionally traces.

    Parameters:
        frame (np.ndarray): The original frame from the video.
        detections (sv.Detections): Detections object with xyxy, confidence, class_id, tracker_id.
        tracker (optional): Tracker object, required if show_trace=True.
        show_trace (bool): Whether to draw trace lines for tracked objects.

    Returns:
        annotated_frame (np.ndarray): Annotated frame ready for display.
    """
    annotated_frame = frame.copy()

    # Annotate boxes
    annotated_frame = box_annotator.annotate(annotated_frame, detections)

    # Annotate labels (tracker IDs)
    labels = [str(tid) for tid in detections.tracker_id]
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

    # Optionally annotate traces
    if show_trace and tracker is not None:
        annotated_frame = trace_annotator.annotate(annotated_frame, tracker.tracks)

    return annotated_frame


def detections_from_results(results, confidence_threshold=0):
    """
    Converts API results into a Supervision Detections object.

    Parameters:
        results (dict): Dictionary returned by call_image_apis().
        confidence_threshold (float): Minimum confidence to keep a detection.

    Returns:
        sv.Detections: Detections object with xyxy, confidence, and class_id.
    """
    print(results)
    # Filter detections by confidence
    filtered_detection = [d for d in results if d.get("confidence", 0) > confidence_threshold]
    print(filtered_detection)

    if not filtered_detection:
        return sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))

    # Convert to numpy arrays
    xyxy = np.array([[d['bbox']['x0'], d['bbox']['y0'], d['bbox']['x1'], d['bbox']['y1']] for d in filtered_detection])
    confidences = np.array([d['confidence'] for d in filtered_detection])
    class_ids = np.array([d['detected_class_id'] for d in filtered_detection])

    return sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)