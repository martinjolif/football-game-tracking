import cv2
import numpy as np
import supervision as sv
from typing import Optional
from pitch_dimensions import PitchDimensions

pitch_dimensions = PitchDimensions()
PLAYER_COLORS = ['#696969', '#FF0000', '#FF6347', '#FFD700']
PLAYER_BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(PLAYER_COLORS),
    thickness=2
)
PLAYER_BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(PLAYER_COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
BALL_COLOR = ["#000000"]
BALL_BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(BALL_COLOR),
    thickness=2
)
BALL_BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(BALL_COLOR),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)

# Define legend items (label -> color)
LEGEND_COLORS = PLAYER_COLORS + list(dict.fromkeys(pitch_dimensions.colors)) + BALL_COLOR
LEGEND_ITEMS = list(zip(['ball', 'goalkeeper', 'player', 'referee', 'left-keypoints', 'center-keypoints', 'right-keypoints', 'ball'], LEGEND_COLORS))


def draw_legend(frame, legend_items, start_pos=(10, 30), box_size=20, spacing=30):
    """
    Draw a simple legend on the frame.

    frame: np.array, the image to draw on
    legend_items: list of tuples (label, hex_color)
    start_pos: top-left corner where the legend starts
    box_size: size of the color box
    spacing: vertical spacing between legend items
    """
    for i, (label, hex_color) in enumerate(legend_items):
        y = start_pos[1] + i * spacing
        color = sv.Color.from_hex(hex_color).as_bgr()
        # Draw color box
        cv2.rectangle(frame, (start_pos[0], y - box_size), (start_pos[0] + box_size, y), color, -1)
        # Draw label text
        cv2.putText(
            frame,
            label,
            (start_pos[0] + box_size + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    return frame

def render_detection_results(
        frame,
        player_detections: sv.Detections = None,
        ball_detection: sv.Detections = None,
        pitch_detection: sv.KeyPoints = None,
        filter: Optional[list[bool]] = None
):
    annotated_frame = frame.copy()
    if filter is None:
        mask = np.ones(len(pitch_dimensions.colors), dtype=bool)
    else:
        mask = np.array(filter)
    if pitch_detection is not None:
        VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
            color=[sv.Color.from_hex(color) for color in np.array(pitch_dimensions.colors)[mask]],
            text_color=sv.Color.from_hex('#FFFFFF'),
            border_radius=5,
            text_thickness=1,
            text_scale=0.5,
            text_padding=5,
        )
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, pitch_detection, np.array(pitch_dimensions.labels)[mask].tolist()
        )
    if player_detections is not None:
        annotated_frame = PLAYER_BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, player_detections
        )
        annotated_frame = PLAYER_BOX_ANNOTATOR.annotate(
            annotated_frame, player_detections
        )
    if ball_detection is not None:
        annotated_frame = BALL_BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, ball_detection
        )
        annotated_frame = BALL_BOX_ANNOTATOR.annotate(
            annotated_frame, ball_detection
        )

    # Draw the legend
    annotated_frame = draw_legend(annotated_frame, LEGEND_ITEMS)

    return annotated_frame



