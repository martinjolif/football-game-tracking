from typing import Optional

import cv2
import numpy as np
import supervision as sv

from src.app.homography import Homography
from src.app.pitch_dimensions import PitchDimensions

def render_pitch_radar(
    player_detection_output: sv.Detections,
    ball_detection_output: sv.Detections,
    pitch_detection_output: sv.KeyPoints,
    keypoint_mask: np.ndarray,
) -> np.ndarray:
    """
        Generates a soccer pitch visualization with player and ball positions overlaid.
        Args:
            player_detection_output (sv.Detections): Player detection results containing anchor coordinates.
            ball_detection_output (sv.Detections): Ball detection results containing anchor coordinates.
            pitch_detection_output (sv.KeyPoints): Keypoint detections for the pitch corners and features.
            keypoint_mask (np.ndarray): Boolean mask selecting valid pitch keypoints.

        Returns:
            np.ndarray: Annotated image of the soccer pitch with player and ball positions.
    """
    pitch_dimensions = PitchDimensions()
    frame_points = pitch_detection_output.xy[0].astype(np.float32)
    pitch_points = np.array(pitch_dimensions.get_vertices())[keypoint_mask].astype(np.float32)
    homography = Homography(source=frame_points, target=pitch_points)

    frame_ball_xy = ball_detection_output.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = homography.transform_points(frame_ball_xy)

    frame_players_xy = player_detection_output.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = homography.transform_points(frame_players_xy)

    annotated_frame = draw_pitch(pitch_dimensions)
    annotated_frame = draw_points_on_pitch(
        config=pitch_dimensions,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_frame
    )
    annotated_frame = draw_points_on_pitch(
        config=pitch_dimensions,
        xy=pitch_players_xy,
        face_color=sv.Color.BLUE,
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame
    )

    return annotated_frame

def draw_pitch(
    config: PitchDimensions,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, and scale.

    Args:
        config (PitchDimensions): Configuration object containing the
            dimensions and layout of the pitch.
        background_color (sv.Color, optional): Color of the pitch background.
            Defaults to sv.Color(34, 139, 34).
        line_color (sv.Color, optional): Color of the pitch lines.
            Defaults to sv.Color.WHITE.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the pitch lines in pixels.
            Defaults to 4.
        point_radius (int, optional): Radius of the penalty spot points in pixels.
            Defaults to 8.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the soccer pitch.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.center_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    config_vertices = config.get_vertices()
    for start, end in config.edges:
        edge_start_point = (int(config_vertices[start - 1][0] * scale) + padding,
                            int(config_vertices[start - 1][1] * scale) + padding)
        edge_end_point = (int(config_vertices[end - 1][0] * scale) + padding,
                          int(config_vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=edge_start_point,
            pt2=edge_end_point,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    center_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=center_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return pitch_image

def draw_points_on_pitch(
    config: PitchDimensions,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (PitchDimensions): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for coordinate in xy:
        scaled_point = (
            int(coordinate[0] * scale) + padding,
            int(coordinate[1] * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch

if __name__ == "__main__":
    config = PitchDimensions()
    pitch = draw_pitch(config)
    sv.plot_image(pitch)