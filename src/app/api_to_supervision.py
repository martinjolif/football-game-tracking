import logging

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

def detections_from_results(
        results: dict,
        detected_class_ids: list[int],
        confidence_threshold: float = 0
) -> sv.Detections:
    """
    Converts API results into a Supervision Detections object.

    Parameters:
        results (dict): Dictionary returned by call_image_apis().
        detected_class_ids (list): List of class ids to select, otherwise it will be ignored.
        confidence_threshold (float): Minimum confidence to keep a detection.

    Returns:
        sv.Detections: Detections object with xyxy, confidence, and class_id.
    """
    # Filter detections by confidence
    filtered_detections = [
        d for d in results
        if d.get("confidence", 0) > confidence_threshold and d.get("detected_class_id", 0) in detected_class_ids
    ]

    valid_detections = [
        d for d in filtered_detections
        if all(k in d.get('bbox', {}) for k in ('x0', 'y0', 'x1', 'y1'))
    ]

    if not valid_detections:
        return sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))

    xyxy = np.array([
        [
            d['bbox']['x0'],
            d['bbox']['y0'],
            d['bbox']['x1'],
            d['bbox']['y1']
        ]
        for d in valid_detections
    ])
    confidences = np.array([d['confidence'] for d in valid_detections])
    class_ids = np.array([d['detected_class_id'] for d in valid_detections])

    return sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)

def keypoints_from_pose_results(
        results: dict,
        confidence_threshold: float = 0
) -> tuple[sv.KeyPoints, list[list[bool]]]:
    """
    Converts API pose results into a Supervision Keypoints object.
    Parameters:
        results (dict): Dictionary returned by call_image_apis(), expected to contain a
                        "poses" key with a list of pose dicts.
        confidence_threshold (float): Minimum confidence to keep a keypoint.

    Returns:
        tuple:
            - sv.KeyPoints: only contain keypoints that have passed the threshold: they
              are therefore shorter than the mask if some keypoints have been excluded.
            - list[list[bool]]: has a length equal to the number of original keypoints
              for each pose. `True` means the keypoint passed the threshold (and is
              included in the returned KeyPoints), `False` means it was filtered out.
    """
    poses_xy = []
    poses_confidences = []
    poses_class_ids = []
    poses_keypoint_masks = []

    for pose in results.get("poses", []):
        keypoints = []
        confidences = []
        keypoint_mask = []

        for kp in pose.get("keypoints", []):
            score = kp.get("score", 0)
            if score > confidence_threshold:
                keypoints.append([kp.get("x", 0), kp.get("y", 0)])
                confidences.append(score)
                keypoint_mask.append(True)
            else:
                keypoint_mask.append(False)

        if not keypoints:
            continue  # skip poses with no keypoints above threshold

        poses_xy.append(np.array(keypoints, dtype=float)[np.newaxis, :, :])       # (1, num_keypoints, 2)
        poses_confidences.append(np.array(confidences, dtype=float)[np.newaxis, :])  # (1, num_keypoints)
        poses_class_ids.append(pose.get("detected_class_id", 0))  # (1,)
        poses_keypoint_masks.append(keypoint_mask)

    if not poses_xy:
        logger.warning("No keypoints pass the threshold.")
        return sv.KeyPoints(
            xy=np.empty((0, 0, 2)),
            confidence=np.empty((0, 0)),
            class_id=np.empty((0,))
        ), []

    # Stack all poses into batch
    xy = np.vstack(poses_xy)                 # (num_poses, num_keypoints, 2)
    confidences = np.vstack(poses_confidences)  # (num_poses, num_keypoints)
    class_ids = np.array(poses_class_ids, dtype=int)  # (num_poses,)

    return sv.KeyPoints(xy=xy, confidence=confidences, class_id=class_ids), poses_keypoint_masks
