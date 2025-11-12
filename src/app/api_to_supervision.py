import numpy as np
import supervision as sv

def detections_from_results(results, detected_class_ids: list[int], confidence_threshold=0):
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
    filtered_detection = [d for d in results if d.get("confidence", 0) > confidence_threshold and d.get("detected_class_id", 0) in detected_class_ids]

    if not filtered_detection:
        return sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))

    # Convert to numpy arrays
    xyxy = np.array([[d['bbox']['x0'], d['bbox']['y0'], d['bbox']['x1'], d['bbox']['y1']] for d in filtered_detection])
    confidences = np.array([d['confidence'] for d in filtered_detection])
    class_ids = np.array([d['detected_class_id'] for d in filtered_detection])

    return sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)

def keypoints_from_pose_results(results, confidence_threshold=0):
    """
    Converts API pose results into a Supervision Keypoints object.
    Returns an empty Keypoints object if no keypoints pass the threshold.
    """
    batch_xy = []
    batch_confidences = []
    batch_class_ids = []
    filter = None

    for pose in results.get("poses", []):
        keypoints = []
        confidences = []
        filter = []

        for kp in pose.get("keypoints", []):
            score = kp.get("score", 0)
            if score > confidence_threshold:
                keypoints.append([kp["x"], kp["y"]])
                confidences.append(score)
                filter.append(True)
            else:
                filter.append(False)

        if not keypoints:
            continue  # skip poses with no keypoints above threshold

        batch_xy.append(np.array(keypoints, dtype=float)[np.newaxis, :, :])       # (1, num_keypoints, 2)
        batch_confidences.append(np.array(confidences, dtype=float)[np.newaxis, :])  # (1, num_keypoints)
        batch_class_ids.append(pose.get("detected_class_id", 0))  # (1,)

    if not batch_xy:
        print("No keypoints pass the threshold.")
        return sv.KeyPoints(
            xy=np.empty((0, 0, 2)),
            confidence=np.empty((0, 0)),
            class_id=np.empty((0,))
        )

    # Stack all poses into batch
    xy = np.vstack(batch_xy)                 # (num_poses, num_keypoints, 2)
    confidences = np.vstack(batch_confidences)  # (num_poses, num_keypoints)
    class_ids = np.array(batch_class_ids, dtype=int)  # (num_poses,)

    if filter is None:
        raise ValueError("No keypoints pass the threshold.")
    return sv.KeyPoints(xy=xy, confidence=confidences, class_id=class_ids), filter
