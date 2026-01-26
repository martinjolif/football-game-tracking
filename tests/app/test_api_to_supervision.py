import pytest
import numpy as np
import supervision as sv
from app.api_to_supervision import detections_from_results, keypoints_from_pose_results

def test_detections_from_results():
    # returns empty detections when no valid detections
    results = []
    detected_class_ids = [1, 2]
    detections = detections_from_results(results, detected_class_ids)
    assert detections.xyxy.shape == (0, 4)
    assert detections.confidence.shape == (0,)
    assert detections.class_id.shape == (0,)

    # converts valid detections to supervision detections
    results = [
        {
            "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
            "confidence": 0.9,
            "detected_class_id": 1
        },
        {
            "bbox": {"x0": 50, "y0": 60, "x1": 70, "y1": 80},
            "confidence": 0.85,
            "detected_class_id": 2
        }
    ]
    detected_class_ids = [1, 2]
    detections = detections_from_results(results, detected_class_ids)
    assert detections.xyxy.shape == (2, 4)
    assert detections.confidence.shape == (2,)
    assert detections.class_id.shape == (2,)
    assert np.array_equal(detections.xyxy[0], [10, 20, 30, 40])
    assert np.array_equal(detections.xyxy[1], [50, 60, 70, 80])
    assert detections.confidence[0] == 0.9
    assert detections.class_id[0] == 1

    # filters detections by class id
    results = [
        {
            "bbox": {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
            "confidence": 0.9,
            "detected_class_id": 1
        },
        {
            "bbox": {"x0": 50, "y0": 60, "x1": 70, "y1": 80},
            "confidence": 0.85,
            "detected_class_id": 3
        }
    ]
    detected_class_ids = [1]
    detections = detections_from_results(results, detected_class_ids)
    assert detections.xyxy.shape == (1, 4)
    assert detections.class_id[0] == 1

    # ignores detections with missing bbox fields
    results = [
        {
            "bbox": {"x0": 10, "y0": 20},  # missing x1, y1
            "confidence": 0.9,
            "detected_class_id": 1
        }
    ]
    detected_class_ids = [1]
    detections = detections_from_results(results, detected_class_ids)
    assert detections.xyxy.shape == (0, 4)


def test_keypoints_from_pose_results():
    # returns empty keypoints when no poses
    results = {"poses": []}
    keypoints, masks = keypoints_from_pose_results(results)
    assert keypoints.xy.shape == (0, 0, 2)
    assert keypoints.confidence.shape == (0, 0)
    assert keypoints.class_id.shape == (0,)
    assert masks == []

    # converts valid poses to supervision keypoints
    results = {
        "poses": [
            {
                "keypoints": [
                    {"x": 10, "y": 20, "score": 0.9},
                    {"x": 30, "y": 40, "score": 0.85}
                ],
                "detected_class_id": 1
            }
        ]
    }
    keypoints, masks = keypoints_from_pose_results(results)
    assert keypoints.xy.shape == (1, 2, 2)
    assert keypoints.confidence.shape == (1, 2)
    assert keypoints.class_id.shape == (1,)
    assert np.array_equal(keypoints.xy[0][0], [10, 20])
    assert np.array_equal(keypoints.xy[0][1], [30, 40])
    assert keypoints.confidence[0][0] == 0.9
    assert masks[0] == [True, True]

    # filters keypoints by confidence threshold
    results = {
        "poses": [
            {
                "keypoints": [
                    {"x": 10, "y": 20, "score": 0.9},
                    {"x": 30, "y": 40, "score": 0.3}
                ],
                "detected_class_id": 1
            }
        ]
    }
    keypoints, masks = keypoints_from_pose_results(results, confidence_threshold=0.5)
    assert keypoints.xy.shape == (1, 1, 2)
    assert np.array_equal(keypoints.xy[0][0], [10, 20])
    assert masks[0] == [True, False]

    # skips poses with no keypoints above threshold
    results = {
        "poses": [
            {
                "keypoints": [
                    {"x": 10, "y": 20, "score": 0.3},
                    {"x": 30, "y": 40, "score": 0.2}
                ],
                "detected_class_id": 1
            }
        ]
    }
    keypoints, masks = keypoints_from_pose_results(results, confidence_threshold=0.5)
    assert keypoints.xy.shape == (0, 0, 2)

    # handles multiple poses
    results = {
        "poses": [
            {
                "keypoints": [
                    {"x": 10, "y": 20, "score": 0.9}
                ],
                "detected_class_id": 1
            },
            {
                "keypoints": [
                    {"x": 50, "y": 60, "score": 0.8}
                ],
                "detected_class_id": 2
            }
        ]
    }
    keypoints, masks = keypoints_from_pose_results(results)
    assert keypoints.xy.shape == (2, 1, 2)
    assert keypoints.class_id.shape == (2,)
    assert keypoints.class_id[0] == 1
    assert keypoints.class_id[1] == 2
