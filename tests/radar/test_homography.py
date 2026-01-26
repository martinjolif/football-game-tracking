import pytest
import numpy as np
from radar.homography import Homography

def test_homography():
    # initializes homography with valid points
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    homography = Homography(source, target)
    assert homography.homography_matrix is not None

    # raises error when source and target shapes differ
    source = np.array([[0, 0], [1, 0], [1, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    with pytest.raises(ValueError, match="Source and target must have the same shape."):
        Homography(source, target)

    # raises error when points are not 2d
    source = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    target = np.array([[0, 0, 1], [2, 0, 1], [2, 2, 1], [0, 2, 1]])
    with pytest.raises(ValueError, match="Source and target points must be 2D coordinates."):
        Homography(source, target)

    # transforms points correctly
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([[0.5, 0.5], [0.25, 0.75]])
    homography = Homography(source, target)
    transformed_points = homography.transform_points(points)
    assert transformed_points.shape == points.shape

    # returns empty array when no points provided
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([])
    homography = Homography(source, target)
    transformed_points = homography.transform_points(points)
    assert transformed_points.size == 0

    # raises error when transforming non 2d points
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([[0.5, 0.5, 1.0]])
    homography = Homography(source, target)
    with pytest.raises(ValueError, match="Points must be 2D coordinates."):
        homography.transform_points(points)

