import cv2
import numpy as np

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initialize the Homography with source and target points.
        Args:
            source: Source points for homography calculation.
            target: Target points for homography calculation.
        Raises:
            ValueError: If source and target do not have the same shape or if they are not 2D coordinates.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform the given points using the homography matrix.
        Args:
            points: Points to be transformed.
        Returns:
            Transformed points.
        Raises:
            ValueError: If points are not 2D coordinates.
        """
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)
