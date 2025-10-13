from enum import StrEnum

import numpy as np


class DistanceMode(StrEnum):
    """Distance calculation modes."""

    EUCLIDEAN = "Euclidean"
    MANHATTAN = "Manhattan"


def tup_distance(
    node1: tuple[float, float],
    node2: tuple[float, float],
    mode: DistanceMode = DistanceMode.EUCLIDEAN,
) -> float:
    """
    Calculate distance between two points.

    Args:
        node1: First point (x, y)
        node2: Second point (x, y)
        mode: Distance mode (Euclidean or Manhattan)
    """
    if mode == DistanceMode.EUCLIDEAN:
        return float(((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5)
    elif mode == DistanceMode.MANHATTAN:
        return float(np.abs(node1[0] - node2[0]) + np.abs(node1[1] - node2[1]))
    else:
        raise ValueError(f"Unrecognized distance mode: {mode}")


def mat_distance(
    mat1: np.ndarray, mat2: np.ndarray, mode: DistanceMode = DistanceMode.EUCLIDEAN
) -> np.ndarray:
    """
    Calculate distance between arrays of points.

    Args:
        mat1: First array of points
        mat2: Second array of points
        mode: Distance mode (Euclidean or Manhattan)
    """
    if mode == DistanceMode.EUCLIDEAN:
        result: np.ndarray = np.sum((mat1 - mat2) ** 2, axis=-1) ** 0.5
        return result
    elif mode == DistanceMode.MANHATTAN:
        result_: np.ndarray = np.sum(np.abs(mat1 - mat2), axis=-1)
        return result_
    else:
        raise ValueError(f"Unrecognized distance mode: {mode}")
