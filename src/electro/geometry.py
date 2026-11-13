import numpy as np
from array_api_compat import array_namespace


def rotation_matrix_3d_y(angle):
    """Computes a 3D matrix that rotates a point about the y-axis.

    Positive angles rotate counterclockwise when axis points toward observer.

    Args:
        angle: Angle to rotate in radians.

    Returns:
        NumPy array with shape (3, 3).
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    out = [
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ]
    return np.array(out)


def rotation_matrix_3d_z(angle):
    """Computes a 3D matrix that rotates a point about the z-axis.

    Positive angles rotate counterclockwise when axis points toward observer.

    Args:
        angle: Angle to rotate in radians.

    Returns:
        NumPy array with shape (3, 3).
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    out = [
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ]
    return np.array(out)


def angular_distances(points_a, points_b):
    """Calculates the angle between pairs of points on the unit sphere.

    The vectors from the center of the sphere to the two points form two sides
    of a rhombus, because both have a length of 1. The diagonals of a rhombus
    are a + b and a - b. They are perpendicular and bisect the angle between
    the two sides, which gives the equation
    tan(angle / 2) = abs(a - b) / abs(a + b).

    This is used instead of the arccos of the dot product. The dot product of
    nearby points is close to 1, where floating-point numbers are too coarse
    to represent the difference. In float32 this makes arccos unreliable at the
    size of the stage tolerances in the optimization.

    Args:
        points_a: Array with shape (..., 3). Points are expected to be on the
            unit sphere.
        points_b: Array with shape (..., 3). Must be broadcastable with
            points_a. Points are expected to be on the unit sphere.

    Returns:
        out: Array of angles in radians with the shape of points_a and points_b
            broadcast together, without the last axis. Values are between 0
            and pi.
    """
    xp = array_namespace(points_a, points_b)
    diff = points_a - points_b
    total = points_a + points_b
    out = 2 * xp.atan2(xp.linalg.vector_norm(diff, axis=-1),
                       xp.linalg.vector_norm(total, axis=-1))
    return out
