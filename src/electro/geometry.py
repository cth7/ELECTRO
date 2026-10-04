import numpy as np


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
