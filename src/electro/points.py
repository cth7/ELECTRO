import numpy as np


def square(point, num_points, offset=0, n0=0, dtype=np.float64):
    """Generates an incremental sequence of points that wrap the unit square.

    Args:
        point: The sequence is incremented by this value.
        num_points: Total number of points to generate in this sequence.
        offset: The sequence is shifted by this value.
        n0: By default the sequence starts with n=0, 1, 2,... Change this
            integer to start the sequence at a later point. For example, use 2
            to start with n=2, 3, 4,...

    Returns:
        out: Array with shape (num_points, len(point)).
    """
    point = np.array(point, dtype=dtype)
    if offset != 0:
        offset = np.array(offset, dtype=dtype)
        assert offset.shape == point.shape

    n = np.arange(n0, num_points + n0).reshape((-1, 1))
    out = np.mod(n * point + offset, 1)
    return out


def square_to_sphere(points, half_sphere=False):
    """Maps a set of points in the unit square to the unit sphere.

    Args:
        points: Array with shape (num_points, 2).
        half_sphere: Boolean. If true points get mapped to the top hemisphere
            only.

    Returns:
        out: Array with shape (num_points, 3) corresponding to (x, y, z)
            coordinates.
    """
    z = 1 - points[:, 0] if half_sphere else 1 - 2 * points[:, 0]
    r = (1 - z ** 2) ** 0.5
    azimuthal_angle = 2 * np.pi * points[:, 1]
    x = r * np.cos(azimuthal_angle)
    y = r * np.sin(azimuthal_angle)
    out = np.stack([x, y, z], axis=1)
    return out


def random_square(num_points, rng=None):
    """Generates a random set of points in the unit square.

    Generates from a uniform distribution.

    Args:
        num_points: Integer specifying the number of points.
        rng: NumPy Generator instance.

    Returns:
        Array with shape (num_points, 2).
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.uniform(low=0, high=1, size=(num_points, 2))
