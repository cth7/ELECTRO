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
