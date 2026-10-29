import numpy as np

from . import seqs


def sphere_sizes_paper(num_points):
    """Returns an array of sphere sizes.

    The sphere size is the number of points in a sphere. The length of the
    array is the number of stages in the optimization. Each term in the array
    is the sphere size to be added for that stage.

    The sphere sizes are the terms in Narayana's cows sequence that are
    greater than 1 and no larger than num_points. If the last of these terms
    is not num_points, num_points is appended.

    Args:
        num_points: Integer specifying the total number of points in the
            optimization.
    """
    tmp = seqs.narayanas_cows(116)
    # Congrats if you can optimize more points than this

    out = [m for m in tmp if m > 1 and m <= num_points]

    if out[-1] != num_points:
        out.append(num_points)

    return np.array(out)


def paper(num_points):
    """Returns all parameter values used in the paper.

    Args:
        num_points: Integer specifying the total number of points in the
            optimization.

    Returns:
        out: Dictionary containing parameter values.
    """
    out = {}
    out["sphere_sizes"] = sphere_sizes_paper(num_points)
    return out
