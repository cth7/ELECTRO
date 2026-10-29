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


def characteristic_length(sphere_sizes):
    """Calculates the characteristic length of a unit sphere with m points.

    The characteristic length is the side length of a square with area
    4 * pi / m, which is the surface area of the unit sphere divided evenly
    among m points. It is limited to a maximum of 2 (the diameter of the unit
    sphere) which applies to sphere sizes 2 and 3.

    Args:
        sphere_sizes: Array of sphere sizes. The sphere size is the number of
            points in a sphere.

    Returns:
        out: Array of characteristic lengths. Array has the same length as the
            number of stages in the optimization.
    """
    sphere_sizes = np.atleast_1d(sphere_sizes)
    sphere_sizes = np.asarray(sphere_sizes)
    out = np.minimum(2.0, (4 * np.pi / sphere_sizes) ** 0.5)
    return out


def characteristic_angle(sphere_sizes):
    """Calculates the characteristic angle of a unit sphere with m points.

    The characteristic angle is the angle subtended at the center of the unit
    sphere by a chord with the characteristic length.

    Args:
        sphere_sizes: Array of sphere sizes. The sphere size is the number of
            points in a sphere.

    Returns:
        out: Array of characteristic angles in radians. Array has the same
            length as the number of stages in the optimization.
    """
    length = characteristic_length(sphere_sizes)
    out = 2 * np.arcsin(length / 2)
    return out


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
