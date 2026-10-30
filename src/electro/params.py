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


def characteristic_lengths(sphere_sizes):
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


def characteristic_angles(sphere_sizes):
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
    length = characteristic_lengths(sphere_sizes)
    out = 2 * np.arcsin(length / 2)
    return out


def alphas(sphere_sizes):
    """Calculates the weighting for the energies of spheres of each size.

    The weighting is the cube of the characteristic length. The force between
    a pair of points at the characteristic length apart is proportional to
    1 / length ** 2, so this weighting makes the forces from each sphere size
    relative to its characteristic length. This effectively scales up the
    forces in smaller spheres and scales down the forces in larger spheres.

    Args:
        sphere_sizes: Array of sphere sizes. The sphere size is the number of
            points in a sphere.

    Returns:
        out: Array of weightings. Array has the same length as the number of
            stages in the optimization.
    """
    out = characteristic_lengths(sphere_sizes) ** 3
    return out


def stepsizes_paper(sphere_sizes, num_points, scale=0.08):
    """Calculates the step size for each stage of the optimization.

    Points in the middle of the sequence are contained in more spheres than
    points at the ends, so the forces on them are larger. The step size
    approximately averages the forces by dividing by the number of spheres
    that contain the middle point. For a sphere size m, this number is
    min(m, num_points - m + 1). The sum is over the sphere sizes included in
    each stage. The scale was determined empirically.

    Args:
        sphere_sizes: Array of sphere sizes. The sphere size is the number of
            points in a sphere.
        num_points: Integer specifying the total number of points in the
            optimization.
        scale: Numerator of the step size.

    Returns:
        out: Array of step sizes. Array has the same length as the number of
            stages in the optimization.
    """
    sphere_sizes = np.atleast_1d(sphere_sizes)
    num_spheres = np.minimum(sphere_sizes, num_points - sphere_sizes + 1)
    # Each stage includes the sphere sizes of all previous stages
    out = scale / np.cumsum(num_spheres)
    return out


def max_displacements_paper(sphere_sizes, fraction=0.5):
    """Calculates the maximum angular displacement for each stage.

    When a stage adds a new sphere size, points that did not previously
    interact may be very close to each other. The resulting large forces would
    displace points far from their near optimal positions, so angular
    displacements larger than this value get clipped. The value is a fraction
    of the characteristic angle of the largest sphere size in the stage.

    Args:
        sphere_sizes: Array of sphere sizes. The sphere size is the number of
            points in a sphere.
        fraction: Fraction of the characteristic angle.

    Returns:
        out: Array of maximum angular displacements in radians. Array has the
            same length as the number of stages in the optimization.
    """
    out = fraction * characteristic_angles(sphere_sizes)
    return out


def tolerances_paper(sphere_sizes, fraction=0.01):
    """Calculates the criterion for advancing past each stage.

    The optimization advances to the next stage when no point moved more than
    this value. The value is a fraction of the characteristic angle of the
    sphere size to be added in the next stage. The last stage has no next
    stage, so its value is 0.

    Args:
        sphere_sizes: Array of sphere sizes. The sphere size is the number of
            points in a sphere.
        fraction: Fraction of the characteristic angle.

    Returns:
        out: Array of tolerances in radians. Array has the same length as the
            number of stages in the optimization.
    """
    next_angles = characteristic_angles(sphere_sizes)[1:]
    out = np.append(fraction * next_angles, 0.0)
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
    out["alphas"] = alphas(out["sphere_sizes"])
    out["stepsizes"] = stepsizes_paper(out["sphere_sizes"], num_points)
    out["max_displacements"] = max_displacements_paper(out["sphere_sizes"])
    out["tolerances"] = tolerances_paper(out["sphere_sizes"])
    return out
