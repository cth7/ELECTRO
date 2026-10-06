import math

import numpy as np

from . import geometry


def crr(points):
    """Calculates the CRR metric for a set of points in the unit square.

    CRR stands for Clustered-Random-Regular. This metric decreases to 0 as
    points become more clustered. This metric has a value of around 1 when
    points are randomly distributed. This metric increases as points become
    more evenly spaced.

    Args:
        points: Array with shape (num_points, 2). Points are expected to be in
            the unit square.

    References:
        P. J. Clark et al. (1954). Distance to Nearest Neighbor as a Measure of
        Spatial Relationships in Populations. Ecology.
        https://doi.org/10.2307/1931034

        M. J. Dry et al. (2012). Clustering, Randomness, and Regularity:
        Spatial Distributions and Human Performance on the Traveling
        Salesperson Problem and Minimum Spanning Tree Problem. The Journal of
        Problem Solving.
    """
    num_points = len(points)

    # First calculate the distance to the nearest neighbor, for each point
    min_d = np.zeros(num_points, dtype=float)
    for i in range(num_points):
        d = np.linalg.norm(points - points[i], ord=2, axis=1)
        d[i] = np.inf  # Exclude distance to itself from the min calculation
        min_d[i] = d.min()

    # Then calculate the normalizing factor. This is the expected value of the
    # nearest neighbor distance for randomly distributed points.
    expected_value = 0.5 / num_points ** 0.5

    out = min_d.mean() / expected_value
    return out


def nmna(points, polar_angle=0, azimuthal_angle=0, cap_angle=np.pi):
    """Calculates the NMNA metric for a set of points on the unit sphere.

    The Normalized Mean Nearest neighbor Angular distance metric. This metric
    decreases to 0 as points become more clustered. This metric has a value of
    around 1 when points are randomly distributed. This metric increases as
    points become more evenly spaced.

    This metric can be calculated for a region of the sphere. The region is a
    spherical cap. Under default parameters, the metric is calculated for the
    entire sphere.

    Args:
        points: Array with shape (num_points, 3). Points are expected to be on
            the unit sphere. This should include all points, even those not on
            the spherical cap.
        polar_angle: Polar angle coordinate of the position of the spherical
            cap in radians.
        azimuthal_angle: Azimuthal angle coordinate of the position of the
            spherical cap in radians.
        cap_angle: Size of the spherical cap. Angle between the center and the
            edge of the cap in radians.
    """
    num_points = len(points)

    # Need to figure out which points are on the spherical cap
    # First rotate points to the top of the sphere
    # Calculate rotation matrix
    R_y = geometry.rotation_matrix_3d_y(-polar_angle)
    R_z = geometry.rotation_matrix_3d_z(-azimuthal_angle)
    R = R_y @ R_z
    # Rotate
    points_r = R @ points.transpose()
    points_r = points_r.transpose()

    # Points that are higher than this z coordinate are on the cap
    z = np.cos(cap_angle)
    on_cap = points_r[:, 2] >= z

    min_d = np.full(num_points, np.inf, dtype=float)
    for i in range(num_points):
        if not on_cap[i]:
            continue

        # Calculate angles
        d = np.sum(points * points[i], axis=1)
        # Need to clip to handle floating-point errors
        d = np.arccos(np.clip(d, -1.0, 1.0))
        d[i] = np.inf  # Exclude angle to itself from the min calculation
        min_d[i] = d.min()

    # Expected value of the nearest neighbor angular distance for randomly
    # distributed points on the sphere
    K = num_points - 1
    expected_value = math.comb(2 * K, K) / 4 ** K * np.pi

    out = min_d[on_cap].mean() / expected_value
    return out
