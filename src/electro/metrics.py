import math

import array_api_compat
from array_api_compat import array_namespace

from . import geometry


def nearest_neighbors(points, indices, block_size):
    """Finds the nearest neighbor of the points specified by indices.

    Distances are calculated in blocks of rows to limit memory use. Each block
    uses arrays with shape (block_size, num_points).

    Args:
        points: Array with shape (num_points, num_dims).
        indices: Integer array of the indices of the points to find the nearest
            neighbors of.
        block_size: Integer specifying the number of rows in each block.

    Returns:
        out: Integer array with the shape of indices. out[k] is the index of
            the nearest neighbor of the point with index indices[k].
    """
    xp = array_namespace(points, indices)
    device = array_api_compat.device(points)
    num_points, num_dims = points.shape

    # Uses all columns for each block
    columns = xp.arange(num_points, device=device)
    inf = xp.asarray(math.inf, dtype=points.dtype, device=device)

    out = []
    for start in range(0, indices.shape[0], block_size):
        rows = indices[start:start + block_size]
        block = xp.take(points, rows, axis=0)
        # Adding one coordinate (e.g. x, y, z) at a time only needs 2D arrays,
        # which is faster on the CPU than a 3D array of differences. Insert
        # axes so that each row broadcasts against each column.
        distances_sq = xp.zeros((rows.shape[0], num_points),
                                dtype=points.dtype, device=device)
        for k in range(num_dims):
            diff = (xp.expand_dims(block[:, k], axis=1)
                    - xp.expand_dims(points[:, k], axis=0))
            distances_sq += diff * diff
        # Exclude the distance from each point to itself
        is_self = xp.expand_dims(rows, axis=1) == xp.expand_dims(columns,
                                                                 axis=0)
        distances_sq = xp.where(is_self, inf, distances_sq)
        out.append(xp.argmin(distances_sq, axis=1))

    return xp.concat(out, axis=0)


def crr(points, block_size=256):
    """Calculates the CRR metric for a set of points in the unit square.

    CRR stands for Clustered-Random-Regular. This metric decreases to 0 as
    points become more clustered. This metric has a value of around 1 when
    points are randomly distributed. This metric increases as points become
    more evenly spaced.

    Args:
        points: Array with shape (num_points, 2). Points are expected to be in
            the unit square.
        block_size: Integer specifying the number of rows in each block of
            the nearest neighbor search. The numerical result does not depend
            on it.

    Returns:
        out: Python float with the value of the metric.

    References:
        P. J. Clark et al. (1954). Distance to Nearest Neighbor as a Measure of
        Spatial Relationships in Populations. Ecology.
        https://doi.org/10.2307/1931034

        M. J. Dry et al. (2012). Clustering, Randomness, and Regularity:
        Spatial Distributions and Human Performance on the Traveling
        Salesperson Problem and Minimum Spanning Tree Problem. The Journal of
        Problem Solving.
    """
    xp = array_namespace(points)
    device = array_api_compat.device(points)
    num_points = points.shape[0]

    # Include all points in the calculation of CRR, i.e. no regional analysis
    # like the NMNA metric.
    indices = xp.arange(num_points, device=device)
    # Find the indices of the nearest points
    nearest = nearest_neighbors(points, indices, block_size)
    # Calculate the distance to the nearest neighbor
    diff = points - xp.take(points, nearest, axis=0)
    min_d = xp.linalg.vector_norm(diff, axis=1)

    # Then calculate the normalizing factor. This is the expected value of the
    # nearest neighbor distance for randomly distributed points.
    expected_value = 0.5 / num_points ** 0.5

    out = float(xp.mean(min_d)) / expected_value
    return out


def nmna(points, polar_angle=0, azimuthal_angle=0, cap_angle=math.pi,
         block_size=256):
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
        block_size: Integer specifying the number of rows in each block of
            the nearest neighbor search. The numerical result does not depend
            on it.

    Returns:
        out: Python float with the value of the metric, or NaN if no points
            are on the spherical cap.
    """
    xp = array_namespace(points)
    device = array_api_compat.device(points)
    num_points = points.shape[0]

    # Need to figure out which points are on the spherical cap
    # First rotate points to the top of the sphere
    # Calculate rotation matrix
    R_y = geometry.rotation_matrix_3d_y(-polar_angle)
    R_z = geometry.rotation_matrix_3d_z(-azimuthal_angle)
    R = xp.asarray(R_y @ R_z, dtype=points.dtype, device=device)
    # Rotate. Each point is a row, so multiply by the transpose of R.
    points_r = points @ R.T

    # Points that are higher than this z coordinate are on the cap
    z = math.cos(cap_angle)
    # Indices of points on the cap
    indices = xp.nonzero(points_r[:, 2] >= z)[0]
    if indices.shape[0] == 0:
        return math.nan

    # The nearest neighbor by straight-line distance is also the nearest
    # neighbor by angle. So this avoids the additional angle calculation during
    # the nearest neighbor search.
    nearest = nearest_neighbors(points, indices, block_size)
    min_d = geometry.angular_distances(xp.take(points, indices, axis=0),
                                       xp.take(points, nearest, axis=0))

    # Expected value of the nearest neighbor angular distance for randomly
    # distributed points on the sphere
    K = num_points - 1
    expected_value = math.comb(2 * K, K) / 4 ** K * math.pi

    out = float(xp.mean(min_d)) / expected_value
    return out
