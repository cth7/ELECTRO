import numpy as np


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
