from array_api_compat import array_namespace, device


def pair_weights(i, j, sphere_sizes, alphas, num_points, dtype=None):
    """Calculates the weights for the forces/energies between pairs of points.

    The weight of a pair of points is the sum over all sphere sizes m of
    alpha_m multiplied by the number of spheres of size m that contain both
    points. A sphere of size m is a set of m consecutive points, so a pair of
    points is only contained in spheres of size m when abs(i - j) + 1 <= m.

    Args:
        i: Integer array of point indices (zero-based).
        j: Integer array of point indices (zero-based). Must be
            broadcastable with i. Pairs are symmetric, so i can be greater
            than j. Pairs where i equals j are not meaningful.
        sphere_sizes: Array of the sphere sizes included in the optimization.
        alphas: Array of weightings of the forces/energies of each sphere size.
            Array has the same length as sphere_sizes.
        num_points: Integer specifying the total number of points in the
            optimization.
        dtype: Floating-point data type of the output. Uses float64 if None.

    Returns:
        out: Array of weights with the shape of i and j broadcast together.
    """
    xp = array_namespace(i, j)
    if dtype is None:
        dtype = xp.float64

    # Orders the indices so that first <= last
    first = xp.minimum(i, j)
    last = xp.maximum(i, j)

    out = xp.zeros(first.shape, dtype=dtype, device=device(first))
    # Convert to Python ints and floats to avoid conflict between different
    # types of arrays.
    sphere_sizes = sphere_sizes.tolist()
    alphas = alphas.tolist()
    for m, alpha in zip(sphere_sizes, alphas, strict=True):
        # A sphere that starts at index s contains points s to s + m - 1.
        # So to contain both points, s must satisfy:
        # 1) s <= first
        # 2) s >= last - m + 1
        # Also s must be between 0 and num_points - m.
        s_max = xp.clip(first, max=num_points - m)
        s_min = xp.clip(last - m + 1, min=0)
        # The following can be negative when the points are too far apart to
        # fit in a sphere of size m, so clip minimum to zero.
        num_spheres = xp.clip(s_max - s_min + 1, min=0)
        out += alpha * xp.astype(num_spheres, dtype)

    return out
