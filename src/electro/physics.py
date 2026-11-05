import array_api_compat
from array_api_compat import array_namespace


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
    device = array_api_compat.device(i)
    if dtype is None:
        dtype = xp.float64

    # Orders the indices so that first <= last
    first = xp.minimum(i, j)
    last = xp.maximum(i, j)

    out = xp.zeros(first.shape, dtype=dtype, device=device)
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


def potential_energy(points, sphere_sizes, alphas, block_size=1024):
    """Calculates the objective function G(S) for a set of points.

    This is the sum over all pairs of points of the pair weight divided by the
    distance between the points. Only pairs of points that are both contained
    in a sphere have non-zero weights, so other pairs are skipped.

    The calculation of all pairs of points can be visualized as filling a
    square matrix, where the indices of the rows, i, correspond to the first
    point in the pair and the indices of the columns, j, correspond to the
    second point in the pair. The matrix is symmetric so only the upper right
    triangle is calculated to avoid redundancy. The diagonal is ignored.

    Pairs are processed in blocks to limit memory use. The result does not
    depend on block_size. A block is a set of consecutive rows in the matrix.
    The rows are truncated/shortened to only include pairs of points that can
    fit in a sphere of the largest size.

    Args:
        points: Array with shape (num_points, 3). Points are expected to be on
            the unit sphere.
        sphere_sizes: Array of the sphere sizes included in the optimization.
        alphas: Array of weightings of the forces/energies of each sphere size.
            Array has the same length as sphere_sizes.
        block_size: Integer specifying the number of points in each block.

    Returns:
        total_energy: Python float containing the value of the objective
            function.
    """
    xp = array_namespace(points)
    device = array_api_compat.device(points)
    num_points = points.shape[0]
    max_size = int(sphere_sizes.max())

    total_energy = 0.0
    for start in range(0, num_points, block_size):
        stop = min(start + block_size, num_points)
        # The last point in the block can only pair with points up to but not
        # including last_j
        last_j = min(stop + max_size - 1, num_points)

        # Calculate the indices of the pairs of points in this block
        i, j = xp.meshgrid(xp.arange(start, stop, device=device),
                           xp.arange(start, last_j, device=device),
                           indexing="ij")
        # Keep each pair once by discarding the lower left triangle, and only
        # keep the pairs that can fit in the largest sphere
        keep = (i < j) & (j - i + 1 <= max_size)
        i = i[keep]
        j = j[keep]
        # i and j are now 1D arrays of the same length

        weights = pair_weights(i, j, sphere_sizes, alphas, num_points,
                               dtype=points.dtype)
        # Using xp.take instead of points[i] and points[j] to conform with the
        # array API standard
        diff = xp.take(points, i, axis=0) - xp.take(points, j, axis=0)
        distances = xp.sqrt(xp.sum(diff * diff, axis=1))
        total_energy += float(xp.sum(weights / distances))

    return total_energy


def net_force(points, sphere_sizes, alphas, block_size=1024):
    """Calculates the net force on each point.

    The force on point i is the sum over all other points of the pair weight
    multiplied by a vector pointing from point j to point i and divided by the
    cube of the distance between them. Only pairs of points that are both
    contained in a sphere have non-zero weights, so other pairs are skipped.

    Similar to potential_energy, the calculation of all pairs of points can be
    visualized as filling a square matrix, where the indices of the rows, i,
    correspond to the point the force acts on and the indices of the columns,
    j, correspond to the point exerting the force. Unlike potential_energy,
    each row needs the columns on both sides of the diagonal, so each pair of
    points is calculated twice. This avoids having to scatter the force of
    each pair to two different rows, which is slow and not part of the array
    API standard.

    Pairs are processed in blocks to limit memory use. The result does not
    depend on block_size. A block is a set of consecutive rows in the matrix.
    The rows are truncated/shortened to only include points j that can fit in a
    sphere of the largest size with at least one of the points i in the block.

    Args:
        points: Array with shape (num_points, 3). Points are expected to be on
            the unit sphere.
        sphere_sizes: Array of the sphere sizes included in the optimization.
        alphas: Array of weightings of the forces/energies of each sphere size.
            Array has the same length as sphere_sizes.
        block_size: Integer specifying the number of points in each block.

    Returns:
        out: Array of forces with shape (num_points, 3).
    """
    xp = array_namespace(points)
    device = array_api_compat.device(points)
    num_points = points.shape[0]
    max_size = int(sphere_sizes.max())

    net_forces = []
    for start in range(0, num_points, block_size):
        stop = min(start + block_size, num_points)
        # The first point in the block can only pair with points starting at
        # first_j, and the last point in the block can only pair with points
        # up to but not including last_j
        first_j = max(start - max_size + 1, 0)
        last_j = min(stop + max_size - 1, num_points)

        # Calculate the indices of the pairs of points in this block
        i, j = xp.meshgrid(xp.arange(start, stop, device=device),
                           xp.arange(first_j, last_j, device=device),
                           indexing="ij")
        # i and j are 2D arrays with shape (num_rows, num_columns)

        weights = pair_weights(i, j, sphere_sizes, alphas, num_points,
                               dtype=points.dtype)
        # A point exerts no force on itself
        is_pair = i != j
        weights = xp.where(is_pair, weights, xp.zeros_like(weights))

        # Insert axes so that each row broadcasts against each column
        diff = (xp.expand_dims(points[start:stop, :], axis=1)
                - xp.expand_dims(points[first_j:last_j, :], axis=0))
        distances_sq = xp.sum(diff * diff, axis=2)
        # The distance from a point to itself is 0. Its weight is 0, so any
        # non-zero distance avoids dividing by 0 without changing the result.
        distances_sq = xp.where(is_pair, distances_sq,
                                xp.ones_like(distances_sq))

        scale = weights / (distances_sq * xp.sqrt(distances_sq))
        net_forces.append(xp.sum(xp.expand_dims(scale, axis=2) * diff, axis=1))

    # TODO: The net force can be split into a radial component and a tangential
    # component. A force with a large radial component moves the point further
    # off the sphere, which reduces the magnitude of its angular movement. This
    # is the negative gradient of the energy and is what was used in the paper.
    # It may be worth trying to discard the radial component and only use the
    # tangential component, which would also require readjusting the step size.

    return xp.concat(net_forces, axis=0)
