import array_api_compat
import numpy as np
from array_api_compat import array_namespace


def counts_table(sphere_sizes, alphas, num_points):
    """Calculates the table used by pair_weights to calculate the weights.

    A sphere of size m is a set of m consecutive points. Points are indexed by
    the zero-based indices. A sphere that starts at index s contains points
    s to s + m - 1, and s must be between 0 and num_points - m. For a pair of
    points with indices i and j where i < j, the spheres of size m that contain
    both points have starting indices s from max(j - m + 1, 0) to
    min(i, num_points - m). The number of these spheres is the sum of three
    types of counts:
    1)  A count of the number of spheres ignoring the ends of the sequence,
        max(0, m - (j - i)).
    2)  Minus a count of the number of spheres that would start before the
        first point in the sequence, max(0, m - (j + 1)).
    3)  Minus a count of the number of spheres that would end after the last
        point in the sequence, max(0, m - (num_points - i)).
    These three counts all have the form max(0, m - x). The value of x is
    generally different for each of the three types of counts, and depends on
    the variables i, j and num_points. Given a pair of points i and j from a
    set of num_points, for a type of count the value of x remains the same when
    performing the count for different sphere sizes. When calculating the
    weight for a pair of points, typically these three counts are first summed,
    and then multiplied by alpha, and then repeated and summed for each sphere
    size. However, using the linearity of this calculation, alpha can be
    distributed to each of the three counts first, then the summation over
    sphere sizes can be performed separately for each of the three types of
    counts, and finally the resulting three terms can be summed to calculate
    the weight. The reason for performing the sphere size summation for each
    type of count separately is that it can be pre-calculated, stored in a
    table, and then looked up by the value of x (recall that the value of x is
    the same throughout each type of count). The value of x is an integer from
    0 to num_points.

    For a given set of sphere sizes, alphas, and the number of points, the
    table holds the value T(x) = sum over m of alpha_m * max(0, m - x). The
    pair weight is then calculated by T(j - i) - T(j + 1) - T(num_points - i).

    Args:
        sphere_sizes: NumPy array of the sphere sizes included in the
            optimization.
        alphas: NumPy array of weightings of the forces/energies of each
            sphere size. Array has the same length as sphere_sizes.
        num_points: Integer specifying the total number of points in the
            optimization.

    Returns:
        table: NumPy float64 array with shape (num_points + 1,).
    """
    x = np.arange(num_points + 1)
    table = np.zeros(num_points + 1)
    for m, alpha in zip(sphere_sizes.tolist(), alphas.tolist(), strict=True):
        table += alpha * np.maximum(0, m - x)
    return table


def pair_weights(i, j, num_points, table):
    """Calculates the weights for the forces/energies between pairs of points.

    The weight of a pair of points is the sum over all sphere sizes m of
    alpha_m multiplied by the number of spheres of size m that contain both
    points.

    Args:
        i: Integer array of point indices (zero-based).
        j: Integer array of point indices (zero-based). Must be
            broadcastable with i. Pairs are symmetric, so i can be greater
            than j. Pairs where i equals j are not meaningful.
        num_points: Integer specifying the total number of points in the
            optimization.
        table: Array with shape (num_points + 1,) calculated by counts_table.
            Must have the array namespace and device of i and j.

    Returns:
        out: Array of weights with the shape of i and j broadcast together,
            and the data type of table.
    """
    xp = array_namespace(i, j, table)

    # Orders the indices so that i <= j
    i, j = xp.minimum(i, j), xp.maximum(i, j)

    def lookup(x):
        # take only accepts 1D indices in the array API standard
        shape = x.shape
        x = xp.reshape(x, (-1,))
        t = xp.take(table, x)
        return xp.reshape(t, shape)

    return lookup(j - i) - lookup(j + 1) - lookup(num_points - i)


def potential_energy(points, sphere_sizes, alphas, table=None,
                     block_size=1024):
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
        sphere_sizes: NumPy array of the sphere sizes included in the
            optimization.
        alphas: NumPy array of weightings of the forces/energies of each
            sphere size. Array has the same length as sphere_sizes.
        table: Array calculated by counts_table from sphere_sizes, alphas and
            num_points, with the array namespace, device and data type of
            points. Calculated in this function if None.
        block_size: Integer specifying the number of points in each block.

    Returns:
        total_energy: Python float containing the value of the objective
            function.
    """
    xp = array_namespace(points)
    device = array_api_compat.device(points)
    num_points = points.shape[0]
    max_size = int(sphere_sizes.max())
    if table is None:
        table = counts_table(sphere_sizes, alphas, num_points)
        table = xp.asarray(table, dtype=points.dtype, device=device)

    total_energy = 0.0
    for first_i in range(0, num_points, block_size):
        # This index last_i is used to calculate the end of the block, which
        # has points up to but not including last_i
        last_i = min(first_i + block_size, num_points)
        first_j = first_i
        # The last point in the block can only pair with points up to but not
        # including last_j
        last_j = min(last_i + max_size - 1, num_points)

        # Calculate the indices of the pairs of points in this block. int32
        # is faster than int64 on GPUs, which emulate 64-bit integer math.
        # Its range of about 2.1 billion is far more than the number of points.
        i, j = xp.meshgrid(
            xp.arange(first_i, last_i, dtype=xp.int32, device=device),
            xp.arange(first_j, last_j, dtype=xp.int32, device=device),
            indexing="ij")
        # Keep each pair once by discarding the lower left triangle, and only
        # keep the pairs that can fit in the largest sphere
        keep = (i < j) & (j - i + 1 <= max_size)
        i = i[keep]
        j = j[keep]
        # i and j are now 1D arrays of the same length

        weights = pair_weights(i, j, num_points, table)
        # Using xp.take instead of points[i] and points[j] to conform with the
        # array API standard
        diff = xp.take(points, i, axis=0) - xp.take(points, j, axis=0)
        distances = xp.linalg.vector_norm(diff, axis=1)
        total_energy += float(xp.sum(weights / distances))

    return total_energy


def net_force(points, sphere_sizes, alphas, table=None, block_size=1024):
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
        sphere_sizes: NumPy array of the sphere sizes included in the
            optimization.
        alphas: NumPy array of weightings of the forces/energies of each
            sphere size. Array has the same length as sphere_sizes.
        table: Array calculated by counts_table from sphere_sizes, alphas and
            num_points, with the array namespace, device and data type of
            points. Calculated in this function if None.
        block_size: Integer specifying the number of points in each block.

    Returns:
        out: Array of forces with shape (num_points, 3).
    """
    xp = array_namespace(points)
    device = array_api_compat.device(points)
    num_points = points.shape[0]
    max_size = int(sphere_sizes.max())
    if table is None:
        table = counts_table(sphere_sizes, alphas, num_points)
        table = xp.asarray(table, dtype=points.dtype, device=device)

    net_forces = []
    for first_i in range(0, num_points, block_size):
        # This index last_i is used to calculate the end of the block, which
        # has points up to but not including last_i
        last_i = min(first_i + block_size, num_points)
        # The first point in the block can only pair with points starting at
        # first_j, and the last point in the block can only pair with points
        # up to but not including last_j
        first_j = max(first_i - max_size + 1, 0)
        last_j = min(last_i + max_size - 1, num_points)

        # Calculate the indices of the pairs of points in this block. int32
        # is faster than int64 on GPUs, which emulate 64-bit integer math.
        # Its range of about 2.1 billion is far more than the number of points.
        i, j = xp.meshgrid(
            xp.arange(first_i, last_i, dtype=xp.int32, device=device),
            xp.arange(first_j, last_j, dtype=xp.int32, device=device),
            indexing="ij")
        # i and j are 2D arrays with shape (num_rows, num_columns)

        weights = pair_weights(i, j, num_points, table)
        # A point exerts no force on itself
        is_self = i == j
        weights = xp.where(is_self, 0, weights)

        # Insert axes so that each row broadcasts against each column
        diff = (xp.expand_dims(points[first_i:last_i, :], axis=1)
                - xp.expand_dims(points[first_j:last_j, :], axis=0))
        distances_sq = xp.sum(diff * diff, axis=2)
        # The distance from a point to itself is 0. Its weight is 0, so any
        # non-zero distance avoids dividing by 0 without changing the result.
        distances_sq = xp.where(is_self, 1, distances_sq)

        scale = weights / (distances_sq * xp.sqrt(distances_sq))
        net_forces.append(xp.sum(xp.expand_dims(scale, axis=2) * diff, axis=1))

    # TODO: The net force can be split into a radial component and a tangential
    # component. A force with a large radial component moves the point further
    # off the sphere, which reduces the magnitude of its angular movement. This
    # is the negative gradient of the energy and is what was used in the paper.
    # It may be worth trying to discard the radial component and only use the
    # tangential component, which would also require readjusting the step size.

    return xp.concat(net_forces, axis=0)
