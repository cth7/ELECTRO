import array_api_compat
import numpy as np
from array_api_compat import array_namespace


def counts_table(sphere_sizes, alphas, num_points):
    """Calculates the table used by pair_weights to calculate the weights.

    A sphere of size m is a set of m consecutive points. Points are indexed by
    zero-based indices.

    First, the counting of the number of spheres is described for a
    non-circular sequence of points. A sphere that starts at index s contains
    points s to s + m - 1, and s must be between 0 and num_points - m. For a
    pair of points with indices i and j where i < j, the spheres of size m that
    contain both points have starting indices s from max(j - m + 1, 0) to
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

    In the second case where the sequence of points is circular (the last point
    is followed by the first point), a sphere can wrap around the ends of the
    sequence. A sphere of size m can start at any index s from 0 to
    num_points - 1 and contains the points s to s + m - 1 modulo num_points.
    There are num_points spheres of each size. There are now two sequence gaps
    between points i and j.
    1)  The first gap j - i is the same as before, and there are
        max(0, m - (j - i)) spheres of size m that contain this gap.
    2)  The second gap num_points - (j - i) is covered by spheres that wrap
        around the ends of the sequence. There are
        max(0, m - (num_points - (j - i))) such spheres.
    Note that a sphere can contain the pair of points in only one of these two
    ways. Both these counts can be non-zero when the sphere size is larger than
    both gaps. The total number of size m spheres that contain both points is
    max(0, m - (j - i)) + max(0, m - (num_points - (j - i))), and the pair
    weight is calculated by T(j - i) + T(num_points - (j - i)).

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


def pair_weights(i, j, num_points, table, circular=True):
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
        circular: Boolean. If True, the sequence of points is circular and
            spheres can wrap around its ends, as described in counts_table.

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

    if circular:
        return lookup(j - i) + lookup(num_points - (j - i))
    else:
        return lookup(j - i) - lookup(j + 1) - lookup(num_points - i)


def _columns(first_i, last_i, max_size, num_points, circular, xp, device):
    """Indices of the points that can share a sphere with points i in a block.

    Args:
        first_i: Integer index of the first point i in the block.
        last_i: Integer index after the last point i in the block.
        max_size: Integer specifying the largest sphere size.
        num_points: Integer specifying the total number of points.
        circular: Boolean. If True, the sequence of points is circular.
        xp: Array namespace.
        device: Device of the indices.

    Returns:
        out: 1D int32 array of point indices (none repeated).
    """
    dtype = xp.int32
    # The first point in the block can only pair with points starting from
    # first_j, and the last point in the block can only pair with points up to
    # but not including last_j
    first_j = first_i - max_size + 1
    last_j = last_i + max_size - 1

    if not circular:
        # Clip the indices to the ends of the sequence
        first_j = max(first_j, 0)
        last_j = min(last_j, num_points)
        return xp.arange(first_j, last_j, dtype=dtype, device=device)

    else:
        # On a circular sequence, the indices can wrap around the ends

        if last_j - first_j >= num_points:
            # When the range of j covers all the points, just calculate all the
            # indices in order and avoid repeating indices
            return xp.arange(num_points, dtype=dtype, device=device)

        # Calculate the (unwrapped) indices then wrap them (if there are any)
        return xp.arange(first_j, last_j, dtype=dtype,
                         device=device) % num_points


def potential_energy(points, sphere_sizes, alphas, circular=True, table=None,
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

    When the sequence is circular, rows can wrap around and appear disjoint.
    Since the bottom left of the matrix is not calculated, j indices that wrap
    on the right side are effectively discarded, while j indices that wrap on
    the left are kept. Pairs that are discarded when j wraps on the right side
    are still counted in the other point's row where j wraps on the left. This
    avoids any double counting.

    Args:
        points: Array with shape (num_points, 3). Points are expected to be on
            the unit sphere.
        sphere_sizes: NumPy array of the sphere sizes included in the
            optimization.
        alphas: NumPy array of weightings of the forces/energies of each
            sphere size. Array has the same length as sphere_sizes.
        circular: Boolean. If True, the sequence of points is circular and
            spheres can wrap around its ends, as described in counts_table.
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

        # Calculate the indices of the pairs of points in this block. int32
        # is faster than int64 on GPUs, which emulate 64-bit integer math.
        # Its range of about 2.1 billion is far more than the number of points.
        i, j = xp.meshgrid(
            xp.arange(first_i, last_i, dtype=xp.int32, device=device),
            _columns(first_i, last_i, max_size, num_points, circular, xp,
                     device),
            indexing="ij")

        # Number of points the pair is apart
        gap = xp.abs(j - i)
        if circular:
            # On a circular sequence, get the shorter way around
            gap = xp.minimum(gap, num_points - gap)

        # Keep each pair once by discarding the lower left triangle, and only
        # keep the pairs that can fit in the largest sphere
        keep = (i < j) & (gap + 1 <= max_size)
        i = i[keep]
        j = j[keep]
        # i and j are now 1D arrays of the same length

        weights = pair_weights(i, j, num_points, table, circular=circular)
        # Using xp.take instead of points[i] and points[j] to conform with the
        # array API standard
        diff = xp.take(points, i, axis=0) - xp.take(points, j, axis=0)
        distances = xp.linalg.vector_norm(diff, axis=1)
        total_energy += float(xp.sum(weights / distances))

    return total_energy


def net_force(points, sphere_sizes, alphas, circular=True, table=None,
              block_size=1024):
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
    When the sequence is circular, these include points j that wrap around
    the ends of the sequence.

    Args:
        points: Array with shape (num_points, 3). Points are expected to be on
            the unit sphere.
        sphere_sizes: NumPy array of the sphere sizes included in the
            optimization.
        alphas: NumPy array of weightings of the forces/energies of each
            sphere size. Array has the same length as sphere_sizes.
        circular: Boolean. If True, the sequence of points is circular and
            spheres can wrap around its ends, as described in counts_table.
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
        # j indices to be included in this block
        columns = _columns(first_i, last_i, max_size, num_points, circular,
                           xp, device)

        # Calculate the indices of the pairs of points in this block. int32
        # is faster than int64 on GPUs, which emulate 64-bit integer math.
        # Its range of about 2.1 billion is far more than the number of points.
        i, j = xp.meshgrid(
            xp.arange(first_i, last_i, dtype=xp.int32, device=device),
            columns,
            indexing="ij")
        # i and j are 2D arrays with shape (num_rows, num_columns)

        weights = pair_weights(i, j, num_points, table, circular=circular)
        # A point exerts no force on itself
        is_self = i == j
        weights = xp.where(is_self, 0, weights)

        # Insert axes so that each row broadcasts against each column
        diff = (xp.expand_dims(points[first_i:last_i, :], axis=1)
                - xp.expand_dims(xp.take(points, columns, axis=0), axis=0))
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
