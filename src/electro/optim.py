import math

import array_api_compat
from array_api_compat import array_namespace

from . import geometry


def apply_forces(points, forces, stepsize):
    """Moves each point by its net force multiplied by the step size.

    Args:
        points: Array with shape (num_points, 3). Points are expected to be on
            the unit sphere.
        forces: Array of net forces with shape (num_points, 3).
        stepsize: Float specifying the step size for the current stage.

    Returns:
        out: Array of moved points with shape (num_points, 3).
    """
    # Convert to a Python float to avoid conflict between different types of
    # arrays and to keep the data type of points.
    out = points + float(stepsize) * forces
    return out


def project_to_sphere(points):
    """Projects each point onto the unit sphere.

    Args:
        points: Array with shape (num_points, 3).

    Returns:
        out: Array of points on the unit sphere with shape (num_points, 3).
    """
    xp = array_namespace(points)
    out = points / xp.sqrt(xp.sum(points * points, axis=1, keepdims=True))
    return out


def clip_displacements(old_points, new_points, max_displacement):
    """Clips the angle that each point moved, preserving the direction.

    Args:
        old_points: Array with shape (num_points, 3). Points are expected to
            be on the unit sphere.
        new_points: Array with shape (num_points, 3). Points are expected to
            be on the unit sphere.
        max_displacement: Float specifying the maximum angular displacement in
            radians. Use math.inf to disable clipping.

    Returns:
        out: Array of points on the unit sphere with shape (num_points, 3).
    """
    xp = array_namespace(old_points, new_points)
    # No angle is larger than pi, so this has no effect on the result. It keeps
    # the value finite for math.cos and math.sin.
    max_displacement = min(float(max_displacement), math.pi)

    angles = geometry.angular_distances(old_points, new_points)
    is_clipped = xp.expand_dims(angles > max_displacement, axis=1)

    # TODO: Add early exit when no points need clipping

    # Calculate the unit vector at each old point that is tangent to the sphere
    # and points along the great circle toward the new point. Starting from the
    # difference between the points keeps precision for small displacements.
    diff = new_points - old_points
    radial = xp.sum(diff * old_points, axis=1, keepdims=True)
    tangents = diff - radial * old_points
    norms = xp.sqrt(xp.sum(tangents * tangents, axis=1, keepdims=True))
    # A point that did not move has no tangent vector. Avoid dividing by 0.
    norms = xp.where(is_clipped, norms, xp.ones_like(norms))
    tangents = tangents / norms

    clipped_points = (math.cos(max_displacement) * old_points
                      + math.sin(max_displacement) * tangents)
    out = xp.where(is_clipped, clipped_points, new_points)
    # TODO: Considering projecting points back to sphere after clipping
    return out
