import math
import time

import numpy as np
from array_api_compat import array_namespace

from . import geometry, physics


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


class Electro:
    """ELECTRO optimizer.

    Performs a multi-stage minimization of the electric potential energy of
    points on the unit sphere. Each stage adds a new, larger sphere size to the
    sum of energies. For example, in the first stage the sphere size of 2 is
    added to the optimization, so the energies between points 1 to 2, 2 to 3,
    3 to 4, etc. are included. In the second stage the sphere size of 3 is
    added. In addition to the energies from the previous stages, the energies
    between points 1 to 3, 2 to 4, 3 to 5, etc. are included.

    Each iteration consists of a gradient update, a projection onto the unit
    sphere and a clipping of angular displacements. When no point moves more
    than the tolerance of the current stage, the optimization advances to the
    next stage. In the last stage, the optimization finishes when no point
    moves more than the tolerance of the last stage.

    Attributes:
        points: Array with shape (num_points, 3) of the current points.
        stage: Index of the current stage (zero-based).
        iteration: Number of completed iterations.
        stage_iteration: Number of completed iterations in the current stage.
        elapsed: Total seconds spent in iterations, excluding the time spent
            evaluating the loss and saving checkpoints.
        converged: True when the last stage met its tolerance.
        stage_starts: List of (iteration, elapsed) at the start of each
            stage that has been reached.
        num_points: Number of points on the sphere.
        num_stages: Number of stages in the optimization.
    """

    def __init__(self, initial_points, sphere_sizes, alphas, stepsizes,
                 max_displacements, tolerances, energy_every=0,
                 block_size=1024):
        """Initializes the optimizer.

        All arrays of parameters have one value per stage.

        Args:
            initial_points: Array with shape (num_points, 3) of points on the
                unit sphere. Its array namespace, device and data type are
                used for the whole optimization.
            sphere_sizes: Array of increasing sphere sizes. Stage k adds the
                sphere size sphere_sizes[k].
            alphas: Array of weightings of the energies of each sphere size.
            stepsizes: Array of step sizes of each stage.
            max_displacements: Array of the maximum angular displacement in
                radians allowed for each stage. Use math.inf to disable
                clipping.
            tolerances: Array of angular displacements in radians. The
                optimization advances past stage k when no point moves more
                than tolerances[k].
            energy_every: Integer. Evaluates the energy function every
                energy_every iterations. 0 never evaluates it. Values not
                evaluated are recorded as NaN. The energy is evaluated after
                the points are updated.
            block_size: Integer passed to the force and energy functions.
                Computations are done in blocks of this size.
        """
        self.sphere_sizes = np.asarray(sphere_sizes, dtype=np.int64)
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.stepsizes = np.asarray(stepsizes, dtype=np.float64)
        self.max_displacements = np.asarray(max_displacements,
                                            dtype=np.float64)
        self.tolerances = np.asarray(tolerances, dtype=np.float64)
        self.energy_every = int(energy_every)
        self.block_size = block_size

        # Attributes
        self.points = initial_points
        self.stage = 0
        self.iteration = 0
        self.stage_iteration = 0
        self.elapsed = 0.0
        self.converged = False
        self.stage_starts = [(0, 0.0)]

    @property
    def num_points(self):
        return self.points.shape[0]

    @property
    def num_stages(self):
        return len(self.sphere_sizes)

    def step(self):
        """Performs one iteration of the current stage.

        Advances to the next stage, or marks the optimization as converged,
        when no point moved more than the tolerance of the current stage.
        """
        if self.converged:
            raise RuntimeError("The optimization has already converged.")

        xp = array_namespace(self.points)
        stage = self.stage

        # Get parameters for the current stage
        sphere_sizes = self.sphere_sizes[:stage + 1]
        alphas = self.alphas[:stage + 1]
        stepsize = self.stepsizes[stage]
        max_displacement = float(self.max_displacements[stage])
        tolerance = float(self.tolerances[stage])

        start_time = time.perf_counter()
        old_points = self.points
        forces = physics.net_force(old_points, sphere_sizes, alphas,
                                   block_size=self.block_size)
        new_points = apply_forces(old_points, forces, stepsize)
        new_points = project_to_sphere(new_points)
        angles = geometry.angular_distances(old_points, new_points)
        new_points = clip_displacements(old_points, new_points,
                                        max_displacement)
        # Clipped points moved exactly max_displacement, so the largest
        # displacement after clipping is known without recomputing angles.
        # Converting to Python scalars also synchronizes GPU devices, so the
        # measured time includes all the queued work.
        largest = min(float(xp.max(angles)), max_displacement)
        num_clipped = int(xp.sum(xp.astype(angles > max_displacement,
                                           xp.int64)))
        step_time = time.perf_counter() - start_time

        if (self.energy_every > 0
                and (self.iteration + 1) % self.energy_every == 0):
            energy = physics.potential_energy(new_points, sphere_sizes, alphas,
                                              block_size=self.block_size)

        self.points = new_points
        self.iteration += 1
        self.stage_iteration += 1
        self.elapsed += step_time

        if largest <= tolerance:
            if stage == self.num_stages - 1:
                self.converged = True
            else:
                self.stage += 1
                self.stage_iteration = 0
                self.stage_starts.append((self.iteration, self.elapsed))

    def run(self, max_iterations):
        """Runs iterations until max_iterations or convergence.

        Args:
            max_iterations: Integer specifying the total number of iterations,
                including those completed before the optimization was saved and
                loaded. Calling run again with the same max_iterations
                continues to the same end point.

        Returns:
            self
        """
        while self.iteration < max_iterations and not self.converged:
            self.step()

        return self
