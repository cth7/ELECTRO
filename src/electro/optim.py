import math
import time
from pathlib import Path

import numpy as np
from array_api_compat import array_namespace

from . import geometry, io, physics


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

    When save_dir is given, the optimization is saved to that directory as it
    runs. The contents of the directory are described in electro.io.
    Electro.resume loads the optimization from the latest saved points.
    Resuming produces the same points as an uninterrupted optimization.

    Attributes:
        points: Array with shape (num_points, 3) of the current points.
        iteration: Number of completed iterations.
        stage: Index of the current stage (zero-based).
        stage_iteration: Number of completed iterations in the current stage.
        elapsed_seconds: Total seconds spent in iterations, excluding the time
            spent evaluating the energy function and saving.
        converged: True when the last stage met its tolerance.
        save_dir: Path of the save directory, or None if not saving.
        num_points: Number of points on the sphere.
        num_stages: Number of stages in the optimization.
        points_dtype: Name of the data type of the points.
    """

    def __init__(self, initial_points, sphere_sizes, alphas, stepsizes,
                 max_displacements, tolerances, save_dir=None, points_every=0,
                 energy_every=0, block_size=1024):
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
            save_dir: Directory to save the optimization to. It must not
                already contain a saved optimization. Nothing is saved if None.
            points_every: Integer. Saves the points every points_every
                iterations. 0 never saves them periodically. The points are
                also saved when run returns or is interrupted.
            energy_every: Integer. Evaluates the energy function every
                energy_every iterations. 0 never evaluates it. Values not
                evaluated are recorded as None. The energy is evaluated after
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
        self.points_every = int(points_every)
        self.energy_every = int(energy_every)
        self.block_size = block_size

        # Attributes
        self.points = initial_points
        self.iteration = 0
        self.stage = 0
        self.stage_iteration = 0
        self.elapsed_seconds = 0.0
        self.converged = False

        # Saving
        self.save_dir = None
        self._history_file = None
        if save_dir is not None:
            self._start_saving(Path(save_dir))

    @property
    def num_points(self):
        return int(self.points.shape[0])

    @property
    def num_stages(self):
        return len(self.sphere_sizes)

    @property
    def points_dtype(self):
        # The array API standard does not define how a data type is printed
        # (e.g. PyTorch prints "torch.float64"), so the name is found by
        # comparing with the data types of the array namespace. Names the
        # namespace does not have are skipped rather than compared with None,
        # because NumPy treats None as float64. If nothing matches, the
        # library's own name is used.
        xp = array_namespace(self.points)
        names = [name for name in ("float16", "bfloat16", "float32", "float64")
                 if hasattr(xp, name)
                 and self.points.dtype == getattr(xp, name)]
        return names[0] if names else str(self.points.dtype)

    def step(self):
        """Performs one iteration of the current stage.

        Advances to the next stage, or marks the optimization as converged,
        when no point moved more than the tolerance of the current stage.

        Returns:
            record: Dictionary of the values used and calculated in this
                iteration. This is also the record written to the history file.
                iteration: Number of completed iterations, including this one.
                stage: Index of the stage of this iteration.
                stage_iteration: Number of completed iterations in the stage,
                    including this one.
                sphere_sizes: List of the sphere sizes included.
                alphas: List of the weightings of the included sphere sizes.
                stepsize: Step size.
                max_displacement: Maximum angular displacement in radians that
                    can be applied. Never larger than pi, which is the same as
                    no clipping.
                tolerance: Tolerance in radians for advancing past this stage.
                largest_displacement: Largest angular displacement in radians
                    of any point, after clipping.
                num_clipped: Number of points whose displacements were clipped.
                step_seconds: Seconds spent in this iteration.
                elapsed_seconds: Total seconds spent in iterations so far.
                energy: Value of the energy function after this iteration
                    with the sphere sizes of this stage, or None if it was not
                    evaluated.
        """
        if self.converged:
            raise RuntimeError("The optimization has already converged.")

        xp = array_namespace(self.points)
        stage = self.stage

        # Get parameters for the current stage
        sphere_sizes = self.sphere_sizes[:stage + 1]
        alphas = self.alphas[:stage + 1]
        stepsize = float(self.stepsizes[stage])
        # JSON files cannot store infinity, so max_displacement is limited to
        # pi, which is the same as no clipping.
        max_displacement = min(float(self.max_displacements[stage]), math.pi)
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
        largest_displacement = min(float(xp.max(angles)), max_displacement)
        num_clipped = int(xp.sum(xp.astype(angles > max_displacement,
                                           xp.int64)))
        step_seconds = time.perf_counter() - start_time

        energy = None
        if (self.energy_every > 0
                and (self.iteration + 1) % self.energy_every == 0):
            energy = physics.potential_energy(new_points, sphere_sizes, alphas,
                                              block_size=self.block_size)

        record = {
            "iteration": self.iteration + 1,
            "stage": stage,
            "stage_iteration": self.stage_iteration + 1,
            "sphere_sizes": sphere_sizes.tolist(),
            "alphas": alphas.tolist(),
            "stepsize": stepsize,
            "max_displacement": max_displacement,
            "tolerance": tolerance,
            "largest_displacement": largest_displacement,
            "num_clipped": num_clipped,
            "step_seconds": step_seconds,
            "elapsed_seconds": self.elapsed_seconds + step_seconds,
            "energy": None if energy is None else float(energy),
        }
        # The record is written before the state is updated. If writing is
        # interrupted, the state still matches the last complete record.
        if self.save_dir is not None:
            self._write_record(record)

        self.points = new_points
        self.iteration += 1
        self.stage_iteration += 1
        self.elapsed_seconds += step_seconds
        self._update_stage(largest_displacement, tolerance)

        if (self.save_dir is not None and self.points_every > 0
                and self.iteration % self.points_every == 0):
            io.save_points(self.points, self.save_dir, self.iteration)

        return record

    def _update_stage(self, largest_displacement, tolerance):
        """Advances to the next stage or converges after an iteration.

        Args:
            largest_displacement: Largest angular displacement of the
                iteration.
            tolerance: Tolerance of the stage of the iteration.
        """
        if largest_displacement <= tolerance:
            if self.stage == self.num_stages - 1:
                self.converged = True
            else:
                self.stage += 1
                self.stage_iteration = 0

    def run(self, max_iterations):
        """Runs iterations until max_iterations or convergence.

        When saving, the points are saved when this returns or is interrupted
        (e.g. with KeyboardInterrupt), and the history file is closed.

        Args:
            max_iterations: Integer specifying the total number of iterations,
                including those completed before the optimization was saved and
                resumed. Calling run again with the same max_iterations
                continues to the same end point.

        Returns:
            self
        """
        try:
            while self.iteration < max_iterations and not self.converged:
                self.step()
        finally:
            if self.save_dir is not None:
                # The points may already have been saved by step, in which
                # case the same file is written again
                io.save_points(self.points, self.save_dir, self.iteration)
                self.close()

        return self

    def close(self):
        """Closes the history file. It is reopened if step is called again."""
        if self._history_file is not None:
            self._history_file.close()
            self._history_file = None

    def _start_saving(self, save_dir):
        """Creates a save directory and saves the parameters and points."""
        if io.params_path(save_dir).exists():
            raise FileExistsError(
                f"{save_dir} already contains a saved optimization. Use "
                "Electro.resume to continue it.")

        self.save_dir = save_dir
        params = {
            "num_points": self.num_points,
            "points_dtype": self.points_dtype,
            "sphere_sizes": self.sphere_sizes,
            "alphas": self.alphas,
            "stepsizes": self.stepsizes,
            "max_displacements": self.max_displacements,
            "tolerances": self.tolerances,
            "points_every": self.points_every,
            "energy_every": self.energy_every,
            "block_size": self.block_size,
        }
        io.save_params(params, self.save_dir)
        io.save_points(self.points, self.save_dir, self.iteration)

    def _write_record(self, record):
        """Appends a record to the history file."""
        assert self.save_dir is not None
        if self._history_file is None:
            # Kept open across iterations and closed by close()
            self._history_file = open(  # noqa: SIM115
                io.history_path(self.save_dir), "a", encoding="utf-8",
                newline="\n")
        io.write_record(self._history_file, record)

    @classmethod
    def resume(cls, save_dir, xp=np, device=None):
        """Resumes an optimization from its save directory.

        The optimization continues from the latest saved points that can be
        loaded and have a complete record in the history file. Normally that
        is the newest points file. After a crash, later records or points may
        be incomplete; they are removed, because those iterations will be
        repeated.

        Args:
            save_dir: Save directory of the optimization.
            xp: Array namespace for the points, e.g. numpy or torch.
            device: Device for the points.

        Returns:
            out: Electro instance that continues saving to save_dir.
        """
        save_dir = Path(save_dir)
        params = io.load_params(save_dir)
        points, record = io.prepare_resume(save_dir)

        out = cls(xp.asarray(points, device=device), params["sphere_sizes"],
                  params["alphas"], params["stepsizes"],
                  params["max_displacements"], params["tolerances"],
                  points_every=params["points_every"],
                  energy_every=params["energy_every"],
                  block_size=params["block_size"])

        # Restore the state after the iteration from its record. The record
        # holds the stage of the iteration, so _update_stage applies the same
        # check as step to advance the stage or converge.
        if record is not None:
            out.iteration = record["iteration"]
            out.stage = record["stage"]
            out.stage_iteration = record["stage_iteration"]
            out.elapsed_seconds = record["elapsed_seconds"]
            out._update_stage(record["largest_displacement"],
                              record["tolerance"])

        out.save_dir = save_dir
        return out
