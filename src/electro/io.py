"""Saving and loading of optimizations run by electro.optim.Electro.

A save directory contains:
    params.yaml: The parameters, written once when the optimization is created.
    history.jsonl: One line of JSON per iteration with the values used and
        calculated in that iteration (see Electro.step).
    points/iter_<iteration>.npy: The points after an iteration.
        iter_00000000.npy holds the initial points.
The paths are given by params_path, history_path and points_path.
"""

import json
from pathlib import Path

import array_api_compat
import numpy as np
import yaml


def params_path(save_dir):
    return Path(save_dir) / "params.yaml"


def history_path(save_dir):
    return Path(save_dir) / "history.jsonl"


def points_path(save_dir, iteration):
    return Path(save_dir) / "points" / f"iter_{iteration:08d}.npy"


def save_params(params, save_dir):
    """Saves parameters to the params file in a save directory.

    Args:
        params: Dictionary of parameters. Arrays, including NumPy scalars, are
            converted to lists and Python numbers with their tolist method,
            because YAML cannot represent them.
        save_dir: Save directory of the optimization.
    """
    params = {name: value.tolist() if hasattr(value, "tolist") else value
              for name, value in params.items()}
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(params_path(save_dir), "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, default_flow_style=None)


def load_params(save_dir):
    """Loads the parameters of an optimization saved by Electro.

    Args:
        save_dir: Save directory of the optimization.

    Returns:
        out: Dictionary of parameters.
    """
    with open(params_path(save_dir), encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_record(file, record):
    """Writes a record as one line of a history file.

    Args:
        file: Text file opened for appending, such as the history file of a
            save directory.
        record: Dictionary of values from one iteration. Values must be
            finite, or None for values that were not computed.
    """
    # allow_nan=False raises an error instead of writing NaN or infinity,
    # which are not valid JSON
    file.write(json.dumps(record, allow_nan=False) + "\n")
    file.flush()


def _read_history_lines(path):
    """Reads the complete records at the start of a history file.

    Reading stops at the first line that is incomplete, is not valid JSON, or
    does not continue the sequence of iterations. Such a line can only be left
    by an interrupted write.

    Args:
        path: Path of the history file.

    Returns:
        records: List of dictionaries, where records[k] is for iteration k + 1.
        offsets: List of the byte offsets after each record.
    """
    records = []
    offsets = []
    if not path.exists():
        return records, offsets

    offset = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if record.get("iteration") != len(records) + 1:
                break
            offset += len(line)
            records.append(record)
            offsets.append(offset)
    return records, offsets


def load_history(save_dir):
    """Loads the history of an optimization saved by Electro.

    The history can also be read with
    pandas.read_json(history_path(save_dir), lines=True).

    Args:
        save_dir: Save directory of the optimization.

    Returns:
        records: List with one dictionary per iteration. records[k] is for
            iteration k + 1. The keys are described in Electro.step.
    """
    records, _ = _read_history_lines(history_path(save_dir))
    return records


def save_points(points, save_dir, iteration):
    """Saves the points after an iteration in a save directory.

    Args:
        points: Array with shape (num_points, 3) from any array API library,
            on any device. It is converted to a NumPy array before saving.
        save_dir: Save directory of the optimization.
        iteration: Integer. Iteration 0 is the initial points.
    """
    path = points_path(save_dir, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array_api_compat.to_device(points, "cpu")))


def load_points(save_dir, iteration):
    """Loads the points saved after an iteration.

    Args:
        save_dir: Save directory of the optimization.
        iteration: Integer. Iteration 0 is the initial points.

    Returns:
        out: NumPy array with shape (num_points, 3).
    """
    return np.load(points_path(save_dir, iteration))


def saved_iterations(save_dir):
    """Returns the iterations whose points were saved by Electro.

    Args:
        save_dir: Save directory of the optimization.

    Returns:
        out: Sorted list of integers. Iteration 0 is the initial points.
    """
    paths = points_path(save_dir, 0).parent.glob("iter_*.npy")
    return sorted(int(path.stem.removeprefix("iter_")) for path in paths)


def prepare_resume(save_dir):
    """Finds the state to resume an optimization from.

    The optimization resumes from the latest saved points that can be loaded
    and have a complete record in the history file. Normally that is the
    newest points file. After a crash, later records or points may be
    incomplete; they are removed from the save directory, because those
    iterations will be repeated.

    Args:
        save_dir: Save directory of the optimization.

    Returns:
        points: NumPy array with shape (num_points, 3) of the points to resume
            from.
        record: Dictionary of the history record of the iteration of the
            points, or None for the initial points (iteration 0).
    """
    path = history_path(save_dir)
    records, offsets = _read_history_lines(path)

    # Find the latest saved points that have a complete record and can be
    # read. A save that was interrupted can leave an unreadable file.
    for iteration in reversed(saved_iterations(save_dir)):
        if iteration > len(records):
            continue
        try:
            points = load_points(save_dir, iteration)
        except (OSError, ValueError, EOFError):
            continue
        break
    else:
        raise FileNotFoundError(f"No resumable points in {save_dir}.")

    # Remove what will be repeated. Normally nothing comes after the
    # iteration, but a crash can leave later records or points: a process
    # killed while saving leaves an unreadable points file, and a power loss
    # can lose the last history lines while some later points files survive.
    if path.exists():
        with open(path, "r+b") as f:
            f.truncate(offsets[iteration - 1] if iteration > 0 else 0)
    for later in saved_iterations(save_dir):
        if later > iteration:
            points_path(save_dir, later).unlink()

    record = records[iteration - 1] if iteration > 0 else None
    return points, record
