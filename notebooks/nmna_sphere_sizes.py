import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    import electro as ele


@app.function
def nearest_records(points, max_offset, block_size=256):
    """Finds the record nearest neighbors to the right of each point.

    For point i, point j > i is a record if it is closer to point i than all
    of the points i + 1, ..., j - 1. Only j - i <= max_offset is considered.
    Point i + 1 is always a record. The nearest neighbor of point i among the
    points i + 1, ..., i + r is the last record with an offset j - i <= r.

    Args:
        points: NumPy array with shape (num_points, 3). Points are expected to
            be on the unit sphere.
        max_offset: Integer specifying the largest offset j - i considered.
        block_size: Integer specifying the number of points i in each block.

    Returns:
        keys: Increasing integer array of i * num_points + (j - i), one for
            each record.
        angles: Array of the angular distance between points i and j of each
            record.
    """
    num_points = points.shape[0]
    record_i = []
    record_j = []
    # The last point has no points to its right
    for first_i in range(0, num_points - 1, block_size):
        last_i = min(first_i + block_size, num_points - 1)
        first_j = first_i + 1
        last_j = min(last_i + max_offset, num_points)

        # Squared distances from points i (rows) to points j (columns). Adding
        # one coordinate at a time only needs 2D arrays.
        distances_sq = np.zeros((last_i - first_i, last_j - first_j))
        for k in range(3):
            diff = (points[first_i:last_i, k, None]
                    - points[None, first_j:last_j, k])
            distances_sq += diff * diff
        i = np.arange(first_i, last_i)[:, None]
        j = np.arange(first_j, last_j)[None, :]
        distances_sq[(j <= i) | (j - i > max_offset)] = np.inf

        # Squared distance to the nearest neighbor among points i + 1 to j.
        # A record is where it decreases. The first finite value, at
        # j = i + 1, is a record.
        nearest_sq = np.minimum.accumulate(distances_sq, axis=1)
        is_record = np.empty(nearest_sq.shape, dtype=bool)
        is_record[:, 0] = nearest_sq[:, 0] < np.inf
        is_record[:, 1:] = nearest_sq[:, 1:] < nearest_sq[:, :-1]
        # Row-major order, so sorted by i and then j
        rows, columns = np.nonzero(is_record)
        record_i.append(first_i + rows)
        record_j.append(first_j + columns)

    record_i = np.concatenate(record_i)
    record_j = np.concatenate(record_j)
    keys = record_i * num_points + (record_j - record_i)
    # Same angle calculation as electro.metrics.nmna
    angles = ele.geometry.angular_distances(points[record_i],
                                            points[record_j])
    return keys, angles


@app.function
def lookup_nearest(records, num_points, i, r):
    """Looks up the angle from point i to its nearest neighbor to the right.

    Args:
        records: Tuple of keys and angles from nearest_records.
        num_points: Integer specifying the total number of points.
        i: Integer array of point indices.
        r: Integer array with the shape of i. The nearest neighbor is among
            the points i + 1, ..., i + r. Must be at most the max_offset of
            nearest_records.

    Returns:
        out: Array of angles with the shape of i. Infinite where r is 0.
    """
    keys, angles = records
    # The last record of point i with an offset of at most r. When r is at
    # least 1 this is a record of point i, because i + 1 is always a record.
    position = np.searchsorted(keys, i * num_points + r, side="right") - 1
    out = angles[position]
    out[r == 0] = np.inf
    return out


@app.function
def sphere_nmna(right, left, num_points, size, starts, expected_values):
    """Calculates the NMNA of spheres of consecutive points.

    Gives the same values as electro.metrics.nmna(points[s:s + size]) for
    each start s.

    Args:
        right: Records from nearest_records for the points.
        left: Records from nearest_records for the points in reverse order.
        num_points: Integer specifying the total number of points.
        size: Integer specifying the sphere size, at least 2.
        starts: Integers of the first point index of each sphere.
        expected_values: Array from electro.metrics.expected_nearest_angle
            with max_points of at least size.

    Returns:
        out: NumPy array with the NMNA of each sphere.
    """
    out = []
    for start in starts:
        i = np.arange(start, start + size)
        # Nearest neighbor of point i among the points to its right and to
        # its left in the sphere. In reverse order, point i has index
        # num_points - 1 - i and the points to its left are to its right.
        to_right = lookup_nearest(right, num_points, i, start + size - 1 - i)
        to_left = lookup_nearest(left, num_points, num_points - 1 - i,
                                 i - start)
        angles = np.minimum(to_right, to_left)
        out.append(np.mean(angles) / expected_values[size])
    return np.array(out)


@app.cell
def _():
    # Save directory of the optimization
    save_dir_input = mo.ui.file_browser(
        initial_path=Path.home(),
        selection_mode="directory",
        multiple=False,
        restrict_navigation=False,
        label="Optimization save directory",
    )
    save_dir_input
    return (save_dir_input,)


@app.cell
def _(save_dir_input):
    save_dir = save_dir_input.path()
    mo.stop(save_dir is None, mo.md("Select a save directory."))
    mo.stop(
        not ele.io.params_path(save_dir).exists(),
        mo.md(f"No optimization found in `{save_dir}`."),
    )
    params = ele.io.load_params(save_dir)
    saved = ele.io.saved_iterations(save_dir)
    mo.stop(not saved, mo.md("No points saved yet."))

    # Defaults to the latest saved points
    iteration_input = mo.ui.dropdown(
        options={str(iteration): iteration for iteration in saved},
        value=str(saved[-1]),
        label="Saved iteration",
    )
    run_button = mo.ui.run_button(label="Find nearest neighbors")
    mo.hstack([iteration_input, run_button], justify="start")
    return iteration_input, params, run_button, save_dir


@app.cell
def _(iteration_input, run_button, save_dir):
    # Only calculates when the button is clicked, because it can take a while
    # for many points. This is the slow part. The NMNA of the spheres below is
    # fast, so their options can be changed without recalculating this.
    mo.stop(not run_button.value,
            mo.md("Click the button to find the nearest neighbors."))

    points_iteration = iteration_input.value
    points = ele.io.load_points(save_dir, points_iteration)
    num_points = points.shape[0]

    # The largest sphere has all points, so offsets up to num_points - 1 are
    # needed
    with mo.status.spinner(title="Finding nearest neighbors to the right"):
        right = nearest_records(points, num_points - 1)
    with mo.status.spinner(title="Finding nearest neighbors to the left"):
        left = nearest_records(points[::-1], num_points - 1)

    # expected_values[N] is for N points
    expected_values = ele.metrics.expected_nearest_angle(num_points)
    return expected_values, left, num_points, points_iteration, right


@app.cell
def _():
    num_spheres_input = mo.ui.number(
        start=1, step=1, value=30, label="Spheres per sphere size")
    num_sizes_input = mo.ui.number(
        start=1, step=1, value=50, label="Number of sphere sizes")
    mo.hstack([num_spheres_input, num_sizes_input], justify="start")
    return num_sizes_input, num_spheres_input


@app.cell
def _(
    expected_values,
    left,
    num_points,
    num_sizes_input,
    num_spheres_input,
    right,
):
    # Sphere sizes evenly spaced on a log scale from 2 to num_points
    sizes = np.unique(np.round(np.geomspace(
        2, num_points, num_sizes_input.value)).astype(np.int64))

    means = []
    stds = []
    counts = []
    for _size in sizes:
        # Starts evenly spaced from the first to the last possible sphere
        _starts = np.unique(np.round(np.linspace(
            0, num_points - _size, num_spheres_input.value)).astype(np.int64))
        _nmna = sphere_nmna(right, left, num_points, _size, _starts,
                            expected_values)
        means.append(np.mean(_nmna))
        # NaN when there is only one sphere
        stds.append(np.std(_nmna, ddof=1) if len(_nmna) > 1 else np.nan)
        counts.append(len(_nmna))
    means = np.array(means)
    stds = np.array(stds)
    return counts, means, sizes, stds


@app.cell
def _(counts, means, params, points_iteration, sizes, stds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=means, mode="lines+markers", name="Mean NMNA",
        line={"color": "#1f77b4"},
        error_y={"type": "data", "array": stds, "visible": True},
        customdata=np.stack([stds, counts], axis=1),
        hovertemplate=("Sphere size %{x}<br>NMNA %{y:.4f} ± "
                       "%{customdata[0]:.4f}<br>%{customdata[1]} spheres"
                       "<extra></extra>"),
    ))

    # Dotted vertical lines mark the sphere sizes of the optimization
    for sphere_size in params["sphere_sizes"]:
        fig.add_vline(x=sphere_size, line={"color": "lightgray",
                                           "dash": "dot", "width": 1})

    fig.update_xaxes(title_text="Sphere size", type="log",
                     showgrid=True, gridcolor="#eee")
    fig.update_yaxes(title_text="NMNA", showgrid=True, gridcolor="#eee")
    fig.update_layout(
        title=(f"NMNA of spheres of consecutive points, iteration "
               f"{points_iteration}. Error bars are standard deviations. "
               "Dotted lines are the optimization's sphere sizes."),
        height=500, plot_bgcolor="white",
    )
    fig
    return


if __name__ == "__main__":
    app.run()
