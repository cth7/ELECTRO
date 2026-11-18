import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import electro as ele


@app.function
def first_points_nmna(points, ends):
    """Calculates the global NMNA of the first N points for several N.

    Gives the same values as electro.metrics.nmna(points[:N]) for each N.

    Args:
        points: NumPy array with shape (num_points, 3). Points are expected to
            be on the unit sphere.
        ends: Increasing integers between 2 and num_points. The NMNA is
            calculated for the first N points for each N in ends.

    Returns:
        out: NumPy array with the NMNA for each N in ends.
    """
    # Calling electro.metrics.nmna for each N would repeat the nearest neighbor
    # search from the start every time. Instead, the points are added in order,
    # one block at a time. Each block of new points is compared with all
    # points so far. This finds the nearest neighbors of the new points, and
    # updates the nearest neighbors of the earlier points when a new point is
    # closer.
    num_points = points.shape[0]
    # Squared distance to, index of, and angle to the nearest neighbor of each
    # point so far
    nearest_sq = np.full(num_points, np.inf)
    nearest = np.zeros(num_points, dtype=np.int64)
    angles = np.zeros(num_points)

    # expected_values[N] is for N points
    expected_values = ele.metrics.expected_nearest_angle(num_points)

    out = []
    start = 0
    for end in ends:
        # Squared distances from the new points (rows) to all points so far
        # (columns). Adding one coordinate at a time only needs 2D arrays.
        distances_sq = np.zeros((end - start, end))
        for k in range(3):
            diff = points[start:end, k, None] - points[None, :end, k]
            distances_sq += diff * diff
        # Exclude the distance from each new point to itself
        rows = np.arange(end - start)
        distances_sq[rows, start + rows] = np.inf

        # Nearest neighbors of the new points
        nearest[start:end] = np.argmin(distances_sq, axis=1)
        nearest_sq[start:end] = distances_sq[rows, nearest[start:end]]
        changed = [np.arange(start, end)]

        # Earlier points whose nearest neighbor is now one of the new points
        if start > 0:
            new_nearest = np.argmin(distances_sq[:, :start], axis=0)
            new_nearest_sq = distances_sq[new_nearest, np.arange(start)]
            closer = np.flatnonzero(new_nearest_sq < nearest_sq[:start])
            nearest[closer] = start + new_nearest[closer]
            nearest_sq[closer] = new_nearest_sq[closer]
            changed.append(closer)

        # Same angle calculation as electro.metrics.nmna, only for the points
        # whose nearest neighbor changed
        changed = np.concatenate(changed)
        angles[changed] = ele.geometry.angular_distances(
            points[changed], points[nearest[changed]])

        out.append(np.mean(angles[:end]) / expected_values[end])
        start = end

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
    # NMNA is calculated for N = step, 2 * step, 3 * step, ...
    step_input = mo.ui.number(start=1, step=1, value=10, label="Step in N")
    run_button = mo.ui.run_button(label="Calculate NMNA")
    mo.hstack([iteration_input, step_input, run_button], justify="start")
    return iteration_input, params, run_button, save_dir, step_input


@app.cell
def _(iteration_input, run_button, save_dir, step_input):
    # Only calculates when the button is clicked, because it can take a while
    # for many points
    mo.stop(not run_button.value,
            mo.md("Click the button to calculate the NMNA."))

    points_iteration = iteration_input.value
    points = ele.io.load_points(save_dir, points_iteration)
    num_points = points.shape[0]

    # The NMNA needs at least 2 points
    ends = np.arange(step_input.value, num_points + 1, step_input.value)
    ends = ends[ends >= 2]
    nmna_first = first_points_nmna(points, mo.status.progress_bar(
        ends.tolist(), title="Calculating NMNA of the first N points"))
    # The last N points are the first N points in reverse order
    nmna_last = first_points_nmna(points[::-1], mo.status.progress_bar(
        ends.tolist(), title="Calculating NMNA of the last N points"))
    return ends, nmna_first, nmna_last, points_iteration


@app.cell
def _(ends, nmna_first, nmna_last, params, points_iteration):
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
        subplot_titles=["First N points", "Last N points"],
    )
    fig.add_trace(go.Scattergl(
        x=ends, y=nmna_first, mode="lines", name="First N points",
        line={"color": "#1f77b4"},
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=ends, y=nmna_last, mode="lines", name="Last N points",
        line={"color": "#ff7f0e"},
    ), row=1, col=2)

    # Dotted vertical lines mark the sphere sizes of the optimization
    for sphere_size in params["sphere_sizes"]:
        fig.add_vline(x=sphere_size, line={"color": "lightgray",
                                           "dash": "dot", "width": 1})

    fig.update_xaxes(title_text="Number of points N", type="log",
                     showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(title_text="NMNA", row=1, col=1)
    fig.update_layout(
        title=(f"NMNA of the first and last N points, iteration "
               f"{points_iteration}. Dotted lines are sphere sizes."),
        height=500, plot_bgcolor="white", hovermode="x unified",
        showlegend=False,
    )
    fig
    return


if __name__ == "__main__":
    app.run()
