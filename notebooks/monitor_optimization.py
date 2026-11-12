import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import electro as ele

    return Path, datetime, ele, go, make_subplots, math, mo, np


@app.function
def format_seconds(seconds):
    """Formats a duration in seconds as hours, minutes and seconds."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@app.cell
def _(Path, mo):
    # Save directory of the optimization to watch. The optimization runs in
    # another process, e.g. the run_optimization notebook or a script.
    save_dir_input = mo.ui.file_browser(
        initial_path=Path.home(),
        selection_mode="directory",
        multiple=False,
        restrict_navigation=False,
        label="Optimization save directory",
    )
    # Reruns the cells that read the save directory at the chosen interval
    refresh = mo.ui.refresh(
        options=["5s", "10s", "30s", "1m", "5m"],
        default_interval="10s",
        label="Refresh Interval",
    )
    mo.vstack([save_dir_input, refresh])
    return refresh, save_dir_input


@app.cell
def _(ele, mo, save_dir_input):
    # The params file is written once, so it is not reloaded on refresh
    save_dir = save_dir_input.path()
    mo.stop(save_dir is None, mo.md("Select a save directory."))
    mo.stop(
        not ele.io.params_path(save_dir).exists(),
        mo.md(f"No optimization found in `{save_dir}`."),
    )
    params = ele.io.load_params(save_dir)
    sphere_sizes = params["sphere_sizes"]
    num_stages = len(sphere_sizes)
    return num_stages, params, save_dir, sphere_sizes


@app.cell
def _(ele, mo, np, refresh, save_dir):
    refresh  # Reruns this cell on every refresh

    # load_history skips a last line that is still being written
    records = ele.io.load_history(save_dir)
    mo.stop(not records, mo.md("No iterations recorded yet."))

    def column(key):
        # Converting to float turns None (no clipping, energy not evaluated)
        # into NaN. Records from before a field was added are also NaN.
        return np.array([record.get(key) for record in records], dtype=float)

    history = {key: column(key) for key in [
        "iteration", "stage", "stage_iteration", "max_displacement",
        "tolerance", "largest_displacement", "num_clipped", "step_seconds",
        "elapsed_seconds", "energy",
    ]}
    last = records[-1]
    return history, last, records


@app.cell
def _(
    datetime,
    history,
    last,
    math,
    mo,
    np,
    num_stages,
    records,
    sphere_sizes,
):
    # Status of the optimization after the last recorded iteration
    stage = last["stage"]
    tolerance = last["tolerance"]
    largest_displacement = last["largest_displacement"]
    # Same condition as Electro._update_stage
    stage_done = largest_displacement <= tolerance

    if stage_done and stage == num_stages - 1:
        stage_caption = "Converged"
    elif stage_done:
        stage_caption = f"Advancing to stage {stage + 2}"
    else:
        stage_caption = (f"Sphere size {sphere_sizes[stage]}, "
                         f"iteration {last['stage_iteration']} of this stage")

    if tolerance > 0:
        displacement_caption = (
            f"{largest_displacement / tolerance:.2f} × tolerance "
            f"{tolerance:.3g}")
    else:
        displacement_caption = "Tolerance 0, runs to max iterations"

    recent_seconds = history["step_seconds"][-30:]
    recent_mean = float(np.mean(recent_seconds))
    recent_std = float(np.std(recent_seconds, ddof=1))
    # Decimal places for 3 significant figures of the mean, also used for the
    # standard deviation
    decimals = max(0, 2 - math.floor(math.log10(recent_mean)))

    # Wall-clock time includes the energy, saving and any time between
    # resumes, while elapsed_seconds only includes the algorithm
    timestamps = [record["timestamp"] for record in records
                  if "timestamp" in record]
    if timestamps:
        first_time = datetime.fromisoformat(timestamps[0])
        last_time = datetime.fromisoformat(timestamps[-1])
        wall_time = format_seconds((last_time - first_time).total_seconds())
        since_last = (datetime.now().astimezone()
                      - last_time).total_seconds()
        last_record_stat = mo.stat(
            f"{format_seconds(since_last)} ago",
            label="Last record",
            caption=last_time.strftime("%Y-%m-%d %H:%M:%S"),
            bordered=True,
        )
    else:
        wall_time = "Unknown"
        last_record_stat = mo.stat("Unknown", label="Last record",
                                   bordered=True)

    mo.hstack(
        [
            mo.stat(f"{last['iteration']:,}",
                    label="Iteration",
                    bordered=True),
            mo.stat(f"{stage + 1} / {num_stages}",
                    label="Stage",
                    caption=stage_caption,
                    bordered=True),
            mo.stat(f"{largest_displacement:.3g}",
                    label="Largest displacement",
                    caption=displacement_caption,
                    bordered=True),
            mo.stat(f"{recent_mean:.{decimals}f} ± {recent_std:.{decimals}f} s",
                    label="Average seconds per iteration",
                    caption=f"Last {len(recent_seconds)} iterations",
                    bordered=True),
            mo.stat(wall_time,
                    label="Wall-clock time",
                    caption=(f"Algorithm "
                             f"{format_seconds(last['elapsed_seconds'])}"),
                    bordered=True),
            last_record_stat,
        ],
        widths="equal",
        wrap=True,
    )
    return


@app.cell
def _(go, history, make_subplots, np):
    # History plots, sharing the iteration axis. Dotted vertical lines mark
    # the first iteration of each stage.
    iterations = history["iteration"]

    fig_history = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[
            "Largest displacement (radians)",
            "Number of clipped points",
            "Seconds per iteration",
            "Energy (sphere sizes of each stage)",
        ],
    )

    fig_history.add_trace(go.Scattergl(
        x=iterations, y=history["largest_displacement"], mode="lines",
        name="Largest displacement", line={"color": "#1f77b4"},
    ), row=1, col=1)

    # The tolerance and max displacement are constant within a stage, so
    # horizontal-then-vertical steps draw a horizontal line for each stage.
    # NaN (no clipping) and 0 (last tolerance) are not drawn on a log axis.
    fig_history.add_trace(go.Scatter(
        x=iterations, y=history["tolerance"], mode="lines", name="Tolerance",
        line={"color": "gray", "dash": "dash", "shape": "hv"},
    ), row=1, col=1)

    fig_history.add_trace(go.Scatter(
        x=iterations, y=history["max_displacement"], mode="lines",
        name="Max displacement",
        line={"color": "#d62728", "dash": "dot", "shape": "hv"},
    ), row=1, col=1)

    fig_history.add_trace(go.Scattergl(
        x=iterations, y=history["num_clipped"], mode="lines",
        name="Clipped points", line={"color": "#ff7f0e"},
    ), row=2, col=1)

    fig_history.add_trace(go.Scattergl(
        x=iterations, y=history["step_seconds"], mode="lines",
        name="Seconds per iteration", line={"color": "#2ca02c"},
    ), row=3, col=1)

    is_evaluated = np.isfinite(history["energy"])
    fig_history.add_trace(go.Scattergl(
        x=iterations[is_evaluated], y=history["energy"][is_evaluated],
        mode="markers", name="Energy", marker={"color": "#9467bd", "size": 4},
    ), row=4, col=1)

    stage_starts = iterations[history["stage_iteration"] == 1]
    for stage_start in stage_starts[1:]:
        fig_history.add_vline(x=stage_start, line={"color": "lightgray",
                                                   "dash": "dot", "width": 1})

    fig_history.update_yaxes(type="log", row=1, col=1)
    fig_history.update_yaxes(type="log", row=4, col=1)

    fig_history.update_xaxes(title_text="Iteration", row=4, col=1)

    fig_history.update_layout(
        height=900, hovermode="x unified", plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.06}, margin={"t": 80},
        # Keeps the zoom when the figure is redrawn on refresh
        uirevision="history",
    )
    fig_history.update_xaxes(showgrid=True, gridcolor="#eee")
    fig_history.update_yaxes(showgrid=True, gridcolor="#eee")
    fig_history
    return


@app.cell
def _(datetime, history, mo, np, num_stages, records, sphere_sizes):
    # Time when each iteration finished, or None for records from before
    # timestamps were added
    finish_times = [
        datetime.fromisoformat(record["timestamp"])
        if "timestamp" in record else None
        for record in records
    ]

    # One row per stage reached so far
    stage_rows = []
    for _stage in range(num_stages):
        in_stage = history["stage"] == _stage
        if not np.any(in_stage):
            break
        _final = history["largest_displacement"][in_stage][-1]
        _tolerance = history["tolerance"][in_stage][-1]

        # A stage starts when the previous stage's last iteration finished.
        # The first stage starts at its first record, so its wall-clock time
        # misses its first iteration.
        _indices = np.flatnonzero(in_stage)
        _start = finish_times[max(_indices[0] - 1, 0)]
        _end = finish_times[_indices[-1]]
        if _start is None or _end is None:
            _wall_time = "Unknown"
        else:
            _wall_time = format_seconds((_end - _start).total_seconds())

        stage_rows.append({
            "Stage": _stage + 1,
            "Sphere size added": sphere_sizes[_stage],
            "First iteration": int(history["iteration"][in_stage][0]),
            "Iterations": int(np.sum(in_stage)),
            "Final largest displacement": f"{_final:.3g}",
            "Tolerance": f"{_tolerance:.3g}",
            "Done": bool(_final <= _tolerance),
            "Wall-clock time": _wall_time,
            "Algorithm time": format_seconds(
                np.sum(history["step_seconds"][in_stage])),
        })
    mo.ui.table(stage_rows, selection=None, pagination=False,
                show_column_summaries=False, label="Stages")
    return


@app.cell
def _(mo, params):
    # The chosen iteration is kept in a state, because the slider is recreated
    # with the new saved iterations on every refresh. None means the latest.
    get_chosen_iteration, set_chosen_iteration = mo.state(None)
    # Suggested marker size, as in the explore_points notebook. It is in this
    # cell so its value is kept when the history is refreshed.
    marker_size = mo.ui.number(
        start=1, step=1, label="Marker size",
        value=max(1, min(16, round(350 / params["num_points"] ** 0.5))),
    )
    return get_chosen_iteration, marker_size, set_chosen_iteration


@app.cell
def _(ele, get_chosen_iteration, last, mo, save_dir, set_chosen_iteration):
    # Points files are saved before their record is written, so only files up
    # to the last record are known to be completely written
    saved = [iteration for iteration in ele.io.saved_iterations(save_dir)
             if iteration <= last["iteration"]]
    mo.stop(not saved, mo.md("No points saved yet."))

    # The slider stays on the chosen iteration, or on the latest saved
    # iteration if the latest was chosen, so it follows new saves
    _chosen = get_chosen_iteration()
    points_slider = mo.ui.slider(
        steps=saved,
        value=_chosen if _chosen in saved else saved[-1],
        show_value=True,
        debounce=True,
        full_width=True,
        label="Saved iteration",
        on_change=lambda value: set_chosen_iteration(
            None if value == saved[-1] else value),
    )
    return (points_slider,)


@app.cell
def _(ele, marker_size, mo, num_stages, points_slider, records, save_dir):
    points_iteration = points_slider.value
    if points_iteration == 0:
        points_title = "Initial points"
    else:
        _stage = records[points_iteration - 1]["stage"]
        points_title = f"Iteration {points_iteration} (stage {_stage + 1}/{num_stages})"

    points = ele.io.load_points(save_dir, points_iteration)
    fig_points = ele.plot.points_on_sphere(
        points, title=points_title, marker_size=marker_size.value,
        colorscale="Turbo", radius=0.97, show_cap=False,
    )
    # Keeps the camera when the figure is redrawn on refresh
    fig_points.update_layout(uirevision="points")

    mo.vstack([
        points_slider,
        mo.hstack([fig_points, marker_size], justify="start", align="start"),
    ])
    return


if __name__ == "__main__":
    app.run()
