import numpy as np
import plotly.graph_objects as go

from . import plot


def points_in_square(points, title="", marker_size=12, colorscale="Viridis"):
    """Animates a set of 2D points in the unit square.

    Args:
        points: Array with shape (num_points, 2).
        title: String with title text.
        marker_size: Size of scatter markers.
        colorscale: Colorscale argument for plotly.

    Returns:
        f: Plotly figure object.
    """
    f = plot.points_in_square(points, title=title, marker_size=marker_size,
                              colorscale=colorscale)

    num_points = len(points)

    # This should be the same marker_dict as the one used in
    # plot.points_in_square
    marker_dict = {}
    marker_dict["size"] = marker_size
    marker_dict["color"] = np.arange(num_points)
    marker_dict["colorscale"] = colorscale
    marker_dict["showscale"] = True
    marker_dict["colorbar"] = {"title": "Point Index"}

    # Create frames
    frames = []
    for t in range(num_points+1):
        trace = go.Scatter(x=points[:t, 0], y=points[:t, 1],
                           mode="markers", marker=marker_dict)
        frames.append(go.Frame(name=str(t), data=[trace]))

    f.frames = frames

    # Play button
    play_button = {}
    play_button["label"] = "▶"
    play_button["method"] = "animate"
    play_button["args"] = [
        # 1. The frames to play (None uses all frames)
        None,
        # 2. Dictionary of animation options
        {"frame": {"duration": 20, "redraw": False},
         "transition": {"duration": 0},
         "fromcurrent": True,
         "mode": "immediate"}
    ]

    # Pause button
    pause_button = {}
    pause_button["label"] = "⏸"
    pause_button["method"] = "animate"
    pause_button["args"] = [
        # 1. Putting None in a list creates a pause button
        [None],
        # 2. Dictionary of animation options
        {"frame": {"duration": 0, "redraw": False},
         "transition": {"duration": 0},
         "mode": "immediate"}
    ]

    # Slider
    slider_steps = []
    for t in range(num_points+1):
        # This dict follows similar format as the buttons
        slider_step = {}
        slider_step["label"] = str(t)
        slider_step["method"] = "animate"
        slider_step["args"] = [
            # 1. Sequence of named frames to animate
            [str(t)],
            # 2. Dictionary of animation options
            {"frame": {"duration": 0, "redraw": False},
             "transition": {"duration": 0},
             "mode": "immediate"}
        ]

        slider_steps.append(slider_step)

    f.update_layout(
        updatemenus=[
            {"type": "buttons",
             "buttons": [play_button, pause_button],
             "showactive": True,
             "direction": "left",
             "x": 1.02,
             "xanchor": "left",
             "y": -0.095,
             "yanchor": "top"}
        ],
        sliders=[
            {"active": num_points,
             "currentvalue": {"prefix": "Points "},
             "transition": {"duration": 0},
             # Set to same color as background to hide these elements
             "tickcolor": "white",
             "font": {"color": "white"},
             "y": -0.01,
             "yanchor": "top",
             "steps": slider_steps}
        ]
    )

    # Help push figure elements around
    f.update_layout(height=600, width=640)
    return f
