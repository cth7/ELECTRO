import numpy as np
import plotly.graph_objects as go


def set_up_points_in_square():
    """Sets up a default frame for the figure.

    For plotting a set of 2D points in the unit square. 

    Returns:
        f: Plotly figure object.
    """
    f = go.Figure()

    margin_dict = {}
    margin_dict["l"] = 40
    margin_dict["r"] = 40
    margin_dict["t"] = 40
    margin_dict["b"] = 40

    title_dict = {}
    title_dict["x"] = 0.5
    title_dict["xanchor"] = "center"
    title_dict["xref"] = "paper"
    title_dict["font"] = {"weight": "bold",
                          "family": "Arial",
                          "size": 22}

    # Settings common to both x and y axes
    axis_dict = {}
    axis_dict["range"] = [0, 1]
    axis_dict["autorange"] = False
    axis_dict["linecolor"] = "black"
    axis_dict["zeroline"] = False
    axis_dict["showline"] = True
    axis_dict["mirror"] = True
    axis_dict["ticks"] = "outside"
    axis_dict["minor_ticks"] = "outside"
    axis_dict["minor"] = {"dtick": 0.1}
    axis_dict["title_font"] = {"family": "Arial",
                               "size": 16}

    f.update_layout(
        height=600,
        width=700,
        margin=margin_dict,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=title_dict,
        xaxis_title_text="x",
        yaxis_title_text="y",
        xaxis=axis_dict,
        yaxis=axis_dict
    )

    # To make square aspect ratio
    f.update_xaxes(
        constrain="domain"
    )
    f.update_yaxes(
        constrain="domain",
        scaleanchor="x",
        scaleratio=1
    )

    return f


def points_in_square(points, title="", marker_size=12, colorscale="Viridis"):
    """Plots a set of 2D points in the unit square.

    Args:
        points: Array with shape (num_points, 2).
        title: String with title text.
        marker_size: Size of scatter markers.
        colorscale: Colorscale argument for plotly.

    Returns:
        f: Plotly figure object.
    """
    f = set_up_points_in_square()

    f.update_layout(title_text=title)

    marker_dict = {}
    marker_dict["size"] = marker_size
    marker_dict["color"] = np.arange(len(points))
    marker_dict["colorscale"] = colorscale
    marker_dict["showscale"] = True
    marker_dict["colorbar"] = {"title": "Point Index"}

    trace = go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="markers",
        marker=marker_dict
    )

    f.add_trace(trace)

    return f
