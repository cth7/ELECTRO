import numpy as np
import plotly.graph_objects as go

from . import geometry


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
                          "size": 22,
                          "color": "dimgray"}

    # Settings common to both x and y axes
    axis_dict = {}
    axis_dict["range"] = [0, 1]
    axis_dict["autorange"] = False
    axis_dict["linecolor"] = "darkgray"
    axis_dict["zeroline"] = False
    axis_dict["showline"] = True
    axis_dict["linewidth"] = 1.5
    axis_dict["mirror"] = True
    axis_dict["ticks"] = "outside"
    axis_dict["minor_ticks"] = "outside"
    axis_dict["minor"] = {"dtick": 0.1}
    axis_dict["title_font"] = {"size": 16}
    axis_dict["color"] = "dimgray"

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
        yaxis=axis_dict,
        font_family="Arial"
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
    marker_dict["colorbar"] = {"title": "Point Index",
                               "title_font_color": "dimgray",
                               "tickfont_color": "dimgray"}

    trace = go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="markers",
        marker=marker_dict,
        text=[f"Point {i + 1}" for i in range(len(points))],
        hovertemplate=(
            "x: %{x}<br>"
            "y: %{y}"
            "<extra><b>%{text}</b></extra>"
        )
    )

    f.add_trace(trace)

    return f


def set_up_points_on_sphere():
    """Sets up a default frame for the figure.

    For plotting a set of 3D points on the unit sphere.

    Returns:
        f: Plotly figure object.
    """
    f = go.Figure()

    margin_dict = {}
    margin_dict["l"] = 0
    margin_dict["r"] = 0
    margin_dict["t"] = 0
    margin_dict["b"] = 0

    title_dict = {}
    title_dict["x"] = 0.5
    title_dict["xanchor"] = "center"
    title_dict["xref"] = "paper"
    title_dict["y"] = 0.985
    title_dict["yanchor"] = "top"
    title_dict["font"] = {"weight": "bold",
                          "size": 22,
                          "color": "dimgray"}

    axis_dict = {}
    axis_dict["range"] = [-1, 1]
    axis_dict["visible"] = False

    scene_dict = {}
    scene_dict["xaxis"] = axis_dict
    scene_dict["yaxis"] = axis_dict
    scene_dict["zaxis"] = axis_dict
    scene_dict["aspectmode"] = "cube"
    scene_dict["bgcolor"] = "white"
    scene_dict["camera"] = {"eye": {"x": 0.9, "y": 0.9, "z": 0.9}}

    f.update_layout(
        height=650,
        width=600,
        margin=margin_dict,
        title=title_dict,
        scene=scene_dict,
        font_family="Arial"
    )
    return f


def points_on_sphere(points, title="", marker_size=12, colorscale="Viridis",
                     radius=0.99, eye_polar=25, eye_azimuthal=45,
                     show_cap=True, cap_polar=0, cap_azimuthal=0, cap_angle=0):
    """Plots a set of 3D points on the unit sphere.

    Args:
        points: Array with shape (num_points, 3).
        title: String with title text.
        marker_size: Size of scatter markers.
        colorscale: Colorscale argument for plotly. Color scale of the scatter.
        radius: Radius of the sphere surface. A sphere is plotted mainly to
            help obscure points that are at the back. The radius should be
            slightly smaller than 1 to prevent markers from being chopped off.
        eye_polar: Polar angle of the camera eye in degrees.
        eye_azimuthal: Azimuthal angle of the camera eye in degrees.
        show_cap: Boolean controlling whether spherical cap is shown.
        cap_polar: Polar angle coordinate of the position of the spherical
            cap in radians.
        cap_azimuthal: Azimuthal angle coordinate of the position of the
            spherical cap in radians.
        cap_angle: Size of the spherical cap. Angle between the center and the
            edge of the cap in radians.

    Returns:
        f: Plotly figure object.
    """
    f = set_up_points_on_sphere()

    # Get current zoom
    eye = f.layout.scene.camera.eye
    eye_radius = (eye["x"] ** 2 + eye["y"] ** 2 + eye["z"] ** 2) ** 0.5
    # Set camera eye
    eye_polar = np.deg2rad(eye_polar)
    eye_azimuthal = np.deg2rad(eye_azimuthal)
    eye_x = eye_radius * np.sin(eye_polar) * np.cos(eye_azimuthal)
    eye_y = eye_radius * np.sin(eye_polar) * np.sin(eye_azimuthal)
    eye_z = eye_radius * np.cos(eye_polar)
    f.update_layout(scene_camera_eye={"x": eye_x, "y": eye_y, "z": eye_z})

    f.update_layout(title_text=title)

    marker_dict = {}
    marker_dict["size"] = marker_size
    marker_dict["color"] = np.arange(len(points))
    marker_dict["colorscale"] = colorscale
    marker_dict["showscale"] = False
    marker_dict["colorbar"] = {"title": "Point Index",
                               "title_font_color": "dimgray",
                               "tickfont_color": "dimgray"}
    marker_dict["opacity"] = 1

    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=marker_dict,
        text=[f"Point {i + 1}" for i in range(len(points))],
        hovertemplate=(
            "x: %{x}<br>"
            "y: %{y}<br>"
            "z: %{z}"
            "<extra><b>%{text}</b></extra>"
        )
    )

    f.add_trace(trace)

    # Add sphere surface. Helps obscure points on the back half of sphere.
    # Calculate a 2D grid of (x, y, z) coordinates. The grid helps define
    # neighboring points.
    sphere_res = 100
    azimuthal_angles = np.linspace(0, 2 * np.pi, 200)
    polar_angles = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(azimuthal_angles), np.sin(polar_angles))
    y = radius * np.outer(np.sin(azimuthal_angles), np.sin(polar_angles))
    z = radius * np.outer(np.ones_like(azimuthal_angles), np.cos(polar_angles))
    # Construct surface
    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=z,
        cmin=-1,
        cmax=1,
        colorscale=[[0, "lightgray"], [1, "white"]],
        showscale=False,
        opacity=1,
        lighting={"ambient": 1, "diffuse": 0, "specular": 0, "fresnel": 0,
                  "roughness": 1},
        hoverinfo="skip",
        contours={"x": {"highlight": False},
                  "y": {"highlight": False},
                  "z": {"highlight": False}}
    )

    f.add_trace(surface)

    if show_cap and cap_angle > 0:
        # Add spherical cap
        azimuthals = np.linspace(0, 2 * np.pi, 2 * sphere_res)
        polars = np.linspace(0, np.pi, sphere_res)
        cap_grid = np.zeros((2 * sphere_res, sphere_res, 3), dtype=float)
        cap_grid[:, :, 0] = np.outer(np.cos(azimuthals), np.sin(polars))
        cap_grid[:, :, 1] = np.outer(np.sin(azimuthals), np.sin(polars))
        cap_grid[:, :, 2] = np.outer(np.ones_like(azimuthals), np.cos(polars))
        # Get points on cap
        on_cap = cap_grid[0, :, 2] >= np.cos(cap_angle)
        cap_grid = cap_grid[:, on_cap, :]
        # Rotate cap to the desired position
        R_y = geometry.rotation_matrix_3d_y(cap_polar)
        R_z = geometry.rotation_matrix_3d_z(cap_azimuthal)
        R = R_z @ R_y
        cap_grid = R @ cap_grid.reshape((-1, 3)).transpose()
        cap_grid = cap_grid.transpose().reshape((2 * sphere_res, -1, 3))
        cap_grid = (radius + 0.01) * cap_grid
        # Construct cap
        spherical_cap = go.Surface(
            x=cap_grid[:, :, 0],
            y=cap_grid[:, :, 1],
            z=cap_grid[:, :, 2],
            surfacecolor=np.ones_like(cap_grid[:, :, 2]),
            colorscale=[[0, "mediumaquamarine"], [1, "mediumaquamarine"]],
            showscale=False,
            opacity=0.5,
            lighting={"ambient": 1, "diffuse": 0, "specular": 0, "fresnel": 0,
                    "roughness": 1},
            hoverinfo="skip",
            contours={"x": {"highlight": False},
                      "y": {"highlight": False},
                      "z": {"highlight": False}}
        )
        f.add_trace(spherical_cap)

    return f
