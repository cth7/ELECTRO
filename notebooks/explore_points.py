import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    import electro as ele

    return ele, mo, np


@app.cell
def _(mo):
    # Set up inputs for points parameters
    num_points = mo.ui.number(value=400, start=1, step=1, debounce=True)
    offset_x = mo.ui.number(value=0, label="x", debounce=True)
    offset_y = mo.ui.number(value=0, label="y", debounce=True)
    n0 = mo.ui.number(value=0, start=0, step=1, debounce=True)
    # Option for reversing the order of the point components
    reverse = mo.ui.switch(value=False)

    # Set up dropdown menu for set of points
    name_of_pts = mo.ui.dropdown(
        options=["Supergolden", "Plastic", "Custom"], value="Supergolden")

    # Set up inputs for the custom point
    custom_point_x = mo.ui.number(value=0, label="x", debounce=True)
    custom_point_y = mo.ui.number(value=0, label="y", debounce=True)

    # Set up dropdown menu for colormap
    colorscale_dict = {}
    colorscale_dict["Viridis"] = "Viridis"
    colorscale_dict["Plasma"] = "Plasma"
    colorscale_dict["Cividis"] = "Cividis"
    colorscale_dict["haline"] = "haline"
    colorscale_dict["Rainbow"] = "Rainbow"
    colorscale_dict["Turbo"] = "Turbo"
    colorscale_dict["Portland"] = "Portland"
    colorscale_dict["Agsunset"] = "Agsunset"
    colorscale_dict["Aggrnyl"] = "Aggrnyl"
    colorscale_dict["BRG"] = [[0.0, "rgb(0, 0, 255)"], [0.5, "rgb(255, 0, 0)"], [1.0, "rgb(0, 255, 0)"]]
    colorscale_dict["RGB"] = [[0.0, "rgb(255, 0, 0)"], [0.5, "rgb(0, 255, 0)"], [1.0, "rgb(0, 0, 255)"]]

    colorscale = mo.ui.dropdown(options=colorscale_dict, value="Turbo")

    # Slider for radius of inner sphere
    radius = mo.ui.slider(value=0.960, start=0.9, stop=1, step=0.001, debounce=True, show_value=True, include_input=True)
    # Option for mapping square of points to half the sphere only
    half_sphere = mo.ui.switch(value=False)
    return (
        colorscale,
        custom_point_x,
        custom_point_y,
        half_sphere,
        n0,
        name_of_pts,
        num_points,
        offset_x,
        offset_y,
        radius,
        reverse,
    )


@app.cell
def _(mo, num_points):
    # Suggested marker size
    sugg_marker_size = round(350 / (num_points.value ** 0.5))

    # Impose a maximum on the marker size or else it looks ridiculous when
    # there are too few points
    max_marker_size = 16
    sugg_marker_size = min(sugg_marker_size, max_marker_size)

    marker_size = mo.ui.number(value=sugg_marker_size, start=0, debounce=True)
    return (marker_size,)


@app.cell
def _(custom_point_x, custom_point_y, ele, np, reverse):
    # Generate points
    point_supergolden = ele.constants.golden_mean_2d(reverse=reverse.value)
    point_plastic = ele.constants.plastic_point(reverse=reverse.value)
    if reverse.value:
        point_custom = np.array([custom_point_y.value, custom_point_x.value])
    else:
        point_custom = np.array([custom_point_x.value, custom_point_y.value])
    return point_custom, point_plastic, point_supergolden


@app.cell
def _(
    ele,
    n0,
    num_points,
    offset_x,
    offset_y,
    point_custom,
    point_plastic,
    point_supergolden,
):
    # Calculate points in square
    square_supergolden = ele.points.square(
        point_supergolden, num_points.value, offset=(offset_x.value, offset_y.value),
        n0=n0.value)

    square_plastic = ele.points.square(
        point_plastic, num_points.value, offset=(offset_x.value, offset_y.value),
        n0=n0.value)

    square_custom = ele.points.square(
        point_custom, num_points.value, offset=(offset_x.value, offset_y.value),
        n0=n0.value)
    return square_custom, square_plastic, square_supergolden


@app.cell
def _(
    colorscale,
    ele,
    marker_size,
    name_of_pts,
    square_custom,
    square_plastic,
    square_supergolden,
):
    # Plot points in unit square
    name = name_of_pts.value

    square = {}
    square["Supergolden"] = square_supergolden
    square["Plastic"] = square_plastic
    square["Custom"] = square_custom

    fig1 = ele.plot.points_in_square(square[name], name, marker_size.value, colorscale.value)
    # fig1 = ele.anim.points_in_square(square[name], title=name,
    #     marker_size=marker_size.value, colorscale=colorscale.value)
    return fig1, name


@app.cell
def _(
    colorscale,
    custom_point_x,
    custom_point_y,
    marker_size,
    mo,
    n0,
    name,
    name_of_pts,
    num_points,
    offset_x,
    offset_y,
    point_custom,
    point_plastic,
    point_supergolden,
    reverse,
):
    # Arrange GUI
    if name in ["Supergolden"]:
        point = point_supergolden
    elif name in ["Plastic"]:
        point = point_plastic
    elif name in ["Custom"]:
        point = point_custom

    params_list = []
    params_list += [mo.md("Name"), name_of_pts]
    if name == "Custom":
        params_list += [mo.md("Custom Point (Increment)"), custom_point_x, custom_point_y]

    params_list += [mo.hstack([reverse, mo.md("Reverse Order of Components")], justify="start")]
    params_list += [mo.md("Point (Increment)"),
                    mo.md(f"x: {point[0]}"), mo.md(f"y: {point[1]}")]
    params_list += [mo.md("Total Number of Points"), num_points]
    params_list += [mo.md("Position Offset"), offset_x, offset_y]
    params_list += [mo.md("Index of Initial Point"), n0]
    params_list += [mo.md("Marker Size"), marker_size]
    params_list += [mo.md("Color Scale"), colorscale]

    params = mo.vstack(params_list)
    return (params,)


@app.cell
def _(fig1, mo, params):
    # Display
    mo.hstack([fig1, params], justify="start", align="start")
    return


@app.cell
def _(ele, half_sphere, square_custom, square_plastic, square_supergolden):
    # Calculate points on sphere
    sphere_supergolden = ele.points.square_to_sphere(square_supergolden, half_sphere=half_sphere.value)

    sphere_plastic = ele.points.square_to_sphere(square_plastic, half_sphere=half_sphere.value)

    sphere_custom = ele.points.square_to_sphere(square_custom, half_sphere=half_sphere.value)
    return sphere_custom, sphere_plastic, sphere_supergolden


@app.cell
def _(
    colorscale,
    ele,
    marker_size,
    name,
    radius,
    sphere_custom,
    sphere_plastic,
    sphere_supergolden,
):
    # Plot points on sphere
    sphere = {}
    sphere["Supergolden"] = sphere_supergolden
    sphere["Plastic"] = sphere_plastic
    sphere["Custom"] = sphere_custom

    fig2 = ele.plot.points_on_sphere(sphere[name], title=name,
        marker_size=marker_size.value, colorscale=colorscale.value, radius=radius.value)
    return (fig2,)


@app.cell
def _(
    colorscale,
    half_sphere,
    marker_size,
    mo,
    n0,
    name_of_pts,
    num_points,
    offset_x,
    offset_y,
    radius,
    reverse,
):
    # Arrange GUI
    controls_list = []
    controls_list += [mo.md("Name"), name_of_pts]
    controls_list += [mo.hstack([reverse, mo.md("Reverse Order of Components")], justify="start")]
    controls_list += [mo.md("Total Number of Points"), num_points]
    controls_list += [mo.md("Position Offset"), offset_x, offset_y]
    controls_list += [mo.md("Index of Initial Point"), n0]
    controls_list += [mo.md("Marker Size"), marker_size]
    controls_list += [mo.md("Color Scale"), colorscale]
    controls_list += [mo.md("Radius of Inner Sphere"), radius]
    controls_list += [mo.md("Map to Half of Sphere Only"), half_sphere]

    controls = mo.vstack(controls_list)
    return (controls,)


@app.cell
def _(controls, fig2, mo):
    # Display
    mo.hstack([fig2, controls], justify="start", align="start")
    return


if __name__ == "__main__":
    app.run()
