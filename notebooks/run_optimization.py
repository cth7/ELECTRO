import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import shutil
    from pathlib import Path

    import torch

    import electro as ele

    return Path, ele, shutil, torch


@app.cell
def _(Path):
    num_points = 2_500
    max_iterations = 5_000
    block_size = 2048
    seed = 7
    # Circular sequence of points, where spheres wrap around its ends. False
    # uses the spheres from the paper.
    circular = True
    save_dir = Path("C:/Users/huynh/Desktop/electro_run")
    return block_size, circular, max_iterations, num_points, save_dir, seed


@app.cell
def _(
    block_size,
    circular,
    ele,
    max_iterations,
    num_points,
    save_dir,
    seed,
    shutil,
    torch,
):
    if save_dir.is_dir():
        shutil.rmtree(save_dir)

    params = ele.params.paper(num_points)
    if circular:
        # The step sizes in the paper are for a non-circular sequence
        params["stepsizes"] = ele.params.stepsizes(
            params["sphere_sizes"], num_points, circular=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = ele.optim.Electro(
        num_points, circular=circular, save_dir=save_dir, points_every=1,
        points_each_stage=True, seed=seed, block_size=block_size, **params)

    optimizer.points = torch.asarray(optimizer.points, device=device)
    print(optimizer.points.device, optimizer.points.dtype,
          optimizer.points.shape)
    # print(optimizer.num_stages)
    # optimizer.stage = 23

    with torch.inference_mode():
        optimizer.run(max_iterations)
    return (optimizer,)


@app.cell
def _(ele, optimizer):
    points_optimized = optimizer.points.cpu().numpy()

    f = ele.plot.points_on_sphere(points_optimized, colorscale="Turbo", radius=0.97, show_cap=False)
    f
    return


if __name__ == "__main__":
    app.run()
