def get_training_grids(model=None, n_t=32, n_x=32, n_ic=None, t_max=1):
    """
    Get training grids either from model or create new ones

    Args:
        model: if SpectralInterpolationND, use its nodes. If None or MLP, create new grids
        n_t: number of time points (if not using model grid)
        n_x: number of space points (if not using model grid)
        n_ic: number of IC points (if None, use n_x)

    Returns:
        t_grid: time points
        x_grid: space points
        ic_x_grid: initial condition points
    """
    if n_ic is None:
        # t_grid = torch.cos(torch.linspace(0, 2*np.pi, n_t))*(t_max/2) + (t_max/2)
        # x_grid = torch.linspace(0, 2*np.pi, n_x)
        # ic_x_grid = torch.linspace(0, 2*np.pi, n_ic)

        t_grid = torch.cos(torch.rand(n_t) * 2 * np.pi) * (t_max / 2) + (t_max / 2)
    if isinstance(model, SpectralInterpolationND):
        t_grid = model.nodes[0]
        x_grid = model.nodes[1]
        ic_x_grid = x_grid  # Could be different if desired
    else:
        t_grid = torch.cos(torch.linspace(0, 2 * np.pi, n_t)) * (t_max / 2) + (
            t_max / 2
        )
        x_grid = torch.linspace(0, 2 * np.pi, n_x)
        ic_x_grid = torch.linspace(0, 2 * np.pi, n_ic)

    return t_grid, x_grid, ic_x_grid
