import argparse
import os
import torch
from typing import List
from src.experiments.interpolation.base_analytical_target import (
    BaseAnalyticalTarget,
)
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger


class AdvectionTarget(BaseAnalyticalTarget):
    def __init__(self, c: float = 80, device: str = "cpu"):
        super().__init__(
            "advection",
            f=lambda t, x: torch.sin(x - c * t),
            domain=[(0, 1), (0, 2 * torch.pi)],
            device=device,
        )
        self.c = c

    def plot_solution(
        self,
        nodes: List[torch.Tensor],
        u: torch.Tensor,
        save_path: str = None,
    ):
        self._plot_solution_default(nodes, u, save_path)


# Compare interpolation of advection using different methods:
# 1. Neural network
# 2. Polynomial interpolation

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("")

    torch.set_default_dtype(torch.float64)

    # Problem setup
    target = AdvectionTarget(c=80)

    # Sanity check: ensure that the interpolation model is expressive enough to fit the target
    save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/advection/sanity_check"
    for n_t in [
        target.c // 4 + 1,
        target.c // 2 + 1,
        target.c + 1,
        2 * target.c + 1,
        4 * target.c + 1,
    ]:
        n_x = n_t - 1
        bases = ["chebyshev", "fourier"]
        model = SpectralInterpolationND(
            Ns=[n_t, n_x],
            bases=bases,
            domains=target.domain,
        )
        # Set values
        t_mesh, x_mesh = torch.meshgrid(model.nodes[0], model.nodes[1], indexing="ij")
        data = target.get_exact_solution(t_mesh, x_mesh)
        model.values.data = data

        # Evaluation setup
        n_eval = 200
        t_eval = torch.linspace(0, 1, n_eval)
        x_eval = torch.linspace(0, 2 * torch.pi, n_eval + 1)[:-1]

        # Plot solution
        target.plot_solution(
            model.interpolate([t_eval, x_eval]).detach(),
            t_eval,
            x_eval,
            save_path=os.path.join(save_dir, f"advection_soln_nt={n_t}_nx={n_x}.png"),
        )
