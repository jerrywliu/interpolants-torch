import matplotlib.pyplot as plt
import os
import torch

from experiments.interpolation.base_analytical_target import (
    BaseAnalyticalTarget,
)
from src.models.interpolant_nd import SpectralInterpolationND


class ReactionTarget(BaseAnalyticalTarget):
    def __init__(self, rho: float = 1):
        self.u_0 = lambda x: torch.exp(
            -((x - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2)
        )
        super().__init__(
            "reaction",
            f=lambda t, x: self.u_0(x)
            * torch.exp(self.rho * t)
            / (self.u_0(x) * torch.exp(self.rho * t) + (1 - self.u_0(x))),
            domain=[(0, 1), (0, 2 * torch.pi)],
        )
        self.rho = rho

    def get_exact_solution(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.f(t, x)

    def plot_solution(
        self,
        u: torch.Tensor,  # (N_t, N_x)
        t_grid: torch.Tensor,
        x_grid: torch.Tensor,
        save_path: str = None,
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Predicted solution
        im1 = ax1.imshow(
            u.T,
            extent=[0, 1, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("Predicted Solution")

        # True solution
        t_mesh, x_mesh = torch.meshgrid(t_grid, x_grid, indexing="ij")
        u_true = self.get_exact_solution(t_mesh, x_mesh)
        im2 = ax2.imshow(
            u_true.T,
            extent=[0, 1, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("True Solution")

        # Error on log scale
        error = torch.abs(u - u_true)
        l2_rel_error = torch.norm(u - u_true, p=2) / torch.norm(u_true, p=2)
        im3 = ax3.imshow(
            error.T,
            extent=[0, 1, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
            norm="log",
        )
        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f"Error. Max: {error.max():.3e}, L2 rel: {l2_rel_error:.3e}")

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    rhos = [1, 2, 5]

    for rho in rhos:

        # Problem setup
        target = ReactionTarget(rho=rho)

        # Sanity check: ensure that the interpolation model is expressive enough to fit the target
        save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/reaction/sanity_check"
        for n_t in [21, 41, 81, 161, 321, 641, 1281, 2001]:
            # n_x = n_t - 1
            n_x = n_t
            # bases = ["chebyshev", "fourier"]
            bases = ["chebyshev", "chebyshev"]
            model = SpectralInterpolationND(
                Ns=[n_t, n_x],
                bases=bases,
                domains=target.domain,
            )
            # Set values
            t_mesh, x_mesh = torch.meshgrid(
                model.nodes[0], model.nodes[1], indexing="ij"
            )
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
                save_path=os.path.join(
                    save_dir, f"reaction_soln_rho={rho}_nt={n_t}_nx={n_x}.png"
                ),
            )
