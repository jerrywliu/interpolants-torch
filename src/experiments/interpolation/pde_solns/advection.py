import matplotlib.pyplot as plt
import os
import torch

from experiments.interpolation.base_analytical_target import (
    BaseAnalyticalTarget,
)
from src.models.interpolant_nd import SpectralInterpolationND


class AdvectionTarget(BaseAnalyticalTarget):
    def __init__(self, c: float = 80):
        super().__init__(
            "advection",
            f=lambda t, x: torch.sin(x - c * t),
            domain=[(0, 1), (0, 2 * torch.pi)],
        )
        self.c = c

    def get_solution(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        t_mesh, x_mesh = torch.meshgrid(nodes[0], nodes[1], indexing="ij")
        t_mesh = t_mesh.to(device=nodes[0].device)
        x_mesh = x_mesh.to(device=nodes[0].device)
        return self.f(t_mesh, x_mesh)

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

    def train_model(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        basis_type: str,  # "chebyshev" or "fourier"
        sample_type: str,  # standard or uniform
        n_samples: int,
        x_eval: torch.Tensor,
        plot_every: int = 100,
        save_dir: str = None,
    ):
        # Training history
        history = {
            "loss": [],
            "eval_l2_error": [],
            "eval_max_error": [],
        }
        loss_fn = nn.MSELoss()

        print("Training model...")
        start_time = time()
        for epoch in tqdm(range(n_epochs)):
            # Sample points
            t_train = self.sample_domain_1d(
                n_samples=n_samples,
                dim=0,
                basis=basis_type,
                type=sample_type,
            )
            x_train = self.sample_domain_1d(
                n_samples=n_samples,
                dim=1,
                basis=basis_type,
                type=sample_type,
            )
            f_train_pred = model([t_train, x_train])
            f_train_true = self.get_exact_solution(t_train, x_train)

            # Train step
            optimizer.zero_grad()
            loss = loss_fn(f_train_pred, f_train_true)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

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
