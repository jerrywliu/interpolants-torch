import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Callable

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND

"""
1D Advection equation:
u_t + c * u_x = 0
t in [0, t_final] (default: t_final = 1)
x in [0, 2*pi]
u(t=0, x) = u_0(x) (default: u_0(x) = sin(x))

Solution:
u(t, x) = u_0(x - c*t)
"""


class Advection(BasePDE):
    def __init__(self, c: float, t_final: float = 1, u_0: Callable = None):
        super().__init__("advection", [(0, 1), (0, 2 * torch.pi)])
        self.c = c
        self.t_final = t_final
        if u_0 is None:
            self.u_0 = lambda x: torch.sin(x)
        else:
            self.u_0 = u_0
        self.exact_solution = lambda t, x: self.u_0(x - self.c * t)

    def get_exact_solution(self, t, x):
        return self.exact_solution(t, x)

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: torch.Tensor,
        ic_weight: float = 1,
        **kwargs,
    ) -> torch.Tensor:
        if ic_nodes is None:
            ic_nodes = pde_nodes[-1]

        n_t, n_x = pde_nodes[0].shape[0], pde_nodes[1].shape[0]
        n_ic = ic_nodes.shape[0]

        if isinstance(model, SpectralInterpolationND):
            # PDE
            u = model.interpolate(pde_nodes)
            u_t = model.derivative(pde_nodes, k=(1, 0))  # (N_t, N_x)
            u_x = model.derivative(pde_nodes, k=(0, 1))  # (N_t, N_x)
            # IC
            u_ic = model.interpolate([torch.tensor([0.0]), ic_nodes])[0]  # (N_ic)
        else:
            # Create mesh grid
            t_grid, x_grid = torch.meshgrid(pde_nodes[0], pde_nodes[1], indexing="ij")
            pde_nodes_grid = torch.stack([t_grid.flatten(), x_grid.flatten()], dim=1)
            pde_nodes_grid.requires_grad = True
            # PDE
            u = model(pde_nodes_grid).reshape(n_t, n_x)
            grads = torch.autograd.grad(u.sum(), pde_nodes_grid, create_graph=True)[
                0
            ]  # (N_t*N_x, 2)
            u_t = grads[:, 0].reshape(n_t, n_x)
            u_x = grads[:, 1].reshape(n_t, n_x)
            # IC
            ic_nodes_grid = torch.stack([torch.zeros_like(ic_nodes), ic_nodes], dim=1)
            u_ic = model(ic_nodes_grid).reshape(n_ic)

        # PDE loss
        pde_residual = u_t + self.c * u_x
        pde_loss = torch.mean(pde_residual**2)
        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes)
        ic_loss = torch.mean(ic_residual**2)
        # Total loss
        loss = pde_loss + ic_weight * ic_loss
        return loss, pde_loss, ic_loss

    # Get the least squares problem equivalent to a spectral solve
    def get_least_squares(self, model: SpectralInterpolationND):
        n_t, n_x = model.nodes[0].shape[0], model.nodes[1].shape[0]

        # PDE operator
        D_t = model.derivative_matrix(k=(1, 0))  # (N_t*N_x, N_t*N_x)
        D_x = model.derivative_matrix(k=(0, 1))  # (N_t*N_x, N_t*N_x)
        L = D_t + self.c * D_x

        # Initial condition: extract t=0 values
        IC = torch.zeros(n_x, n_t * n_x).to(dtype=model.values.dtype)
        for i in range(n_x):
            IC[i, n_x * (n_t - 1) + i] = 1  # Set t=0 value to 1 for each x

        # Right hand side
        b = torch.zeros(n_t * n_x + n_x, dtype=model.values.dtype)
        b[n_t * n_x :] = self.u_0(model.nodes[1])

        # Full system
        A = torch.cat([L, IC], dim=0)
        return A, b

    def fit_least_squares(self, model: SpectralInterpolationND):
        A, b = self.get_least_squares(model)
        u = torch.linalg.lstsq(A, b).solution
        u = u.reshape(model.nodes[0].shape[0], model.nodes[1].shape[0]).to(
            dtype=model.values.dtype
        )
        model.values.data = u
        return model

    def plot_solution(
        self,
        u: torch.Tensor,
        t_grid: torch.Tensor,
        x_grid: torch.Tensor,
        save_path: str = None,
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Predicted solution
        im1 = ax1.imshow(
            u, extent=[0, self.t_final, 0, 2 * torch.pi], origin="lower", aspect="auto"
        )
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("Predicted Solution")

        # True solution
        t_mesh, x_mesh = torch.meshgrid(t_grid, x_grid, indexing="ij")
        u_true = self.get_exact_solution(t_mesh, x_mesh)
        im2 = ax2.imshow(
            u_true,
            extent=[0, self.t_final, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("True Solution")

        # Error on log scale
        error = torch.abs(u - u_true)
        im3 = ax3.imshow(
            error,
            extent=[0, self.t_final, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
            norm="log",
        )
        plt.colorbar(im3, ax=ax3)
        ax3.set_title("Log Error")

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
        sample_type: List[str],
        n_t_samples: int,
        n_x_samples: int,
        n_ic_samples: int,
        ic_weight: float,
        t_eval: torch.Tensor,
        x_eval: torch.Tensor,
        plot_every: int = 100,
        save_dir: str = None,
    ):
        # Training history
        history = {
            "loss": [],
            "pde_loss": [],
            "ic_loss": [],
            "eval_l2_error": [],
            "eval_max_error": [],
            "eval_pde_loss": [],
        }

        print("Training model...")
        start_time = time()
        for epoch in tqdm(range(n_epochs)):
            # Sample points
            t_nodes = self.sample_domain_1d(
                n_samples=n_t_samples, dim=0, basis=model.bases[0], type=sample_type[0]
            )
            x_nodes = self.sample_domain_1d(
                n_samples=n_x_samples, dim=1, basis=model.bases[1], type=sample_type[1]
            )
            ic_nodes = self.sample_domain_1d(
                n_samples=n_ic_samples, dim=1, basis=model.bases[1], type=sample_type[1]
            )
            pde_nodes = [t_nodes, x_nodes]

            # Train step
            optimizer.zero_grad()
            # Get PDE loss
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model, pde_nodes, ic_nodes, ic_weight
            )
            loss.backward()
            optimizer.step()

            # Evaluate solution
            u_eval = model.interpolate([t_eval, x_eval])
            u_true = self.get_exact_solution(t_eval, x_eval)
            eval_l2_error = torch.mean((u_eval - u_true) ** 2)
            eval_max_error = torch.max(torch.abs(u_eval - u_true))
            # Evaluate PDE loss
            _, eval_pde_loss, _ = self.get_pde_loss(
                model, [t_eval, x_eval], ic_nodes, ic_weight
            )

            # Update history
            history["loss"].append(loss.item())
            history["eval_l2_error"].append(eval_l2_error.item())
            history["eval_max_error"].append(eval_max_error.item())
            history["eval_pde_loss"].append(eval_pde_loss.item())
            history["pde_loss"].append(pde_loss.item())
            history["ic_loss"].append(ic_loss.item())

            # Print and plot progress
            if (epoch + 1) % plot_every == 0:
                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"PDE loss: {history['pde_loss'][-1]:1.3e}")
                print(f"IC loss: {history['ic_loss'][-1]:1.3e}")
                print(f"Evaluation L2 error: {history['eval_l2_error'][-1]:1.3e}")
                self.plot_solution(
                    u_eval.detach(),
                    t_eval,
                    x_eval,
                    save_path=os.path.join(save_dir, f"advection_solution_{epoch}.png"),
                )


if __name__ == "__main__":
    # Problem setup
    c = 8
    t_final = 1
    u_0 = lambda x: torch.sin(x)
    pde = Advection(c=c, t_final=t_final, u_0=u_0)
    save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/pdes/advection"

    # Model setup
    n_t = 2 * c + 1
    n_x = 2 * c
    bases = ["chebyshev", "fourier"]
    model = SpectralInterpolationND(
        Ns=[n_t, n_x],
        bases=bases,
        domains=pde.domain,
    )

    # Training setup
    n_epochs = 10000
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # PDE
    sample_type = ["uniform", "uniform"]
    n_t_train = 4 * c + 1
    n_x_train = 4 * c
    n_ic_train = 4 * c
    ic_weight = 10

    # Evaluation setup
    n_eval = 200
    t_eval = torch.linspace(0, t_final, n_eval)
    x_eval = torch.linspace(0, 2 * torch.pi, n_eval + 1)[:-1]

    # Train model
    pde.train_model(
        model,
        n_epochs=n_epochs,
        optimizer=optimizer,
        sample_type=sample_type,
        n_t_samples=n_t_train,
        n_x_samples=n_x_train,
        n_ic_samples=n_ic_train,
        ic_weight=ic_weight,
        t_eval=t_eval,
        x_eval=x_eval,
        plot_every=100,
        save_dir=save_dir,
    )
