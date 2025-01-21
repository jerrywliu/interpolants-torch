import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from typing import List, Callable, Tuple

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.utils.metrics import l2_error, max_error, l2_relative_error

"""
1D Advection equation:
u_t + c * u_x = 0
t in [0, t_final] (default: t_final = 1)
x in [0, 2*pi]
u(t=0, x) = u_0(x) (default: u_0(x) = sin(x))
u(t, x=0) = u(t, x=2*pi)

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

    def get_solution(self, nodes: List[torch.Tensor]):
        t_mesh, x_mesh = torch.meshgrid(nodes[0], nodes[1], indexing="ij")
        return self.exact_solution(t_mesh, x_mesh)

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ic_nodes is None:
            ic_nodes = [torch.tensor([0.0]), pde_nodes[-1]]

        n_t, n_x = pde_nodes[0].shape[0], pde_nodes[1].shape[0]
        n_ic = ic_nodes[1].shape[0]

        if isinstance(model, SpectralInterpolationND):
            # PDE
            u = model.interpolate(pde_nodes)
            u_t = model.derivative(pde_nodes, k=(1, 0))  # (N_t, N_x)
            u_x = model.derivative(pde_nodes, k=(0, 1))  # (N_t, N_x)
            # IC
            u_ic = model.interpolate(ic_nodes)[0]  # (N_ic)
        else:
            # PDE
            u = model(pde_nodes).reshape(n_t, n_x)
            grads = torch.autograd.grad(u.sum(), pde_nodes, create_graph=True)[
                0
            ]  # (N_t*N_x, 2)
            u_t = grads[:, 0].reshape(n_t, n_x)
            u_x = grads[:, 1].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes).reshape(n_ic)

        # PDE loss
        pde_residual = u_t + self.c * u_x
        pde_loss = torch.mean(pde_residual**2)
        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
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
        nodes: List[torch.Tensor],
        u: torch.Tensor,  # (N_t, N_x)
        save_path: str = None,
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Predicted solution
        im1 = ax1.imshow(
            u.T,
            extent=[
                self.domain[0][0],
                self.domain[0][1],
                self.domain[1][0],
                self.domain[1][1],
            ],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("Predicted Solution")

        # True solution
        u_true = self.get_solution(nodes)
        im2 = ax2.imshow(
            u_true.T,
            extent=[
                self.domain[0][0],
                self.domain[0][1],
                self.domain[1][0],
                self.domain[1][1],
            ],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("True Solution")

        # Error on log scale
        error = torch.abs(u - u_true)
        im3 = ax3.imshow(
            error.T,
            extent=[
                self.domain[0][0],
                self.domain[0][1],
                self.domain[1][0],
                self.domain[1][1],
            ],
            origin="lower",
            aspect="auto",
            norm="log",
        )
        plt.colorbar(im3, ax=ax3)
        ax3.set_title("Error")

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--c", type=int, default=80)
    args.add_argument("--nt", type=int, required=False)
    args.add_argument("--nx", type=int, required=False)
    args.add_argument("--method", type=str, default="adam")
    args = args.parse_args()

    torch.set_default_dtype(torch.float64)

    # Problem setup
    c = args.c
    t_final = 1
    u_0 = lambda x: torch.sin(x)
    pde = Advection(c=c, t_final=t_final, u_0=u_0)
    save_dir = f"/pscratch/sd/j/jwl50/interpolants-torch/plots/pdes/advection_c={c}"

    # Eval
    n_eval = 200
    t_eval = torch.linspace(pde.domain[0][0], pde.domain[0][1], n_eval)
    x_eval = torch.linspace(pde.domain[1][0], pde.domain[1][1], n_eval + 1)[:-1]

    # Baseline: least squares
    print("Fitting model with least squares...")
    n_t_ls = args.nt if args.nt is not None else c + 1
    n_x_ls = args.nx if args.nx is not None else c
    bases_ls = ["chebyshev", "fourier"]
    model_ls = SpectralInterpolationND(
        Ns=[n_t_ls, n_x_ls],
        bases=bases_ls,
        domains=pde.domain,
    )
    model_ls = pde.fit_least_squares(model_ls)
    pde.plot_solution(
        [t_eval, x_eval],
        model_ls.interpolate([t_eval, x_eval]).detach(),
        save_path=os.path.join(save_dir, "advection_ls_solution.png"),
    )

    # Model setup
    print("Training model with first-order method...")
    n_t = args.nt if args.nt is not None else c + 1
    n_x = args.nx if args.nx is not None else c
    bases = ["chebyshev", "fourier"]
    model = SpectralInterpolationND(
        Ns=[n_t, n_x],
        bases=bases,
        domains=pde.domain,
    )

    # Training setup
    n_epochs = 100000
    plot_every = 1000
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # PDE
    sample_type = ["uniform", "uniform"]
    n_t_train = 2 * c + 1
    n_x_train = 2 * c
    n_ic_train = 2 * c
    ic_weight = 10

    def pde_sampler():
        t_nodes = pde.sample_domain_1d(
            n_samples=n_t_train,
            dim=0,
            basis=bases[0],
            type=sample_type[0],
        )
        x_nodes = pde.sample_domain_1d(
            n_samples=n_x_train,
            dim=1,
            basis=bases[1],
            type=sample_type[1],
        )
        return [t_nodes, x_nodes]

    def ic_sampler():
        ic_nodes = pde.sample_domain_1d(
            n_samples=n_ic_train,
            dim=1,
            basis=bases[1],
            type=sample_type[1],
        )
        return [torch.tensor([0.0]), ic_nodes]

    def eval_sampler():
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    if args.method == "adam":
        # Train model
        pde.train_model(
            model,
            n_epochs=n_epochs,
            optimizer=optimizer,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            plot_every=plot_every,
            save_dir=save_dir,
        )

    elif args.method == "lbfgs":
        # Train model with L-BFGS
        optimizer = torch.optim.LBFGS(model.parameters())
        pde.train_model_lbfgs(
            model,
            max_iter=n_epochs,
            optimizer=optimizer,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            plot_every=100,
            save_dir=save_dir,
        )
