import argparse
import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Callable, Tuple, Dict

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.utils.metrics import l2_error, max_error, l2_relative_error

from src.optimizers.nys_newton_cg import NysNewtonCG

"""
1D Wave equation:
u_tt - c^2 * u_xx = 0
t in [0, 1]
x in [0, 1]
u(t=0, x) = sin(pi*x) + 1/2 * sin(beta*pi*x)
u_t(t=0, x) = 0
u(t, x=0) = u(t, x=1) = 0

Solution:
u(t, x) = sin(pi*x) cos(2*pi*t) + 1/2 * sin(beta*pi*x) cos(2*beta*pi*t)
"""


class Wave(BasePDE):
    def __init__(
        self,
        c: float = 2,
        beta: float = 5,
        device: str = "cpu",
        **base_kwargs,
    ):
        super().__init__("wave", [(0, 1), (0, 1)], **base_kwargs)
        self.device = torch.device(device)
        self.c = c
        self.beta = beta
        self.u_0 = lambda x: torch.sin(torch.pi * x) + 0.5 * torch.sin(
            beta * torch.pi * x
        )
        self.u_0_t = lambda x: 0
        self.exact_solution = lambda t, x: torch.sin(torch.pi * x) * torch.cos(
            2 * torch.pi * t
        ) + 0.5 * torch.sin(beta * torch.pi * x) * torch.cos(2 * beta * torch.pi * t)

    def get_solution(self, nodes: List[torch.Tensor]):
        t_mesh, x_mesh = torch.meshgrid(nodes[0], nodes[1], indexing="ij")
        return self.exact_solution(t_mesh, x_mesh)

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if ic_nodes is None:
            ic_nodes = [torch.tensor([0.0]), pde_nodes[-1]]

        n_t, n_x = pde_nodes[0].shape[0], pde_nodes[1].shape[0]
        n_ic = ic_nodes[1].shape[0]

        if isinstance(model, SpectralInterpolationND):
            # PDE
            u = model.interpolate(pde_nodes)
            u_tt = model.derivative(pde_nodes, k=(2, 0))
            u_xx = model.derivative(pde_nodes, k=(0, 2))
            # IC
            u_ic = model.interpolate(ic_nodes)[0]
            u_t_ic = model.derivative(ic_nodes, k=(1, 0))[0]
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model.interpolate(
                [pde_nodes[0], torch.tensor([model.domains[1][0]]).to(model.device)]
            )
            u_periodic_t1 = model.interpolate(
                [pde_nodes[0], torch.tensor([model.domains[1][1]]).to(model.device)]
            )
        else:
            # PDE
            u = model(pde_nodes).reshape(n_t, n_x)
            grads = torch.autograd.grad(u.sum(), pde_nodes, create_graph=True)[
                0
            ]  # (N_t*N_x, 2)
            u_tt = grads[:, 0].reshape(n_t, n_x)
            u_xx = grads[:, 1].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes)[0]
            u_t_ic = model.derivative(ic_nodes, k=(1, 0))[0]

        # PDE loss
        pde_residual = u_tt - self.c**2 * u_xx
        pde_loss = torch.mean(pde_residual**2)
        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_dt_residual = u_t_ic - self.u_0_t(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2) + torch.mean(ic_dt_residual**2)
        # Periodic boundary conditions loss
        pbc_loss = torch.mean((u_periodic_t0 - u_periodic_t1) ** 2)

        loss_names = ["pde_loss", "ic_loss", "pbc_loss"]
        return dict(zip(loss_names, [pde_loss, ic_loss, pbc_loss]))

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Total loss
        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes, ic_weight)

        pde_weight = self.loss_weights.get("pde_loss_weight", 1.0)
        ic_weight = self.loss_weights.get("ic_loss_weight", 1.0)
        pbc_weight = self.loss_weights.get("pbc_loss_weight", 1.0)

        loss = (
            (pde_weight * loss_dict["pde_loss"])
            + (ic_weight * ic_weight * loss_dict["ic_loss"])
            + (ic_weight * pbc_weight * loss_dict["pbc_loss"])
        )

        return (
            loss,
            loss_dict["pde_loss"],
            loss_dict["ic_loss"] + loss_dict["pbc_loss"],
        )

    # Get the least squares problem equivalent to a spectral solve
    # TODO JL 1/22/25: add periodic boundary conditions
    def get_least_squares(self, model: SpectralInterpolationND):
        n_t, n_x = model.nodes[0].shape[0], model.nodes[1].shape[0]

        # PDE operator
        D_t = model.derivative_matrix(k=(1, 0))  # (N_t, N_t*N_x)
        D_tt = model.derivative_matrix(k=(2, 0))  # (N_t*N_x, N_t*N_x)
        D_xx = model.derivative_matrix(k=(0, 2))  # (N_t*N_x, N_t*N_x)
        L = D_tt - self.c**2 * D_xx

        # Initial condition: extract t=0 values
        IC = torch.zeros(n_x, n_t * n_x).to(dtype=model.values.dtype)
        for i in range(n_x):
            IC[i, n_x * (n_t - 1) + i] = 1  # Set t=0 value to 1 for each x
        D_t_IC = D_t[n_x * (n_t - 1) : n_x * n_t, :]

        # Right hand side
        b = torch.zeros(n_t * n_x + n_x + n_x, dtype=model.values.dtype)
        b[n_t * n_x : n_t * n_x + n_x] = self.u_0(model.nodes[1])
        b[n_t * n_x + n_x :] = self.u_0_t(model.nodes[1])

        # Full system
        A = torch.cat([L, IC, D_t_IC], dim=0)
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
        u_cpu = u.detach().cpu()

        # Predicted solution
        im1 = ax1.imshow(
            u.detach().T.cpu().numpy(),
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
        u_true = self.get_solution(nodes).cpu()
        im2 = ax2.imshow(
            u_true.detach().T.cpu().numpy(),
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
        error = torch.abs(u_cpu - u_true)
        im3 = ax3.imshow(
            error.detach().T.cpu().numpy(),
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
    args.add_argument("--c", type=float, default=2)
    args.add_argument("--beta", type=float, default=5)
    args.add_argument("--n_t", type=int, default=41)
    args.add_argument("--n_x", type=int, default=41)
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=100000)

    # lwup is one of [grad_norm, none].
    args.add_argument("--loss_weight_update_policy", "-lwup", type=str, default="none")
    args.add_argument("--loss_weight_update_interval", "-lwui", type=int, default=-1)
    args.add_argument("--loss_weight_max", "-lmw", type=float, default=100.0)

    args = args.parse_args()

    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    c = args.c
    beta = args.beta
    pde = Wave(
        c=c,
        beta=beta,
        loss_weight_update_policy=args.loss_weight_update_policy,
        loss_weight_update_interval=args.loss_weight_update_interval,
        loss_weight_max=args.loss_weight_max,
        device=device,
    )

    base_save_dir = "/common/results/pdes/wave"
    # base_save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/pdes/wave"
    base_save_dir = os.path.join(
        base_save_dir,
        f"c={c}_beta={beta}_method={args.method}_n_t={args.n_t}_n_x={args.n_x}",
    )

    # Evaluation setup
    n_eval = 200
    t_eval = torch.linspace(
        pde.domain[0][0], pde.domain[0][1], n_eval, requires_grad=True, device=device
    )
    x_eval = torch.linspace(
        pde.domain[1][0], pde.domain[1][1], n_eval, requires_grad=True, device=device
    )

    # Training setup
    n_epochs = args.n_epochs
    plot_every = 1000

    # PDE
    bases = ["chebyshev", "chebyshev"]
    if args.sample_type == "standard":
        sample_type = ["standard", "standard"]
    elif args.sample_type == "uniform":
        sample_type = ["uniform", "uniform"]
    else:
        raise ValueError(f"Invalid sample type: {args.sample_type}")

    n_t_train = 161
    # n_x_train = 160
    # n_ic_train = 160
    n_x_train = 161
    n_ic_train = 161
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
        return [torch.tensor([0.0], requires_grad=True, device=device), ic_nodes]

    def eval_sampler():
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    # # 1. Least Squares Polynomial Interpolant
    # save_dir = os.path.join(base_save_dir, "polynomial_least_squares")
    # print("Fitting model with least squares...")
    # model_ls = SpectralInterpolationND(
    #     Ns=[args.n_t, args.n_x],
    #     bases=bases,
    #     domains=pde.domain,
    #     device=device,
    # )
    # model_ls = pde.fit_least_squares(model_ls)
    # pde.plot_solution(
    #     [t_eval, x_eval],
    #     model_ls.interpolate([t_eval, x_eval]),
    #     save_path=os.path.join(save_dir, "wave_ls_solution.png"),
    # )

    # 2. Polynomial Interpolant
    save_dir = os.path.join(base_save_dir, "polynomial")

    model_polynomial = SpectralInterpolationND(
        Ns=[args.n_t, args.n_x],
        bases=bases,
        domains=pde.domain,
        device=device,
    )

    optimizer = pde.get_optimizer(model_polynomial, args.method)
    pde.train_model(
        model_polynomial,
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
