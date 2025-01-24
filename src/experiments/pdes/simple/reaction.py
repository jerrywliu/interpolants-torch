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
from src.models.mlp import MLP
from src.utils.metrics import l2_error, max_error, l2_relative_error

from src.optimizers.shampoo import Shampoo
from src.optimizers.nys_newton_cg import NysNewtonCG

"""
1D Reaction equation:
u_t - rho * u(1-u) = 0
t in [0, t_final] (default: t_final = 1)
x in [0, 2*pi]
u(t=0, x) = u_0(x) (default: u_0(x) = exp(-(x - pi)**2 / (2 * (pi / 4)**2)))
u(t, x=0) = u(t, x=2*pi)

Solution:
u(t, x) = u_0(x) exp(rho * t) / [u_0(x) exp(rho * t) + (1 - u_0(x))]
"""


class Reaction(BasePDE):
    def __init__(
        self,
        rho: float,
        t_final: float = 1,
        u_0: Callable = None,
        loss_weight_update_policy: str = "grad_norm",
        loss_weight_update_interval: int = -1,
        device: str = "cpu",
    ):
        super().__init__("reaction", [(0, 1), (0, 2 * torch.pi)])
        self.device = torch.device(device)
        self.rho = rho
        self.t_final = t_final
        if u_0 is None:
            self.u_0 = lambda x: torch.exp(
                -((x - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2)
            )
        else:
            self.u_0 = u_0
        self.exact_solution = (
            lambda t, x: self.u_0(x)
            * torch.exp(self.rho * t)
            / (self.u_0(x) * torch.exp(self.rho * t) + (1 - self.u_0(x)))
        )

        self.loss_weights = [1.0, 1.0, 1.0]
        self.loss_weight_update_policy = loss_weight_update_policy
        self.loss_weight_update_interval = loss_weight_update_interval

    def get_solution(self, nodes: List[torch.Tensor]):
        t_mesh, x_mesh = torch.meshgrid(nodes[0], nodes[1], indexing="ij")
        return self.exact_solution(t_mesh, x_mesh)

    def maybe_update_loss_weights(self, epoch, model, optimizer, pde_nodes, ic_nodes):
        if (
            self.loss_weight_update_interval == -1
            or epoch % self.loss_weight_update_interval != 0
        ):
            return

        if self.loss_weight_update_policy == "grad_norm":
            num_params = sum(p.numel() for p in model.parameters())

            optimizer.zero_grad()
            loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes)
            pde_loss = loss_dict["pde_loss"]
            optimizer.zero_grad()
            pde_loss.backward(retain_graph=True)
            avg_pde_loss_grad_norm = (
                torch.sqrt(sum([torch.sum(p.grad**2) for p in model.parameters()]))
                / num_params
            )

            optimizer.zero_grad()
            loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes)
            ic_loss = loss_dict["ic_loss"]
            optimizer.zero_grad()
            ic_loss.backward(retain_graph=True)
            avg_ic_loss_grad_norm = (
                torch.sqrt(sum([torch.sum(p.grad**2) for p in model.parameters()]))
                / num_params
            )

            optimizer.zero_grad()
            loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes)
            pbc_loss = loss_dict["pbc_loss"]
            optimizer.zero_grad()
            pbc_loss.backward(retain_graph=True)
            avg_pbc_loss_grad_norm = (
                torch.sqrt(sum([torch.sum(p.grad**2) for p in model.parameters()]))
                / num_params
            )

            self.loss_weights = [
                1.0 / avg_pde_loss_grad_norm,
                1.0 / avg_ic_loss_grad_norm,
                1.0 / avg_pbc_loss_grad_norm,
            ]

            print(self.loss_weights)

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
            u_t = model.derivative(pde_nodes, k=(1, 0))  # (N_t, N_x)
            # IC
            u_ic = model.interpolate(ic_nodes)[0]  # (N_ic)
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model.interpolate(
                [pde_nodes[0], torch.tensor([model.domains[1][0]]).to(model.device)]
            )
            u_periodic_t1 = model.interpolate(
                [pde_nodes[0], torch.tensor([model.domains[1][1]]).to(model.device)]
            )
        else:
            # PDE
            grid = model.make_grid(pde_nodes)
            u = model.forward_grid(grid).reshape(n_t, n_x)
            grads = torch.autograd.grad(
                u.sum(), grid, create_graph=True
            )  # (N_t*N_x, 2)
            u_t = grads[0][:, :, 0].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes).reshape(n_ic)
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model(
                [pde_nodes[0], torch.tensor([self.domain[1][0]]).to(model.device)]
            )
            u_periodic_t1 = model(
                [pde_nodes[0], torch.tensor([self.domain[1][1]]).to(model.device)]
            )

        # PDE loss
        pde_residual = u_t - self.rho * u * (1 - u)
        pde_loss = torch.mean(pde_residual**2)

        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2)

        # Periodic boundary conditions loss.
        pbc_loss = torch.mean((u_periodic_t0 - u_periodic_t1) ** 2)

        # Total loss
        pde_loss_weight = self.loss_weights[0]
        ic_loss_weight = self.loss_weights[1]
        pbc_loss_weight = self.loss_weights[1]

        loss = (
            (pde_loss_weight * pde_loss)
            + (ic_weight * ic_loss_weight * ic_loss)
            + (ic_weight * pbc_loss_weight * pbc_loss)
        )
        loss_names = "loss, pde_loss, ic_loss, pbc_loss".split(", ")

        return dict(zip(loss_names, [loss, pde_loss, ic_loss, pbc_loss]))

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes, ic_weight)
        return (
            loss_dict["loss"],
            loss_dict["pde_loss"],
            loss_dict["ic_loss"] + loss_dict["pbc_loss"],
        )

    # Get the least squares problem equivalent to a spectral solve
    # Since this problem is nonlinear, we perform Picard iteration using the model's current guess for u
    def get_least_squares(self, model: SpectralInterpolationND):
        n_t, n_x = model.nodes[0].shape[0], model.nodes[1].shape[0]

        # PDE operator
        D_t = model.derivative_matrix(k=(1, 0))  # (N_t*N_x, N_t*N_x)
        L = D_t

        # Initial condition: extract t=0 values
        IC = torch.zeros(n_x, n_t * n_x).to(dtype=model.values.dtype)
        for i in range(n_x):
            IC[i, n_x * (n_t - 1) + i] = 1  # Set t=0 value to 1 for each x

        # Right hand side
        b = torch.zeros(n_t * n_x + n_x, dtype=model.values.dtype)
        # Picard iteration: calculate rho * u_0(x) * (1 - u_0(x))
        L_rhs = self.rho * model(model.nodes) * (1 - model(model.nodes))  # (N_t*N_x)
        b[: n_t * n_x] = L_rhs
        # IC
        b[n_t * n_x :] = self.u_0(model.nodes[1])

        # Full system
        A = torch.cat([L, IC], dim=0)
        return A, b

    def fit_least_squares(self, model: SpectralInterpolationND, max_iter: int = 100):
        # Set model values to initial condition
        IC = self.u_0(model.nodes[1])
        model.values.data = IC[None, :]
        # Perform Picard iteration
        for i in range(self.max_iter):
            A, b = self.get_least_squares(model)
            u = torch.linalg.lstsq(A, b).solution
            model.values.data = u.reshape(
                model.nodes[0].shape[0], model.nodes[1].shape[0]
            )
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


def get_optimizer(model, optimizer_name, **override_kwargs):
    optimizer_dict = {
        "adam": {
            "constructor": torch.optim.Adam,
            "kwargs": {"lr": 1e-3},
        },
        "lbfgs": {
            "constructor": torch.optim.LBFGS,
            "kwargs": {"history_size": 1000},
        },
        "shampoo": {
            "constructor": Shampoo,
            "kwargs": {"lr": 1e-3, "update_freq": 1},
        },
        "nys_newton": {
            "constructor": NysNewtonCG,
            "kwargs": {
                "lr": 1.0,
                "rank": 100,
                "mu": 1e-4,
                "line_search_fn": "armijo",
            },
        },
    }

    entry = optimizer_dict[optimizer_name]
    constructor = entry["constructor"]
    kwargs = entry["kwargs"]
    kwargs.update(override_kwargs)

    return constructor(model.parameters(), **kwargs)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--rho", type=float, default=5)
    args.add_argument("--n_t", type=int, default=81)
    args.add_argument("--n_x", type=int, default=81)
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=100000)
    args = args.parse_args()

    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    rho = args.rho
    t_final = 1

    def u_0(x):
        return torch.exp(-((x - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2))

    pde = Reaction(
        rho=rho,
        t_final=t_final,
        u_0=u_0,
        loss_weight_update_policy="grad_norm",
        loss_weight_update_interval=-1,
        device=device,
    )

    device = torch.device("cuda")

    base_save_dir = "/common/results/pdes/reaction"
    # base_save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/pdes/reaction"

    # Evaluation setup
    n_eval = 200
    t_eval = torch.linspace(
        pde.domain[0][0], pde.domain[0][1], n_eval, requires_grad=True, device=device
    )

    x_eval = torch.linspace(
        pde.domain[1][0],
        pde.domain[1][1],
        n_eval,
        requires_grad=True,
        device=device,
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
        return [t_nodes.to(device), x_nodes.to(device)]

    def ic_sampler():
        ic_nodes = pde.sample_domain_1d(
            n_samples=n_ic_train,
            dim=1,
            basis=bases[1],
            type=sample_type[1],
        )
        return [
            torch.tensor([0.0], device=device, requires_grad=True),
            ic_nodes.to(device),
        ]

    def eval_sampler():
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    # 1. Neural network
    save_dir = os.path.join(base_save_dir, "mlp")
    model_mlp = MLP(n_dim=2, hidden_dim=32, device=device, activation=torch.tanh)

    optimizer = get_optimizer(model_mlp, args.method)
    pde.train_model(
        model=model_mlp,
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

    # 2. Polynomial interpolation.
    # save_dir = os.path.join(base_save_dir, "polynomial")

    # model_polynomial = SpectralInterpolationND(
    #     Ns=[args.n_t, args.n_x],
    #     bases=bases,
    #     domains=pde.domain,
    #     device=device,
    # )

    # optimizer = get_optimizer(model_polynomial, args.method)
    # pde.train_model(
    #     model_polynomial,
    #     n_epochs=n_epochs,
    #     optimizer=optimizer,
    #     pde_sampler=pde_sampler,
    #     ic_sampler=ic_sampler,
    #     ic_weight=ic_weight,
    #     eval_sampler=eval_sampler,
    #     eval_metrics=eval_metrics,
    #     plot_every=plot_every,
    #     save_dir=save_dir,
    # )
