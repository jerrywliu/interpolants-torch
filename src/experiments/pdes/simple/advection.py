import argparse
import os
import torch
import torch.nn as nn
from typing import List, Callable, Tuple, Dict

from src.experiments.pdes.base_pde import BasePDE
from src.models.mlp import MLP
from src.models.interpolant_nd import SpectralInterpolationND
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger

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
    def __init__(
        self,
        c: float,
        t_final: float = 1,
        u_0: Callable = None,
        device: str = "cpu",
        **base_kwargs,
    ):
        super().__init__(
            "advection", [(0, 1), (0, 2 * torch.pi)], device=device, **base_kwargs
        )
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

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

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
            grid = model.make_grid(pde_nodes)
            u = model.forward_grid(grid).reshape(n_t, n_x)
            grads = torch.autograd.grad(
                u.sum(), grid, create_graph=True
            )  # (N_t*N_x, 2)
            u_t = grads[0][:, :, 0].reshape(n_t, n_x)
            u_x = grads[0][:, :, 1].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes).reshape(n_ic)

        # PDE loss
        pde_residual = u_t + self.c * u_x
        pde_loss = torch.mean(pde_residual**2)
        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2)

        loss_names = ["pde_loss", "ic_loss"]
        return dict(zip(loss_names, [pde_loss, ic_loss]))

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes, ic_weight)

        pde_weight = self.loss_weights.get("pde_loss_weight", 1.0)
        ic_weight = self.loss_weights.get("ic_loss_weight", 1.0)

        loss = (pde_weight * loss_dict["pde_loss"]) + (
            ic_weight * ic_weight * loss_dict["ic_loss"]
        )

        return (
            loss,
            loss_dict["pde_loss"],
            loss_dict["ic_loss"],
        )

    # Get the least squares problem equivalent to a spectral solve
    def get_least_squares(self, model: SpectralInterpolationND):
        n_t, n_x = model.nodes[0].shape[0], model.nodes[1].shape[0]

        # PDE operator
        D_t = model.derivative_matrix(k=(1, 0))  # (N_t*N_x, N_t*N_x)
        D_x = model.derivative_matrix(k=(0, 1))  # (N_t*N_x, N_t*N_x)
        L = D_t + self.c * D_x

        # Initial condition: extract t=0 values
        IC = torch.zeros(
            n_x, n_t * n_x, device=model.values.device, dtype=model.values.dtype
        )
        for i in range(n_x):
            IC[i, n_x * (n_t - 1) + i] = 1  # Set t=0 value to 1 for each x

        # Right hand side
        b = torch.zeros(
            n_t * n_x + n_x, device=model.values.device, dtype=model.values.dtype
        )
        b[n_t * n_x :] = self.u_0(model.nodes[1])

        # Full system
        A = torch.cat([L, IC], dim=0)
        return A, b

    def fit_least_squares(self, model: SpectralInterpolationND):
        A, b = self.get_least_squares(model)
        u = torch.linalg.lstsq(A, b).solution
        u = u.reshape(model.nodes[0].shape[0], model.nodes[1].shape[0])
        model.values.data = u
        return model

    def plot_solution(
        self,
        nodes: List[torch.Tensor],
        u: torch.Tensor,  # (N_t, N_x)
        save_path: str = None,
    ):
        self._plot_solution_default(nodes, u, save_path)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--c", type=int, default=80)
    args.add_argument("--n_t", type=int, required=False)
    args.add_argument("--n_x", type=int, required=False)
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=100000)
    args.add_argument("--eval_every", type=int, default=1000)

    # lwup is one of [grad_norm, none].
    args.add_argument("--loss_weight_update_policy", "-lwup", type=str, default="none")
    args.add_argument("--loss_weight_update_interval", "-lwui", type=int, default=-1)
    args.add_argument("--loss_weight_max", "-lmw", type=float, default=100.0)

    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    c = args.c
    t_final = 1
    u_0 = torch.sin
    pde = Advection(
        c=c,
        t_final=t_final,
        u_0=u_0,
        device=device,
        loss_weight_update_policy=args.loss_weight_update_policy,
        loss_weight_update_interval=args.loss_weight_update_interval,
        loss_weight_max=args.loss_weight_max,
    )

    base_save_dir = (
        f"/pscratch/sd/j/jwl50/interpolants-torch/plots/pdes/advection/c={c}"
    )

    # Evaluation setup (shared for all methods)
    eval_every = args.eval_every
    n_eval = 200
    t_eval = torch.linspace(
        pde.domain[0][0], pde.domain[0][1], n_eval, device=device, requires_grad=True
    )
    x_eval = torch.linspace(
        pde.domain[1][0],
        pde.domain[1][1],
        n_eval + 1,
        device=device,
        requires_grad=True,
    )[:-1]

    def eval_sampler():
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    #########################################################
    # 1. Least Squares Polynomial Interpolant
    #########################################################
    save_dir = os.path.join(
        base_save_dir, f"polynomial_least_squares/n_t={args.n_t}_n_x={args.n_x}"
    )
    print("Fitting model with least squares...")
    n_t_ls = args.n_t if args.n_t is not None else c + 1
    n_x_ls = args.n_x if args.n_x is not None else c
    bases_ls = ["chebyshev", "fourier"]
    model_ls = SpectralInterpolationND(
        Ns=[n_t_ls, n_x_ls],
        bases=bases_ls,
        domains=pde.domain,
        device=device,
    )
    model_ls = pde.fit_least_squares(model_ls)
    pde.plot_solution(
        [t_eval, x_eval],
        model_ls.interpolate([t_eval, x_eval]),
        save_path=os.path.join(save_dir, "advection_ls_solution.png"),
    )

    #########################################################
    # 2. Neural network
    #########################################################
    save_dir = os.path.join(base_save_dir, f"mlp")
    # Logger setup
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    # Model setup
    model_mlp = MLP(
        n_dim=2,
        hidden_dim=32,
        activation=torch.tanh,
        device=device,
    )

    # Training setup
    n_epochs = args.n_epochs
    # lr = 1e-3
    optimizer = pde.get_optimizer(model_mlp, args.method)

    n_t_train = 2 * c + 1
    n_x_train = 2 * c
    n_ic_train = 2 * c
    ic_weight = 10

    def pde_sampler():
        t_nodes = pde.sample_domain_1d(
            n_samples=n_t_train,
            dim=0,
            basis="fourier",
            type=args.sample_type,
        )
        x_nodes = pde.sample_domain_1d(
            n_samples=n_x_train,
            dim=1,
            basis="fourier",
            type=args.sample_type,
        )
        return [t_nodes, x_nodes]

    def ic_sampler():
        ic_nodes = pde.sample_domain_1d(
            n_samples=n_ic_train,
            dim=1,
            basis="fourier",
            type=args.sample_type,
        )
        return [torch.tensor([0.0], requires_grad=True, device=device), ic_nodes]

    print(f"Training MLP with {args.method} optimizer...")
    pde.train(
        model_mlp,
        n_epochs=args.n_epochs,
        optimizer=optimizer,
        pde_sampler=pde_sampler,
        ic_sampler=ic_sampler,
        ic_weight=ic_weight,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
    )

    #########################################################
    # 3. Polynomial Interpolant
    #########################################################
    save_dir = os.path.join(base_save_dir, "polynomial")
    # Logger setup
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    # Model setup
    n_t = args.n_t if args.n_t is not None else c + 1
    n_x = args.n_x if args.n_x is not None else c
    bases = ["chebyshev", "fourier"]
    model = SpectralInterpolationND(
        Ns=[n_t, n_x],
        bases=bases,
        domains=pde.domain,
        device=device,
    )

    # Training setup
    n_epochs = args.n_epochs
    # lr = 1e-3
    optimizer = pde.get_optimizer(model, args.method)

    n_t_train = 2 * c + 1
    n_x_train = 2 * c
    n_ic_train = 2 * c
    ic_weight = 10

    def pde_sampler():
        t_nodes = pde.sample_domain_1d(
            n_samples=n_t_train,
            dim=0,
            basis=bases[0],
            type=args.sample_type,
        )
        x_nodes = pde.sample_domain_1d(
            n_samples=n_x_train,
            dim=1,
            basis=bases[1],
            type=args.sample_type,
        )
        return [t_nodes, x_nodes]

    def ic_sampler():
        ic_nodes = pde.sample_domain_1d(
            n_samples=n_ic_train,
            dim=1,
            basis=bases[1],
            type=args.sample_type,
        )
        return [torch.tensor([0.0], requires_grad=True, device=device), ic_nodes]

    # Handle the case for cyclical optimizers seperately 
    if args.method == "dual":
        print("Training Polynomial Interpolant with cycling optimizer...")

        pde.train_model_dual_optimizers(
            model, 
            n_epochs=args.n_epochs,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=eval_every,
            save_dir=save_dir,
        )

    else: 
        print(f"Training Polynomial Interpolant with {args.method} optimizer...")
        pde.train(
            model,
            n_epochs=args.n_epochs,
            optimizer=optimizer,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=eval_every,
            save_dir=save_dir,
        )
