import argparse
import os
import scipy.io
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.models.rational_2d import RationalInterpolation2D
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger

"""
1D Burgers equation:
u_t + u * u_x - nu * u_xx = 0
t in [0, 1]
x in [-1, 1]
u(t=0, x) = -sin(pi*x)
u(t, x=-1) = u(t, x=1) = 0
"""


class Burgers(BasePDE):
    def __init__(
        self,
        nu: float = 0.01 / torch.pi,
        device: str = "cpu",
        **base_kwargs,
    ):
        super().__init__("burgers", [(0, 1), (-1, 1)], device=device, **base_kwargs)
        self.nu = nu
        self.u_0 = lambda x: -torch.sin(torch.pi * x)
        self.ref_u, self.ref_t, self.ref_x = self.load_ref_solution()

    def load_ref_solution(
        self,
        path: str = "/pscratch/sd/j/jwl50/interpolants-torch/src/experiments/pdes/benchmarks/ref/burgers_1d.mat",
    ):
        mat = scipy.io.loadmat(path)
        u_ref = torch.tensor(mat["usol"], dtype=torch.float64, device=self.device)
        t = torch.tensor(mat["t"][0], dtype=torch.float64, device=self.device)
        x = torch.tensor(mat["x"][0], dtype=torch.float64, device=self.device)
        return u_ref, t, x

    # Hack: assume t and x are the same as the reference solution
    def get_solution(self, nodes: List[torch.Tensor]):
        return self.ref_u

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

        if isinstance(model, SpectralInterpolationND) or isinstance(
            model, RationalInterpolation2D
        ):
            # PDE
            u = model.interpolate(pde_nodes)
            u_t = model.derivative(pde_nodes, k=(1, 0))
            u_x = model.derivative(pde_nodes, k=(0, 1))
            u_xx = model.derivative(pde_nodes, k=(0, 2))
            # IC
            u_ic = model.interpolate(ic_nodes)[0]
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][0]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )
            u_periodic_t1 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][1]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )
        else:
            # PDE
            grid = model.make_grid(pde_nodes)  # (N_t*N_x, 2)
            u = model.forward_grid(grid).reshape(n_t, n_x)  # (N_t, N_x)
            grads = torch.autograd.grad(
                u.sum(), grid, create_graph=True
            )  # (N_t*N_x, 2)
            # First derivatives
            u_t = grads[0][..., 0].reshape(n_t, n_x)
            u_x = grads[0][..., 1].reshape(n_t, n_x)
            # Second derivative: u_xx
            grad_xx = torch.autograd.grad(u_x.sum(), grid, create_graph=True)
            u_xx = grad_xx[0][..., 1].reshape(n_t, n_x)
            # IC
            u_ic = model(ic_nodes)[0]
            # Enforce periodic boundary conditions at t nodes
            u_periodic_t0 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][0]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )
            u_periodic_t1 = model.interpolate(
                [
                    pde_nodes[0],
                    torch.tensor(
                        [self.domain[1][1]],
                        dtype=pde_nodes[1].dtype,
                        device=model.device,
                        requires_grad=True,
                    ),
                ]
            )

        # PDE loss
        pde_residual = u_t + u * u_x - self.nu * u_xx
        pde_loss = torch.mean(pde_residual**2)

        # IC loss
        ic_residual = u_ic - self.u_0(ic_nodes[1])
        ic_loss = torch.mean(ic_residual**2)

        # Dirichlet boundary conditions loss
        pbc_loss = torch.mean(u_periodic_t0**2) + torch.mean(u_periodic_t1**2)

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

    def get_least_squares(self, model: SpectralInterpolationND):
        raise NotImplementedError("Least squares not implemented for Burgers equation")

    def fit_least_squares(self, model: SpectralInterpolationND):
        raise NotImplementedError("Least squares not implemented for Burgers equation")

    def plot_solution(
        self,
        nodes: List[torch.Tensor],
        u: torch.Tensor,  # (N_t, N_x)
        save_path: str = None,
    ):
        self._plot_solution_default(nodes, u, save_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--nu", type=float, default=0.01 / torch.pi)
    args.add_argument("--n_t", type=int, default=81)
    args.add_argument("--n_x", type=int, default=81)
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=100000)
    args.add_argument("--eval_every", type=int, default=1000)
    args.add_argument("--model", type=str, default=None)

    # lwup is one of [grad_norm, none].
    args.add_argument("--loss_weight_update_policy", "-lwup", type=str, default="none")
    args.add_argument("--loss_weight_update_interval", "-lwui", type=int, default=-1)
    args.add_argument("--loss_weight_max", "-lmw", type=float, default=100.0)

    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    nu = args.nu
    pde = Burgers(
        nu=nu,
        device=device,
        loss_weight_update_policy=args.loss_weight_update_policy,
        loss_weight_update_interval=args.loss_weight_update_interval,
        loss_weight_max=args.loss_weight_max,
    )

    base_save_dir = (
        f"/pscratch/sd/j/jwl50/interpolants-torch/plots/pdes/burgers/nu={nu}"
    )

    # Evaluation setup
    eval_every = args.eval_every
    n_eval = 200
    t_eval = pde.ref_t
    x_eval = pde.ref_x

    def eval_sampler():
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    #########################################################
    # 1. Neural network
    #########################################################
    if args.model is None or args.model == "mlp":
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

        n_t_train = 161
        n_x_train = 161
        n_ic_train = 161
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
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

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
            logger=logger,
        )

    #########################################################
    # 2. Polynomial interpolation
    #########################################################
    if args.model is None or args.model == "polynomial":
        save_dir = os.path.join(
            base_save_dir,
            f"polynomial/method={args.method}_nt={args.n_t}_nx={args.n_x}_sample={args.sample_type}",
        )

        # Logger setup
        logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Model setup
        n_t = args.n_t
        n_x = args.n_x
        bases = ["chebyshev", "chebyshev"]
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

        n_t_train = 161
        n_x_train = 161
        n_ic_train = 161
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
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

        print(f"Training Polynomial Interpolant with {args.method} optimizer...")
        pde.train(
            model,
            n_epochs=n_epochs,
            optimizer=optimizer,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=eval_every,
            save_dir=save_dir,
            logger=logger,
        )

    #########################################################
    # 3. Rational interpolation
    #########################################################
    if args.model is None or args.model == "rational":
        save_dir = os.path.join(
            base_save_dir,
            f"rational/method={args.method}_nt={args.n_t}_nx={args.n_x}_sample={args.sample_type}",
        )

        # Logger setup
        logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Model setup
        n_t = args.n_t
        n_x = args.n_x
        bases = ["chebyshev", "chebyshev"]
        model = RationalInterpolation2D(
            N_1=n_t,
            N_2=n_x,
            bases_1=bases[0],
            domain_1=pde.domain[0],
            domain_2=pde.domain[1],
            num_poles=1,  # TODO does nothing
            device=device,
        )

        # Training setup
        n_epochs = args.n_epochs
        # lr = 1e-3
        optimizer = pde.get_optimizer(model, args.method)

        n_t_train = 161
        n_x_train = 161
        n_ic_train = 161
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
            return [torch.tensor([0.0], device=device, requires_grad=True), ic_nodes]

        print(f"Training Rational Interpolant with {args.method} optimizer...")
        pde.train(
            model,
            n_epochs=n_epochs,
            optimizer=optimizer,
            pde_sampler=pde_sampler,
            ic_sampler=ic_sampler,
            ic_weight=ic_weight,
            eval_sampler=eval_sampler,
            eval_metrics=eval_metrics,
            eval_every=eval_every,
            save_dir=save_dir,
            logger=logger,
        )
