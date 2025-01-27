import argparse
import os
import torch
import torch.nn as nn
from typing import List

from src.experiments.interpolation.pde_solns.advection import AdvectionTarget
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger


# This is the same as the AdvectionTarget, but we cut a hole in the domain
# Domain = [0, 1] x [0, 2*pi] minus a circle of radius 0.25 centered at (0.5, 0.5)
class AdvectionIrregularTarget(AdvectionTarget):
    def __init__(self, c: float = 80, device: str = "cpu"):
        super().__init__(c, device)

    def _in_domain(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Check if outside the circle of radius 0.25 at (0.5, 0.5)
        outside_circle = (x - 0.5) ** 2 + (t - 0.5) ** 2 > 0.25**2
        # Check if within the rectangular domain [0, 1] x [0, 2π]
        in_rectangle = (
            (t >= self.domain[0][0])
            & (t <= self.domain[0][1])
            & (x >= self.domain[1][0])
            & (x <= self.domain[1][1])
        )
        return outside_circle & in_rectangle  # Combine conditions

    def in_domain(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        mesh = torch.meshgrid(*nodes, indexing="ij")
        return self._in_domain(*mesh)

    def get_loss(self, model: nn.Module, nodes: List[torch.Tensor]) -> torch.Tensor:
        u_pred = model(nodes)
        u_true = self.get_function(nodes)
        mask = self.in_domain(nodes).float()  # Convert boolean mask to float
        return torch.mean((u_pred - u_true) ** 2 * mask) / torch.mean(mask)


# Compare interpolation of advection using different methods:
# 1. Neural network
# 2. Polynomial interpolation

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--c", type=int, default=80)
    args.add_argument("--n_t", type=int, default=81)
    args.add_argument("--n_x", type=int, default=81)
    args.add_argument("--sample_type", type=str, default="uniform")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=10000)
    args.add_argument("--eval_every", type=int, default=100)
    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    c = args.c
    target = AdvectionIrregularTarget(c=c, device=device)
    base_save_dir = f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/advection_irregular/c={c}"

    # Evaluation setup (shared for all methods)
    eval_every = args.eval_every
    n_eval = 200
    t_eval = torch.linspace(
        target.domain[0][0],
        target.domain[0][1],
        n_eval,
        device=device,
        requires_grad=True,
    )
    x_eval = torch.linspace(
        target.domain[1][0],
        target.domain[1][1],
        n_eval + 1,
        device=device,
        requires_grad=True,
    )[:-1]

    def eval_sampler():
        return t_eval, x_eval

    eval_metrics = [l2_error, max_error, l2_relative_error]

    #########################################################
    # 1. Neural network
    #########################################################
    save_dir = os.path.join(base_save_dir, "mlp")
    # Logger setup
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    # Model setup
    model_mlp = MLP(n_dim=2, hidden_dim=32, activation=torch.tanh, device=device)

    # Training setup
    n_epochs = args.n_epochs
    optimizer = target.get_optimizer(model_mlp, args.method)

    n_t_train = 2 * c + 1
    n_x_train = 2 * c

    def train_sampler():
        t_nodes = target.sample_domain_1d(
            n_samples=n_t_train,
            dim=0,
            basis="fourier",
            type=args.sample_type,
        )
        x_nodes = target.sample_domain_1d(
            n_samples=n_x_train,
            dim=1,
            basis="fourier",
            type=args.sample_type,
        )
        return [t_nodes, x_nodes]

    print(f"Training MLP...")
    target.train_model(
        model=model_mlp,
        n_epochs=n_epochs,
        optimizer=optimizer,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
        in_domain=target.in_domain,
    )

    #########################################################
    # 2. Polynomial interpolation
    #########################################################
    save_dir = os.path.join(base_save_dir, "chebyshev")
    # Logger setup
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    # Model setup
    n_t = args.n_t if args.n_t is not None else c + 1
    n_x = args.n_x if args.n_x is not None else c
    bases = ["chebyshev", "fourier"]
    domains = target.domain
    model = SpectralInterpolationND(
        Ns=[n_t, n_x],
        bases=bases,
        domains=domains,
        device=device,
    )

    # Training setup
    n_epochs = args.n_epochs
    # lr = 1e-3
    optimizer = target.get_optimizer(model, args.method)

    n_t_train = 2 * c + 1
    n_x_train = 2 * c

    def train_sampler():
        t_nodes = target.sample_domain_1d(
            n_samples=n_t_train,
            dim=0,
            basis=bases[0],
            type=args.sample_type,
        )
        x_nodes = target.sample_domain_1d(
            n_samples=n_x_train,
            dim=1,
            basis=bases[1],
            type=args.sample_type,
        )
        return [t_nodes, x_nodes]

    print(f"Training Polynomial Interpolant...")
    target.train(
        model=model,
        n_epochs=n_epochs,
        optimizer=optimizer,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
        in_domain=target.in_domain,
    )
