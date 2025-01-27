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
    args.add_argument("--c", type=int, default=80)
    args.add_argument("--n_t", type=int, default=81)
    args.add_argument("--n_x", type=int, default=81)
    args.add_argument("--sample_type", type=str, default="uniform")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=10000)
    args.add_argument("--eval_every", type=int, default=100)
    args.add_argument("--model", type=str, default="mlp")
    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    c = args.c
    target = AdvectionTarget(c=c, device=device)
    base_save_dir = (
        f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/advection/c={c}"
    )

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
    # lr = 1e-3
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
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    target.train_model(
        model=model,
        n_epochs=n_epochs,
        optimizer=optimizer,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
    )
