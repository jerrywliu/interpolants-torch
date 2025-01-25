import argparse
import os
import torch
from typing import List

from src.experiments.interpolation.base_analytical_target import (
    BaseAnalyticalTarget,
)
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.models.rational_1d import RationalInterpolation1D
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger


class Abs1DTarget(BaseAnalyticalTarget):
    def __init__(self, device: str = "cpu"):
        super().__init__(
            "abs_1d",
            f=lambda x: torch.abs(x),
            domain=[(-1, 1)],
            derivative=lambda x: torch.sign(x),
            second_derivative=lambda x: torch.zeros_like(x),
            device=device,
        )

    def plot_solution(
        self,
        nodes: List[torch.Tensor],
        u: torch.Tensor,
        save_path: str = None,
    ):
        self._plot_solution_default(nodes, u, save_path)


# Compare interpolation of abs(x) using different methods:
# 1. Neural network
# 2. Polynomial interpolation
# 3. Barycentric rational interpolation

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--n_epochs", type=int, default=10000)
    args.add_argument("--eval_every", type=int, default=100)
    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    target = Abs1DTarget(device=device)

    base_save_dir = (
        f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/abs_1d"
    )

    # Evaluation setup (shared for all methods)
    eval_every = args.eval_every
    n_eval = 200
    x_eval = torch.linspace(
        target.domain[0][0], target.domain[0][1], n_eval, device=device
    )

    def eval_sampler():
        return [x_eval]

    eval_metrics = [l2_error, max_error, l2_relative_error]

    #########################################################
    # 1. Neural network
    #########################################################
    save_dir = os.path.join(base_save_dir, "mlp")
    n_epochs = args.n_epochs
    lr = 1e-3
    n_samples = 41
    basis_type = "fourier"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    model_mlp = MLP(n_dim=1, hidden_dim=32, activation=torch.tanh, device=device)
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=lr)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

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
    n_epochs = args.n_epochs
    lr = 1e-3
    n_samples = 41
    basis_type = "chebyshev"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    n_x = 41
    bases = ["chebyshev"]
    domains = target.domain
    model_cheb_uniform = SpectralInterpolationND(
        Ns=[n_x],
        bases=bases,
        domains=domains,
        device=device,
    )
    optimizer = torch.optim.Adam(model_cheb_uniform.parameters(), lr=lr)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train_model(
        model=model_cheb_uniform,
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
    # 3. Barycentric rational interpolation
    #########################################################
    save_dir = os.path.join(base_save_dir, "rational")
    n_epochs = args.n_epochs
    lr = 1e-3
    n_samples = 41
    basis_type = "chebyshev"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    n_x = 21
    model_rational = RationalInterpolation1D(
        N=n_x,
        domain=target.domain[0],
        device=device,
    )
    optimizer = torch.optim.Adam(model_rational.parameters(), lr=lr)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train_model(
        model=model_rational,
        n_epochs=n_epochs,
        optimizer=optimizer,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
    )
