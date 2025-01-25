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


class Sine1DTarget(BaseAnalyticalTarget):
    def __init__(self, device: str = "cpu"):
        super().__init__(
            "sine_1d",
            f=lambda x: torch.sin(x),
            domain=[(0, 2 * torch.pi)],
            derivative=lambda x: torch.cos(x),
            second_derivative=lambda x: -torch.sin(x),
            device=device,
        )

    def plot_solution(
        self,
        nodes: List[torch.Tensor],
        u: torch.Tensor,
        save_path: str = None,
    ):
        self._plot_solution_default(nodes, u, save_path)


# Compare interpolation of sin(x) using different methods:
# 1. Neural network
# 2a. Polynomial interpolation (Chebyshev, training points are uniformly distributed)
# 2. Polynomial interpolation (Chebyshev, training points are Chebyshev distributed)
# 3. Polynomial interpolation (Fourier)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--n_epochs", type=int, default=10000)
    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    target = Sine1DTarget(device=device)
    n_samples = 21
    n_eval = 200
    x_eval = torch.linspace(
        target.domain[0][0], target.domain[0][1], n_eval, device=device
    )

    def eval_sampler():
        return [x_eval]

    # 1. Neural network
    save_dir = f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/sine_1d/mlp_sample={args.sample_type}"
    n_epochs = args.n_epochs
    eval_every = 100
    lr = 1e-3
    basis_type = "fourier"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    eval_metrics = [l2_error, max_error, l2_relative_error]

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

    # 2a. Polynomial interpolation (Chebyshev, training points are uniformly distributed)
    save_dir = f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/sine_1d/chebyshev_uniform_sample={args.sample_type}"
    n_epochs = args.n_epochs
    eval_every = 100
    lr = 1e-3
    basis_type = "fourier"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    eval_metrics = [l2_error, max_error, l2_relative_error]

    n_x = 21
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

    # 2b. Polynomial interpolation (Chebyshev, training points are Chebyshev distributed)
    save_dir = f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/sine_1d/chebyshev_chebyshev_sample={args.sample_type}"
    n_epochs = args.n_epochs
    eval_every = 100
    lr = 1e-3
    basis_type = "chebyshev"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    eval_metrics = [l2_error, max_error, l2_relative_error]

    n_x = 21
    bases = ["chebyshev"]
    domains = target.domain
    model_cheb_chebyshev = SpectralInterpolationND(
        Ns=[n_x],
        bases=bases,
        domains=domains,
        device=device,
    )
    optimizer = torch.optim.Adam(model_cheb_chebyshev.parameters(), lr=lr)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train_model(
        model=model_cheb_chebyshev,
        n_epochs=n_epochs,
        optimizer=optimizer,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
    )

    # 3. Polynomial interpolation (Fourier)
    save_dir = f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/sine_1d/fourier_sample={args.sample_type}"
    n_epochs = args.n_epochs
    eval_every = 100
    lr = 1e-3
    basis_type = "fourier"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    eval_metrics = [l2_error, max_error, l2_relative_error]

    n_x = 20
    bases = ["fourier"]
    domains = target.domain
    model_fourier = SpectralInterpolationND(
        Ns=[n_x],
        bases=bases,
        domains=domains,
        device=device,
    )
    optimizer = torch.optim.Adam(model_fourier.parameters(), lr=lr)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train_model(
        model=model_fourier,
        n_epochs=n_epochs,
        optimizer=optimizer,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        eval_metrics=eval_metrics,
        eval_every=eval_every,
        save_dir=save_dir,
        logger=logger,
    )
