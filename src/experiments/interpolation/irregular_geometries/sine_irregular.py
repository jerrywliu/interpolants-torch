import argparse
import os
import torch
import torch.nn as nn
from typing import List

from src.experiments.interpolation.simple_fcns.sine_1d import Sine1DTarget
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.utils.metrics import l2_error, max_error, l2_relative_error
from src.loggers.logger import Logger


# This is the same as the Sine1DTarget, but we cut a hole in the domain
# Domain = [0, 2π] minus [π/2, 3π/2]
class SineIrregularTarget(Sine1DTarget):
    def __init__(self, device: str = "cpu"):
        super().__init__(
            device=device,
        )

    def _in_domain(self, x: torch.Tensor) -> torch.Tensor:
        # Check if outside the interval [π/2, 3π/2]
        outside_interval = (x < torch.pi / 2) | (x > 3 * torch.pi / 2)
        # Check if within the domain [0, 2π]
        in_domain = (x >= 0) & (x <= 2 * torch.pi)
        return outside_interval & in_domain  # Combine conditions

    def in_domain(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        mesh = torch.meshgrid(*nodes, indexing="ij")
        return self._in_domain(*mesh)

    def get_loss(self, model: nn.Module, nodes: List[torch.Tensor]) -> torch.Tensor:
        u_pred = model(nodes)
        u_true = self.get_function(nodes)
        mask = self.in_domain(nodes).float()  # Convert boolean mask to float
        return torch.sum((u_pred - u_true) ** 2 * mask) / torch.sum(mask)


# Compare interpolation of sin(x) using different methods:
# 1. Neural network
# 2a. Polynomial interpolation (Chebyshev, training points are uniformly distributed)
# 2. Polynomial interpolation (Chebyshev, training points are Chebyshev distributed)
# 3. Polynomial interpolation (Fourier)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--sample_type", type=str, default="standard")
    args.add_argument("--method", type=str, default="adam")
    args.add_argument("--n_epochs", type=int, default=20000)
    args.add_argument("--eval_every", type=int, default=100)
    args = args.parse_args()

    torch.random.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    # Problem setup
    target = SineIrregularTarget(device=device)

    base_save_dir = (
        f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/sine_irregular"
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
    # lr = 1e-3
    n_samples = 81
    basis_type = "fourier"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    model_mlp = MLP(n_dim=1, hidden_dim=32, activation=torch.tanh, device=device)
    optimizer = target.get_optimizer(model_mlp, args.method)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train(
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
    # 2a. Polynomial interpolation (Chebyshev, training points are uniformly distributed)
    #########################################################
    save_dir = os.path.join(base_save_dir, "chebyshev_uniform")
    n_epochs = args.n_epochs
    eval_every = 100
    # lr = 1e-3
    n_samples = 81
    basis_type = "fourier"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    n_x = 21
    bases = ["chebyshev"]
    domains = target.domain
    model_cheb_uniform = SpectralInterpolationND(
        Ns=[n_x],
        bases=bases,
        domains=domains,
        device=device,
    )
    optimizer = target.get_optimizer(model_cheb_uniform, args.method)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train(
        model=model_cheb_uniform,
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
    # 2b. Polynomial interpolation (Chebyshev, training points are Chebyshev distributed)
    #########################################################
    save_dir = os.path.join(base_save_dir, "chebyshev_chebyshev")
    n_epochs = args.n_epochs
    eval_every = 100
    # lr = 1e-3
    n_samples = 81
    basis_type = "chebyshev"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    n_x = 21
    bases = ["chebyshev"]
    domains = target.domain
    model_cheb_chebyshev = SpectralInterpolationND(
        Ns=[n_x],
        bases=bases,
        domains=domains,
        device=device,
    )
    optimizer = target.get_optimizer(model_cheb_chebyshev, args.method)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train(
        model=model_cheb_chebyshev,
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
    # 3. Polynomial interpolation (Fourier)
    #########################################################
    save_dir = os.path.join(base_save_dir, "fourier")
    n_epochs = args.n_epochs
    eval_every = 100
    # lr = 1e-3
    n_samples = 81
    basis_type = "fourier"
    sample_type = args.sample_type

    def train_sampler():
        return target.sample_domain(
            n_samples,
            basis=[basis_type],
            type=[sample_type],
        )

    n_x = 20
    bases = ["fourier"]
    domains = target.domain
    model_fourier = SpectralInterpolationND(
        Ns=[n_x],
        bases=bases,
        domains=domains,
        device=device,
    )
    optimizer = target.get_optimizer(model_fourier, args.method)
    logger = Logger(path=os.path.join(save_dir, "logger.json"))

    target.train(
        model=model_fourier,
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