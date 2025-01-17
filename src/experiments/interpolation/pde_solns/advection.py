import os
import matplotlib.pyplot as plt
from time import time

import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Callable

from src.experiments.interpolation.simple_fcns.base_analytical_target import (
    BaseAnalyticalTarget,
)
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.models.rational_1d import RationalInterpolation1D


def make_grid(t_pts, x_pts):
    num_t_pts = t_pts.shape[0]
    num_x_pts = x_pts.shape[0]

    x_grid = torch.zeros(num_t_pts, num_x_pts)
    x_grid[:] = x_pts.view(1, -1)

    t_grid = torch.zeros(num_t_pts, num_x_pts)
    t_grid[:] = t_pts.view(-1, 1)

    grid = torch.stack([t_grid, x_grid], dim=-1)
    return grid.reshape(-1, 2)


class AdvectionTarget(BaseAnalyticalTarget):
    def __init__(self):
        T = 2.0
        L = 2 * np.pi
        c = 50

        def advection_forward(inputs):
            t = inputs[:, 0]
            x = inputs[:, 1]
            return torch.sin(torch.remainder(x - c * t, L))

        super().__init__(
            "adv",
            f=advection_forward,
            domain=[(0.0, T), (0.0, L)],
            derivative=lambda x: 0.0,  # Dummy value.
            second_derivative=lambda x: 0.0,  # Dummy value.
        )

    def train_model(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        t_basis_type: str,  # "chebyshev" or "fourier"
        x_basis_type: str,  # "chebyshev" or "fourier"
        t_sample_type: str,  # standard or uniform
        x_sample_type: str,  # standard or uniform
        t_n_samples: int,
        x_n_samples: int,
        eval_t_pts: torch.Tensor,
        eval_x_pts: torch.Tensor,
        plot_every: int = 100,
        save_dir: str = None,
    ):
        # Training history
        history = {
            "loss": [],
            "eval_l2_error": [],
            "eval_max_error": [],
        }
        loss_fn = nn.MSELoss()

        eval_grid = make_grid(eval_t_pts, eval_x_pts)

        print("Training model...")
        start_time = time()
        for epoch in tqdm(range(n_epochs)):
            # Sample points
            sampled_pts_per_dim = self.sample_domain(
                n_samples=[t_n_samples, x_n_samples],
                # dim=0,
                basis=[t_basis_type, x_basis_type],
                type=[t_sample_type, x_sample_type],
            )

            train_grid = make_grid(sampled_pts_per_dim[0], sampled_pts_per_dim[1])

            # f_train_pred = model([train_grid])
            f_train_pred = model(sampled_pts_per_dim)
            f_train_true = self.get_function(train_grid).reshape(
                t_n_samples, x_n_samples
            )

            # Train step
            optimizer.zero_grad()
            loss = loss_fn(f_train_pred, f_train_true)
            loss.backward()
            optimizer.step()

            # Evaluate solution
            f_eval_pred = model([eval_t_pts, eval_x_pts]).detach()
            f_eval_true = self.get_function(eval_grid).reshape(
                eval_t_pts.shape[0], eval_x_pts.shape[0]
            )

            eval_l2_error = torch.mean((f_eval_pred - f_eval_true) ** 2)
            eval_max_error = torch.max(torch.abs(f_eval_pred - f_eval_true))

            # Update history
            history["loss"].append(loss.item())
            history["eval_l2_error"].append(eval_l2_error.item())
            history["eval_max_error"].append(eval_max_error.item())

            # Print and plot progress
            if (epoch + 1) % plot_every == 0:
                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"Evaluation L2 error: {history['eval_l2_error'][-1]:1.3e}")
                # self.plot_comparison(
                #     x_train,
                #     x_eval,
                #     f_train_true,
                #     f_eval_pred,
                #     save_path=os.path.join(save_dir, f"abs_1d_solution_{epoch}.png"),
                # )

        # Plot loss history
        plt.figure()
        plt.semilogy(history["loss"], label="Loss")
        plt.semilogy(history["eval_l2_error"], label="Eval L2 Error")
        plt.semilogy(history["eval_max_error"], label="Eval Max Error")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_history.png"))
        plt.close()


# Compare interpolation of abs(x) using different methods:
# 1. Neural network
# 2. Polynomial interpolation
# 3. Barycentric rational interpolation

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Problem setup
    target = AdvectionTarget()
    t_n_samples = 200
    x_n_samples = 50

    t_n_samples_eval = 100
    x_n_samples_eval = 100

    t_eval = torch.linspace(target.domain[0][0], target.domain[0][1], t_n_samples_eval)
    x_eval = torch.linspace(target.domain[1][0], target.domain[1][1], x_n_samples_eval)
    # eval_grid = make_grid(t_eval, x_eval)

    # 1. Neural network
    save_dir = "/common/results/interpolation/advection/mlp"
    model_mlp = MLP(n_dim=2, hidden_dim=32, activation=torch.tanh)
    lr = 1e-3
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=lr)
    n_epochs = 10000
    plot_every = 100
    basis_type_t = "fourier"
    basis_type_x = "fourier"
    sample_type_t = "uniform"
    sample_type_x = "uniform"
    target.train_model(
        model=model_mlp,
        n_epochs=n_epochs,
        optimizer=optimizer,
        t_basis_type=basis_type_t,
        x_basis_type=basis_type_x,
        t_sample_type=sample_type_t,
        x_sample_type=sample_type_x,
        t_n_samples=t_n_samples,
        x_n_samples=x_n_samples,
        eval_t_pts=t_eval,
        eval_x_pts=x_eval,
        plot_every=plot_every,
        save_dir=save_dir,
    )

    # # 2. Polynomial interpolation
    # save_dir = (
    #     "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/abs_1d/chebyshev"
    # )
    # n_x = 41
    # bases = ["chebyshev"]
    # domains = target.domain
    # model_cheb_uniform = SpectralInterpolationND(
    #     Ns=[n_x],
    #     bases=bases,
    #     domains=domains,
    # )
    # lr = 1e-3
    # optimizer = torch.optim.Adam(model_cheb_uniform.parameters(), lr=lr)
    # n_epochs = 10000
    # plot_every = 100
    # basis_type = "chebyshev"
    # sample_type = "uniform"
    # target.train_model(
    #     model=model_cheb_uniform,
    #     n_epochs=n_epochs,
    #     optimizer=optimizer,
    #     basis_type=basis_type,
    #     sample_type=sample_type,
    #     n_samples=n_samples,
    #     x_eval=x_eval,
    #     plot_every=plot_every,
    #     save_dir=save_dir,
    # )

    # # 3. Barycentric rational interpolation
    # save_dir = (
    #     "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/abs_1d/rational"
    # )
    # n_x = 21
    # model_rational = RationalInterpolation1D(N=n_x, domain=target.domain[0])
    # lr = 1e-3
    # optimizer = torch.optim.Adam(model_rational.parameters(), lr=lr)
    # n_epochs = 20000
    # plot_every = 100
    # basis_type = "chebyshev"
    # sample_type = "standard"
    # target.train_model(
    #     model=model_rational,
    #     n_epochs=n_epochs,
    #     optimizer=optimizer,
    #     basis_type=basis_type,
    #     sample_type=sample_type,
    #     n_samples=n_samples,
    #     x_eval=x_eval,
    #     plot_every=plot_every,
    #     save_dir=save_dir,
    # )
