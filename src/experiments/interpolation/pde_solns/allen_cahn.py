import os
import matplotlib.pyplot as plt
from time import time

import numpy as np
import scipy.io

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


class AllenCahnTarget(BaseAnalyticalTarget):
    def __init__(self):
        t_range = (0.0, 1.0)
        x_range = (-1.0, 1.0)

        self.true_fn_vals, self.t_vals, self.x_vals, self.full_grid = (
            self.load_allen_cahn_dataset()
        )

        super().__init__(
            "allen-cahn",
            f=lambda x: 0.0,  # Dummy function.
            domain=[t_range, x_range],
            derivative=lambda x: 0.0,  # Dummy function.
            second_derivative=lambda x: 0.0,  # Dummy function.
        )

    def load_allen_cahn_dataset(self):
        data = scipy.io.loadmat(
            "src/experiments/interpolation/pde_solns/allen-cahn-data/allen_cahn.mat"
        )
        u_ref = torch.as_tensor(data["usol"])
        t_star = torch.as_tensor(data["t"].flatten())
        x_star = torch.as_tensor(data["x"].flatten())
        full_grid = make_grid(t_star, x_star)

        return u_ref, t_star, x_star, full_grid

    def plot_comparison(
        self,
        f_eval_pred,
        f_eval_true,
        save_path,
    ):
        f_eval_pred = f_eval_pred.detach().cpu().numpy()
        f_eval_true = f_eval_true.detach().cpu().numpy()

        # Create a figure with 1x2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))

        ax1.imshow(f_eval_pred.T, cmap="viridis")
        ax1.set_title("Predicted")

        ax2.imshow(f_eval_true.T, cmap="viridis")
        ax2.set_title("True")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()

    def train_model(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        t_basis_type: str,  # "chebyshev" or "fourier"
        x_basis_type: str,  # "chebyshev" or "fourier"
        t_sample_type: str,  # standard or uniform
        x_sample_type: str,  # standard or uniform
        plot_every: int = 100,
        save_dir: str = None,
    ):
        assert t_basis_type == "fourier" and x_basis_type == "chebyshev", (
            "Allen-cahn data was generated with fourier basis for t and "
            + "chebyshev for x."
        )

        assert t_sample_type == "standard" and x_sample_type == "standard", (
            "Allen-cahn data was generated with standard sample type."
        )

        # Train / eval split.

        train_t_vals = self.t_vals[0::2]
        eval_t_vals = self.t_vals[1::2]

        train_x_vals = self.x_vals[0::2]
        eval_x_vals = self.x_vals[1::2]

        # train_grid = self.full_grid[0::2, 0::2]
        # eval_grid = self.full_grid[1::2, 1::2]

        f_train_true = torch.as_tensor(self.true_fn_vals[0::2, 0::2])
        f_eval_true = torch.as_tensor(self.true_fn_vals[1::2, 1::2])

        # Training history
        history = {
            "loss": [],
            "eval_l2_error": [],
            "eval_max_error": [],
        }
        loss_fn = nn.MSELoss()

        print("Training model...")
        start_time = time()
        for epoch in tqdm(range(n_epochs)):
            f_train_pred = model([train_t_vals, train_x_vals])

            # Train step
            optimizer.zero_grad()
            loss = loss_fn(f_train_pred, f_train_true)
            loss.backward()
            optimizer.step()

            # Evaluate solution
            f_eval_pred = model([eval_t_vals, eval_x_vals]).detach()
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
                self.plot_comparison(
                    f_eval_pred,
                    f_eval_true,
                    save_path=os.path.join(save_dir, f"allen-cahn_{epoch}.png"),
                )

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
    target = AllenCahnTarget()
    base_save_dir = "/common/results/interpolation/allen-cahn"

    # 1. Neural network
    save_dir = os.path.join(base_save_dir, "mlp")
    model_mlp = MLP(n_dim=2, hidden_dim=32, activation=torch.tanh)
    lr = 1e-3
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=lr)
    n_epochs = 10000
    plot_every = 100
    basis_type_t = "fourier"
    basis_type_x = "chebyshev"
    sample_type_t = "standard"
    sample_type_x = "standard"

    target.train_model(
        model=model_mlp,
        n_epochs=n_epochs,
        optimizer=optimizer,
        t_basis_type=basis_type_t,
        x_basis_type=basis_type_x,
        t_sample_type=sample_type_t,
        x_sample_type=sample_type_x,
        plot_every=plot_every,
        save_dir=save_dir,
    )

    # # 2. Polynomial interpolation
    # save_dir = os.path.join(base_save_dir, "chebyshev")
    # n_t = 41
    # n_x = 41
    # bases = ["fourier", "chebyshev"]
    # domains = target.domain
    # model_cheb_uniform = SpectralInterpolationND(
    #     Ns=[n_t, n_x],
    #     bases=bases,
    #     domains=domains,
    # )
    # lr = 1e-3
    # optimizer = torch.optim.Adam(model_cheb_uniform.parameters(), lr=lr)
    # n_epochs = 10000
    # plot_every = 100
    # basis_type_t = "fourier"
    # basis_type_x = "chebyshev"
    # sample_type_t = "standard"
    # sample_type_x = "standard"
    # target.train_model(
    #     model=model_cheb_uniform,
    #     n_epochs=n_epochs,
    #     optimizer=optimizer,
    #     t_basis_type=basis_type_t,
    #     x_basis_type=basis_type_x,
    #     t_sample_type=sample_type_t,
    #     x_sample_type=sample_type_x,
    #     t_n_samples=t_n_samples,
    #     x_n_samples=x_n_samples,
    #     eval_t_pts=t_eval,
    #     eval_x_pts=x_eval,
    #     plot_every=plot_every,
    #     save_dir=save_dir,
    # )

    # TODO: Looks like rational interpolation model still is only 1D?
    # 3. Barycentric rational interpolation
    # save_dir = os.path.join(base_save_dir, "rational")
    # n_t = 21
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
    #     t_basis_type=basis_type_t,
    #     x_basis_type=basis_type_x,
    #     t_sample_type=sample_type_t,
    #     x_sample_type=sample_type_x,
    #     t_n_samples=t_n_samples,
    #     x_n_samples=x_n_samples,
    #     eval_t_pts=t_eval,
    #     eval_x_pts=x_eval,
    #     plot_every=plot_every,
    #     save_dir=save_dir,
    # )
