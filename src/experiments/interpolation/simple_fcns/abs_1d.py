import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Callable

from src.experiments.interpolation.simple_fcns.base_analytical_target import (
    BaseAnalyticalTarget,
)
from models.interpolant_nd import SpectralInterpolationND
from src.models.mlp import MLP
from src.models.rational_1d import RationalInterpolation1D


class Abs1DTarget(BaseAnalyticalTarget):
    def __init__(self):
        super().__init__(
            "abs_1d",
            f=lambda x: torch.abs(x),
            domain=[(-1, 1)],
            derivative=lambda x: torch.sign(x),
            second_derivative=lambda x: torch.zeros_like(x),
        )

    def plot_comparison(
        self,
        train_points: torch.Tensor,  # (N_train, 1)
        eval_points: torch.Tensor,  # (N_eval, 1)
        train_values: torch.Tensor,  # (N_train, 1)
        eval_values: torch.Tensor,  # (N_eval, 1)
        save_path: str = None,
        f: Callable = None,
    ):
        # Create a figure with 1x2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Get true values for comparison
        if f is None:
            true_values = self.f(eval_points)
        else:
            true_values = f(eval_points)

        # Left subplot: True vs Predicted
        ax1.plot(eval_points, true_values, "k:", label="True function")
        ax1.plot(eval_points, eval_values, "b-", label="Predicted")
        ax1.plot(train_points, train_values, "ro", label="Training points")
        ax1.set_title("True vs Predicted")
        ax1.legend()
        ax1.grid(True)

        # Right subplot: Error (log scale)
        errors = torch.abs(eval_values - true_values)
        ax2.semilogy(eval_points, errors, "b-", label="Absolute Error")
        ax2.set_title("Absolute Error (log scale)")
        ax2.legend()
        ax2.grid(True)

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
        basis_type: str,  # "chebyshev" or "fourier"
        sample_type: str,  # standard or uniform
        n_samples: int,
        x_eval: torch.Tensor,
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

        print("Training model...")
        start_time = time()
        for epoch in tqdm(range(n_epochs)):
            # Sample points
            x_train = self.sample_domain_1d(
                n_samples=n_samples,
                dim=0,
                basis=basis_type,
                type=sample_type,
            )
            f_train_pred = model([x_train])
            f_train_true = self.get_function(x_train)

            # Train step
            optimizer.zero_grad()
            loss = loss_fn(f_train_pred, f_train_true)
            loss.backward()
            optimizer.step()

            # Evaluate solution
            f_eval_pred = model([x_eval]).detach()
            f_eval_true = self.get_function(x_eval)
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
                    x_train,
                    x_eval,
                    f_train_true,
                    f_eval_pred,
                    save_path=os.path.join(save_dir, f"abs_1d_solution_{epoch}.png"),
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
    target = Abs1DTarget()
    n_samples = 21
    n_eval = 200
    x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)

    # 1. Neural network
    save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/abs_1d/mlp"
    model_mlp = MLP(n_dim=1, hidden_dim=32, activation=torch.tanh)
    lr = 1e-3
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=lr)
    n_epochs = 10000
    plot_every = 100
    basis_type = "fourier"
    sample_type = "uniform"
    target.train_model(
        model=model_mlp,
        n_epochs=n_epochs,
        optimizer=optimizer,
        basis_type=basis_type,
        sample_type=sample_type,
        n_samples=n_samples,
        x_eval=x_eval,
        plot_every=plot_every,
        save_dir=save_dir,
    )

    # 2. Polynomial interpolation
    save_dir = (
        "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/abs_1d/chebyshev"
    )
    n_x = 21
    bases = ["chebyshev"]
    domains = target.domain
    model_cheb_uniform = SpectralInterpolationND(
        Ns=[n_x],
        bases=bases,
        domains=domains,
    )
    lr = 1e-3
    optimizer = torch.optim.Adam(model_cheb_uniform.parameters(), lr=lr)
    n_epochs = 10000
    plot_every = 100
    basis_type = "chebyshev"
    sample_type = "uniform"
    target.train_model(
        model=model_cheb_uniform,
        n_epochs=n_epochs,
        optimizer=optimizer,
        basis_type=basis_type,
        sample_type=sample_type,
        n_samples=n_samples,
        x_eval=x_eval,
        plot_every=plot_every,
        save_dir=save_dir,
    )

    # 3. Barycentric rational interpolation
    save_dir = (
        "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/abs_1d/rational"
    )
    n_x = 21
    bases = ["chebyshev"]
    domains = target.domain
    model_rational = RationalInterpolation1D(N=n_x, domain=target.domain)
    lr = 1e-3
    optimizer = torch.optim.Adam(model_rational.parameters(), lr=lr)
    n_epochs = 10000
    plot_every = 100
    basis_type = "chebyshev"
    sample_type = "uniform"
    target.train_model(
        model=model_rational,
        n_epochs=n_epochs,
        optimizer=optimizer,
        basis_type=basis_type,
        sample_type=sample_type,
        n_samples=n_samples,
        x_eval=x_eval,
        plot_every=plot_every,
        save_dir=save_dir,
    )
