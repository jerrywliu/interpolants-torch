import torch
import matplotlib.pyplot as plt
import os
from typing import Callable
from experiments.interpolation.simple_fcns.base_analytical_target import (
    BaseAnalyticalTarget,
)


class Sine1DTarget(BaseAnalyticalTarget):
    def __init__(self):
        super().__init__(
            "sine_1d",
            f=lambda x: torch.sin(x),
            domain=[(-2 * torch.pi, 2 * torch.pi)],
            derivative=lambda x: torch.cos(x),
            second_derivative=lambda x: -torch.sin(x),
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


# Compare interpolation of sin(x) using different methods:
# 1. Neural network
# 2. Polynomial interpolation (Chebyshev)
# 3. Polynomial interpolation (Fourier)

if __name__ == "__main__":
    target = Sine1DTarget()
