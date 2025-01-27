import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List

from src.experiments.interpolation.base_analytical_target import (
    BaseAnalyticalTarget,
)
from src.models.interpolant_nd import SpectralInterpolationND
from src.models.rational_2d import RationalInterpolation2D


class ReactionTarget(BaseAnalyticalTarget):
    def __init__(self, rho: float = 1, device: str = "cpu"):
        self.device = torch.device(device)
        self.u_0 = lambda x: torch.exp(
            -((x - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2)
        )
        super().__init__(
            "reaction",
            f=lambda t, x: self.u_0(x)
            * torch.exp(self.rho * t)
            / (self.u_0(x) * torch.exp(self.rho * t) + (1 - self.u_0(x))),
            domain=[(0, 1), (0, 2 * torch.pi)],
        )
        self.rho = rho

    def get_solution(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        t_mesh, x_mesh = torch.meshgrid(nodes[0], nodes[1], indexing="ij")
        return self.f(t_mesh, x_mesh)

    def plot_solution(
        self,
        u: torch.Tensor,  # (N_t, N_x)
        t_grid: torch.Tensor,
        x_grid: torch.Tensor,
        save_path: str = None,
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        u_cpu = u.detach().cpu()

        # Predicted solution
        im1 = ax1.imshow(
            u_cpu.T,
            extent=[0, 1, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("Predicted Solution")

        # True solution
        u_true = self.get_solution([t_grid, x_grid]).cpu()
        im2 = ax2.imshow(
            u_true.T,
            extent=[0, 1, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im2, ax=ax2)
        ax2.set_title("True Solution")

        # Error on log scale
        error = torch.abs(u_cpu - u_true)
        l2_rel_error = torch.norm(u_cpu - u_true, p=2) / torch.norm(u_true, p=2)
        im3 = ax3.imshow(
            error.T,
            extent=[0, 1, 0, 2 * torch.pi],
            origin="lower",
            aspect="auto",
            norm="log",
        )
        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f"Error. Max: {error.max():.3e}, L2 rel: {l2_rel_error:.3e}")

        plt.tight_layout()

        if save_path is not None:
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
        eval_nodes: List[torch.Tensor],
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
            t_train = self.sample_domain_1d(
                n_samples=n_samples,
                dim=0,
                basis=basis_type,
                type=sample_type,
            )
            x_train = self.sample_domain_1d(
                n_samples=n_samples,
                dim=1,
                basis=basis_type,
                type=sample_type,
            )
            f_train_pred = model([t_train, x_train])
            f_train_true = self.get_solution([t_train, x_train])

            # Train step
            optimizer.zero_grad()
            loss = loss_fn(f_train_pred, f_train_true)
            loss.backward()
            optimizer.step()

            # Evaluate solution
            f_eval_pred = model(eval_nodes).detach()
            f_eval_true = self.get_solution(eval_nodes)
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
                self.plot_solution(
                    f_eval_pred,
                    eval_nodes[0],
                    eval_nodes[1],
                    save_path=os.path.join(
                        save_dir,
                        f"reaction_soln_rho={self.rho}_nt={n_samples}_nx={n_samples}.png",
                    ),
                )

        # Plot loss history
        plt.figure()
        plt.semilogy(history["loss"], label="Loss")
        plt.semilogy(history["eval_l2_error"], label="Eval L2 Error")
        plt.semilogy(history["eval_max_error"], label="Eval Max Error")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_history.png"))
        plt.close()


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # rhos = [1, 2, 5]
    rhos = [5]

    for rho in rhos:

        # Problem setup
        target = ReactionTarget(rho=rho, device=device)

        # # 1. Sanity check: ensure that the interpolation model is expressive enough to fit the target
        # save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/reaction/sanity_check"
        # for n_t in [21, 41, 81, 161, 321, 641, 1281, 2001]:
        #     # n_x = n_t - 1
        #     n_x = n_t
        #     # bases = ["chebyshev", "fourier"]
        #     bases = ["chebyshev", "chebyshev"]
        #     model = SpectralInterpolationND(
        #         Ns=[n_t, n_x],
        #         bases=bases,
        #         domains=target.domain,
        #     )
        #     # Set values
        #     t_mesh, x_mesh = torch.meshgrid(
        #         model.nodes[0], model.nodes[1], indexing="ij"
        #     )
        #     data = target.get_exact_solution(t_mesh, x_mesh)
        #     model.values.data = data

        #     # Evaluation setup
        #     n_eval = 200
        #     t_eval = torch.linspace(0, 1, n_eval)
        #     x_eval = torch.linspace(0, 2 * torch.pi, n_eval + 1)[:-1]

        #     # Plot solution
        #     target.plot_solution(
        #         model.interpolate([t_eval, x_eval]).detach(),
        #         t_eval,
        #         x_eval,
        #         save_path=os.path.join(
        #             save_dir, f"reaction_soln_rho={rho}_nt={n_t}_nx={n_x}.png"
        #         ),
        #     )

        # # 2. Train polynomial model
        # save_dir = (
        #     "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/reaction/train"
        # )
        # n_t = 41
        # n_x = 41
        # bases = ["chebyshev", "chebyshev"]
        # model = SpectralInterpolationND(
        #     Ns=[n_t, n_x],
        #     bases=bases,
        #     domains=target.domain,
        #     device=device,
        # )
        # lr = 1e-3
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # n_epochs = 1000
        # plot_every = 100
        # n_samples = 100
        # eval_nodes = [
        #     torch.linspace(0, 1, 200).to(device),
        #     torch.linspace(0, 2 * torch.pi, 200).to(device),
        # ]
        # target.train_model(
        #     model=model,
        #     n_epochs=n_epochs,
        #     optimizer=optimizer,
        #     basis_type="chebyshev",
        #     sample_type="uniform",
        #     n_samples=n_samples,
        #     eval_nodes=eval_nodes,
        #     plot_every=plot_every,
        #     save_dir=save_dir,
        # )

        # 3. Train rational model
        # save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/reaction/rational"
        save_dir = "/scratch/interpolants-torch/plots/interpolation/reaction/rational"
        n_t = 41
        n_x = 41
        bases = ["chebyshev", "rational"]
        num_poles = 2
        model = RationalInterpolation2D(
            N_1=n_t,
            N_2=n_x,
            bases_1=bases[0],
            domain_1=target.domain[0],
            domain_2=target.domain[1],
            num_poles=num_poles,
            device=device,
        )
        lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        n_epochs = 100
        plot_every = 10
        n_samples = 100
        eval_nodes = [
            torch.linspace(0, 1, 200).to(device),
            torch.linspace(0, 2 * torch.pi, 200).to(device),
        ]
        target.train_model(
            model=model,
            n_epochs=n_epochs,
            optimizer=optimizer,
            basis_type="chebyshev",
            sample_type="uniform",
            n_samples=n_samples,
            eval_nodes=eval_nodes,
            plot_every=plot_every,
            save_dir=save_dir,
        )
