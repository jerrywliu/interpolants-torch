import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Callable


class BasePDE:
    def __init__(self, name: str, domain: List[Tuple[float, float]]):
        self.name = name
        self.domain = domain
        self.n_dims = len(domain)
        self.domain_lengths = [domain[i][1] - domain[i][0] for i in range(self.n_dims)]
        # Set up domain mapping functions for each dimension, for both chebyshev and fourier
        # Default domains are [-1, 1] for chebyshev and [0, 2*pi] for fourier
        self._to_cheb = (
            lambda x, d: 2 * (x - self.domain[d][0]) / (self.domain_lengths[d]) - 1
        )
        self._from_cheb = (
            lambda x, d: self.domain[d][0] + (x + 1) * (self.domain_lengths[d]) / 2
        )
        self._to_fourier = (
            lambda x, d: 2
            * torch.pi
            * (x - self.domain[d][0])
            / (self.domain_lengths[d])
        )
        self._from_fourier = (
            lambda x, d: self.domain[d][0]
            + (x / (2 * torch.pi)) * self.domain_lengths[d]
        )

    # Input: x = [(n_1,), ..., (n_d,)]
    # Output: u = (n_1 * ... * n_d,)
    def get_solution(self, x: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def get_pde_loss(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],
        ic_weight: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_domain(self, dim: int) -> Tuple[float, float]:
        return self.domain[dim]

    def get_domain_lengths(self) -> List[float]:
        return self.domain_lengths

    def get_n_dims(self) -> int:
        return self.n_dims

    def get_name(self) -> str:
        return self.name

    # Sample points from one dimension of the domain:
    # basis: "chebyshev" or "fourier"
    # type: "standard" or "uniform"
    def sample_domain_1d(
        self, n_samples: int, dim: int, basis: str, type: str
    ) -> torch.Tensor:
        if type == "standard":
            if basis == "chebyshev":
                cheb_nodes = torch.cos(torch.linspace(0, torch.pi, n_samples, requires_grad=True))
                return self._from_cheb(cheb_nodes, dim)
            elif basis == "fourier":
                fourier_nodes = torch.linspace(0, 2 * torch.pi, n_samples + 1, requires_grad=True)[:-1]
                return self._from_fourier(fourier_nodes, dim)
            else:
                raise ValueError(f"Invalid basis: {basis}")
        elif type == "uniform":
            if basis == "chebyshev":
                uniform_nodes = torch.cos(torch.rand(n_samples, requires_grad=True) * torch.pi)
                return self._from_cheb(uniform_nodes, dim)
            elif basis == "fourier":
                uniform_nodes = torch.rand(n_samples, requires_grad=True) * 2 * torch.pi
                return self._from_fourier(uniform_nodes, dim)
            else:
                raise ValueError(f"Invalid basis: {basis}")
        else:
            raise ValueError(f"Invalid type: {type}")

    def sample_domain(
        self, n_samples: int, basis: List[str], type: List[str]
    ) -> List[torch.Tensor]:
        return [
            self.sample_domain_1d(n_samples, dim, basis[dim], type[dim])
            for dim in range(self.n_dims)
        ]

    def plot_solution(
        self, nodes: List[torch.Tensor], u: torch.Tensor, save_path: str = None
    ):
        raise NotImplementedError

    def train_model(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        plot_every: int = 100,
        save_dir: str = None,
    ):
        # Training history
        history = {
            "loss": [],
            "train_pde_loss": [],
            "train_ic_loss": [],
            "eval_pde_loss": [],
        }
        for eval_metric in eval_metrics:
            history[f"eval_{eval_metric.__name__}"] = []

        print("Training model...")
        start_time = time()
        for epoch in tqdm(range(n_epochs)):
            # Sample points
            pde_nodes = pde_sampler()
            ic_nodes = ic_sampler()
            # Train step
            optimizer.zero_grad()
            # Get PDE loss
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model, pde_nodes, ic_nodes, ic_weight
            )
            loss.backward()
            optimizer.step()

            # Evaluate solution
            eval_nodes = eval_sampler()
            u_eval = model.interpolate(eval_nodes)
            u_true = self.get_solution(eval_nodes)
            for eval_metric in eval_metrics:
                eval_metric_value = eval_metric(u_eval, u_true)
                history[f"eval_{eval_metric.__name__}"].append(eval_metric_value)
            # Evaluate PDE loss
            _, eval_pde_loss, _ = self.get_pde_loss(
                model, eval_nodes, ic_nodes, ic_weight
            )

            # Update history
            history["loss"].append(loss.item())
            history["train_pde_loss"].append(pde_loss.item())
            history["train_ic_loss"].append(ic_loss.item())
            history["eval_pde_loss"].append(eval_pde_loss.item())

            # Print and plot progress
            if (epoch + 1) % plot_every == 0:
                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"PDE loss: {history['train_pde_loss'][-1]:1.3e}")
                print(f"IC loss: {history['train_ic_loss'][-1]:1.3e}")
                print(f"Evaluation L2 error: {history['eval_l2_error'][-1]:1.3e}")
                self.plot_solution(
                    eval_nodes,
                    u_eval.detach(),
                    save_path=os.path.join(
                        save_dir, f"{self.name}_solution_{epoch}.png"
                    ),
                )

                # Save history
                torch.save(history, os.path.join(save_dir, "history.pth"))

        return history

    def train_model_lbfgs(
        self,
        model: nn.Module,
        max_iter: int,
        optimizer: torch.optim.LBFGS,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        plot_every: int = 10,
        save_dir: str = None,
    ):
        # Training history
        history = {
            "loss": [],
            "train_pde_loss": [],
            "train_ic_loss": [],
            "eval_pde_loss": [],
        }
        for eval_metric in eval_metrics:
            history[f"eval_{eval_metric.__name__}"] = []

        # Sample points once since L-BFGS works better with fixed points
        pde_nodes = pde_sampler()
        ic_nodes = ic_sampler()
        eval_nodes = eval_sampler()

        print("Training model with L-BFGS...")
        start_time = time()

        # Define closure for L-BFGS
        def closure():
            optimizer.zero_grad()
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model, pde_nodes, ic_nodes, ic_weight
            )
            loss.backward()
            return loss

        # Training loop
        for i in tqdm(range(max_iter)):
            # Optimize
            loss = optimizer.step(closure)

            # Evaluate solution
            with torch.no_grad():
                u_eval = model.interpolate(eval_nodes)
                u_true = self.get_solution(eval_nodes)
                for eval_metric in eval_metrics:
                    eval_metric_value = eval_metric(u_eval, u_true)
                    history[f"eval_{eval_metric.__name__}"].append(eval_metric_value)
                # Get losses for history
                total_loss, pde_loss, ic_loss = self.get_pde_loss(
                    model, pde_nodes, ic_nodes, ic_weight
                )
                _, eval_pde_loss, _ = self.get_pde_loss(
                    model, eval_nodes, ic_nodes, ic_weight
                )

            # Update history
            history["loss"].append(total_loss.item())
            history["train_pde_loss"].append(pde_loss.item())
            history["train_ic_loss"].append(ic_loss.item())
            history["eval_pde_loss"].append(eval_pde_loss.item())

            # Print and plot progress
            if (i + 1) % plot_every == 0:
                current_time = time() - start_time
                print(f"Iteration {i + 1} completed in {current_time:.2f} seconds")
                print(f"PDE loss: {history['train_pde_loss'][-1]:1.3e}")
                print(f"IC loss: {history['train_ic_loss'][-1]:1.3e}")
                print(f"Evaluation L2 error: {history['eval_l2_error'][-1]:1.3e}")
                self.plot_solution(
                    eval_nodes,
                    u_eval,
                    save_path=os.path.join(save_dir, f"{self.name}_solution_{i}.png"),
                )

                # Save history
                torch.save(history, os.path.join(save_dir, "history.pth"))

        return history
