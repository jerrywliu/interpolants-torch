import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Callable

from src.experiments.base_fcn import BaseFcn
from src.loggers.logger import Logger


class BasePDE(BaseFcn):
    def __init__(
        self,
        name: str,
        domain: List[Tuple[float, float]],
        device: str = "cpu",
    ):
        super().__init__(name, domain, device)

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
        eval_every: int = 1000,
        save_dir: str = None,
        logger: Logger = None,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

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

            # Backprop
            loss.backward()

            # Update parameters
            optimizer.step()

            # Log
            logger.log("loss", loss.item(), epoch)
            logger.log("train_pde_loss", pde_loss.item(), epoch)
            logger.log("train_ic_loss", ic_loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Evaluate solution
                with torch.no_grad():
                    eval_nodes = eval_sampler()
                    u_eval = model(eval_nodes)
                    u_true = self.get_solution(eval_nodes)
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    # Evaluate PDE loss
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model, eval_nodes, ic_nodes, ic_weight
                    )
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                self.plot_solution(
                    eval_nodes,
                    u_eval,
                    save_path=os.path.join(
                        save_dir, f"{self.name}_solution_{epoch}.png"
                    ),
                )

                # Save history
                logger.save()

    def train_model_lbfgs(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.LBFGS,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

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
        for epoch in tqdm(range(n_epochs)):

            # Optimize
            loss = optimizer.step(closure)

            # Log
            logger.log("loss", loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model.interpolate(eval_nodes)
                    u_true = self.get_solution(eval_nodes)
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    # Evaluate PDE losses
                    total_loss, pde_loss, ic_loss = self.get_pde_loss(
                        model, pde_nodes, ic_nodes, ic_weight
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model, eval_nodes, ic_nodes, ic_weight
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Iteration {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 error: {logger.get_most_recent_value('eval_l2_error'):1.3e}"
                )
                self.plot_solution(
                    eval_nodes,
                    u_eval,
                    save_path=os.path.join(
                        save_dir, f"{self.name}_solution_{epoch}.png"
                    ),
                )

                # Save history
                logger.save()

    def train_model_nys_newton(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,  # NysNewtonCG optimizer
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since Newton methods work better with fixed points
        pde_nodes = pde_sampler()
        ic_nodes = ic_sampler()
        eval_nodes = eval_sampler()

        print("Training model with Nyström Newton-CG...")
        start_time = time()

        # Define closure for NysNewtonCG that returns both loss and gradient
        def closure():
            optimizer.zero_grad()
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model, pde_nodes, ic_nodes, ic_weight
            )
            # Compute gradient with create_graph=True for Hessian computation
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Make gradients contiguous and reshape them
            grads = [g.contiguous() for g in grads]

            # Update preconditioner every iteration
            optimizer.update_preconditioner(grads)

            return loss, grads

        # Training loop
        for epoch in tqdm(range(n_epochs)):

            # Optimize
            loss, _ = optimizer.step(closure)

            # Log
            logger.log("loss", loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model.interpolate(eval_nodes)
                    u_true = self.get_solution(eval_nodes)
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    # Get losses for history
                    total_loss, pde_loss, ic_loss = self.get_pde_loss(
                        model, pde_nodes, ic_nodes, ic_weight
                    )
                    _, eval_pde_loss, _ = self.get_pde_loss(
                        model, eval_nodes, ic_nodes, ic_weight
                    )
                    logger.log("train_pde_loss", pde_loss.item(), epoch)
                    logger.log("train_ic_loss", ic_loss.item(), epoch)
                    logger.log("eval_pde_loss", eval_pde_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Iteration {epoch + 1} completed in {current_time:.2f} seconds")
                print(
                    f"PDE loss: {logger.get_most_recent_value('train_pde_loss'):1.3e}"
                )
                print(f"IC loss: {logger.get_most_recent_value('train_ic_loss'):1.3e}")
                print(
                    f"Evaluation L2 rel error: {logger.get_most_recent_value('eval_l2_relative_error'):1.3e}"
                )
                self.plot_solution(
                    eval_nodes,
                    u_eval,
                    save_path=os.path.join(
                        save_dir, f"{self.name}_solution_{epoch}.png"
                    ),
                )

                # Save history
                logger.save()
