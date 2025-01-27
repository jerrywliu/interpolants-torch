import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Callable, List, Tuple

from src.experiments.base_fcn import BaseFcn
from src.loggers.logger import Logger
from src.optimizers.nys_newton_cg import NysNewtonCG


class BaseAnalyticalTarget(BaseFcn):
    def __init__(
        self,
        name: str,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: List[Tuple[float, float]],
        derivative: Callable[[torch.Tensor], torch.Tensor] = None,
        second_derivative: Callable[[torch.Tensor], torch.Tensor] = None,
        device: str = "cpu",
    ):
        super().__init__(name, domain, device)
        self.f = f
        self.derivative = derivative
        self.second_derivative = second_derivative

    def get_function(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        mesh = torch.meshgrid(*nodes, indexing="ij")
        return self.f(*mesh)

    def get_solution(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        return self.get_function(nodes)

    def get_derivative(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        if self.derivative is None:
            raise ValueError("Derivative not implemented")
        mesh = torch.meshgrid(*nodes, indexing="ij")
        return self.derivative(*mesh)

    def get_second_derivative(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        if self.second_derivative is None:
            raise ValueError("Second derivative not implemented")
        mesh = torch.meshgrid(*nodes, indexing="ij")
        return self.second_derivative(*mesh)

    def get_loss(self, model: nn.Module, nodes: List[torch.Tensor]) -> torch.Tensor:
        u_pred = model(nodes)
        u_true = self.get_function(nodes)
        return torch.mean((u_pred - u_true) ** 2)

    def plot_solution(
        self, nodes: List[torch.Tensor], u: torch.Tensor, save_path: str = None
    ):
        raise NotImplementedError

    def train_model(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        train_sampler: Callable,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        in_domain: Callable = None,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        print("Training model...")
        start_time = time()
        for epoch in tqdm(range(n_epochs)):

            # Sample points
            train_nodes = train_sampler()

            # Train step
            optimizer.zero_grad()

            # Get loss
            loss = self.get_loss(model, train_nodes)

            # Backprop
            loss.backward()

            # Update parameters
            optimizer.step()

            # Log
            logger.log("loss", loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:
                with torch.no_grad():
                    eval_nodes = eval_sampler()
                    u_eval = model(eval_nodes)
                    u_true = self.get_function(eval_nodes)
                    mask = in_domain(eval_nodes) if in_domain is not None else None
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true, mask)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    eval_loss = self.get_loss(model, eval_nodes)
                    logger.log("eval_loss", eval_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Epoch {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"Loss: {logger.get_most_recent_value('loss'):1.3e}")
                self.plot_solution(
                    eval_nodes,
                    u_eval,
                    save_path=os.path.join(save_dir, f"{self.name}_{epoch}.png"),
                )

                # Save history
                logger.save()

        # Plot loss history
        plt.figure()
        plt.semilogy(logger.get_iters("loss"), logger.get_values("loss"), label="Loss")
        plt.semilogy(
            logger.get_iters("eval_l2_error"),
            logger.get_values("eval_l2_error"),
            label="Eval L2 Error",
        )
        plt.semilogy(
            logger.get_iters("eval_max_error"),
            logger.get_values("eval_max_error"),
            label="Eval Max Error",
        )
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_history.png"))
        plt.close()

    def train_model_lbfgs(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.LBFGS,
        train_sampler: Callable,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        in_domain: Callable = None,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since L-BFGS works better with fixed points
        train_nodes = train_sampler()
        eval_nodes = eval_sampler()

        print("Training model with L-BFGS...")
        start_time = time()

        # Define closure for L-BFGS
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(model, train_nodes)
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
                with torch.no_grad():
                    u_eval = model(eval_nodes)
                    u_true = self.get_function(eval_nodes)
                    mask = in_domain(eval_nodes) if in_domain is not None else None
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true, mask)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    eval_loss = self.get_loss(model, eval_nodes)
                    logger.log("eval_loss", eval_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Iteration {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"Loss: {logger.get_most_recent_value('loss'):1.3e}")
                self.plot_solution(
                    eval_nodes,
                    u_eval,
                    save_path=os.path.join(save_dir, f"{self.name}_{epoch}.png"),
                )

                # Save history
                logger.save()

    def train_model_nys_newton(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: NysNewtonCG,
        train_sampler: Callable,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        in_domain: Callable = None,
    ):
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        # Sample points once since Newton methods work better with fixed points
        train_nodes = train_sampler()
        eval_nodes = eval_sampler()

        print("Training model with Nyström Newton-CG...")
        start_time = time()

        # Define closure for NysNewtonCG that returns both loss and gradient
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(model, train_nodes)
            # Compute gradient with create_graph=True for Hessian computation
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Make gradients contiguous
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
                with torch.no_grad():
                    u_eval = model(eval_nodes)
                    u_true = self.get_function(eval_nodes)
                    mask = in_domain(eval_nodes) if in_domain is not None else None
                    for eval_metric in eval_metrics:
                        eval_metric_value = eval_metric(u_eval, u_true, mask)
                        logger.log(
                            f"eval_{eval_metric.__name__}", eval_metric_value, epoch
                        )

                    eval_loss = self.get_loss(model, eval_nodes)
                    logger.log("eval_loss", eval_loss.item(), epoch)

                current_time = time() - start_time
                print(f"Iteration {epoch + 1} completed in {current_time:.2f} seconds")
                print(f"Loss: {logger.get_most_recent_value('loss'):1.3e}")
                self.plot_solution(
                    eval_nodes,
                    u_eval,
                    save_path=os.path.join(save_dir, f"{self.name}_{epoch}.png"),
                )

                # Save history
                logger.save()

    def train(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        train_sampler: Callable,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
        in_domain: Callable = None,
    ):
        """Routes to appropriate training method based on optimizer type"""
        if isinstance(optimizer, NysNewtonCG):
            self.train_model_nys_newton(
                model=model,
                n_epochs=n_epochs,
                optimizer=optimizer,
                train_sampler=train_sampler,
                eval_sampler=eval_sampler,
                eval_metrics=eval_metrics,
                eval_every=eval_every,
                save_dir=save_dir,
                logger=logger,
                in_domain=in_domain,
            )
        elif isinstance(optimizer, torch.optim.LBFGS):
            self.train_model_lbfgs(
                model=model,
                n_epochs=n_epochs,
                optimizer=optimizer,
                train_sampler=train_sampler,
                eval_sampler=eval_sampler,
                eval_metrics=eval_metrics,
                eval_every=eval_every,
                save_dir=save_dir,
                logger=logger,
                in_domain=in_domain,
            )
        else:
            self.train_model(
                model=model,
                n_epochs=n_epochs,
                optimizer=optimizer,
                train_sampler=train_sampler,
                eval_sampler=eval_sampler,
                eval_metrics=eval_metrics,
                eval_every=eval_every,
                save_dir=save_dir,
                logger=logger,
                in_domain=in_domain,
            )


if __name__ == "__main__":
    name = "xy"
    f = lambda x: x[0] * x[1]
    domain = [(0, 1), (0, 1)]
    basis = ["chebyshev", "fourier"]
    type = ["standard", "standard"]
    n_samples = 10

    target = BaseAnalyticalTarget(name, f, domain)
    print(target.sample_domain(n_samples, basis, type))
