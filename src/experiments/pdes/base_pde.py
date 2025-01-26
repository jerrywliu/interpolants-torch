import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Callable

from src.optimizers.nys_newton_cg import NysNewtonCG

from src.experiments.base_fcn import BaseFcn
from src.loggers.logger import Logger


class BasePDE(BaseFcn):
    def __init__(
        self,
        name: str,
        domain: List[Tuple[float, float]],
        device: str = "cpu",
        loss_weight_update_policy: str = "grad_norm",
        loss_weight_update_interval: int = -1,
        loss_weight_max: float = 100.0,
    ):
        super().__init__(name, domain, device)

        self.loss_weights = {}
        self.loss_weight_max = loss_weight_max
        self.loss_weight_update_policy = loss_weight_update_policy
        self.loss_weight_update_interval = loss_weight_update_interval

    # Input: x = [(n_1,), ..., (n_d,)]
    # Output: u = (n_1 * ... * n_d,)
    def get_solution(self, x: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def sample_domain_1d(
        self, n_samples: int, dim: int, basis: str, type: str
    ) -> torch.Tensor:
        """Sample points from domain with requires_grad=True"""
        points = super().sample_domain_1d(n_samples, dim, basis, type)
        points.requires_grad_(True)
        return points

    def sample_domain(
        self, n_samples: int, dim: int, basis: str, type: str
    ) -> List[torch.Tensor]:
        """Sample points from domain with requires_grad=True"""
        points = super().sample_domain(n_samples, dim, basis, type)
        points = [p.requires_grad_(True) for p in points]
        return points

    def get_loss_dict(
        self,
        model: nn.Module,
        pde_nodes: List[torch.Tensor],
        ic_nodes: List[torch.Tensor],  # [torch.tensor(0), nodes]
        ic_weight: float = 1,
        **kwargs,
    ):
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

    def update_loss_weights(self, epoch, model, optimizer, pde_nodes, ic_nodes):
        if (
            self.loss_weight_update_interval == -1
            or epoch % self.loss_weight_update_interval != 0
        ):
            return

        loss_dict = self.get_loss_dict(model, pde_nodes, ic_nodes)
        loss_names = loss_dict.keys()

        if self.loss_weight_update_policy != "grad_norm":
            return

        self.loss_weights = {}
        num_params = sum(p.numel() for p in model.parameters())

        for loss_name in loss_names:
            optimizer.zero_grad()
            curr_loss = self.get_loss_dict(model, pde_nodes, ic_nodes)[loss_name]
            optimizer.zero_grad()
            curr_loss.backward(retain_graph=True)
            curr_loss_avg_grad_norm = (
                torch.sqrt(
                    sum(
                        [
                            torch.sum(p.grad**2)
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    )
                )  # TODO JL 1/25/25 why is the bias of MLP.fc2's grad=None?
                / num_params
            )
            # TODO JL 1/25/25 I don't think this logic is correct. We want one of the loss weights to be 1.0 and the rest to be scaled accordingly.
            self.loss_weights[f"{loss_name}_weight"] = min(
                1.0 / curr_loss_avg_grad_norm, self.loss_weight_max
            )

        print(self.loss_weights)

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

            # Update loss weights
            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

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

            # Update loss weights
            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

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
    
    def train_model_dual_optimizers(
        self,
        model: nn.Module,
        n_epochs: int,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 1000,
        switch_every: int = 1000,  # Number of epochs to switch optimizers
        save_dir: str = None,
        logger: Logger = None,
    ):
    
        if logger is None:
            logger = Logger(path=os.path.join(save_dir, "logger.json"))

        print("Training model with dual optimizers...")
        start_time = time()

        ## TODO: Make this settable from arguments 
        optimizer1 = self.get_optimizer(model, 'adam')
        optimizer2 = self.get_optimizer(model, 'lbfgs')
        
        current_optimizer = optimizer1  # Start with optimizer1

        for epoch in tqdm(range(n_epochs)):

            # Switch optimizer based on the epoch
            if epoch % (2 * switch_every) < switch_every:
                current_optimizer = optimizer1
            else:
                current_optimizer = optimizer2

            # Sample points
            pde_nodes = pde_sampler()
            ic_nodes = ic_sampler()

            # Update loss weights
            self.update_loss_weights(epoch, model, current_optimizer, pde_nodes, ic_nodes)

            # Train step
            current_optimizer.zero_grad()

            # Get PDE loss
            loss, pde_loss, ic_loss = self.get_pde_loss(
                model, pde_nodes, ic_nodes, ic_weight
            )

            # Backprop
            loss.backward()

            # Update parameters
            current_optimizer.step()

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

    def train_model_nys_newton(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: NysNewtonCG,  # NysNewtonCG optimizer
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

            # Update loss weights
            self.update_loss_weights(epoch, model, optimizer, pde_nodes, ic_nodes)

            # Log
            logger.log("loss", loss.item(), epoch)

            # Eval, print, and plot progress
            if (epoch + 1) % eval_every == 0:

                # Evaluate solution
                with torch.no_grad():
                    u_eval = model(eval_nodes)
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

    def train(
        self,
        model: nn.Module,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        pde_sampler: Callable,
        ic_sampler: Callable,
        ic_weight: float,
        eval_sampler: Callable,
        eval_metrics: List[Callable],
        eval_every: int = 100,
        save_dir: str = None,
        logger: Logger = None,
    ):
        if isinstance(optimizer, NysNewtonCG):
            self.train_model_nys_newton(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every,
                save_dir,
                logger,
            )
        elif isinstance(optimizer, torch.optim.LBFGS):
            self.train_model_lbfgs(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every,
                save_dir,
                logger,
            )
        else:
            self.train_model(
                model,
                n_epochs,
                optimizer,
                pde_sampler,
                ic_sampler,
                ic_weight,
                eval_sampler,
                eval_metrics,
                eval_every,
                save_dir,
                logger,
            )
