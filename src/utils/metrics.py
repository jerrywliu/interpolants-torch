import torch


def l2_error(u_eval: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((u_eval - u_true) ** 2).item()


def max_error(u_eval: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(u_eval - u_true)).item()


def l2_relative_error(u_eval: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    return (torch.norm(u_eval - u_true, p=2) / torch.norm(u_true, p=2)).item()
