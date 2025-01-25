import torch
from typing import Callable, List, Tuple

from src.experiments.base_fcn import BaseFcn


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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

    def get_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

    def get_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.derivative(x)

    def get_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_derivative(x)


if __name__ == "__main__":
    name = "xy"
    f = lambda x: x[0] * x[1]
    domain = [(0, 1), (0, 1)]
    basis = ["chebyshev", "fourier"]
    type = ["standard", "standard"]
    n_samples = 10

    target = BaseAnalyticalTarget(name, f, domain)
    print(target.sample_domain(n_samples, basis, type))
