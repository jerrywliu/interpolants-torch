import torch
from typing import Callable, List, Tuple


# TODO JL 1/17/25: change inputs to be list of tensors?
class BaseAnalyticalTarget:
    def __init__(
        self,
        name: str,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: List[Tuple[float, float]],
        derivative: Callable[[torch.Tensor], torch.Tensor] = None,
        second_derivative: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        self.name = name
        self.f = f
        self.derivative = derivative
        self.second_derivative = second_derivative
        # Domain is a list of tuples, where each tuple is a pair of floats
        # representing the lower and upper bounds of the domain for each dimension
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

    def get_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

    def get_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.derivative(x)

    def get_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_derivative(x)

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
                cheb_nodes = torch.cos(torch.linspace(0, torch.pi, n_samples))
                return self._from_cheb(cheb_nodes, dim)
            elif basis == "fourier":
                fourier_nodes = torch.linspace(0, 2 * torch.pi, n_samples + 1)[:-1]
                return self._from_fourier(fourier_nodes, dim)
            else:
                raise ValueError(f"Invalid basis: {basis}")
        elif type == "uniform":
            if basis == "chebyshev":
                uniform_nodes = torch.cos(torch.rand(n_samples) * torch.pi)
                return self._from_cheb(uniform_nodes, dim)
            elif basis == "fourier":
                uniform_nodes = torch.rand(n_samples) * 2 * torch.pi
                return self._from_fourier(uniform_nodes, dim)
            else:
                raise ValueError(f"Invalid basis: {basis}")
        else:
            raise ValueError(f"Invalid type: {type}")

    def sample_domain(
        self, n_samples: List[int], basis: List[str], type: List[str]
    ) -> List[torch.Tensor]:
        return [
            self.sample_domain_1d(n_samples[dim], dim, basis[dim], type[dim])
            for dim in range(self.n_dims)
        ]


if __name__ == "__main__":
    name = "xy"
    f = lambda x: x[0] * x[1]
    domain = [(0, 1), (0, 1)]
    basis = ["chebyshev", "fourier"]
    type = ["standard", "standard"]
    n_samples = 10

    target = BaseAnalyticalTarget(name, f, domain)
    print(target.sample_domain(n_samples, basis, type))
