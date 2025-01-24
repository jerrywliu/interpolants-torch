import torch
import torch.nn as nn
from typing import Tuple, Callable, List, Union


def get_cheb_nodes(n: int) -> torch.Tensor:
    """Get Chebyshev points of the second kind"""
    i = torch.linspace(0, 1, n)
    return torch.cos(torch.pi * i)


# TODO JL 1/16/25: double-check these poles are correct
def compute_barycentric_weights(
    nodes: torch.Tensor,
    poles_real: torch.Tensor = None,
    poles_imag: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute barycentric weights maintaining gradients through poles.

    Args:
        nodes: Tensor of shape (N,) containing nodes
        poles_real: Tensor of shape (M,) containing real parts of poles
        poles_imag: Tensor of shape (M,) containing imaginary parts of poles (conjugate pairs will be added)
    """
    n = len(nodes)
    log_weights = torch.zeros_like(nodes)
    signs = torch.ones_like(nodes)

    for j in range(n):
        # Compute standard node differences
        diffs = nodes[j] - nodes
        diffs = diffs[torch.arange(n) != j]
        log_weights[j] = -torch.sum(torch.log(torch.abs(diffs)))
        signs[j] = (-1) ** (torch.sum(diffs < 0))

        # Add pole contributions maintaining gradients
        if poles_real is not None and poles_imag is not None:
            for r, i in zip(poles_real, poles_imag):
                # Add contribution from pole and its conjugate
                pole_diff = torch.sqrt((nodes[j] - r) ** 2 + i**2)  # distance to p
                log_weights[j] += torch.log(pole_diff)  # p contribution
                # Conjugate is same distance so just multiply by 2
                log_weights[j] += torch.log(pole_diff)  # p̄ contribution

    # Convert back from log space
    log_weights = log_weights - log_weights.max()
    weights = signs * torch.exp(log_weights)
    weights = weights / torch.abs(weights[0])

    return weights


def compute_barycentric_weights_vect(
    nodes: torch.Tensor,
    poles_real: torch.Tensor = None,
    poles_imag: torch.Tensor = None,
) -> torch.Tensor:
    n = len(nodes)

    # Compute node differences using broadcasting
    nodes_diff = nodes.unsqueeze(1) - nodes.unsqueeze(0)  # Shape: (n,n)
    mask = ~torch.eye(n, dtype=bool)  # Exclude diagonal
    diffs = nodes_diff[mask].view(n, n - 1)  # Shape: (n,n-1)

    # Compute log weights and signs
    log_weights = -torch.sum(torch.log(torch.abs(diffs)), dim=1)
    signs = (-1) ** torch.sum(diffs < 0, dim=1)

    # Add pole contributions
    if poles_real is not None and poles_imag is not None:
        # Compute pole distances using broadcasting
        pole_diffs = torch.sqrt(
            (nodes.unsqueeze(1) - poles_real) ** 2 + poles_imag**2
        )  # Shape: (n,m)
        log_weights += 2 * torch.sum(
            torch.log(pole_diffs), dim=1
        )  # Factor 2 for conjugates

    # Convert from log space
    log_weights = log_weights - log_weights.max()
    weights = signs * torch.exp(log_weights)
    weights = weights / torch.abs(weights[0])

    return weights


# This model's learnable parameters are the values at the nodes and the barycentric weights.
class RationalInterpolation1D(nn.Module):
    def __init__(self, N: int, domain: Tuple[float, float]):
        """
        Rational interpolation with Chebyshev backbone and learnable weights

        Args:
            N: Number of Chebyshev points
            domain: Tuple of floats (min, max) specifying the domain
        """
        super().__init__()

        # Chebyshev backbone
        self.N = N
        self.domain = domain
        self.domain_length = domain[1] - domain[0]
        # Standard domain is [-1, 1]
        self.to_standard = lambda x: 2 * (x - self.domain[0]) / self.domain_length - 1
        self.from_standard = lambda x: (x + 1) * self.domain_length / 2 + self.domain[0]
        self.nodes = get_cheb_nodes(N)
        self.nodes_standard = self.to_standard(self.nodes)

        # Values at nodes
        self.values = nn.Parameter(torch.zeros(N))

        # Initialize weights using standard barycentric formula
        init_weights = compute_barycentric_weights(self.nodes)

        # Make weights learnable parameters
        self.weights = nn.Parameter(init_weights)

    def _compute_derivative_matrix(
        self,
        nodes: torch.Tensor,
        domain_length: float,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the differentiation matrix for the barycentric interpolant
        given learnable weights.

        Args:
            nodes: Tensor of shape (N,) containing nodes
            domain_length: Length of the domain
            weights: Tensor of shape (N,) containing learnable weights. If None, use the initialized weights.
        Returns:
            Tensor of shape (N, N) containing the differentiation matrix
        """
        n = len(nodes)
        D = torch.zeros((n, n), dtype=nodes.dtype, device=nodes.device)

        if weights is None:
            weights = self.weights

        # Compute off-diagonal entries using learnable weights
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = weights[j] / weights[i] / (nodes[i] - nodes[j])

        # Fill diagonal using negative sum trick
        D.diagonal().copy_(-torch.sum(D, dim=1))

        # Scale for domain transformation
        D = D * (2.0 / domain_length)

        return D

    def derivative_matrix(self, k: int = 1) -> torch.Tensor:
        """
        Get k-th derivative matrix
        """
        if k == 0:
            n = len(self.nodes)
            return torch.eye(n, dtype=self.nodes.dtype, device=self.nodes.device)

        else:
            D = self._compute_derivative_matrix(
                self.nodes, self.domain_length, self.weights
            )
            Dk = D.clone()

            for i in range(k - 1):
                Dk = Dk @ D

            return Dk

    def _interpolate(
        self,
        x_eval: torch.Tensor,
        values: torch.Tensor,
        nodes_std: torch.Tensor,
        to_std: Callable,
        weights: torch.Tensor,
        eps: float = 1e-14,
    ) -> torch.Tensor:
        """
        Interpolate values using barycentric formula

        Args:
            x_eval: shape (N_eval) - points to evaluate at
            values: shape (N) - values at self.nodes to interpolate from
            nodes_std: shape (N) - standard Chebyshev nodes
            to_std: function - maps from physical to standard domain
            weights: shape (N) - barycentric weights
            eps: small number to handle numerical instability

        Returns:
            shape (N_eval) - interpolated values
        """
        x_eval_standard = to_std(x_eval)  # (N_eval)

        # Compute difference matrix
        d_x = x_eval_standard.unsqueeze(0) - nodes_std.unsqueeze(1)

        # Handle numerical instability
        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=0).values
        d_x = torch.where(small_diff_max[None, :], torch.zeros_like(d_x), 1.0 / d_x)
        d_x[small_diff] = 1

        # Compute barycentric terms using learnable weights
        f_eval_num = torch.einsum("...n,nm,n->...m", values, d_x, weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, weights)

        return f_eval_num / f_eval_denom

    def forward(self, x_eval: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Evaluate rational interpolant at arbitrary evaluation points

        Args:
            x_eval: shape (N_eval) - points to evaluate at

        Returns:
            shape (N_eval) - interpolated values
        """
        if isinstance(x_eval, List):
            assert (
                len(x_eval) == 1
            ), "Only one dimension supported for 1D rational interpolation"
            return self._interpolate(
                x_eval[0],
                self.values,
                self.nodes_standard,
                self.to_standard,
                self.weights,
            )

        return self._interpolate(
            x_eval,
            self.values,
            self.nodes_standard,
            self.to_standard,
            self.weights,
        )

    def derivative(self, x_eval: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Compute k-th derivative of rational interpolant at arbitrary evaluation points

        Args:
            x_eval: Tensor of shape (..., n_dim) containing coordinates to evaluate at
            k: Integer specifying derivative order

        Returns:
            Tensor of shape (...) containing derivative values at x_eval points
        """
        if k == 0:
            return self(x_eval)

        # Get k-th derivative matrix
        Dk = self.derivative_matrix(k)

        # Compute derivative values at nodes (differentiable w.r.t self.values)
        dk_nodes = Dk @ self.values

        # Interpolate to evaluation points using the derivative values
        return self._interpolate(x_eval, dk_nodes)


# This model's learnable parameters are the values at the nodes and the poles.
class RationalInterpolationPoles1D(nn.Module):
    def __init__(self, N: int, domain: Tuple[float, float], num_poles: int = 1):
        """
        Rational interpolation with Chebyshev backbone and learnable poles (in conjugate pairs)

        Args:
            N: Number of Chebyshev points
            domain: Tuple of floats (min, max) specifying the domain
            num_poles: Number of conjugate pairs of poles
        """
        super().__init__()

        # Chebyshev backbone
        self.N = N
        self.domain = domain
        self.domain_length = domain[1] - domain[0]
        # Standard domain is [-1, 1]
        self.to_standard = lambda x: 2 * (x - self.domain[0]) / self.domain_length - 1
        self.from_standard = lambda x: (x + 1) * self.domain_length / 2 + self.domain[0]
        self.nodes = get_cheb_nodes(N)
        self.nodes_standard = self.to_standard(self.nodes)

        # Values at nodes
        self.values = nn.Parameter(torch.zeros(N))

        # Initialize poles
        self.poles_real = nn.Parameter(torch.randn(num_poles))
        self.poles_imag = nn.Parameter(torch.randn(num_poles))

    def _compute_derivative_matrix(
        self,
        nodes: torch.Tensor,
        domain_length: float,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the differentiation matrix for the barycentric interpolant
        given learnable weights.

        Args:
            nodes: Tensor of shape (N,) containing nodes
            domain_length: Length of the domain
            weights: Tensor of shape (N,) containing barycentric weights
        Returns:
            Tensor of shape (N, N) containing the differentiation matrix
        """
        n = len(nodes)
        D = torch.zeros((n, n), dtype=nodes.dtype, device=nodes.device)

        # Compute off-diagonal entries using barycentric weights
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = weights[j] / weights[i] / (nodes[i] - nodes[j])

        # Fill diagonal using negative sum trick
        D.diagonal().copy_(-torch.sum(D, dim=1))

        # Scale for domain transformation
        D = D * (2.0 / domain_length)

        return D

    def derivative_matrix(
        self, k: int = 1, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get k-th derivative matrix
        """
        if k == 0:
            n = len(self.nodes)
            return torch.eye(n, dtype=self.nodes.dtype, device=self.nodes.device)

        else:
            D = self._compute_derivative_matrix(self.nodes, self.domain_length, weights)
            Dk = D.clone()

            for i in range(k - 1):
                Dk = Dk @ D

            return Dk

    def _interpolate(
        self,
        x_eval: torch.Tensor,
        values: torch.Tensor,
        nodes_std: torch.Tensor,
        to_std: Callable,
        weights: torch.Tensor,
        eps: float = 1e-14,
    ) -> torch.Tensor:
        """
        Interpolate values using barycentric formula

        Args:
            x_eval: shape (N_eval) - points to evaluate at
            values: shape (N) - values at self.nodes to interpolate from
            nodes_std: shape (N) - standard Chebyshev nodes
            to_std: function - maps from physical to standard domain
            weights: shape (N) - barycentric weights
            eps: small number to handle numerical instability

        Returns:
            shape (N_eval) - interpolated values
        """
        x_eval_standard = to_std(x_eval)  # (N_eval)

        # Compute difference matrix
        d_x = x_eval_standard.unsqueeze(0) - nodes_std.unsqueeze(1)

        # Handle numerical instability
        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=0).values
        d_x = torch.where(small_diff_max[None, :], torch.zeros_like(d_x), 1.0 / d_x)
        d_x[small_diff] = 1

        # Compute barycentric terms using learnable weights
        f_eval_num = torch.einsum("...n,nm,n->...m", values, d_x, weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, weights)

        return f_eval_num / f_eval_denom

    def forward(self, x_eval: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Evaluate rational interpolant at arbitrary evaluation points

        Args:
            x_eval: shape (N_eval) - points to evaluate at

        Returns:
            shape (N_eval) - interpolated values
        """
        weights = compute_barycentric_weights(
            self.nodes, self.poles_real, self.poles_imag
        )

        if isinstance(x_eval, List):
            assert (
                len(x_eval) == 1
            ), "Only one dimension supported for 1D rational interpolation"
            return self._interpolate(
                x_eval[0],
                self.values,
                self.nodes_standard,
                self.to_standard,
                weights,
            )

        return self._interpolate(
            x_eval,
            self.values,
            self.nodes_standard,
            self.to_standard,
            weights,
        )

    def derivative(self, x_eval: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Compute k-th derivative of rational interpolant at arbitrary evaluation points

        Args:
            x_eval: Tensor of shape (..., n_dim) containing coordinates to evaluate at
            k: Integer specifying derivative order

        Returns:
            Tensor of shape (...) containing derivative values at x_eval points
        """
        weights = compute_barycentric_weights(
            self.nodes, self.poles_real, self.poles_imag
        )

        if k == 0:
            return self(x_eval)

        # Get k-th derivative matrix
        Dk = self.derivative_matrix(k, weights)

        # Compute derivative values at nodes (differentiable w.r.t self.values)
        dk_nodes = Dk @ self.values

        # Interpolate to evaluation points using the derivative values
        return self._interpolate(
            x_eval,
            dk_nodes,
            self.nodes_standard,
            self.to_standard,
            weights,
        )
