import torch
import torch.nn as nn
from typing import Tuple, Callable, List, Union


def get_cheb_nodes(n: int) -> torch.Tensor:
    """Get Chebyshev points of the second kind"""
    i = torch.linspace(0, 1, n)
    return torch.cos(torch.pi * i)


def get_polynomial_barycentric_weights(nodes: torch.Tensor, **kwargs):
    """
    Compute barycentric weights for polynomial interpolation (up to a constant factor)
    Formula:
        w_i = \prod_{j=0, \neq i}^{n} 1/(x_i - x_j)
    """
    n = len(nodes)
    log_weights = torch.zeros_like(nodes)
    signs = torch.ones_like(nodes)

    # Compute weights in log space
    for j in range(n):
        diffs = nodes[j] - nodes
        diffs = diffs[torch.arange(n) != j]
        log_weights[j] = -torch.sum(torch.log(torch.abs(diffs)))
        signs[j] = (-1) ** (torch.sum(diffs < 0))

    # Convert back from log space and normalize
    log_weights = log_weights - log_weights.max()
    weights = signs * torch.exp(log_weights)
    weights = weights / torch.abs(weights[0])

    return weights


# TODO JL 1/19/25: any way to make this more numerically stable?
def get_floater_hormann_barycentric_weights(nodes: torch.Tensor, d: int, **kwargs):
    """
    Compute barycentric weights for Floater-Hormann rational interpolation
    n: number of nodes - 1
    d ranges from 0 to n
    Formula:
        w_i = \sum_{k=min(0, i-d)}^{max(n-d, i)} (-1)^k \prod_{j=i, \neq k}^{n} 1/(x_i - x_j)
    """
    n = len(nodes) - 1
    weights = torch.zeros_like(nodes)

    # TODO JL 1/19/25: since we only compute this once, let's manually loop.
    for i in range(n + 1):
        for k in range(max(0, i - d), min(n - d, i) + 1):
            # Compute innermost product in log space
            diffs = nodes[i] - nodes[k : k + d + 1]  # shape (d+1,)
            diffs = diffs[torch.arange(d + 1) != i - k]
            log_weights = -torch.sum(torch.log(torch.abs(diffs)))
            signs = (-1) ** (torch.sum(diffs < 0))
            weights[i] += (-1) ** k * signs * torch.exp(log_weights)

    # Normalize
    weights_max = torch.max(weights)
    weights = weights / weights_max

    return weights


class RationalInterpolation1D(nn.Module):
    def __init__(
        self,
        nodes: torch.Tensor,
        domain: Tuple[float, float],
        weights_type: str = "polynomial",
        floater_hormann_d: int = None,  # Floater-Hormann parameter
    ):
        """
        Rational interpolation with Chebyshev backbone and learnable weights

        Args:
            num_points: Number of Chebyshev points
        """
        super().__init__()

        # Chebyshev backbone
        self.nodes = nodes
        self.N = len(nodes)
        self.domain = domain
        self.domain_length = domain[1] - domain[0]
        # Standard domain is [-1, 1]
        self.to_standard = lambda x: 2 * (x - self.domain[0]) / self.domain_length - 1
        self.from_standard = lambda x: (x + 1) * self.domain_length / 2 + self.domain[0]
        self.nodes_standard = self.to_standard(self.nodes)

        self.floater_hormann_d = floater_hormann_d

        # Initialize weights using standard barycentric formula
        if weights_type == "polynomial":
            self._get_barycentric_weights = get_polynomial_barycentric_weights
            self.weights = nn.Parameter(
                self._get_barycentric_weights(self.nodes), requires_grad=False
            )
        elif weights_type == "floater_hormann":
            self._get_barycentric_weights = get_floater_hormann_barycentric_weights
            self.weights = nn.Parameter(
                self._get_barycentric_weights(
                    self.nodes,
                    d=(
                        self.floater_hormann_d
                        if self.floater_hormann_d is not None
                        else self.N - 1
                    ),
                ),
                requires_grad=False,
            )
        elif weights_type == "learnable":
            self._get_barycentric_weights = get_polynomial_barycentric_weights
            self.weights = nn.Parameter(
                self._get_barycentric_weights(self.nodes), requires_grad=True
            )
        else:
            raise ValueError(f"Invalid weights type: {weights_type}")

        # Values at nodes
        self.values = nn.Parameter(torch.zeros(self.N))

        # TODO JL 1/16/25: do we cache derivative matrices?

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


if __name__ == "__main__":
    n = 81
    d = 80
    cheb = get_cheb_nodes(n)

    print("Barycentric weights for polynomial interpolation")
    print(get_polynomial_barycentric_weights(cheb))

    print("For d=0, the Floater-Hormann weights should be +-1")
    print(get_floater_hormann_barycentric_weights(cheb, d=0))

    print(
        "For d=n, the Floater-Hormann weights should be the same as the polynomial weights"
    )
    print(get_floater_hormann_barycentric_weights(cheb, d=n - 1))

    print("Sanity check: make sure that the weights are symmetric for some d")
    print(get_floater_hormann_barycentric_weights(cheb, d=10))
