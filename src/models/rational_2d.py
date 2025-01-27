import torch
import torch.nn as nn
from typing import List, Tuple, Callable

from src.models.rational_1d import (
    RationalInterpolation1D,
    RationalInterpolationPoles1D,
    compute_barycentric_weights_vect,
)


class RationalInterpolation2D(nn.Module):
    def __init__(
        self,
        N_1: int,
        N_2: int,
        bases_1: str,
        domain_1: Tuple[float, float],
        domain_2: Tuple[float, float],
        num_poles: int = 1,
        device: str = "cpu",
    ):
        """
        Naive 2D rational interpolation using 1D rational interpolants

        Args:
            N_1: Number of points in the first dimension
            N_2: Number of points in the second dimension
            bases_1: Basis for the first dimension, either 'chebyshev' or 'fourier'
            (bases_2 is always 'rational')
            domain_1: Tuple of (min, max) for the first dimension
            domain_2: Tuple of (min, max) for the second dimension
        """
        super().__init__()
        self.device = torch.device(device)

        self.num_poles = (
            num_poles  # Number of conjugate pairs of poles for the second dimension
        )
        self.n_dim = 2
        self.N_1 = N_1
        self.N_2 = N_2
        self.bases_1 = bases_1
        self.domain_1 = domain_1
        self.domain_2 = domain_2
        self.domain_length_1 = domain_1[1] - domain_1[0]
        self.domain_length_2 = domain_2[1] - domain_2[0]

        # Set up nodes and weight/frequencies for interpolation (as needed)
        self.nodes = [None for _ in range(self.n_dim)]
        self.nodes_standard = [None for _ in range(self.n_dim)]
        self._from_standard = [None for _ in range(self.n_dim)]
        self._to_standard = [None for _ in range(self.n_dim)]

        # Utilities for the first dimension
        self.cheb_weight_1 = None
        self.k_1 = None

        if self.bases_1 == "chebyshev":
            i = torch.linspace(0, 1, self.N_1)
            self.nodes_standard[0] = torch.cos(torch.pi * i).to(self.device)
            # Compute barycentric weights for Chebyshev
            N = self.N_1
            # Initialize weights using standard barycentric formula
            init_weights = compute_barycentric_weights_vect(self.nodes_standard[0]).to(
                self.device
            )
            # weights = torch.ones(N, device=self.device)
            # weights[0] *= 0.5
            # weights[-1] *= 0.5
            # weights[1::2] = -1
            self.cheb_weights_1 = init_weights
            self.k_1 = None
        else:
            self.nodes_standard[0] = torch.linspace(0, 2 * torch.pi, self.N_1 + 1)[
                :-1
            ].to(self.device)
            # Compute FFT frequencies
            self.k_1 = torch.fft.fftfreq(self.N_1) * self.N_1
            self.k_1 = self.k_1.to(self.device)
            self.cheb_weights_1 = None

        # Set up domain mapping functions for this dimension
        if self.bases_1 == "chebyshev":
            self._to_standard[0] = (
                lambda x: 2 * (x - self.domain_1[0]) / self.domain_length_1 - 1
            )
            self._from_standard[0] = (
                lambda x: self.domain_1[0] + (x + 1) * self.domain_length_1 / 2
            )
        else:  # fourier
            self._to_standard[0] = (
                lambda x: 2 * torch.pi * (x - self.domain_1[0]) / self.domain_length_1
            )
            self._from_standard[0] = lambda x: self.domain_1[
                0
            ] + self.domain_length_1 * x / (2 * torch.pi)

        # Map standard nodes to physical domain
        self.nodes[0] = self._from_standard[0](self.nodes_standard[0]).to(self.device)

        # Second dimension is always rational
        i = torch.linspace(0, 1, self.N_2)
        self.nodes_standard[1] = torch.cos(torch.pi * i).to(self.device)
        self._to_standard[1] = (
            lambda x: 2 * (x - self.domain_2[0]) / self.domain_length_2 - 1
        )
        self._from_standard[1] = (
            lambda x: self.domain_2[0] + (x + 1) * self.domain_length_2 / 2
        )
        self.nodes[1] = self._from_standard[1](self.nodes_standard[1]).to(self.device)

        # Set up diff matrices cache
        self._diff_matrices_1 = {}
        # Second dimension is always rational so we don't need to compute diff matrices

        # Create mesh grid of nodes
        mesh_args = [self.nodes[d] for d in range(self.n_dim)]
        self.mesh = torch.meshgrid(*mesh_args, indexing="ij")

        # Stack of 1D rational interpolants along the first dimension
        self.interpolants = nn.ModuleList(
            [
                # RationalInterpolationPoles1D(
                #     N=self.N_2,
                #     domain=self.domain_2,
                #     num_poles=self.num_poles,
                #     device=self.device,
                # )
                RationalInterpolation1D(
                    N=self.N_2,
                    domain=self.domain_2,
                    device=self.device,
                )
                for _ in range(self.N_1)
            ]
        )

    def _compute_cheb_derivative_matrix(
        self, nodes: torch.Tensor, domain_length: float
    ) -> torch.Tensor:
        """
        Compute the differentiation matrix for 1D Chebyshev points
        """
        N = len(nodes)
        D = torch.zeros((N, N), dtype=nodes.dtype, device=nodes.device)

        # Compute weights for endpoints
        c = torch.ones(N, dtype=nodes.dtype, device=nodes.device)
        c[0] = 2
        c[-1] = 2

        # Compute off-diagonal entries
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = c[i] / c[j] * (-1) ** (i + j) / (nodes[i] - nodes[j])

        # Fill diagonal using negative sum trick
        D.diagonal().copy_(-torch.sum(D, dim=1))

        # Scale for domain transformation
        D = D * (2.0 / domain_length)

        return D

    def _compute_fourier_derivative_matrix(
        self, nodes: torch.Tensor, domain_length: float
    ) -> torch.Tensor:
        """
        Compute the differentiation matrix for 1D equispaced Fourier
        """
        N = len(nodes)
        D = torch.zeros((N, N), dtype=nodes.dtype, device=nodes.device)

        # Create index matrices
        i, j = torch.meshgrid(
            torch.arange(N, dtype=nodes.dtype, device=nodes.device),
            torch.arange(N, dtype=nodes.dtype, device=nodes.device),
            indexing="ij",
        )

        # Compute off-diagonal elements using cotangent formula
        mask = i != j
        diff = (i[mask] - j[mask]) * (-1) ** (i[mask] - j[mask])
        D[mask] = 0.5 * torch.tan(torch.pi * diff / N).reciprocal()

        # Diagonal elements are 0 for periodic functions
        D.diagonal().zero_()

        # Scale for domain transformation
        D = D * (2 * torch.pi / domain_length)

        return D

    def derivative_matrix(self, k: int) -> torch.Tensor:
        """
        Get derivative matrix D^k for the first dimension
        """

        if k == 0:
            return torch.eye(
                self.N_1, dtype=self.nodes[0].dtype, device=self.nodes[0].device
            )
        else:
            # Compute/get cached derivative matrix
            if k not in self._diff_matrices_1:
                if 1 not in self._diff_matrices_1:
                    if self.bases_1 == "chebyshev":
                        D = self._compute_cheb_derivative_matrix(
                            self.nodes[0], self.domain_length_1
                        )
                    else:  # fourier
                        D = self._compute_fourier_derivative_matrix(
                            self.nodes[0], self.domain_length_1
                        )
                    self._diff_matrices_1[1] = D

                # Compose for higher derivatives
                Dk = self._diff_matrices_1[1]
                for _ in range(k - 1):
                    Dk = Dk @ self._diff_matrices_1[1]
                self._diff_matrices_1[k] = Dk

            return self._diff_matrices_1[k]

    def _cheb_interpolate_1d(
        self,
        x_eval: torch.Tensor,
        values: torch.Tensor,
        nodes_std: torch.Tensor,
        to_std: Callable,
        weights: torch.Tensor,
        eps: float = 1e-14,
    ):
        """Helper for 1D Chebyshev interpolation along last axis

        Args:
            x_eval: shape (B1, B) - points to evaluate at
            values: shape (B2, B, N) - function values at nodes
            nodes_std: shape (N,) - standard Chebyshev nodes
            to_std: function - maps from physical to standard domain
            weights: shape (N,) - barycentric weights

        Returns:
            shape (B1, B2, B) - interpolated values
        """
        x_eval_standard = to_std(x_eval)  # (B1, B)

        # Reshape inputs for broadcasting:
        # x_eval: (B1, 1, B, 1)
        # values: (1, B2, B, N)
        # nodes: (1, 1, 1, N)
        # weights: (1, 1, 1, N)

        # Implementation 1
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        values_expanded = values.unsqueeze(0)  # (1, B2, B, N)
        nodes_expanded = nodes_std.reshape(1, 1, 1, -1)
        weights_expanded = weights.reshape(1, 1, 1, -1)

        # Compute distances - result is (B1, B2, B, N)
        d_x = x_eval_expanded - nodes_expanded

        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=-1, keepdim=True).values

        d_x = torch.where(small_diff_max, torch.zeros_like(d_x), 1.0 / d_x)
        d_x[small_diff] = 1

        # Compute weighted sum along last axis
        f_eval_num = torch.sum(
            values_expanded * d_x * weights_expanded, dim=-1
        )  # (B1, B2, B)
        f_eval_denom = torch.sum(d_x * weights_expanded, dim=-1)  # (B1, B2, B)

        return f_eval_num / f_eval_denom

    def _fourier_interpolate_1d(
        self,
        x_eval: torch.Tensor,
        values: torch.Tensor,
        to_std: Callable,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Helper for 1D Fourier interpolation along last axis

        Args:
            x_eval: shape (B1, B) - points to evaluate at
            values: shape (B2, B, N) - function values at nodes
            to_std: function - maps from physical to standard domain
            k: shape (N,) - frequency modes

        Returns:
            shape (B1, B2, B) - interpolated values
        """
        N = values.shape[-1]
        x_eval_standard = to_std(x_eval)  # (B1, B)

        # Compute FFT along last axis
        coeffs = torch.fft.fft(values, dim=-1)  # (B2, B, N)

        # Reshape inputs for broadcasting:
        # x_eval: (B1, 1, B, 1)
        # coeffs: (1, B2, B, N)
        # k: (1, 1, 1, N)
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        coeffs_expanded = coeffs.unsqueeze(0)  # (1, B2, B, N)
        k_expanded = k.reshape(1, 1, 1, -1)

        # Compute Fourier matrix - result is (B1, 1, B, N)
        x_matrix = x_eval_expanded * k_expanded
        fourier_matrix = torch.exp(1j * x_matrix)

        # Matrix multiply and sum along last axis - result is (B1, B2, B)
        result = torch.sum(fourier_matrix * coeffs_expanded, dim=-1)
        return torch.real(result) / N

    def _interpolate_first_dim(
        self, x_eval: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate along the first dimension

        Args:
            x_eval: shape (B1,) - points to evaluate at
            values: shape (N_1, B2) - function values at nodes

        Returns:
            shape (B1, B2) - interpolated values
        """
        # Interpolate along the first dimension
        if self.bases_1 == "chebyshev":
            interpolated = self._cheb_interpolate_1d(
                x_eval=x_eval.unsqueeze(-1),  # shape (m1,) - > (m1, 1)
                values=values.transpose(0, 1).unsqueeze(
                    1
                ),  # shape (N_1, m2) - > (m2, 1, N_1)
                nodes_std=self.nodes_standard[0],  # shape (N_1,)
                to_std=self._to_standard[0],  # function
                weights=self.cheb_weights_1,  # shape (N_1,)
            ).squeeze(
                -1
            )  # (m1, m2, 1) -> (m1, m2)
        else:  # fourier
            interpolated = self._fourier_interpolate_1d(
                x_eval=x_eval.unsqueeze(-1),  # shape (m1,) - > (m1, 1)
                values=values.transpose(0, 1).unsqueeze(
                    1
                ),  # shape (N_1, m2) - > (m2, 1, N_1)
                to_std=self._to_standard[0],  # function
                k=self.k_1,  # shape (N_1,)
            ).squeeze(
                -1
            )  # (m1, m2, 1) -> (m1, m2)

        return interpolated

    def interpolate(self, x_eval: List[torch.Tensor]) -> torch.Tensor:
        """
        Interpolate the function at the given points
        We perform interpolation along the second dimension first (for each rational interpolant), then along the first dimension

        Args:
            x_eval: List of tensors of shapes (m1,), (m2,) - points to evaluate at

        Returns:
            Tensor of shape (m1, m2) - interpolated values
        """
        # Interpolate along the second dimension first
        interpolated = torch.stack(
            [self.interpolants[i](x_eval[1]) for i in range(len(self.interpolants))],
            dim=0,
        )  # shape (N_1, m2)

        # Interpolate along the first dimension
        return self._interpolate_first_dim(x_eval[0], interpolated)

    def forward(self, x_eval: List[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the interpolant at arbitrary evaluation points

        Args:
            x_eval: List of tensors of shapes (m1,), (m2,) - points to evaluate at

        Returns:
            Tensor of shape (m1, m2) - interpolated values
        """
        return self.interpolate(x_eval)

    def derivative(
        self, x_eval: List[torch.Tensor], k: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Compute mixed derivative of interpolant at arbitrary evaluation points

        Args:
            x_eval: List of tensors of shapes (m1,), (m2,) - points to evaluate at
            k: Tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,1) means second derivative in x, first in y

        Returns:
            Tensor of shape (m1, m2) - interpolated values
        """
        # Compute derivative along the second dimension first
        dk_nodes = torch.stack(
            [
                self.interpolants[i].derivative(x_eval[1], k[1])
                for i in range(len(self.interpolants))
            ],
            dim=0,
        )  # shape (N_1, m2)

        # Compute derivative along the first dimension
        Dk = self.derivative_matrix(k[0])  # shape (N_1, N_1)
        dk_nodes = torch.einsum("ij,jk->ik", Dk, dk_nodes)  # shape (N_1, m2)

        # Interpolate to evaluation points
        return self._interpolate_first_dim(x_eval[0], dk_nodes)

    def set_values(self, values: torch.Tensor):
        """
        Set the values of the interpolant

        Args:
            values: shape (N_1, N_2) - function values at nodes
        """
        for i in range(self.N_1):
            self.interpolants[i].values = values[i]

    def get_values(self) -> torch.Tensor:
        """
        Get the values of the interpolant

        Returns:
            Tensor of shape (N_1, N_2) - function values at nodes
        """
        return torch.stack([i.values for i in self.interpolants], dim=0)
