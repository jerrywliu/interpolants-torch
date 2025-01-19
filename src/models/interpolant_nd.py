import torch
import torch.nn as nn
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt


# TODO JL 1/14/2025: add function to get entries corresponding to IC
class SpectralInterpolationND(nn.Module):
    def __init__(
        self, Ns: List[int], bases: List[str], domains: List[Tuple[float, float]]
    ):
        """
        ND interpolation using spectral methods

        Args:
            Ns: List of number of points per direction
            bases: List of bases per direction, either 'fourier' or 'chebyshev'
            domains: List of tuples of (min, max) per direction
        """
        super().__init__()

        # Store domain information
        assert len(Ns) == len(bases) == len(domains)
        self.n_dim = len(Ns)

        self.Ns = Ns
        self.bases = bases
        self.domains = domains
        self.domain_lengths = [domain[1] - domain[0] for domain in domains]

        # Set up nodes and weight/frequencies for interpolation (as needed)
        self.nodes = [None for _ in range(self.n_dim)]
        self.nodes_standard = [None for _ in range(self.n_dim)]
        self._from_standard = [None for _ in range(self.n_dim)]
        self._to_standard = [None for _ in range(self.n_dim)]
        self.cheb_weights = [None for _ in range(self.n_dim)]
        self.k = [None for _ in range(self.n_dim)]

        for dim in range(self.n_dim):
            if self.bases[dim] == "chebyshev":
                i = torch.linspace(0, 1, self.Ns[dim])
                self.nodes_standard[dim] = torch.cos(torch.pi * i)
                # Compute barycentric weights for Chebyshev
                N = self.Ns[dim]
                weights = torch.ones(N)
                weights[0] *= 0.5
                weights[-1] *= 0.5
                weights[1::2] = -1
                self.cheb_weights[dim] = weights

                self.k[dim] = None
            else:
                self.nodes_standard[dim] = torch.linspace(
                    0, 2 * torch.pi, self.Ns[dim] + 1
                )[:-1]
                # Compute FFT frequencies
                self.k[dim] = torch.fft.fftfreq(self.Ns[dim]) * self.Ns[dim]
                self.cheb_weights[dim] = None

            # Set up domain mapping functions for this dimension
            domain = self.domains[dim]
            if self.bases[dim] == "chebyshev":
                self._to_standard[dim] = (
                    lambda x, d=dim: 2
                    * (x - self.domains[d][0])
                    / self.domain_lengths[d]
                    - 1
                )
                self._from_standard[dim] = (
                    lambda x, d=dim: self.domains[d][0]
                    + (x + 1) * self.domain_lengths[d] / 2
                )
            else:  # fourier
                self._to_standard[dim] = (
                    lambda x, d=dim: 2
                    * torch.pi
                    * (x - self.domains[d][0])
                    / self.domain_lengths[d]
                )
                self._from_standard[dim] = lambda x, d=dim: self.domains[d][
                    0
                ] + self.domain_lengths[d] * x / (2 * torch.pi)

            # Map standard nodes to physical domain
            self.nodes[dim] = self._from_standard[dim](self.nodes_standard[dim])

        # Set up diff matrices cache
        self._diff_matrices = [{} for _ in range(self.n_dim)]

        # Create mesh grid of nodes
        mesh_args = [self.nodes[d] for d in range(self.n_dim)]
        self.mesh = torch.meshgrid(*mesh_args, indexing="ij")

        # Learnable values at node points
        self.values = nn.Parameter(torch.zeros(self.Ns))

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

    def derivative_matrix(self, k: Tuple[int, ...]) -> torch.Tensor:
        """
        Get mixed derivative matrix D^k where k is a tuple of derivative orders

        Args:
            k: List/tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Matrix operator for the mixed derivative
        """
        assert (
            len(k) == self.n_dim
        ), f"Expected {self.n_dim} derivative orders, got {len(k)}"

        # Get 1D matrices for each dimension
        matrices = []
        for dim in range(self.n_dim):
            if k[dim] == 0:
                # Identity matrix for this dimension
                matrices.append(
                    torch.eye(
                        self.Ns[dim],
                        dtype=self.nodes_standard[dim].dtype,
                        device=self.nodes_standard[dim].device,
                    )
                )
            else:
                # Compute/get cached derivative matrix
                if k[dim] not in self._diff_matrices[dim]:
                    if 1 not in self._diff_matrices[dim]:
                        if self.bases[dim] == "chebyshev":
                            D = self._compute_cheb_derivative_matrix(
                                self.nodes_standard[dim], self.domain_lengths[dim]
                            )
                        else:  # fourier
                            D = self._compute_fourier_derivative_matrix(
                                self.nodes_standard[dim], self.domain_lengths[dim]
                            )
                        self._diff_matrices[dim][1] = D

                    # Compose for higher derivatives
                    Dk = self._diff_matrices[dim][1]
                    for _ in range(k[dim] - 1):
                        Dk = Dk @ self._diff_matrices[dim][1]
                    self._diff_matrices[dim][k[dim]] = Dk

                matrices.append(self._diff_matrices[dim][k[dim]])

        # Compute Kronecker product
        D = matrices[0]
        for dim in range(1, self.n_dim):
            D = torch.kron(D, matrices[dim])

        return D

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

        # # Implementation 1
        # time_start = time()
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        values_expanded = values.unsqueeze(0)  # (1, B2, B, N)
        nodes_expanded = nodes_std.reshape(1, 1, 1, -1)
        weights_expanded = weights.reshape(1, 1, 1, -1)
        # print(f"Version 1: {time() - time_start}")

        # # Implementation 2
        # time_start = time()
        # x_eval_expanded, nodes_expanded, values_expanded, weights_expanded = torch.broadcast_tensors(
        #     x_eval_standard[:, None, :, None], nodes_std[None, None, None, :], values[None, ...], weights[None, None, None, :]
        # )
        # print(f"Version 2: {time() - time_start}")

        # Compute distances - result is (B1, B2, B, N)
        d_x = x_eval_expanded - nodes_expanded

        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=-1, keepdim=True).values

        d_x = torch.where(small_diff_max, torch.zeros_like(d_x), 1.0 / d_x)
        d_x[small_diff] = 1

        # Compute weighted sum along last axis
        # # Implementation 1
        f_eval_num = torch.sum(
            values_expanded * d_x * weights_expanded, dim=-1
        )  # (B1, B2, B)
        f_eval_denom = torch.sum(d_x * weights_expanded, dim=-1)  # (B1, B2, B)

        # # Implementation 2
        # f_eval_num = torch.einsum('...ij,...ij->...i', values_expanded * weights_expanded, d_x)  # (B1, B2, B)
        # f_eval_denom = torch.einsum('...ij,...ij->...i', weights_expanded, d_x)  # (B1, B2, B)

        return f_eval_num / f_eval_denom

    def _cheb_interpolate_1ofnd(
        self,
        values: torch.Tensor,
        x_eval: torch.Tensor,
        dim: int,
        nodes_std: torch.Tensor,
        to_std: Callable,
        weights: torch.Tensor,
        eps: float = 1e-14,
    ):
        """
        Interpolate along a specific Chebyshev dimension of a tensor.

        Args:
            values: Tensor of shape (..., N, ...), where N is the size of the Chebyshev dim.
            x_eval: Tensor of shape (m,), points to evaluate along the Chebyshev dimension.
            dim: Integer, the axis corresponding to the Chebyshev dimension in `values`.
            nodes_std: Tensor of shape (N,), the Chebyshev nodes.
            to_std: Function mapping physical to standard domain.
            weights: Tensor of shape (N,), the barycentric weights.
            eps: Small value to handle division by zero.

        Returns:
            Tensor of shape (..., m, ...), with the Chebyshev dimension replaced by interpolated values.
        """

        # Step 1: Move the Chebyshev axis to the last position for simplicity
        values_moved = values.movedim(dim, -1)  # (..., N)
        batch_shape, N = values_moved.shape[:-1], values_moved.shape[-1]
        m = x_eval.shape[0]  # Number of evaluation points

        # Step 2: Reshape for batch broadcasting
        # - Add a singleton dimension to `values` for x_eval
        # - Add a singleton dimension to `x_eval` for values
        values_reshaped = values_moved.reshape(-1, 1, N)  # (..., None, N)
        x_eval_reshaped = x_eval[:, None]  # (m, 1)

        # Step 3: Call the 1D Chebyshev interpolation helper
        interpolated = self._cheb_interpolate_1d(
            x_eval=x_eval_reshaped,  # Shape (m, 1)
            values=values_reshaped,  # Shape (..., None, N)
            nodes_std=nodes_std,  # Shape (N,)
            to_std=to_std,  # Function
            weights=weights,  # Shape (N,)
            eps=eps,
        )  # Output shape: (..., m, 1)

        # Step 4: Restore the original dimension layout
        interpolated = interpolated.reshape(m, *batch_shape).movedim(
            0, dim
        )  # (..., m, ...)

        return interpolated

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

    def _fourier_interpolate_1ofnd(
        self,
        values: torch.Tensor,
        x_eval: torch.Tensor,
        dim: int,
        to_std: Callable,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate along a specific Fourier dimension of a tensor.

        Args:
            values: Tensor of shape (..., N, ...), where N is the size of the Fourier dim.
            x_eval: Tensor of shape (m,), points to evaluate along the Fourier dimension.
            dim: Integer, the axis corresponding to the Fourier dimension in `values`.
            to_std: Function mapping physical to standard domain.
            k: Tensor of shape (N,), the Fourier frequencies.

        Returns:
            Tensor of shape (..., m, ...), with the Fourier dimension replaced by interpolated values.
        """

        # Step 1: Move the Fourier axis to the last position for simplicity
        values_moved = values.movedim(dim, -1)
        batch_shape, N = values_moved.shape[:-1], values_moved.shape[-1]
        m = x_eval.shape[0]  # Number of evaluation points

        # Step 2: Reshape for batch broadcasting
        # - Add a singleton dimension to `values` for x_eval
        # - Add a singleton dimension to `x_eval` for values
        values_reshaped = values_moved.reshape(-1, 1, N)
        x_eval_reshaped = x_eval[:, None]

        # Step 3: Call the 1D Fourier interpolation helper
        interpolated = self._fourier_interpolate_1d(
            x_eval=x_eval_reshaped,  # Shape (m, 1)
            values=values_reshaped,  # Shape (..., None, N)
            to_std=to_std,  # Function
            k=k,  # Shape (..., N, 1)
        )

        # Step 4: Restore the original dimension layout
        interpolated = interpolated.reshape(m, *batch_shape).movedim(0, dim)

        return interpolated

    def interpolate(self, x_eval: List[torch.Tensor], values=None) -> torch.Tensor:
        """
        Interpolate the function at the given points

        Args:
            x_eval: List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at

        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) - interpolated values
        """
        if values is not None:
            assert values.shape == self.values.shape
            interpolated = values
        else:
            interpolated = self.values
        for dim in range(self.n_dim):
            if self.bases[dim] == "chebyshev":
                interpolated = self._cheb_interpolate_1ofnd(
                    values=interpolated,
                    x_eval=x_eval[dim],
                    dim=dim,
                    nodes_std=self.nodes_standard[dim],
                    to_std=self._to_standard[dim],
                    weights=self.cheb_weights[dim],
                )
            elif self.bases[dim] == "fourier":
                interpolated = self._fourier_interpolate_1ofnd(
                    values=interpolated,
                    x_eval=x_eval[dim],
                    dim=dim,
                    to_std=self._to_standard[dim],
                    k=self.k[dim],
                )
            else:
                raise ValueError(f"Unknown basis: {self.bases[dim]}")

        return interpolated

    def forward(self, x_eval: List[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the interpolant at arbitrary evaluation points

        Args:
            x_eval: List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at

        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) - interpolated values
        """
        return self.interpolate(x_eval, values=self.values)

    def _derivative_interpolant(self, k: Tuple[int, ...]) -> torch.Tensor:
        """
        Compute mixed derivative of interpolant

        Args:
            k: Tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Tensor containing derivative values at grid points
        """
        # Handle the case where k is a single integer (apply to first dimension)
        if isinstance(k, int):
            k = (k,) + (0,) * (self.n_dim - 1)

        assert (
            len(k) == self.n_dim
        ), f"Expected {self.n_dim} derivative orders, got {len(k)}"

        # If all derivatives are zero, return values
        if all(ki == 0 for ki in k):
            return self.values

        # Get mixed derivative matrix
        Dk = self.derivative_matrix(k)

        # Compute derivative values at nodes (differentiable w.r.t self.values)
        # Reshape values to a flat vector for matrix multiplication
        values_flat = self.values.reshape(-1)
        dk_nodes = Dk @ values_flat

        # Reshape back to grid shape
        dk_nodes = dk_nodes.reshape(*self.values.shape)

        return dk_nodes

    def derivative(
        self, x_eval: List[torch.Tensor], k: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Compute mixed derivative of interpolant at arbitrary evaluation points

        Args:
            x_eval: List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at
            k: Tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) - interpolated values
        """
        # Compute derivative at nodes
        dk_nodes = self._derivative_interpolant(k)

        # Interpolate to evaluation points
        return self.interpolate(x_eval, values=dk_nodes)
