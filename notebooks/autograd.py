# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: icon
#     language: python
#     name: icon
# ---

# %%
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
import numpy as np

from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
import os, sys

BASE_DIR = "/pscratch/sd/j/jwl50/interpolants-torch"
sys.path.append(BASE_DIR)

# %%
# from src.models.interpolant_nd import SpectralInterpolationND
from src.experiments.interpolation.simple_fcns.sine_1d import Sine1DTarget

# %%
import torch
import torch.nn as nn
from typing import List, Tuple, Callable


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

    def _compute_fourier_derivative_matrix(self, nodes: torch.Tensor, domain_length: float) -> torch.Tensor:
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

    # def _cheb_interpolate_1d(
    #     self,
    #     x_eval: torch.Tensor,
    #     values: torch.Tensor,
    #     nodes_std: torch.Tensor,
    #     to_std: Callable,
    #     weights: torch.Tensor,
    #     eps: float = 1e-14,
    # ):
    #     """Helper for 1D Chebyshev interpolation along last axis

    #     Args:
    #         x_eval: shape (B1, B) - points to evaluate at
    #         values: shape (B2, B, N) - function values at nodes
    #         nodes_std: shape (N,) - standard Chebyshev nodes
    #         to_std: function - maps from physical to standard domain
    #         weights: shape (N,) - barycentric weights

    #     Returns:
    #         shape (B1, B2, B) - interpolated values
    #     """
    #     x_eval_standard = to_std(x_eval)  # (B1, B)

    #     # Reshape inputs for broadcasting:
    #     # x_eval: (B1, 1, B, 1)
    #     # values: (1, B2, B, N)
    #     # nodes: (1, 1, 1, N)
    #     # weights: (1, 1, 1, N)

    #     # # Implementation 1
    #     # time_start = time()
    #     x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
    #     values_expanded = values.unsqueeze(0)  # (1, B2, B, N)
    #     nodes_expanded = nodes_std.reshape(1, 1, 1, -1)
    #     weights_expanded = weights.reshape(1, 1, 1, -1)
    #     # print(f"Version 1: {time() - time_start}")

    #     # # Implementation 2
    #     # time_start = time()
    #     # x_eval_expanded, nodes_expanded, values_expanded, weights_expanded = torch.broadcast_tensors(
    #     #     x_eval_standard[:, None, :, None], nodes_std[None, None, None, :], values[None, ...], weights[None, None, None, :]
    #     # )
    #     # print(f"Version 2: {time() - time_start}")

    #     # Compute distances - result is (B1, B2, B, N)
    #     d_x = x_eval_expanded - nodes_expanded

    #     small_diff = torch.abs(d_x) < eps
    #     small_diff_max = torch.max(small_diff, dim=-1, keepdim=True).values

    #     d_x = torch.where(small_diff_max, torch.zeros_like(d_x), 1.0 / d_x)
    #     # d_x[small_diff] = 1
    #     d_x = torch.where(small_diff, torch.ones_like(d_x), d_x)

    #     # Compute weighted sum along last axis
    #     # # Implementation 1
    #     f_eval_num = torch.sum(
    #         values_expanded * d_x * weights_expanded, dim=-1
    #     )  # (B1, B2, B)
    #     f_eval_denom = torch.sum(d_x * weights_expanded, dim=-1)  # (B1, B2, B)

    #     # # Implementation 2
    #     # f_eval_num = torch.einsum('...ij,...ij->...i', values_expanded * weights_expanded, d_x)  # (B1, B2, B)
    #     # f_eval_denom = torch.einsum('...ij,...ij->...i', weights_expanded, d_x)  # (B1, B2, B)

    #     return f_eval_num / f_eval_denom

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
        
        # Reshape inputs for broadcasting
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        values_expanded = values.unsqueeze(0)  # (1, B2, B, N)
        nodes_expanded = nodes_std.reshape(1, 1, 1, -1)
        weights_expanded = weights.reshape(1, 1, 1, -1)
        
        # Compute distances
        d_x = x_eval_expanded - nodes_expanded  # (B1, 1, B, N)
        
        # Get sign and magnitude
        abs_diff = torch.abs(d_x)
        signs = torch.sign(d_x)
        
        # Replace small values with eps while preserving sign
        d_x = signs * torch.max(abs_diff, torch.tensor(eps, device=d_x.device))
        
        # Use reciprocal for barycentric weights
        d_x_reciprocal = 1.0 / d_x
        
        # Compute weighted sum along last axis
        f_eval_num = torch.sum(
            values_expanded * d_x_reciprocal * weights_expanded, dim=-1
        )  # (B1, B2, B)
        f_eval_denom = torch.sum(d_x_reciprocal * weights_expanded, dim=-1)  # (B1, B2, B)
        
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



# %% [markdown]
# ## Sanity check: does the new differentiable 1D interpolate work?

# %%
def test_interpolation(eps_values=[1e-14, 1e-10, 1e-6]):
    """Test the differentiable interpolation implementation"""
    
    # Setup problem
    target = Sine1DTarget()
    n_x = 21
    n_eval = 200
    
    # Create model
    model = SpectralInterpolationND(
        Ns=[n_x],
        bases=['chebyshev'],
        domains=target.domain
    )
    
    # Set the function values at nodes
    model.values.data = target.get_function(model.nodes[0])
    
    # Create evaluation points including points very close to nodes
    x_eval_base = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
    # Add points very close to nodes
    # delta = 1e-10
    # delta = 1e-14
    delta = 0
    x_near_nodes = model.nodes[0].unsqueeze(1) + torch.tensor([-delta, delta]).reshape(1, -1)
    x_near_nodes = x_near_nodes.reshape(-1)
    x_eval = torch.cat([x_eval_base, x_near_nodes])
    # x_eval = x_eval.sort()[0]
    
    results = {}
        
    # Create figure for comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get true values
    true_values = target.get_function(x_eval)
    true_deriv = target.derivative(x_eval)
    
    # Test interpolation
    interp_values = model([x_eval])
    value_error = torch.abs(interp_values - true_values)
    
    # Plot function values
    axes[0,0].plot(x_eval.detach(), true_values.detach(), 'k-', label='True')
    axes[0,0].plot(x_eval.detach(), interp_values.detach(), 'b--', label='Interpolated')
    axes[0,0].plot(model.nodes[0].detach(), model.values.detach(), 'ko', label='Nodes')
    axes[0,0].set_title('Function Values')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot value errors
    axes[0,1].semilogy(x_eval.detach(), value_error.detach(), 'b-', label='Error')
    axes[0,1].semilogy(x_near_nodes.detach(), value_error[x_eval_base.shape[0]:].detach(), 'o', color='red')
    axes[0,1].set_title('Function Value Errors')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Test derivatives
    x_eval_grad = x_eval.clone().requires_grad_(True)
    try:
        values = model([x_eval_grad])
        autograd_deriv = torch.autograd.grad(values.sum(), x_eval_grad)[0]
        deriv_error = torch.abs(autograd_deriv - true_deriv)
        deriv_success = True
        
        # Plot derivatives
        axes[1,0].plot(x_eval.detach(), true_deriv.detach(), 'k-', label='True')
        axes[1,0].plot(x_eval.detach(), autograd_deriv.detach(), 'b--', label='Autograd')
        axes[1,0].set_title('Derivatives')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot derivative errors
        axes[1,1].semilogy(x_eval.detach(), deriv_error.detach(), 'b-', label='Error')
        axes[1,1].semilogy(x_near_nodes.detach(), deriv_error[x_eval_base.shape[0]:].detach(), 'o', color='red')
        axes[1,1].set_title('Derivative Errors')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
    except Exception as e:
        print(f"Derivative computation failed: {e}")
        deriv_success = False
    
    plt.tight_layout()
    plt.show()
    
    # Store results
    results[eps] = {
        'value_l2_error': torch.mean(value_error**2).sqrt().item(),
        'value_max_error': torch.max(value_error).item(),
        'deriv_success': deriv_success,
    }
    if deriv_success:
        results[eps].update({
            'deriv_l2_error': torch.mean(deriv_error**2).sqrt().item(),
            'deriv_max_error': torch.max(deriv_error).item(),
        })
    
    # Print summary
    print("\nResults Summary:")
    for eps, res in results.items():
        print(f"\nEpsilon = {eps}")
        for key, val in res.items():
            print(f"  {key}: {val}")
    
    return results

if __name__ == "__main__":
    results = test_interpolation()

# %% [markdown]
# ## Sanity check: do the differentiable interpolants match the original?

# %%
import torch
import matplotlib.pyplot as plt

def compare_interpolation_methods(model, target, n_eval_points=100, eps_values=[1e-14, 1e-10, 1e-6]):
    """Compare original vs new interpolation methods"""
    
    # Set function values at nodes
    model.values.data = target.get_function(model.nodes[0])
    
    # Create evaluation points
    x_eval = torch.linspace(model.domains[0][0], model.domains[0][1], n_eval_points)
    x_eval_grad = x_eval.requires_grad_(True)
    
    # Get true values and derivatives
    true_values = target.get_function(x_eval)
    true_deriv = target.derivative(x_eval)
    
    # Store results for different eps values
    results = {}
    
    plt.figure(figsize=(15, 10))
    
    # Test different eps values for new methods
    for i, eps in enumerate(eps_values):
        # Store current eps results
        curr_results = {}
        
        # Original method (for reference)
        try:
            # Set the original interpolation method
            model._cheb_interpolate_1d = model.cheb_interpolate_1d_differentiable
            orig_values = model([x_eval_grad])
            orig_deriv = torch.autograd.grad(orig_values.sum(), x_eval_grad, retain_graph=True)[0]
            
            curr_results['orig_values'] = orig_values.detach()
            curr_results['orig_deriv'] = orig_deriv.detach()
        except Exception as e:
            print(f"Error with original method: {e}")
            curr_results['orig_values'] = None
            curr_results['orig_deriv'] = None
        
        # New differentiable method
        try:
            model._cheb_interpolate_1d = lambda *args, **kwargs: model.cheb_interpolate_1d_differentiable(*args, **kwargs)
            diff_values = model([x_eval_grad])
            diff_deriv = torch.autograd.grad(diff_values.sum(), x_eval_grad, retain_graph=True)[0]
            
            curr_results['diff_values'] = diff_values.detach()
            curr_results['diff_deriv'] = diff_deriv.detach()
        except Exception as e:
            print(f"Error with differentiable method (eps={eps}): {e}")
            curr_results['diff_values'] = None
            curr_results['diff_deriv'] = None
        
        # Alternative method
        try:
            model._cheb_interpolate_1d = lambda *args, **kwargs: model.cheb_interpolate_1d_alternative(*args, **kwargs)
            alt_values = model([x_eval_grad])
            alt_deriv = torch.autograd.grad(alt_values.sum(), x_eval_grad, retain_graph=True)[0]
            
            curr_results['alt_values'] = alt_values.detach()
            curr_results['alt_deriv'] = alt_deriv.detach()
        except Exception as e:
            print(f"Error with alternative method (eps={eps}): {e}")
            curr_results['alt_values'] = None
            curr_results['alt_deriv'] = None
        
        # Plot function values
        plt.subplot(len(eps_values), 2, 2*i + 1)
        plt.plot(x_eval, true_values, 'k-', label='True', alpha=0.5)
        if curr_results['orig_values'] is not None:
            plt.plot(x_eval, curr_results['orig_values'], 'b--', label='Original')
        if curr_results['diff_values'] is not None:
            plt.plot(x_eval, curr_results['diff_values'], 'r:', label='Differentiable')
        if curr_results['alt_values'] is not None:
            plt.plot(x_eval, curr_results['alt_values'], 'g-.', label='Alternative')
        plt.title(f'Function Values (eps={eps})')
        plt.legend()
        plt.grid(True)
        
        # Plot derivatives
        plt.subplot(len(eps_values), 2, 2*i + 2)
        plt.plot(x_eval, true_deriv, 'k-', label='True', alpha=0.5)
        if curr_results['orig_deriv'] is not None:
            plt.plot(x_eval, curr_results['orig_deriv'], 'b--', label='Original')
        if curr_results['diff_deriv'] is not None:
            plt.plot(x_eval, curr_results['diff_deriv'], 'r:', label='Differentiable')
        if curr_results['alt_deriv'] is not None:
            plt.plot(x_eval, curr_results['alt_deriv'], 'g-.', label='Alternative')
        plt.title(f'Derivatives (eps={eps})')
        plt.legend()
        plt.grid(True)
        
        # Compute error metrics
        for method in ['orig', 'diff', 'alt']:
            if curr_results[f'{method}_values'] is not None:
                val_error = torch.abs(curr_results[f'{method}_values'] - true_values)
                deriv_error = torch.abs(curr_results[f'{method}_deriv'] - true_deriv)
                
                curr_results[f'{method}_val_l2'] = torch.mean(val_error**2).sqrt().item()
                curr_results[f'{method}_val_max'] = torch.max(val_error).item()
                curr_results[f'{method}_deriv_l2'] = torch.mean(deriv_error**2).sqrt().item()
                curr_results[f'{method}_deriv_max'] = torch.max(deriv_error).item()
        
        results[eps] = curr_results
    
    plt.tight_layout()
    
    # Print error statistics
    print("\nError Statistics:")
    for eps, curr_results in results.items():
        print(f"\nEpsilon = {eps}")
        for method in ['orig', 'diff', 'alt']:
            if all(k in curr_results for k in [f'{method}_val_l2', f'{method}_deriv_l2']):
                print(f"\n{method.capitalize()} method:")
                print(f"  Value L2 Error: {curr_results[f'{method}_val_l2']:.2e}")
                print(f"  Value Max Error: {curr_results[f'{method}_val_max']:.2e}")
                print(f"  Derivative L2 Error: {curr_results[f'{method}_deriv_l2']:.2e}")
                print(f"  Derivative Max Error: {curr_results[f'{method}_deriv_max']:.2e}")
    
    return results

# Example usage:
if __name__ == "__main__":
    # Create interpolation model
    n_points = 31
    model = SpectralInterpolationND(
        Ns=[n_points],
        bases=['chebyshev'],
        domains=[(0, 2*torch.pi)]
    )
    
    # Create target function
    target = Sine1DTarget()
    
    # Run comparison
    results = compare_interpolation_methods(model, target)

# %%
import torch
import matplotlib.pyplot as plt

def compare_derivatives(model, target, n_eval_points=100):
    """Compare spectral and autograd derivatives for a 1D spectral interpolant"""
    
    # Set the function values at nodes
    model.values.data = target.get_function(model.nodes[0])
    
    # Create evaluation points
    x_eval = torch.linspace(model.domains[0][0], model.domains[0][1], n_eval_points+1)[:-1]
    # model_nodes = model.nodes[0]
    # x_eval = torch.cat([x_eval, model_nodes])
    
    # Get true derivative
    true_deriv = target.derivative(x_eval)
    
    # Get spectral derivative
    spectral_deriv = model.derivative([x_eval], k=(1,))
    
    # Get autograd derivative
    x_eval_grad = x_eval.requires_grad_(True)
    y = model([x_eval_grad])
    autograd_deriv = torch.autograd.grad(y.sum(), x_eval_grad)[0]

    interp = model([x_eval])
    print(f"Interp: {interp}")

    print(f"True: {true_deriv}")
    print(f"Spectral: {spectral_deriv}")
    print(f"Autograd: {autograd_deriv}")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Plot derivatives
    plt.subplot(121)
    plt.plot(x_eval.detach(), true_deriv.detach(), 'k-', label='True derivative')
    plt.plot(x_eval.detach(), spectral_deriv.detach(), 'r--', label='Spectral derivative')
    plt.plot(x_eval.detach(), autograd_deriv.detach(), 'b:', label='Autograd derivative')
    plt.title('Derivative Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot errors
    plt.subplot(122)
    spectral_error = torch.abs(spectral_deriv - true_deriv)
    autograd_error = torch.abs(autograd_deriv - true_deriv)
    
    plt.semilogy(x_eval.detach(), spectral_error.detach(), 'r--', label='Spectral error')
    plt.semilogy(x_eval.detach(), autograd_error.detach(), 'b:', label='Autograd error')
    plt.title('Derivative Errors (log scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Compute error statistics
    stats = {
        'spectral_l2_error': torch.mean(spectral_error**2).sqrt().item(),
        'autograd_l2_error': torch.mean(autograd_error**2).sqrt().item(),
        'spectral_max_error': torch.max(spectral_error).item(),
        'autograd_max_error': torch.max(autograd_error).item(),
    }
    
    print("\nError Statistics:")
    print(f"Spectral L2 Error: {stats['spectral_l2_error']:.2e}")
    print(f"Autograd L2 Error: {stats['autograd_l2_error']:.2e}")
    print(f"Spectral Max Error: {stats['spectral_max_error']:.2e}")
    print(f"Autograd Max Error: {stats['autograd_max_error']:.2e}")
    
    return stats

# Example usage:
if __name__ == "__main__":
    # Create interpolation model
    n_points = 31
    model = SpectralInterpolationND(
        Ns=[n_points],
        bases=['chebyshev'],  # or ['fourier']
        domains=[(0, 2*torch.pi)]
    )
    
    # Create target function (sine)
    target = Sine1DTarget()
    
    # Run comparison
    stats = compare_derivatives(model, target)

# %% [markdown]
# # Problem setup

# %%
# Problem setup
target = Sine1DTarget()

# %%
# Polynomial interpolation
n_x = 21
bases = ["chebyshev"]
domains = target.domain
model = SpectralInterpolationND(
    Ns=[n_x],
    bases=bases,
    domains=domains,
)

# %%
import torch
import matplotlib.pyplot as plt
import os

class DerivativeComparison:
    def __init__(self, target, model):
        self.target = target
        self.model = model
        
    def get_autograd_derivative(self, x):
        """Compute derivative using autograd"""
        x_tensor = x.clone().requires_grad_(True)
        y = self.model([x_tensor])
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        return gradients
    
    def compare_derivatives(
        self,
        x_eval: torch.Tensor,
        save_path: str = None
    ):
        """Compare spectral and autograd derivatives"""
        # Get true derivative values
        true_derivative = self.target.derivative(x_eval)
        
        # Get autograd derivative
        autograd_derivative = self.get_autograd_derivative(x_eval).detach()
        
        # Get spectral derivative (assuming model has this method)
        spectral_derivative = self.model.derivative([x_eval], (1)).detach()
        
        # Compute errors
        autograd_errors = torch.abs(autograd_derivative - true_derivative)
        spectral_errors = torch.abs(spectral_derivative - true_derivative)
        
        print(f"Autograd: {autograd_derivative}")
        print(f"Spectral: {spectral_derivative}")
        print(f"True: {true_derivative}")

        print(f"Autograd errors: {autograd_errors}")
        print(f"Spectral errors: {spectral_errors}")
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot derivatives
        ax1.plot(x_eval, true_derivative, 'k:', label='True derivative')
        ax1.plot(x_eval, autograd_derivative, 'b-', label='Autograd')
        ax1.plot(x_eval, spectral_derivative, 'r--', label='Spectral')
        ax1.set_title('Derivative Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Plot errors
        ax2.semilogy(x_eval, autograd_errors, 'b-', label='Autograd Error')
        ax2.semilogy(x_eval, spectral_errors, 'r--', label='Spectral Error')
        ax2.set_title('Derivative Errors (log scale)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            
        # Return error statistics
        return {
            'autograd_l2_error': torch.mean(autograd_errors**2).item(),
            'spectral_l2_error': torch.mean(spectral_errors**2).item(),
            'autograd_max_error': torch.max(autograd_errors).item(),
            'spectral_max_error': torch.max(spectral_errors).item()
        }

# Example usage (add to the main block of your code):
def run_derivative_comparison(target, model, x_eval, save_dir):
    # Initialize comparison class
    derivative_comp = DerivativeComparison(target, model)
    
    # Run comparison
    error_stats = derivative_comp.compare_derivatives(
        x_eval,
        save_path=os.path.join(save_dir, 'derivative_comparison.png')
    )
    
    # Print results
    print("\nDerivative Comparison Results:")
    print(f"Autograd L2 Error: {error_stats['autograd_l2_error']:1.3e}")
    print(f"Spectral L2 Error: {error_stats['spectral_l2_error']:1.3e}")
    print(f"Autograd Max Error: {error_stats['autograd_max_error']:1.3e}")
    print(f"Spectral Max Error: {error_stats['spectral_max_error']:1.3e}")
    
    return error_stats

# Add this to your main code after training each model:
# For example, after training the Fourier model:
model.values.data = target.get_function(model.nodes[0])
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/notebooks/figures/"
run_derivative_comparison(
    target=target,
    model=model,
    x_eval=x_eval,
    save_dir=save_dir
)

# %%
# Problem setup
c = 8
t_final = 1
u_0 = lambda x: torch.sin(x)
pde = Advection(c=c, t_final=t_final, u_0=u_0)
save_dir = os.path.join(BASE_DIR, "plots/advection")

# Model setup
# n_t = 2 * c + 1
# n_x = 2 * c
n_t = 4 * c + 1
n_x = 4 * c
bases = ["chebyshev", "fourier"]
model = SpectralInterpolationND(
    Ns=[n_t, n_x],
    bases=bases,
    domains=pde.domain,
)

# Training setup
n_epochs = 10000
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# PDE
sample_type = ["uniform", "uniform"]
n_t_train = 4 * c + 1
n_x_train = 4 * c
n_ic_train = 4 * c
ic_weight = 10

# Evaluation setup
n_eval = 200
t_eval = torch.linspace(0, t_final, n_eval)
x_eval = torch.linspace(0, 2 * torch.pi, n_eval + 1)[:-1]

# %% [markdown]
# ## Sanity check: does autograd work with interpolant?

# %%
import torch


def test_spectral_autograd(model):
    """Test if autograd can compute derivatives through spectral interpolation."""

    # Get model nodes
    t_nodes, x_nodes = model.nodes

    # Create evaluation points that require gradients
    eval_points = [t_nodes, x_nodes]

    # Get interpolated values
    u = model.interpolate(eval_points)

    # Try to compute derivative with autograd
    try:
        # Sum to get scalar for backward pass
        loss = u.sum()
        # Compute gradient with respect to model values
        grad = torch.autograd.grad(loss, model.values, create_graph=True)[0]
        print("Successfully computed gradient through interpolation!")
        print(f"Gradient shape: {grad.shape}")
        return True, grad
    except Exception as e:
        print("Failed to compute gradient through interpolation")
        print(f"Error: {str(e)}")
        return False, None


def compare_derivatives(model):
    """Compare autograd derivatives with spectral derivatives."""

    t_nodes, x_nodes = model.nodes

    # Get interpolated values with gradient tracking
    u = model.interpolate([t_nodes, x_nodes])
    loss = u.sum()

    # Compute x-derivative using autograd
    try:
        grad_values = torch.autograd.grad(loss, model.values, create_graph=True)[0]
        print("Successfully got gradient w.r.t values!")

        # Now try to get derivative w.r.t x through interpolation
        u = model.interpolate([t_nodes, x_nodes])
        u_x_spectral = model.derivative([t_nodes, x_nodes], k=(0, 1))

        # Compare with spectral derivative
        print(f"\nSpectral derivative shape: {u_x_spectral.shape}")
        print(f"Max spectral derivative value: {u_x_spectral.abs().max():.2e}")

        return True, u_x_spectral, grad_values
    except Exception as e:
        print(f"Error in comparison: {str(e)}")
        return False, None, None


# %%
def plot_derivative_comparison(
    u_x_spectral, u_x_autograd, t_grid, x_grid, save_path=None
):
    """
    Plot comparison between spectral and autograd derivatives.

    Args:
        u_x_spectral: Derivative computed using spectral method
        u_x_autograd: Derivative computed using autograd
        t_grid: Time points
        x_grid: Spatial points
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plot spectral derivatives
    im1 = ax1.imshow(
        u_x_spectral.detach(),
        extent=[
            x_grid[0].item(),
            x_grid[-1].item(),
            t_grid[0].item(),
            t_grid[-1].item(),
        ],
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(im1, ax=ax1)
    ax1.set_title("Spectral Derivative")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")

    # Plot autograd derivatives
    im2 = ax2.imshow(
        u_x_autograd.detach(),
        extent=[
            x_grid[0].item(),
            x_grid[-1].item(),
            t_grid[0].item(),
            t_grid[-1].item(),
        ],
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(im2, ax=ax2)
    ax2.set_title("Autograd Derivative")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")

    # Plot error on log scale
    error = torch.abs(u_x_spectral - u_x_autograd)
    im3 = ax3.imshow(
        error.detach(),
        extent=[
            x_grid[0].item(),
            x_grid[-1].item(),
            t_grid[0].item(),
            t_grid[-1].item(),
        ],
        origin="lower",
        aspect="auto",
        norm="log",
    )
    plt.colorbar(im3, ax=ax3)
    ax3.set_title("Log Error")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def test_and_plot_derivatives(model, save_path=None):
    """
    Test derivatives and create visualization.

    Args:
        model: SpectralInterpolationND model
        save_path: Optional path to save the figure
    """
    t_nodes, x_nodes = model.nodes

    # Get interpolated values
    u = model.interpolate([t_nodes, x_nodes])

    # Compute spectral derivative
    u_x_spectral = model.derivative([t_nodes, x_nodes], k=(0, 1))

    # Compute autograd derivative through interpolation
    loss = u.sum()
    grad_values = torch.autograd.grad(loss, model.values, create_graph=True)[0]

    # Plot comparison
    plot_derivative_comparison(u_x_spectral, grad_values, t_nodes, x_nodes, save_path)

    # Print error metrics
    error = torch.abs(u_x_spectral - grad_values)
    print(f"Maximum error: {error.max().item():.2e}")
    print(f"Mean error: {error.mean().item():.2e}")
    print(f"Error std: {error.std().item():.2e}")

    return u_x_spectral, grad_values, error


model = SpectralInterpolationND(
    Ns=[33, 32], bases=["chebyshev", "fourier"], domains=[(0, 1), (0, 2 * torch.pi)]
)

# Initialize with test function (e.g., sin(x))
x_nodes = model.nodes[1]
model.values.data = torch.sin(x_nodes).unsqueeze(0).expand(33, -1)

# Test and plot
u_x_spectral, grad_values, error = test_and_plot_derivatives(model)

# %% [markdown]
# ## Least squares

# %%
A, b = pde.get_least_squares(model)

# %%
torch.linalg.cond(A)

# %%
A.shape

# %%
pde.fit_least_squares(model)

# %%
u_eval = model.interpolate([t_eval, x_eval])
pde.plot_solution(
    u_eval.detach(),
    t_eval,
    x_eval,
)

# %% [markdown]
# ## Preconditioned least squares

# %%

# %% [markdown]
# ## GD

# %%
# Train model
pde.train_model(
    model,
    n_epochs=n_epochs,
    optimizer=optimizer,
    sample_type=sample_type,
    n_t_samples=n_t_train,
    n_x_samples=n_x_train,
    n_ic_samples=n_ic_train,
    ic_weight=ic_weight,
    t_eval=t_eval,
    x_eval=x_eval,
    plot_every=100,
    save_dir=save_dir,
)

# %%
