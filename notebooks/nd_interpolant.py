# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: icon
#     language: python
#     name: icon
# ---

# %% [markdown]
# # In this notebook, we implement a cleaned version of an N-D spectral interpolant.

# %%
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from time import time

# %%
import torch
import torch.nn as nn


class SpectralInterpolationND(nn.Module):
    def __init__(self, Ns, bases, domains):
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

    def _compute_cheb_derivative_matrix(self, nodes, domain_length):
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

    def _compute_fourier_derivative_matrix(self, nodes, domain_length):
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

    def derivative_matrix(self, k):
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
        self, x_eval, values, nodes_std, to_std, weights, eps=1e-14
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
        self, values, x_eval, dim, nodes_std, to_std, weights, eps=1e-14
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

    def _fourier_interpolate_1d(self, x_eval, values, to_std, k):
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

    def _fourier_interpolate_1ofnd(self, values, x_eval, dim, to_std, k):
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

    def interpolate(self, x_eval, values=None):
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

    def forward(self, x_eval):
        return self.interpolate(x_eval, values=self.values)
    
    def _derivative_interpolant(self, k):
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
    
    def derivative(self, x_eval, k):
        """
        Compute mixed derivative of interpolant at arbitrary evaluation points

        Args:
            x_eval: Tensor of shape (..., n_dim) containing coordinates to evaluate at
            k: Tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Tensor of shape (...) containing derivative values at x_eval points
        """
        # Compute derivative at nodes
        dk_nodes = self._derivative_interpolant(k)

        # Interpolate to evaluation points
        return self.interpolate(x_eval, values=dk_nodes)


# %% [markdown]
# # Test: interpolation

# %% [markdown]
# ### 1D interpolation

# %%
import torch
import matplotlib.pyplot as plt

# Define a test function for interpolation
def test_function(x):
    # Example: A sinusoidal function
    return torch.sin(2 * torch.pi * x)

# Create interpolant
Ns = [21]  # Number of Chebyshev nodes 
bases = ["chebyshev"]  # Basis type
domains = [(0, 1)]  # Domain for the interpolation
interp = SpectralInterpolationND(Ns, bases, domains)

# Evaluate the test function on the interpolation nodes
x_grid = interp.mesh[0]
f_grid = test_function(x_grid)
print(f"x_grid shape: {x_grid.shape}")
print(f"f_grid shape: {f_grid.shape}")

# Create a fine grid for visualization
n_vis = 100
x_vis = torch.linspace(0, 1, n_vis)
f_vis = test_function(x_vis)
print(f"x_vis: {x_vis.shape}")

# Interpolate
f_interp = interp._cheb_interpolate_1d(
# f_interp = interp._cheb_interpolate_1ofnd(
    # x_eval=x_vis,
    x_eval=x_vis.reshape(-1, 1),
    values=f_grid.unsqueeze(0).unsqueeze(0),  # Add batch dimensions
    # dim=0,
    nodes_std=interp.nodes_standard[0],
    to_std=interp._to_standard[0],
    weights=interp.cheb_weights[0],
    eps=1e-14
)
print(f"f_interp shape: {f_interp.shape}")
f_interp = f_interp.squeeze()  # Remove batch dimensions
print(f"f_interp shape: {f_interp.shape}")

# Compute error
error = torch.abs(f_interp - f_vis)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot solutions
ax1.plot(x_vis.numpy(), f_vis.numpy(), 'b-', label='Exact')
ax1.plot(x_vis.numpy(), f_interp.detach().numpy(), 'r--', label='Interpolated')
ax1.plot(x_grid.numpy(), f_grid.numpy(), 'k.', label='Nodes')
ax1.set_title('Function')
ax1.legend()

# Plot error
ax2.semilogy(x_vis.numpy(), error.detach().numpy(), 'k-')
ax2.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print error statistics
print(f"Maximum absolute error: {error.max().item():.2e}")
print(f"Average absolute error: {error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")

# %% [markdown]
# ## 2D interpolation
# First, implement by calling 1D interpolate each time.

# %%
import torch
import matplotlib.pyplot as plt

# Define a test function for interpolation
def test_function(x, y):
    return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)

# Create interpolant
Ns = [21, 21]  # Number of Chebyshev nodes in each dimension
bases = ["chebyshev", "chebyshev"]  # Basis type
domains = [(0, 1), (0, 1)]  # Domain for the interpolation
interp = SpectralInterpolationND(Ns, bases, domains)

# Evaluate the test function on the interpolation nodes
x_grid, y_grid = interp.mesh
f_grid = test_function(x_grid, y_grid)
interp.values.data = f_grid

# Create a fine grid for visualization
n_vis = 17
x_vis = torch.linspace(0.1, 0.9, n_vis)
y_vis = torch.linspace(0.1, 0.9, n_vis+2)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
f_vis = test_function(X_vis, Y_vis)

# Interpolate in first dimension
f_interp_intermediate = interp._cheb_interpolate_1ofnd(
    values=f_grid,
    x_eval=x_vis,
    dim=0,
    nodes_std=interp.nodes_standard[0],
    to_std=interp._to_standard[0],
    weights=interp.cheb_weights[0],
    eps=1e-14
)
print(f"f_interp_intermediate: {f_interp_intermediate.shape}")

# Interpolate in second dimension
f_interp = interp._cheb_interpolate_1ofnd(
    values=f_interp_intermediate,
    x_eval=y_vis,
    dim=1,
    nodes_std=interp.nodes_standard[1],
    to_std=interp._to_standard[1],
    weights=interp.cheb_weights[1],
    eps=1e-14
)
print(f"f_interp: {f_interp.shape}")

# Compute interpolated and exact values
f_vis = test_function(X_vis, Y_vis)
error = torch.abs(f_interp - f_vis)

# Plotting
fig = plt.figure(figsize=(20, 5))

# Plot exact solution
ax1 = fig.add_subplot(141, projection='3d')
surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), f_vis.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot intermediate interpolant
ax2 = fig.add_subplot(142, projection='3d')
X_intermediate, Y_intermediate = torch.meshgrid(x_vis, interp.nodes_standard[1], indexing='ij')
surf2 = ax2.plot_surface(X_intermediate.numpy(), Y_intermediate.numpy(), f_interp_intermediate.detach().numpy(), 
                       cmap='viridis')
ax2.set_title('After First Interpolation')

# Plot final interpolated solution
ax3 = fig.add_subplot(143, projection='3d')
surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), f_interp.detach().numpy(), 
                       cmap='viridis')
ax3.set_title('Final Interpolated')

# Plot error
ax4 = fig.add_subplot(144, projection='3d')
surf4 = ax4.plot_surface(X_vis.numpy(), Y_vis.numpy(), error.detach().numpy(), 
                       cmap='magma')
ax4.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print error statistics
print(f"Maximum absolute error: {error.max().item():.2e}")
print(f"Average absolute error: {error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")


# %% [markdown]
# ## 2D interpolation: Cheb

# %%
import torch
import matplotlib.pyplot as plt

# Define a test function for interpolation
def test_function(x, y):
    return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)

# Create interpolant
Ns = [21, 21]  # Number of Chebyshev nodes in each dimension
bases = ["chebyshev", "chebyshev"]  # Basis type
domains = [(0, 1), (0, 1)]  # Domain for the interpolation
interp = SpectralInterpolationND(Ns, bases, domains)

# Evaluate the test function on the interpolation nodes
x_grid, y_grid = interp.mesh
f_grid = test_function(x_grid, y_grid)
interp.values.data = f_grid

# Create a fine grid for visualization
n_vis = 17
x_vis = torch.linspace(0.1, 0.9, n_vis)
y_vis = torch.linspace(0.1, 0.9, n_vis+2)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
f_vis = test_function(X_vis, Y_vis)

# Interpolate
f_interp = interp.interpolate([x_vis, y_vis])
print(f"f_interp: {f_interp.shape}")

# Compute interpolated and exact values
f_vis = test_function(X_vis, Y_vis)
error = torch.abs(f_interp - f_vis)

# Plotting
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), f_vis.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot final interpolated solution
ax3 = fig.add_subplot(132, projection='3d')
surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), f_interp.detach().numpy(), 
                       cmap='viridis')
ax3.set_title('Final Interpolated')

# Plot error
ax4 = fig.add_subplot(133, projection='3d')
surf4 = ax4.plot_surface(X_vis.numpy(), Y_vis.numpy(), error.detach().numpy(), 
                       cmap='magma')
ax4.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print error statistics
print(f"Maximum absolute error: {error.max().item():.2e}")
print(f"Average absolute error: {error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")

# %% [markdown]
# ## 2D interpolation: Fourier

# %%
import torch
import matplotlib.pyplot as plt

# Define a test function for interpolation
def test_function(x, y):
    return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)

# Create interpolant
Ns = [20, 20]  # Number of Chebyshev nodes in each dimension
bases = ["fourier", "fourier"]  # Basis type
domains = [(0, 1), (0, 1)]  # Domain for the interpolation
interp = SpectralInterpolationND(Ns, bases, domains)

# Evaluate the test function on the interpolation nodes
x_grid, y_grid = interp.mesh
f_grid = test_function(x_grid, y_grid)
interp.values.data = f_grid

# Create a fine grid for visualization
n_vis = 17
x_vis = torch.linspace(0.1, 0.9, n_vis)
y_vis = torch.linspace(0.1, 0.9, n_vis+2)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
f_vis = test_function(X_vis, Y_vis)

# Interpolate
f_interp = interp.interpolate([x_vis, y_vis])
print(f"f_interp: {f_interp.shape}")

# Compute interpolated and exact values
f_vis = test_function(X_vis, Y_vis)
error = torch.abs(f_interp - f_vis)

# Plotting
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), f_vis.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot final interpolated solution
ax3 = fig.add_subplot(132, projection='3d')
surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), f_interp.detach().numpy(), 
                       cmap='viridis')
ax3.set_title('Final Interpolated')

# Plot error
ax4 = fig.add_subplot(133, projection='3d')
surf4 = ax4.plot_surface(X_vis.numpy(), Y_vis.numpy(), error.detach().numpy(), 
                       cmap='magma')
ax4.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print error statistics
print(f"Maximum absolute error: {error.max().item():.2e}")
print(f"Average absolute error: {error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")

# %% [markdown]
# # Test: derivatives + interpolation

# %% [markdown]
# ## 2D derivative: Cheb

# %%
import torch
import matplotlib.pyplot as plt

# Define a test function and its derivatives for comparison
def test_function(x, y):
    return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)

def test_function_dx(x, y):
    return 2*torch.pi*torch.cos(2*torch.pi*x)

def test_function_dy(x, y):
    return -2*torch.pi*torch.sin(2*torch.pi*y)

def test_function_dxx(x, y):
    return -4*torch.pi**2*torch.sin(2*torch.pi*x)

def test_function_dyy(x, y):
    return -4*torch.pi**2*torch.cos(2*torch.pi*y)

# Create interpolant
Ns = [21, 21]  # Number of Chebyshev nodes in each dimension
bases = ["chebyshev", "chebyshev"]  # Basis type
domains = [(0, 1), (0, 1)]  # Domain for the interpolation
interp = SpectralInterpolationND(Ns, bases, domains)

# Evaluate the test function on the interpolation nodes
x_grid, y_grid = interp.mesh
f_grid = test_function(x_grid, y_grid)
interp.values.data = f_grid

# Create a fine grid for visualization
n_vis = 17
x_vis = torch.linspace(0.1, 0.9, n_vis)
y_vis = torch.linspace(0.1, 0.9, n_vis+2)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')

# Test both function values and derivatives
tests = {
    'function': (lambda x, y: interp.interpolate([x, y]), test_function),
    'dx': (lambda x, y: interp.derivative([x, y], k=(1,0)), test_function_dx),
    'dy': (lambda x, y: interp.derivative([x, y], k=(0,1)), test_function_dy),
    'dxx': (lambda x, y: interp.derivative([x, y], k=(2,0)), test_function_dxx),
    'dyy': (lambda x, y: interp.derivative([x, y], k=(0,2)), test_function_dyy)
}

# Plot results for function and each derivative
for name, (spectral_func, exact_func) in tests.items():
    # Compute spectral and exact values
    val_spectral = spectral_func(x_vis, y_vis)
    val_exact = exact_func(X_vis, Y_vis)
    
    # Compute error
    error = torch.abs(val_spectral - val_exact)
    
    # Plotting
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{name.upper()}')
    
    # Plot exact solution
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), val_exact.numpy(), 
                           cmap='viridis')
    ax1.set_title('Exact')
    
    # Plot spectral solution
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X_vis.numpy(), Y_vis.numpy(), val_spectral.detach().numpy(), 
                           cmap='viridis')
    ax2.set_title('Spectral')
    
    # Plot error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), error.detach().numpy(), 
                           cmap='magma')
    ax3.set_title('Absolute Error')
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    print(f"\n{name.upper()} errors:")
    print(f"Maximum absolute error: {error.max().item():.2e}")
    print(f"Average absolute error: {error.mean().item():.2e}")
    print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")

# %% [markdown]
# ## 2D derivative: Fourier

# %%
import torch
import matplotlib.pyplot as plt

# Define a test function and its derivatives for comparison
def test_function(x, y):
    return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)

def test_function_dx(x, y):
    return 2*torch.pi*torch.cos(2*torch.pi*x)

def test_function_dy(x, y):
    return -2*torch.pi*torch.sin(2*torch.pi*y)

def test_function_dxx(x, y):
    return -4*torch.pi**2*torch.sin(2*torch.pi*x)

def test_function_dyy(x, y):
    return -4*torch.pi**2*torch.cos(2*torch.pi*y)

# Create interpolant
Ns = [20, 20]  # Number of Fourier nodes in each dimension
bases = ["fourier", "fourier"]  # Basis type
domains = [(0, 1), (0, 1)]  # Domain for the interpolation
interp = SpectralInterpolationND(Ns, bases, domains)

# Evaluate the test function on the interpolation nodes
x_grid, y_grid = interp.mesh
f_grid = test_function(x_grid, y_grid)
interp.values.data = f_grid

# Create a fine grid for visualization
n_vis = 17
x_vis = torch.linspace(0.1, 0.9, n_vis)
y_vis = torch.linspace(0.1, 0.9, n_vis+2)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')

# Test both function values and derivatives
tests = {
    'function': (lambda x, y: interp.interpolate([x, y]), test_function),
    'dx': (lambda x, y: interp.derivative([x, y], k=(1,0)), test_function_dx),
    'dy': (lambda x, y: interp.derivative([x, y], k=(0,1)), test_function_dy),
    'dxx': (lambda x, y: interp.derivative([x, y], k=(2,0)), test_function_dxx),
    'dyy': (lambda x, y: interp.derivative([x, y], k=(0,2)), test_function_dyy)
}

# Plot results for function and each derivative
for name, (spectral_func, exact_func) in tests.items():
    # Compute spectral and exact values
    val_spectral = spectral_func(x_vis, y_vis)
    val_exact = exact_func(X_vis, Y_vis)
    
    # Compute error
    error = torch.abs(val_spectral - val_exact)
    
    # Plotting
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{name.upper()}')
    
    # Plot exact solution
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), val_exact.numpy(), 
                           cmap='viridis')
    ax1.set_title('Exact')
    
    # Plot spectral solution
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X_vis.numpy(), Y_vis.numpy(), val_spectral.detach().numpy(), 
                           cmap='viridis')
    ax2.set_title('Spectral')
    
    # Plot error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), error.detach().numpy(), 
                           cmap='magma')
    ax3.set_title('Absolute Error')
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    print(f"\n{name.upper()} errors:")
    print(f"Maximum absolute error: {error.max().item():.2e}")
    print(f"Average absolute error: {error.mean().item():.2e}")
    print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")

# %% [markdown]
# # Learning the interpolant with gradient descent

# %% [markdown]
# ## Learning interpolant: Cheb
# Note something important (and perhaps obvious). Using this framework we can decouple the distribution of the model parameters (node locations) from the distribution of the training points.
# For near-optimal polynomial interpolation, we're using Cheb-distributed mode node locations.
# But to get speedy convergence, you need to select your training points in a Cheb-like distribution too. If you don't, convergence is quite slow.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time

# Define test function (same as before)
def test_function(x, y):
    # return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)
    return torch.sin(2*x) + torch.cos(2*y)

# Early stopping with best model tracking
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_values = None

    def __call__(self, val_loss, values):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_values = values.detach().clone()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_values = values.detach().clone()
            self.counter = 0
            
        return self.best_loss

# Create interpolant
Ns = [21, 21]  # Number of nodes in each dimension
bases = ["chebyshev", "chebyshev"]  
domains = [(0, 1), (0, 1)]
interp = SpectralInterpolationND(Ns, bases, domains)

# Generate training grid points
n_train_per_dim = 32
# Cheb distributed
x_train = y_train = torch.cos(torch.pi*torch.linspace(0, 1, n_train_per_dim))*0.5 + 0.5
X_train, Y_train = torch.meshgrid(x_train, y_train, indexing='ij')
f_train = test_function(X_train, Y_train)

# Training parameters
n_epochs = 10000
initial_lr = 0.01
optimizer = optim.Adam(interp.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=50,
    verbose=True,
    min_lr=1e-6
)

early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
loss_fn = nn.MSELoss()

# Lists to store training history
train_losses = []
validation_errors = []
times = [0.0]
learning_rates = []

# Create validation grid
n_val = 50
x_val = torch.linspace(0, 1, n_val)
y_val = torch.linspace(0, 1, n_val)
X_val, Y_val = torch.meshgrid(x_val, y_val, indexing='ij')
f_val_exact = test_function(X_val, Y_val)

# Training loop
print("Starting training...")
print(f"Training points per dimension: {n_train_per_dim}")
print(f"Total training points: {n_train_per_dim**2}")
time_start = time()

for epoch in range(n_epochs):
    epoch_start = time()
    
    # Forward pass
    f_pred = interp([x_train, y_train])
    loss = loss_fn(f_pred, f_train)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record timing and training loss
    times.append(time() - time_start)
    train_losses.append(loss.item())
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Compute validation error
    with torch.no_grad():
        f_val_pred = interp([x_val, y_val])
        val_error = torch.abs(f_val_pred - f_val_exact).mean().item()
        validation_errors.append(val_error)
        
        # Early stopping check with values storage
        early_stopping(val_error, interp.values)
    
    # Learning rate scheduling
    scheduler.step(loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{n_epochs}, Time: {times[-1]:.2f}s, Loss: {loss.item():.2e}, "
              f"Val Error: {val_error:.2e}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        print(f"Best validation error: {early_stopping.best_loss:.2e}")
        break

# Load best values
print("\nLoading best model...")
interp.values.data = early_stopping.best_values
time_elapsed = time() - time_start
print(f"Training completed in {time_elapsed:.2f} seconds")

# Plot training history with timing
plt.figure(figsize=(15, 8))

plt.subplot(221)
plt.semilogy(times[1:], train_losses)
plt.title('Training Loss vs Time')
plt.xlabel('Time (s)')
plt.ylabel('MSE Loss')

plt.subplot(222)
plt.semilogy(range(len(train_losses)), train_losses)
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(223)
plt.semilogy(range(len(validation_errors)), validation_errors)
plt.title('Validation Error vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')

plt.subplot(224)
plt.semilogy(range(len(learning_rates)), learning_rates)
plt.title('Learning Rate vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# Evaluate final (best) model
with torch.no_grad():
    f_val_pred = interp([x_val, y_val])
    final_error = torch.abs(f_val_pred - f_val_exact)

# Visualize final result
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_exact.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot learned solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_pred.numpy(), 
                       cmap='viridis')
ax2.set_title('Learned (Best Model)')

# Plot error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_val.numpy(), Y_val.numpy(), final_error.numpy(), 
                       cmap='magma')
ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print final error statistics
print("\nFinal Error Statistics (Best Model):")
print(f"Maximum absolute error: {final_error.max().item():.2e}")
print(f"Average absolute error: {final_error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((final_error**2).mean()).item():.2e}")

# %% [markdown]
# ### Sanity check. Learning interpolant with Cheb, but use equispaced training points.
# The convergence is really slow!

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time

# Define test function (same as before)
def test_function(x, y):
    # return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)
    return torch.sin(2*x) + torch.cos(2*y)

# Early stopping with best model tracking
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_values = None

    def __call__(self, val_loss, values):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_values = values.detach().clone()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_values = values.detach().clone()
            self.counter = 0
            
        return self.best_loss

# Create interpolant
Ns = [21, 21]  # Number of nodes in each dimension
bases = ["chebyshev", "chebyshev"]  
domains = [(0, 1), (0, 1)]
interp = SpectralInterpolationND(Ns, bases, domains)

# Generate training grid points
n_train_per_dim = 32
# Equispaced
x_train = torch.linspace(0, 1, n_train_per_dim)
y_train = torch.linspace(0, 1, n_train_per_dim)
X_train, Y_train = torch.meshgrid(x_train, y_train, indexing='ij')
f_train = test_function(X_train, Y_train)

# Training parameters
n_epochs = 10000
initial_lr = 0.01
optimizer = optim.Adam(interp.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=50,
    verbose=True,
    min_lr=1e-6
)

early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
loss_fn = nn.MSELoss()

# Lists to store training history
train_losses = []
validation_errors = []
times = [0.0]
learning_rates = []

# Create validation grid
n_val = 50
x_val = torch.linspace(0, 1, n_val)
y_val = torch.linspace(0, 1, n_val)
X_val, Y_val = torch.meshgrid(x_val, y_val, indexing='ij')
f_val_exact = test_function(X_val, Y_val)

# Training loop
print("Starting training...")
print(f"Training points per dimension: {n_train_per_dim}")
print(f"Total training points: {n_train_per_dim**2}")
time_start = time()

for epoch in range(n_epochs):
    epoch_start = time()
    
    # Forward pass
    f_pred = interp([x_train, y_train])
    loss = loss_fn(f_pred, f_train)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record timing and training loss
    times.append(time() - time_start)
    train_losses.append(loss.item())
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Compute validation error
    with torch.no_grad():
        f_val_pred = interp([x_val, y_val])
        val_error = torch.abs(f_val_pred - f_val_exact).mean().item()
        validation_errors.append(val_error)
        
        # Early stopping check with values storage
        early_stopping(val_error, interp.values)
    
    # Learning rate scheduling
    scheduler.step(loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{n_epochs}, Time: {times[-1]:.2f}s, Loss: {loss.item():.2e}, "
              f"Val Error: {val_error:.2e}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        print(f"Best validation error: {early_stopping.best_loss:.2e}")
        break

# Load best values
print("\nLoading best model...")
interp.values.data = early_stopping.best_values
time_elapsed = time() - time_start
print(f"Training completed in {time_elapsed:.2f} seconds")

# Plot training history with timing
plt.figure(figsize=(15, 8))

plt.subplot(221)
plt.semilogy(times[1:], train_losses)
plt.title('Training Loss vs Time')
plt.xlabel('Time (s)')
plt.ylabel('MSE Loss')

plt.subplot(222)
plt.semilogy(range(len(train_losses)), train_losses)
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(223)
plt.semilogy(range(len(validation_errors)), validation_errors)
plt.title('Validation Error vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')

plt.subplot(224)
plt.semilogy(range(len(learning_rates)), learning_rates)
plt.title('Learning Rate vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# Plot training history with timing
plt.figure(figsize=(15, 8))

plt.subplot(221)
plt.plot(times[1:], train_losses)
plt.title('Training Loss vs Time')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('MSE Loss')

plt.subplot(222)
plt.plot(range(len(train_losses)), train_losses)
plt.title('Training Loss vs Epoch')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(223)
plt.plot(range(len(validation_errors)), validation_errors)
plt.title('Validation Error vs Epoch')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')

plt.subplot(224)
plt.plot(range(len(learning_rates)), learning_rates)
plt.title('Learning Rate vs Epoch')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# Evaluate final (best) model
with torch.no_grad():
    f_val_pred = interp([x_val, y_val])
    final_error = torch.abs(f_val_pred - f_val_exact)

# Visualize final result
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_exact.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot learned solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_pred.numpy(), 
                       cmap='viridis')
ax2.set_title('Learned (Best Model)')

# Plot error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_val.numpy(), Y_val.numpy(), final_error.numpy(), 
                       cmap='magma')
ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print final error statistics
print("\nFinal Error Statistics (Best Model):")
print(f"Maximum absolute error: {final_error.max().item():.2e}")
print(f"Average absolute error: {final_error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((final_error**2).mean()).item():.2e}")

# %% [markdown]
# ## Learning interpolant: Fourier

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time

# Define test function (same as before)
def test_function(x, y):
    return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)

# Early stopping with best model tracking
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_values = None

    def __call__(self, val_loss, values):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_values = values.detach().clone()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_values = values.detach().clone()
            self.counter = 0
            
        return self.best_loss

# Create interpolant
Ns = [20, 20]  # Number of nodes in each dimension
bases = ["fourier", "fourier"]  
domains = [(0, 1), (0, 1)]
interp = SpectralInterpolationND(Ns, bases, domains)

# Generate training grid points
n_train_per_dim = 32
x_train = torch.linspace(0, 1, n_train_per_dim)
y_train = torch.linspace(0, 1, n_train_per_dim)
X_train, Y_train = torch.meshgrid(x_train, y_train, indexing='ij')
f_train = test_function(X_train, Y_train)

# Training parameters
n_epochs = 10000
initial_lr = 0.01
optimizer = optim.Adam(interp.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=50,
    verbose=True,
    min_lr=1e-6
)

early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
loss_fn = nn.MSELoss()

# Lists to store training history
train_losses = []
validation_errors = []
times = [0.0]
learning_rates = []

# Create validation grid
n_val = 50
# Equispaced
x_val = torch.linspace(0, 1, n_val)
y_val = torch.linspace(0, 1, n_val)
X_val, Y_val = torch.meshgrid(x_val, y_val, indexing='ij')
f_val_exact = test_function(X_val, Y_val)

# Training loop
print("Starting training...")
print(f"Training points per dimension: {n_train_per_dim}")
print(f"Total training points: {n_train_per_dim**2}")
time_start = time()

for epoch in range(n_epochs):
    epoch_start = time()
    
    # Forward pass
    f_pred = interp([x_train, y_train])
    loss = loss_fn(f_pred, f_train)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record timing and training loss
    times.append(time() - time_start)
    train_losses.append(loss.item())
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Compute validation error
    with torch.no_grad():
        f_val_pred = interp([x_val, y_val])
        val_error = torch.abs(f_val_pred - f_val_exact).mean().item()
        validation_errors.append(val_error)
        
        # Early stopping check with values storage
        early_stopping(val_error, interp.values)
    
    # Learning rate scheduling
    scheduler.step(loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{n_epochs}, Time: {times[-1]:.2f}s, Loss: {loss.item():.2e}, "
              f"Val Error: {val_error:.2e}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        print(f"Best validation error: {early_stopping.best_loss:.2e}")
        break

# Load best values
print("\nLoading best model...")
interp.values.data = early_stopping.best_values
time_elapsed = time() - time_start
print(f"Training completed in {time_elapsed:.2f} seconds")

# Plot training history with timing
plt.figure(figsize=(15, 8))

plt.subplot(221)
plt.semilogy(times[1:], train_losses)
plt.title('Training Loss vs Time')
plt.xlabel('Time (s)')
plt.ylabel('MSE Loss')

plt.subplot(222)
plt.semilogy(range(len(train_losses)), train_losses)
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(223)
plt.semilogy(range(len(validation_errors)), validation_errors)
plt.title('Validation Error vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')

plt.subplot(224)
plt.semilogy(range(len(learning_rates)), learning_rates)
plt.title('Learning Rate vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# Evaluate final (best) model
with torch.no_grad():
    f_val_pred = interp([x_val, y_val])
    final_error = torch.abs(f_val_pred - f_val_exact)

# Visualize final result
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_exact.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot learned solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_pred.numpy(), 
                       cmap='viridis')
ax2.set_title('Learned (Best Model)')

# Plot error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_val.numpy(), Y_val.numpy(), final_error.numpy(), 
                       cmap='magma')
ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print final error statistics
print("\nFinal Error Statistics (Best Model):")
print(f"Maximum absolute error: {final_error.max().item():.2e}")
print(f"Average absolute error: {final_error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((final_error**2).mean()).item():.2e}")

# %% [markdown]
# ## Learning interpolant: MLP

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time

class MLP(nn.Module):
    def __init__(self, n_dim=1, hidden_dim=32, activation=torch.tanh):
        """
        2-layer MLP that maps (B, n_dim) -> (B, 1)
        Args:
            hidden_dim: Dimension of hidden layer
            activation: Activation function to use (default: tanh)
        """
        super().__init__()
        self.n_dim = n_dim
        self.activation = activation
        self.fc1 = nn.Linear(self.n_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass of the network
        Args:
            x: Input tensor of shape (B, n_dim)
        Returns:
            Output tensor of shape (B, 1)
        """
        assert x.shape[-1] == self.n_dim, f"Input shape must be (..., {self.n_dim})"
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Define test function (same as before)
def test_function(x, y):
    return torch.sin(2*torch.pi*x) + torch.cos(2*torch.pi*y)

# Early stopping with best model tracking
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            
        return self.best_loss

# Create MLP
model = MLP(n_dim=2, hidden_dim=64)

# Generate training grid points
n_train_per_dim = 32
x_train = torch.linspace(0, 1, n_train_per_dim)
y_train = torch.linspace(0, 1, n_train_per_dim)
X_train, Y_train = torch.meshgrid(x_train, y_train, indexing='ij')

# Reshape inputs for MLP
train_inputs = torch.stack([X_train.flatten(), Y_train.flatten()], dim=1)  # (N, 2)
f_train = test_function(train_inputs[:, 0], train_inputs[:, 1]).unsqueeze(1)  # (N, 1)

# Training parameters
n_epochs = 10000
initial_lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=50,
    verbose=True,
    min_lr=1e-6
)

early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
loss_fn = nn.MSELoss()

# Lists to store training history
train_losses = []
validation_errors = []
times = [0.0]
learning_rates = []

# Create validation grid
n_val = 50
# Cheb distributed
x_train = y_train = torch.cos(torch.pi*torch.linspace(0, 1, n_train_per_dim))*0.5 + 0.5
X_val, Y_val = torch.meshgrid(x_val, y_val, indexing='ij')
val_inputs = torch.stack([X_val.flatten(), Y_val.flatten()], dim=1)  # (N, 2)
f_val_exact = test_function(val_inputs[:, 0], val_inputs[:, 1]).reshape(n_val, n_val)

# Training loop
print("Starting training...")
print(f"Training points per dimension: {n_train_per_dim}")
print(f"Total training points: {n_train_per_dim**2}")
time_start = time()

for epoch in range(n_epochs):
    epoch_start = time()
    
    # Forward pass
    f_pred = model(train_inputs)
    loss = loss_fn(f_pred, f_train)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record timing and training loss
    times.append(time() - time_start)
    train_losses.append(loss.item())
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Compute validation error
    with torch.no_grad():
        f_val_pred = model(val_inputs).reshape(n_val, n_val)
        val_error = torch.abs(f_val_pred - f_val_exact).mean().item()
        validation_errors.append(val_error)
        
        # Early stopping check
        early_stopping(val_error, model)
    
    # Learning rate scheduling
    scheduler.step(loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{n_epochs}, Time: {times[-1]:.2f}s, Loss: {loss.item():.2e}, "
              f"Val Error: {val_error:.2e}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        print(f"Best validation error: {early_stopping.best_loss:.2e}")
        break

# Load best model
print("\nLoading best model...")
model.load_state_dict(early_stopping.best_state)
time_elapsed = time() - time_start
print(f"Training completed in {time_elapsed:.2f} seconds")

# Plot training history
plt.figure(figsize=(15, 8))

plt.subplot(221)
plt.semilogy(times[1:], train_losses)
plt.title('Training Loss vs Time')
plt.xlabel('Time (s)')
plt.ylabel('MSE Loss')

plt.subplot(222)
plt.semilogy(range(len(train_losses)), train_losses)
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(223)
plt.semilogy(range(len(validation_errors)), validation_errors)
plt.title('Validation Error vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')

plt.subplot(224)
plt.semilogy(range(len(learning_rates)), learning_rates)
plt.title('Learning Rate vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()


# Plot training history (log)
plt.figure(figsize=(15, 8))

plt.subplot(221)
plt.plot(times[1:], train_losses)
plt.title('Training Loss vs Time')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('MSE Loss')

plt.subplot(222)
plt.plot(range(len(train_losses)), train_losses)
plt.title('Training Loss vs Epoch')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(223)
plt.plot(range(len(validation_errors)), validation_errors)
plt.title('Validation Error vs Epoch')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')

plt.subplot(224)
plt.plot(range(len(learning_rates)), learning_rates)
plt.title('Learning Rate vs Epoch')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# Evaluate final (best) model
with torch.no_grad():
    f_val_pred = model(val_inputs).reshape(n_val, n_val)
    final_error = torch.abs(f_val_pred - f_val_exact)

# Visualize final result
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_exact.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot learned solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_val.numpy(), Y_val.numpy(), f_val_pred.numpy(), 
                       cmap='viridis')
ax2.set_title('Learned (Best Model)')

# Plot error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_val.numpy(), Y_val.numpy(), final_error.numpy(), 
                       cmap='magma')
ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print final error statistics
print("\nFinal Error Statistics (Best Model):")
print(f"Maximum absolute error: {final_error.max().item():.2e}")
print(f"Average absolute error: {final_error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((final_error**2).mean()).item():.2e}")

# %%
