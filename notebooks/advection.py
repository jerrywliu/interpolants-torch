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


# %% [markdown]
# # Models

# %% [markdown]
# ## ND polynomial interpolant

# %%
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
# ## MLP

# %%
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


# %% [markdown]
# # PDE

# %% [markdown]
# We look at the 1D advection equation (1 space + 1 time variable, both of which we treat spectrally):
# $$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0, \quad t \in [0,1], \quad x \in [0,2\pi]$$
# $$u(0,x) = \sin(x)$$
#
# Solution:
# $$u(t,x) = \sin(x - ct)$$
#
# Below, we use $c = 8$ for simplicity. Note that the test case in Wang et al. (2023) uses $c = 80$.
#
# Note that we use periodic boundary conditions in x since displacement of a sine wave should be periodic.

# %% [markdown]
#

# %%
def compute_pde_loss(model, t_grid, x_grid, ic_x_grid=None, ic_weight=1.0, c=80, **kwargs):
    """
    Compute loss for advection equation u_t + c*u_x = 0
    
    Args:
        model: SpectralInterpolationND or MLP model
        t_grid: time points of shape (nt,)
        x_grid: space points of shape (nx,)
        ic_x_grid: space points for initial condition of shape (nx_ic,). If None, uses x_grid
        ic_weight: weight for initial condition loss
        c: advection speed
        
    Returns:
        total_loss: combined PDE and IC loss
        pde_residual: residual at collocation points
        ic_residual: residual at initial condition points
    """
    if ic_x_grid is None:
        ic_x_grid = x_grid
        
    n_t, n_x = len(t_grid), len(x_grid)
    
    # Compute solution and derivatives based on model type
    if isinstance(model, SpectralInterpolationND):
        # Spectral model takes separate coordinates
        u = model.interpolate([t_grid, x_grid])
        u_t = model.derivative([t_grid, x_grid], k=(1,0))
        u_x = model.derivative([t_grid, x_grid], k=(0,1))
        
        # Compute initial condition on ic_x_grid
        u_ic = model.interpolate([torch.tensor([0.0]), ic_x_grid])[0]
    else:
        # Create mesh grid points for PDE
        T, X = torch.meshgrid(t_grid, x_grid, indexing='ij')
        points = torch.stack([T.flatten(), X.flatten()], dim=1)
        points.requires_grad_(True)
        
        # Forward pass and compute gradients
        u = model(points).reshape(n_t, n_x)
        grads = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
        u_t = grads[:, 0].reshape(n_t, n_x)
        u_x = grads[:, 1].reshape(n_t, n_x)
        
        # Compute initial condition
        ic_points = torch.stack([torch.zeros_like(ic_x_grid), ic_x_grid], dim=1)
        u_ic = model(ic_points).squeeze()
    
    # PDE residual: u_t + c*u_x = 0
    pde_residual = u_t + c*u_x
    pde_loss = torch.mean(pde_residual**2)
    
    # Initial condition: u(0,x) = sin(x)
    ic_residual = u_ic - torch.sin(ic_x_grid)
    ic_loss = torch.mean(ic_residual**2)
    
    total_loss = pde_loss + ic_loss * ic_weight
    return total_loss, pde_residual, ic_residual


# %%
def get_training_grids(model=None, n_t=32, n_x=32, n_ic=32, t_max=1):
    """
    Get training grids either from model or create new ones
    
    Args:
        model: if SpectralInterpolationND, use its nodes. If None or MLP, create new grids
        n_t: number of time points (if not using model grid)
        n_x: number of space points (if not using model grid)
        n_ic: number of IC points (if None, use n_x)
    
    Returns:
        t_grid: time points
        x_grid: space points
        ic_x_grid: initial condition points
    """
    if model is None:
        # Cheb
        # t_grid = torch.cos(torch.linspace(0, np.pi, n_t))*(t_max/2) + (t_max/2)
        t_grid = torch.cos(torch.rand(n_t)*np.pi)*(t_max/2) + (t_max/2)
        # Fourier
        x_grid = torch.linspace(0, 2*np.pi, n_x+1)[:-1]
        ic_x_grid = torch.linspace(0, 2*np.pi, n_ic+1)[:-1]
    if isinstance(model, SpectralInterpolationND):
        t_grid = model.nodes[0]
        x_grid = model.nodes[1]
        ic_x_grid = x_grid  # Could be different if desired
    else:
        t_grid = torch.cos(torch.linspace(0, 2*np.pi, n_t))*(t_max/2) + (t_max/2)
        x_grid = torch.linspace(0, 2*np.pi, n_x+1)[:-1]
        ic_x_grid = torch.linspace(0, 2*np.pi, n_ic+1)[:-1]
        
    return t_grid, x_grid, ic_x_grid


# %%
def test_solution_interpolation_and_pde(c=80):
    """
    Test if we can:
    1. Interpolate the true solution using SpectralInterpolationND
    2. Verify the true solution satisfies the PDE
    """
    # Setup grids
    n_t, n_x = 2*c+1, 2*c
    t_grid, x_grid, _ = get_training_grids(None, n_t=n_t, n_x=n_x)
    
    # Create spectral interpolation model
    model = SpectralInterpolationND(
        Ns=[c+1, c],
        bases=['chebyshev', 'fourier'],
        domains=[[0, 1], [0, 2*np.pi]]
    )
    
    # Set values to exact solution
    T, X = torch.meshgrid(model.nodes[0], model.nodes[1], indexing='ij')
    true_solution = torch.sin(X - c*T)
    model.values = torch.nn.Parameter(true_solution)
    
    # Test 1: Interpolation accuracy
    print("Testing interpolation accuracy...")
    # Create fine grid for testing
    n_fine = 200
    t_fine = torch.linspace(0, 1, n_fine)
    x_fine = torch.linspace(0, 2*np.pi, n_fine)
    
    # Compute interpolated and exact solutions
    interp_solution = model.interpolate([t_fine, x_fine]).detach()
    T_fine, X_fine = torch.meshgrid(t_fine, x_fine, indexing='ij')
    exact_solution = torch.sin(X_fine - c*T_fine)
    
    # Compute interpolation error
    interp_error = torch.mean((interp_solution - exact_solution)**2).sqrt()
    print(f"L2 interpolation error: {interp_error:.2e}")
    
    # Test 2: PDE satisfaction
    print("\nTesting PDE satisfaction...")
    # Compute PDE residual using our loss function
    total_loss, pde_residual, ic_residual = compute_pde_loss(
        model, t_grid, x_grid, ic_weight=1.0, c=c
    )
    pde_residual = pde_residual.detach()
    
    # Compute statistics of the residual
    mean_residual = torch.mean(torch.abs(pde_residual))
    max_residual = torch.max(torch.abs(pde_residual))
    print(f"Mean absolute PDE residual: {mean_residual:.2e}")
    print(f"Max absolute PDE residual: {max_residual:.2e}")
    print(f"Initial condition residual: {torch.mean(ic_residual**2).sqrt():.2e}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot interpolated solution
    plt.subplot(131)
    plt.imshow(interp_solution.T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Interpolated Solution')
    plt.xlabel('t')
    plt.ylabel('x')
    
    # Plot interpolation error
    plt.subplot(132)
    plt.imshow(torch.abs(interp_solution - exact_solution).T, 
              aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
    plt.colorbar(label='|error|')
    plt.title('Interpolation Error')
    plt.xlabel('t')
    plt.ylabel('x')
    
    # Plot PDE residual
    plt.subplot(133)
    plt.imshow(torch.abs(pde_residual).T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='magma')
    plt.colorbar(label='|residual|')
    plt.title('PDE Residual')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'interp_error': interp_error.item(),
        'mean_residual': mean_residual.item(),
        'max_residual': max_residual.item(),
        'ic_error': torch.mean(ic_residual**2).sqrt().item()
    }


# %%
results = test_solution_interpolation_and_pde(c=80)


# %% [markdown]
# ## Learn the interpolant

# %%
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


# %%
def train_interpolation(model, n_epochs=1000, lr=1e-3, c=80, plot_every=100):
    """
    Train model to interpolate the true solution u(t,x) = sin(x - ct)
    while monitoring interpolation and PDE errors
    """
    # Setup training grid
    t_grid, x_grid, _ = get_training_grids(None, n_t=2*c+1, n_x=2*c)
    T_train, X_train = torch.meshgrid(t_grid, x_grid, indexing='ij')
    train_solution = torch.sin(X_train - c*T_train)
    
    # Setup evaluation grid (finer)
    n_eval = 200
    t_eval = torch.linspace(0, 1, n_eval)
    x_eval = torch.linspace(0, 2*np.pi, n_eval)
    T_eval, X_eval = torch.meshgrid(t_eval, x_eval, indexing='ij')
    eval_solution = torch.sin(X_eval - c*T_eval)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
    
    # Training history
    history = {
        'interp_loss': [], 'eval_error': [], 'pde_residual': [], 
        'learning_rates': []
    }
    
    print("Starting training...")
    time_start = time()
    
    for epoch in tqdm(range(n_epochs)):
        # Training step
        optimizer.zero_grad()
        
        # Compute interpolation on training grid
        pred_train = model.interpolate([t_grid, x_grid])
        interp_loss = torch.mean((pred_train - train_solution)**2)
        
        # Backward and optimize
        interp_loss.backward()
        optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            # Interpolation error on eval grid
            pred_eval = model.interpolate([t_eval, x_eval])
            eval_error = torch.mean((pred_eval - eval_solution)**2).sqrt()
            
            # Check PDE satisfaction
            _, pde_residual, _ = compute_pde_loss(model, t_grid, x_grid, c=c)
            pde_error = torch.mean(pde_residual**2).sqrt()
        
        # Update history
        history['interp_loss'].append(interp_loss.item())
        history['eval_error'].append(eval_error.item())
        history['pde_residual'].append(pde_error.item())
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling and early stopping
        scheduler.step(interp_loss)
        early_stopping(eval_error, model)
        
        # Print and plot progress
        if (epoch + 1) % plot_every == 0:
            current_time = time() - time_start
            print(f"\nEpoch {epoch+1}/{n_epochs}, Time: {current_time:.2f}s")
            print(f"Interpolation Loss: {interp_loss:.2e}")
            print(f"Evaluation Error: {eval_error:.2e}")
            print(f"PDE Residual: {pde_error:.2e}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Visualization
            plt.figure(figsize=(20, 5))
            
            # Plot predicted solution
            plt.subplot(141)
            plt.imshow(pred_eval.T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Predicted Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot exact solution
            plt.subplot(142)
            plt.imshow(eval_solution.T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Exact Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot interpolation error
            plt.subplot(143)
            plt.imshow(torch.abs(pred_eval - eval_solution).T, 
                      aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|error|')
            plt.title('Interpolation Error')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot PDE residual
            plt.subplot(144)
            plt.imshow(torch.abs(pde_residual).T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|residual|')
            plt.title('PDE Residual')
            plt.xlabel('t')
            plt.ylabel('x')
            
            plt.tight_layout()
            plt.show()
            
            # Plot training curves
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.semilogy(history['interp_loss'], label='Train')
            plt.semilogy(history['eval_error'], label='Eval')
            plt.title('Interpolation Error vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.legend()
            
            plt.subplot(132)
            plt.semilogy(history['pde_residual'])
            plt.title('PDE Residual vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Residual')
            
            plt.subplot(133)
            plt.semilogy(history['learning_rates'])
            plt.title('Learning Rate vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            
            plt.tight_layout()
            plt.show()
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(early_stopping.best_state)
    time_elapsed = time() - time_start
    
    print(f"\nTraining completed in {time_elapsed:.2f} seconds")
    print(f"Best interpolation error: {min(history['eval_error']):.2e}")
    print(f"Final PDE residual: {history['pde_residual'][-1]:.2e}")
    
    return history


# %%
# Create model
c = 80
model = SpectralInterpolationND(
    Ns=[c+1, c],  # Number of points in t and x
    bases=['chebyshev', 'fourier'],  # Basis types
    domains=[[0, 1], [0, 2*np.pi]]  # Domain bounds
)

# Train on interpolation
history = train_interpolation(
    model,
    n_epochs=10000,
    lr=1e-2,
    c=c,
    plot_every=50,
)


# %% [markdown]
# ## PINN training

# %%
def train_advection_pde(model, n_epochs=1000, lr=1e-3, ic_weight=1.0, c=80, plot_every=100):
    """
    Train model to solve the advection equation using clean training framework
    """
    # Setup training grid
    t_grid, x_grid, ic_x_grid = get_training_grids(None, n_t=4*c+1, n_x=4*c, n_ic=4*c)
    
    # Setup evaluation grid (finer)
    n_eval = 200
    t_eval = torch.linspace(0, 1, n_eval)
    x_eval = torch.linspace(0, 2*np.pi, n_eval)
    T_eval, X_eval = torch.meshgrid(t_eval, x_eval, indexing='ij')
    eval_solution = torch.sin(X_eval - c*T_eval)  # True solution for comparison
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-3
    )
    early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
    
    # Training history
    history = {
        'total_loss': [], 'pde_residual': [], 'ic_error': [], 
        'l2_error': [], 'learning_rates': []
    }
    
    print("Starting training...")
    time_start = time()
    
    for epoch in tqdm(range(n_epochs)):
        # Training step
        optimizer.zero_grad()
        
        # Compute PDE and IC losses
        total_loss, pde_residual, ic_residual = compute_pde_loss(
            model, t_grid, x_grid, ic_x_grid, ic_weight=ic_weight, c=c
        )
        pde_residual = pde_residual.detach()
        
        # Backward and optimize
        total_loss.backward()
        optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            # Solution error on evaluation grid
            pred_eval = model.interpolate([t_eval, x_eval])
            l2_error = torch.mean((pred_eval - eval_solution)**2).sqrt()
            
            # Mean residuals
            mean_pde_residual = torch.mean(pde_residual**2).sqrt()
            mean_ic_error = torch.mean(ic_residual**2).sqrt()
        
        # Update history
        history['total_loss'].append(total_loss.item())
        history['pde_residual'].append(mean_pde_residual.item())
        history['ic_error'].append(mean_ic_error.item())
        history['l2_error'].append(l2_error.item())
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling and early stopping
        scheduler.step(total_loss)
        early_stopping(l2_error, model)
        
        # Print and plot progress
        if (epoch + 1) % plot_every == 0:
            current_time = time() - time_start
            print(f"\nEpoch {epoch+1}/{n_epochs}, Time: {current_time:.2f}s")
            print(f"Total Loss: {total_loss:.2e}")
            print(f"PDE Residual: {mean_pde_residual:.2e}")
            print(f"IC Error: {mean_ic_error:.2e}")
            print(f"L2 Error: {l2_error:.2e}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Visualization
            plt.figure(figsize=(20, 5))
            
            # Plot predicted solution
            plt.subplot(141)
            plt.imshow(pred_eval.T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Predicted Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot exact solution
            plt.subplot(142)
            plt.imshow(eval_solution.T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Exact Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot solution error
            plt.subplot(143)
            plt.imshow(torch.abs(pred_eval - eval_solution).T, 
                      aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|error|')
            plt.title('Solution Error')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot PDE residual
            plt.subplot(144)
            plt.imshow(torch.abs(pde_residual).T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|residual|')
            plt.title('PDE Residual')
            plt.xlabel('t')
            plt.ylabel('x')
            
            plt.tight_layout()
            plt.show()
            
            # Plot training curves
            plt.figure(figsize=(20, 5))
            
            plt.subplot(141)
            plt.semilogy(history['total_loss'])
            plt.title('Total Loss vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(142)
            plt.semilogy(history['pde_residual'], label='PDE')
            plt.semilogy(history['ic_error'], label='IC')
            plt.title('Residuals vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.legend()
            
            plt.subplot(143)
            plt.semilogy(history['l2_error'])
            plt.title('L2 Error vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            
            plt.subplot(144)
            plt.semilogy(history['learning_rates'])
            plt.title('Learning Rate vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            
            plt.tight_layout()
            plt.show()
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(early_stopping.best_state)
    time_elapsed = time() - time_start
    
    print(f"\nTraining completed in {time_elapsed:.2f} seconds")
    print(f"Best L2 error: {min(history['l2_error']):.2e}")
    print(f"Final PDE residual: {history['pde_residual'][-1]:.2e}")
    print(f"Final IC error: {history['ic_error'][-1]:.2e}")
    
    return history


# %%
c = 8
model = SpectralInterpolationND(
    Ns=[2*c+1, 2*c],
    bases=['chebyshev', 'fourier'],
    domains=[[0, 1], [0, 2*np.pi]]
)

history = train_advection_pde(
    model,
    n_epochs=10000,
    lr=1e-3,
    ic_weight=1.0,
    c=c,
    plot_every=50,
)


# %%

# %% [markdown]
# # Hacking below

# %%
def train_one_epoch(model, optimizer, t_grid, x_grid, ic_x_grid, ic_weight=1.0, c=80, temporal_eps=0):
    """
    Train model for one epoch on given grids
    
    Args:
        model: SpectralInterpolationND or MLP model
        optimizer: PyTorch optimizer
        t_grid: time points (nt,)
        x_grid: space points (nx,)
        ic_x_grid: initial condition points (nx_ic,)
        ic_weight: weight for initial condition loss
        c: advection speed
        
    Returns:
        dict containing loss values and residuals
    """
    optimizer.zero_grad()

    # Compute loss
    loss, pde_residual, ic_residual = compute_pde_loss(
        model,
        t_grid,
        x_grid,
        ic_x_grid,
        ic_weight=ic_weight,
        c=c,
        temporal_eps=temporal_eps,
    )

    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'pde_residual': torch.mean(pde_residual**2).item(),
        'ic_residual': torch.mean(ic_residual**2).item()
    }


# %%
def evaluate_model(model, t_eval, x_eval, c=80):
    """
    Evaluate model against exact solution
    
    Args:
        model: SpectralInterpolationND or MLP model
        t_eval: evaluation time points
        x_eval: evaluation space points
        c: advection speed
        
    Returns:
        dict containing predicted and exact solutions and error
    """
    with torch.no_grad():
        if isinstance(model, SpectralInterpolationND):
            u_pred = model.interpolate([t_eval, x_eval])
        else:
            T, X = torch.meshgrid(t_eval, x_eval, indexing='ij')
            points = torch.stack([T.flatten(), X.flatten()], dim=1)
            u_pred = model(points).reshape(T.shape)
            
        u_exact = torch.sin(x_eval[None, :] - c*t_eval[:, None])
        l2_error = torch.mean((u_pred - u_exact)**2).sqrt()
        
    return {
        'u_pred': u_pred,
        'u_exact': u_exact,
        'l2_error': l2_error.item()
    }


# %%
def train_advection(model, n_epochs=1000, lr=1e-3, ic_weight=1.0, c=80, plot_every=100, n=80):
    """
    Train model to solve the advection equation
    """
    # Get training and evaluation grids
    t_grid, x_grid, ic_x_grid = get_training_grids(
        None,
        n_t=n+1,
        n_x=n,
        n_ic=n,
    )
    n_eval = 100
    # t_eval = torch.linspace(0, 1, n_eval)
    # x_eval = torch.linspace(0, 2*np.pi, n_eval+1)[:-1]
    t_eval = t_grid
    x_eval = x_grid
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
    
    # Training history
    history = {
        'loss': [], 'pde_residual': [], 'ic_residual': [], 
        'l2_error': [], 'learning_rates': [],
    }
    
    print("Starting training...")
    time_start = time()
    
    for epoch in tqdm(range(n_epochs)):
        
        # Get training and evaluation grids
        t_grid, x_grid, ic_x_grid = get_training_grids(
            None,
            n_t=n+1,
            n_x=n,
            n_ic=n
        )

        # Train one epoch
        epoch_results = train_one_epoch(
            model, optimizer, t_grid, x_grid, ic_x_grid, ic_weight, c
        )
        
        # Evaluate
        eval_results = evaluate_model(model, t_eval, x_eval, c)
        
        # Update history
        current_time = time() - time_start
        history['loss'].append(epoch_results['loss'])
        history['pde_residual'].append(epoch_results['pde_residual'])
        history['ic_residual'].append(epoch_results['ic_residual'])
        history['l2_error'].append(eval_results['l2_error'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling and early stopping
        scheduler.step(epoch_results['loss'])
        early_stopping(eval_results['l2_error'], model)
        
        # Print and plot progress
        if (epoch + 1) % plot_every == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}, Time: {current_time:.2f}s")
            print(f"Loss: {epoch_results['loss']:.2e}, L2 Error: {eval_results['l2_error']:.2e}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Plot current solution
            plt.figure(figsize=(15, 5))
            
            # Plot predicted solution
            plt.subplot(131)
            plt.imshow(eval_results['u_pred'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Predicted Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot exact solution
            plt.subplot(132)
            plt.imshow(eval_results['u_exact'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Exact Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot error
            plt.subplot(133)
            plt.imshow(torch.abs(eval_results['u_pred'] - eval_results['u_exact']).T, 
                      aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|error|')
            plt.title('Absolute Error')
            plt.xlabel('t')
            plt.ylabel('x')
            
            plt.tight_layout()
            plt.show()
        
        # if early_stopping.early_stop:
        #     print(f"\nEarly stopping triggered at epoch {epoch}")
        #     break
    
    # Load best model
    model.load_state_dict(early_stopping.best_state)
    time_elapsed = time() - time_start
    
    # Final plotting - Training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.semilogy(range(len(history['loss'])), history['loss'])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(132)
    plt.semilogy(range(len(history['l2_error'])), history['l2_error'])
    plt.title('L2 Error vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')
    
    plt.subplot(133)
    plt.semilogy(range(len(history['learning_rates'])), history['learning_rates'])
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.show()
    
    # Final solution evaluation and plot
    final_eval = evaluate_model(model, t_eval, x_eval, c)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(final_eval['u_pred'].T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Final Predicted Solution')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.subplot(132)
    plt.imshow(final_eval['u_exact'].T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Exact Solution')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.subplot(133)
    plt.imshow(torch.abs(final_eval['u_pred'] - final_eval['u_exact']).T, 
              aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
    plt.colorbar(label='|error|')
    plt.title('Final Absolute Error')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.tight_layout()
    plt.show()

    print(f"\nTraining completed in {time_elapsed:.2f} seconds")
    print(f"Best L2 error: {min(history['l2_error']):.2e}")

    return history


# %%
import torch
import numpy as np
from time import time

# Set random seed for reproducibility
torch.manual_seed(42)

# Spectral model
Ns = [9, 8]  # 21 (odd) Chebyshev points in time, 20 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

# MLP model
mlp_model = MLP(n_dim=2, hidden_dim=64, activation=torch.tanh)

# Train both models
print("Training Spectral Model...")
history_spectral = train_advection(
    spectral_model,
    n_epochs=10000,
    lr=0.01,
    ic_weight=10.0,
    c=8,
    plot_every=100,
    n=8,
)

print("\nTraining MLP Model...")
history_mlp = train_advection(
    mlp_model,
    n_epochs=1000,
    lr=0.01,
    ic_weight=10.0,
    c=8,
    plot_every=100
)

# Compare final L2 errors
print("\nFinal Results:")
print(f"Spectral Model - Best L2 error: {min(history_spectral['l2_error']):.2e}")
print(f"MLP Model - Best L2 error: {min(history_mlp['l2_error']):.2e}")

# Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.semilogy(range(len(history_spectral['loss'])), history_spectral['loss'], label='Spectral')
plt.semilogy(range(len(history_mlp['loss'])), history_mlp['loss'], label='MLP')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(132)
plt.semilogy(range(len(history_spectral['l2_error'])), history_spectral['l2_error'], label='Spectral')
plt.semilogy(range(len(history_mlp['l2_error'])), history_mlp['l2_error'], label='MLP')
plt.title('L2 Error vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('L2 Error')
plt.legend()

plt.subplot(133)
plt.semilogy(range(len(history_spectral['learning_rates'])), history_spectral['learning_rates'], label='Spectral')
plt.semilogy(range(len(history_mlp['learning_rates'])), history_mlp['learning_rates'], label='MLP')
plt.title('Learning Rate vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()


# %%

# %%

# %%

# %%
def compute_pde_loss(model, t_grid, x_grid, ic_x_grid=None, ic_weight=1.0, c=80, temporal_eps=0):
    """
    Compute loss for advection equation u_t + c*u_x = 0 with temporal weighting
    
    Args:
        model: SpectralInterpolationND or MLP model
        t_grid: time points of shape (nt,)
        x_grid: space points of shape (nx,)
        ic_x_grid: space points for initial condition of shape (nx_ic,). If None, uses x_grid
        ic_weight: weight for initial condition loss
        c: advection speed
        temporal_eps: parameter for temporal weighting scheme
        
    Returns:
        total_loss: combined PDE and IC loss
        pde_residual: residual at collocation points
        ic_residual: residual at initial condition points
    """
    if ic_x_grid is None:
        ic_x_grid = x_grid
        
    n_t, n_x = len(t_grid), len(x_grid)
    
    # Compute solution and derivatives based on model type
    if isinstance(model, SpectralInterpolationND):
        # Spectral model takes separate coordinates
        u = model.interpolate([t_grid, x_grid])
        u_t = model.derivative([t_grid, x_grid], k=(1,0))
        u_x = model.derivative([t_grid, x_grid], k=(0,1))
        
        # Compute initial condition on ic_x_grid
        u_ic = model.interpolate([torch.tensor([0.0]), ic_x_grid])
        u_ic = u_ic[0]
    else:
        # Create mesh grid points for PDE
        T, X = torch.meshgrid(t_grid, x_grid, indexing='ij')
        points = torch.stack([T.flatten(), X.flatten()], dim=1)
        points.requires_grad_(True)
        
        # Forward pass and compute gradients
        u = model(points).reshape(n_t, n_x)
        grads = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
        u_t = grads[:, 0].reshape(n_t, n_x)
        u_x = grads[:, 1].reshape(n_t, n_x)
        
        # Compute initial condition
        ic_points = torch.stack([torch.zeros_like(ic_x_grid), ic_x_grid], dim=1)
        u_ic = model(ic_points).squeeze()
    
    # PDE residual: u_t + c*u_x = 0
    pde_residual = u_t + c*u_x
    
    pde_loss = torch.mean(pde_residual**2)
    ic_residual = u_ic - torch.sin(ic_x_grid)
    ic_loss = torch.mean(ic_residual**2)
    total_loss = pde_loss + ic_loss * ic_weight
    #print(f"total {total_loss}, pde {pde_loss}, ic {ic_loss}")
    return total_loss, pde_residual, ic_residual
    
    
    # Compute per-timestep losses
    timestep_losses = torch.mean(pde_residual**2, dim=1)  # Mean over space
    
    # Compute temporal weights
    cumsum = torch.cumsum(timestep_losses, dim=0)
    previous_losses = torch.cat([torch.zeros(1, device=cumsum.device), cumsum[:-1]])
    weights = torch.exp(-temporal_eps * previous_losses)
    weights = weights.detach()  # Stop gradient through weights
    
    # Apply weights to get final PDE loss
    pde_loss = torch.mean(weights * timestep_losses)
    
    # Initial condition: u(0,x) = sin(x)
    ic_residual = u_ic - torch.sin(ic_x_grid)
    ic_loss = torch.mean(ic_residual**2)
    
    total_loss = pde_loss + ic_loss * ic_weight
    
    # print(f"total {total_loss}, pde {pde_loss}, ic {ic_loss}")
    return total_loss, pde_residual, ic_residual


# %%
def get_training_grids(model=None, n_t=32, n_x=32, n_ic=32, t_max=1):
    """
    Get training grids either from model or create new ones
    
    Args:
        model: if SpectralInterpolationND, use its nodes. If None or MLP, create new grids
        n_t: number of time points (if not using model grid)
        n_x: number of space points (if not using model grid)
        n_ic: number of IC points (if None, use n_x)
    
    Returns:
        t_grid: time points
        x_grid: space points
        ic_x_grid: initial condition points
    """
    if model is None:
        # Cheb
        # t_grid = torch.cos(torch.linspace(0, 2*np.pi, n_t))*(t_max/2) + (t_max/2)
        # Random cheb
        t_grid = torch.cos(torch.rand(n_t)*2*np.pi)*(t_max/2) + (t_max/2)
        # Fourier
        # x_grid = torch.linspace(0, 2*np.pi, n_x)
        # ic_x_grid = torch.linspace(0, 2*np.pi, n_ic)
        # Random Fourier
        x_grid = torch.cat([
            torch.linspace(0, 2*np.pi, n_x//2),
            torch.rand(n_x//2)*2*np.pi,
        ])
        # ic_x_grid = torch.cat([
        #     torch.linspace(0, 2*np.pi, n_ic//2),
        #     torch.rand(n_ic//2)*2*np.pi,
        # ])
        ic_x_grid = x_grid
        # print(f"t_grid {t_grid}, x_grid {x_grid}")
    if isinstance(model, SpectralInterpolationND):
        t_grid = model.nodes[0]
        x_grid = model.nodes[1]
        ic_x_grid = x_grid  # Could be different if desired
    else:
        t_grid = torch.cos(torch.linspace(0, 2*np.pi, n_t))*(t_max/2) + (t_max/2)
        x_grid = torch.linspace(0, 2*np.pi, n_x)
        ic_x_grid = torch.linspace(0, 2*np.pi, n_ic)
        
    return t_grid, x_grid, ic_x_grid


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ### Run the models

# %%
import torch
import numpy as np
from time import time

# Set random seed for reproducibility
torch.manual_seed(42)

# Spectral model
Ns = [21, 20]  # 21 (odd) Chebyshev points in time, 20 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

# MLP model
mlp_model = MLP(n_dim=2, hidden_dim=64, activation=torch.tanh)

# Train both models
print("Training Spectral Model...")
history_spectral = train_advection(
    spectral_model,
    n_epochs=1000,
    lr=0.01,
    ic_weight=10.0,
    c=80,
    plot_every=100
)

print("\nTraining MLP Model...")
history_mlp = train_advection(
    mlp_model,
    n_epochs=1000,
    lr=0.01,
    ic_weight=10.0,
    c=80,
    plot_every=100
)

# Compare final L2 errors
print("\nFinal Results:")
print(f"Spectral Model - Best L2 error: {min(history_spectral['l2_error']):.2e}")
print(f"MLP Model - Best L2 error: {min(history_mlp['l2_error']):.2e}")

# Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.semilogy(range(len(history_spectral['loss'])), history_spectral['loss'], label='Spectral')
plt.semilogy(range(len(history_mlp['loss'])), history_mlp['loss'], label='MLP')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(132)
plt.semilogy(range(len(history_spectral['l2_error'])), history_spectral['l2_error'], label='Spectral')
plt.semilogy(range(len(history_mlp['l2_error'])), history_mlp['l2_error'], label='MLP')
plt.title('L2 Error vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('L2 Error')
plt.legend()

plt.subplot(133)
plt.semilogy(range(len(history_spectral['learning_rates'])), history_spectral['learning_rates'], label='Spectral')
plt.semilogy(range(len(history_mlp['learning_rates'])), history_mlp['learning_rates'], label='MLP')
plt.title('Learning Rate vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# # Hacking: L-BFGS. Also doesn't really work at all

# %%
def train_advection_lbfgs(model, n_epochs=1000, max_iter=20, lr=1.0, history_size=50, 
                           ic_weight=1.0, c=80, plot_every=100):
    """
    Train model to solve the advection equation using L-BFGS optimization
    
    Args:
        model: Neural network model
        n_epochs: Maximum number of epochs
        max_iter: Maximum number of iterations per L-BFGS step
        lr: Learning rate for L-BFGS
        history_size: Number of iterations to keep for L-BFGS history
        ic_weight: Weight for initial condition loss
        c: Advection speed
        plot_every: Plot interval
    """
    # Get training and evaluation grids
    t_grid, x_grid, ic_x_grid = get_training_grids(
        None,
        n_t=2*c+1,
        n_x=2*c,
        n_ic=2*c
    )
    n_eval = 100
    t_eval = torch.linspace(0, 1, n_eval)
    x_eval = torch.linspace(0, 2*np.pi, n_eval)
    
    # Setup L-BFGS optimizer
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn='strong_wolfe'
    )
    
    early_stopping = EarlyStopping(patience=20, min_delta=1e-20)
    
    # Training history
    history = {
        'loss': [], 'pde_residual': [], 'ic_residual': [], 
        'l2_error': [], 'function_evals': []
    }
    
    print("Starting training...")
    time_start = time()
    
    # Current loss values for closure
    current_loss = {'total': None, 'pde': None, 'ic': None}
    
    def closure():
        optimizer.zero_grad()
        loss, pde_residual, ic_residual = compute_pde_loss(
            model, t_grid, x_grid, ic_x_grid, ic_weight, c
        )
        loss.backward()
        
        # Store current loss values
        current_loss['total'] = loss.item()
        current_loss['pde'] = torch.mean(pde_residual**2).item()
        current_loss['ic'] = torch.mean(ic_residual**2).item()
        
        return loss
    
    for epoch in tqdm(range(n_epochs)):
        # Perform L-BFGS step
        optimizer.step(closure)
        
        # Get loss values from last closure call
        epoch_results = {
            'loss': current_loss['total'],
            'pde_residual': current_loss['pde'],
            'ic_residual': current_loss['ic']
        }
        
        # Evaluate
        eval_results = evaluate_model(model, t_eval, x_eval, c)
        
        # Update history
        current_time = time() - time_start
        history['loss'].append(epoch_results['loss'])
        history['pde_residual'].append(epoch_results['pde_residual'])
        history['ic_residual'].append(epoch_results['ic_residual'])
        history['l2_error'].append(eval_results['l2_error'])
        history['function_evals'].append(optimizer.state_dict()['state'][0]['n_iter'])
        
        # Early stopping
        early_stopping(eval_results['l2_error'], model)
        
        # Print and plot progress
        if (epoch + 1) % plot_every == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}, Time: {current_time:.2f}s")
            print(f"Loss: {epoch_results['loss']:.2e}, L2 Error: {eval_results['l2_error']:.2e}")
            print(f"Function evaluations: {optimizer.state_dict()['state'][0]['n_iter']}")
            
            # Plot current solution
            plt.figure(figsize=(15, 5))
            
            # Plot predicted solution
            plt.subplot(131)
            plt.imshow(eval_results['u_pred'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Predicted Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot exact solution
            plt.subplot(132)
            plt.imshow(eval_results['u_exact'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Exact Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot error
            plt.subplot(133)
            plt.imshow(torch.abs(eval_results['u_pred'] - eval_results['u_exact']).T, 
                      aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|error|')
            plt.title('Absolute Error')
            plt.xlabel('t')
            plt.ylabel('x')
            
            plt.tight_layout()
            plt.show()
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(early_stopping.best_state)
    time_elapsed = time() - time_start
    
    # Final plotting - Training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.semilogy(range(len(history['loss'])), history['loss'])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(132)
    plt.semilogy(range(len(history['l2_error'])), history['l2_error'])
    plt.title('L2 Error vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')
    
    plt.subplot(133)
    plt.plot(range(len(history['function_evals'])), history['function_evals'])
    plt.title('Cumulative Function Evaluations')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Evaluations')
    
    plt.tight_layout()
    plt.show()
    
    # Final solution evaluation and plot
    final_eval = evaluate_model(model, t_eval, x_eval, c)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(final_eval['u_pred'].T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Final Predicted Solution')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.subplot(132)
    plt.imshow(final_eval['u_exact'].T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Exact Solution')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.subplot(133)
    plt.imshow(torch.abs(final_eval['u_pred'] - final_eval['u_exact']).T, 
              aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
    plt.colorbar(label='|error|')
    plt.title('Final Absolute Error')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.tight_layout()
    plt.show()

    print(f"\nTraining completed in {time_elapsed:.2f} seconds")
    print(f"Best L2 error: {min(history['l2_error']):.2e}")
    print(f"Total function evaluations: {history['function_evals'][-1]}")

    return history


# %%
# Spectral model
Ns = [101, 20]  # 101 (odd) Chebyshev points in time, 20 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

history = train_advection_lbfgs(
    spectral_model,
    n_epochs=1000,
    max_iter=100,  # Max L-BFGS iterations per step
    lr=1.0,       # Initial step size
    history_size=100,  # Number of previous iterations to use
    ic_weight=10.0,
    c=80
)

# %%
# Spectral model
Ns = [101, 80]  # 101 (odd) Chebyshev points in time, 80 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

history = train_advection_lbfgs(
    spectral_model,
    n_epochs=1000,
    max_iter=100,  # Max L-BFGS iterations per step
    lr=1.0,       # Initial step size
    history_size=100,  # Number of previous iterations to use
    ic_weight=10.0,
    c=80
)

# %%
history_spectral = train_advection(
    spectral_model,
    n_epochs=1000,
    lr=1e-3,
    ic_weight=10.0,
    c=80,
    plot_every=100
)

# %%
history_spectral = train_advection(
    spectral_model,
    n_epochs=1000,
    lr=1e-3,
    ic_weight=1000.0,
    c=80,
    plot_every=100
)

# %%
Ns = [101, 20]  # 101 (odd) Chebyshev points in time, 20 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

history_spectral = train_advection(
    spectral_model,
    n_epochs=1000,
    lr=1e-2,
    ic_weight=np.inf,
    c=80,
    plot_every=100
)

# %%
# Eval points (from code)
# n_eval = 100
# t_eval = torch.linspace(0, 1, n_eval)
# x_eval = torch.linspace(0, 2*np.pi, n_eval)

# Train points (from code)
c = 80
t_eval, x_eval, _ = get_training_grids(
    None,
    n_t=2*c+1,
    n_x=2*c,
    n_ic=2*c
)
model = spectral_model

# Final solution evaluation and plot
final_eval = evaluate_model(model, t_eval, x_eval, c=80)
print(model.values[-1])

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(final_eval['u_pred'].T, aspect='auto', 
          extent=[0, 1, 0, 2*np.pi], cmap='viridis')
plt.colorbar(label='u')
plt.title('Final Predicted Solution')
plt.xlabel('t')
plt.ylabel('x')

plt.subplot(132)
plt.imshow(final_eval['u_exact'].T, aspect='auto', 
          extent=[0, 1, 0, 2*np.pi], cmap='viridis')
plt.colorbar(label='u')
plt.title('Exact Solution')
plt.xlabel('t')
plt.ylabel('x')

plt.subplot(133)
plt.imshow(torch.abs(final_eval['u_pred'] - final_eval['u_exact']).T, 
          aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
plt.colorbar(label='|error|')
plt.title('Final Absolute Error')
plt.xlabel('t')
plt.ylabel('x')

plt.tight_layout()
plt.show()

print(f"\nTraining completed in {time_elapsed:.2f} seconds")
print(f"Best L2 error: {min(history['l2_error']):.2e}")

# %%
print(spectral_model.values[-1])
print(spectral_model.values[-1].shape)

# %%
torch.sin(torch.linspace(0, 2*torch.pi, 20))


# %% [markdown]
# ## ------ Hacking: Try fixing initial condition

# %%
def fit_initial_condition(model, x_grid):
    """
    Fit initial condition u(0,x) = sin(x) exactly using 1D spectral interpolation
    
    Args:
        model: SpectralInterpolationND instance
        x_grid: Spatial grid points
    """
    # Extract initial values at t=0 from the model's values
    t0_idx = 0  # Index for t=0 in the model's grid
    ic_values = model.values[t0_idx]
    
    # Set these values to match sin(x)
    with torch.no_grad():
        ic_values[:] = torch.sin(x_grid)
    
    # Zero out all other time slices
    model.values[1:] = 0.0

def compute_pde_loss_fixed_ic(model, t_grid, x_grid, c=80):
    """
    Compute PDE loss without IC term (since IC is exactly satisfied)
    """
    n_t, n_x = len(t_grid), len(x_grid)
    
    # Compute solution and derivatives
    u = model.interpolate([t_grid, x_grid])
    u_t = model.derivative([t_grid, x_grid], k=(1,0))
    u_x = model.derivative([t_grid, x_grid], k=(0,1))
    
    # PDE residual: u_t + c*u_x = 0
    pde_residual = u_t + c*u_x
    
    # Compute loss
    pde_loss = torch.mean(pde_residual**2)
    
    return pde_loss, pde_residual

def train_one_epoch_fixed_ic(model, optimizer, t_grid, x_grid, c=80):
    """
    Train model for one epoch with fixed IC
    """
    optimizer.zero_grad()
    
    # Compute loss (only PDE term since IC is fixed)
    loss, pde_residual = compute_pde_loss_fixed_ic(model, t_grid, x_grid, c)
    
    # Backward and optimize
    loss.backward()
    
    # Zero out gradients for IC parameters (t=0 slice)
    with torch.no_grad():
        print(model.values.grad.shape)
        model.values.grad[-1] = 0.0
    
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'pde_residual': torch.mean(pde_residual**2).item()
    }

def train_advection_fixed_ic(model, n_epochs=1000, lr=1e-3, c=80, plot_every=100):
    """
    Train model to solve advection equation with fixed initial condition
    """
    # Get training and evaluation grids
    t_grid, x_grid, _ = get_training_grids(
        None,
        n_t=2*c+1,
        n_x=2*c,
        n_ic=2*c
    )
    n_eval = 100
    t_eval = torch.linspace(0, 1, n_eval)
    x_eval = torch.linspace(0, 2*np.pi, n_eval)
    
    print("Phase 1: Fitting initial condition...")
    # First phase: Fit initial condition exactly
    fit_initial_condition(model, x_grid)
    
    print("Phase 2: Training PDE solution...")
    # Second phase: Setup optimizer for PDE training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=150, min_delta=1e-20)
    
    # Training history
    history = {
        'loss': [], 'pde_residual': [], 'l2_error': [], 'learning_rates': []
    }
    
    print("Starting PDE training...")
    time_start = time()
    
    for epoch in tqdm(range(n_epochs)):
        # Train one epoch
        epoch_results = train_one_epoch_fixed_ic(
            model, optimizer, t_grid, x_grid, c
        )
        
        # Evaluate
        eval_results = evaluate_model(model, t_eval, x_eval, c)
        
        # Update history
        current_time = time() - time_start
        history['loss'].append(epoch_results['loss'])
        history['pde_residual'].append(epoch_results['pde_residual'])
        history['l2_error'].append(eval_results['l2_error'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling and early stopping
        scheduler.step(epoch_results['loss'])
        early_stopping(eval_results['l2_error'], model)
        
        # Print and plot progress
        if (epoch + 1) % plot_every == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}, Time: {current_time:.2f}s")
            print(f"Loss: {epoch_results['loss']:.2e}, L2 Error: {eval_results['l2_error']:.2e}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Plot current solution
            plt.figure(figsize=(15, 5))
            
            # Plot predicted solution
            plt.subplot(131)
            plt.imshow(eval_results['u_pred'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Predicted Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot exact solution
            plt.subplot(132)
            plt.imshow(eval_results['u_exact'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Exact Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot error
            plt.subplot(133)
            plt.imshow(torch.abs(eval_results['u_pred'] - eval_results['u_exact']).T, 
                      aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|error|')
            plt.title('Absolute Error')
            plt.xlabel('t')
            plt.ylabel('x')
            
            plt.tight_layout()
            plt.show()
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(early_stopping.best_state)
    time_elapsed = time() - time_start
    
    # Final plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.semilogy(range(len(history['loss'])), history['loss'])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(132)
    plt.semilogy(range(len(history['l2_error'])), history['l2_error'])
    plt.title('L2 Error vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')
    
    plt.subplot(133)
    plt.semilogy(range(len(history['learning_rates'])), history['learning_rates'])
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining completed in {time_elapsed:.2f} seconds")
    print(f"Best L2 error: {min(history['l2_error']):.2e}")

    return history


# %%
# Spectral model
Ns = [101, 20]  # 101 (odd) Chebyshev points in time, 20 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

history = train_advection_fixed_ic(
    model,
    n_epochs=1000,
    lr=1e-3,
    c=80
)


# %% [markdown]
# # ------------------------- Hacking below: early stopping, adaptive step rate, weighting the loss function to be causal, changing the domain

# %%
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


# %%
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class AdaptiveStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        step_size,
        gamma=0.1,
        threshold=1e-6,
        last_epoch=-1,
        # For adaptive threshold
        gradient_variance_metric_lambda=0.9,
        gradient_variance_threshold=0.9,
        no_increase_length=1e3,
    ):
        self.step_size = step_size
        self.gamma = gamma
        self.threshold = threshold

        self.gradient_variance_metric_ema = 1.0
        self.gradient_variance_metric_lambda = gradient_variance_metric_lambda
        self.gradient_variance_threshold = gradient_variance_threshold
        self.no_increase_length = no_increase_length
        self.should_decrement = True
        super(AdaptiveStepLR, self).__init__(optimizer, last_epoch)

    def update_gradient_variance_metric(self, gradient_variance_metric):
        self.gradient_variance_metric_ema = self.gradient_variance_metric_lambda * self.gradient_variance_metric_ema + (1 - self.gradient_variance_metric_lambda) * gradient_variance_metric
        if self.gradient_variance_metric_ema < self.gradient_variance_threshold and self.last_epoch > self.no_increase_length:
            self.should_decrement = False
        else:
            self.should_decrement = True

    def get_lr(self):
        new_lrs = []
        for base_lr in self.base_lrs:
            # Start with current learning rate
            lr = self.optimizer.param_groups[0]['lr']

            # Apply multiplicative factor based on the flags
            if self.last_epoch % self.step_size == 0:
                if self.should_decrement:
                    lr = max(lr * self.gamma, self.threshold)  # Decrease LR
                else:
                    lr = lr / self.gamma  # Increase LR

            new_lrs.append(lr)

        return new_lrs



# %%
def get_current_gradients(model, n_batches=16, _helper_get_training_grids=None, ic_weight=1.0, c=80, temporal_eps=0.1):
    """
    Compute gradients for PDE problem multiple times to get statistics
    """
    gradients = []
    
    prev_t_grid = None
    for _ in range(n_batches):
        
        # Get training grids
        t_max=1/100
        n_t = 2*11+1
        n_x = 2*c
        n_ic = 2*c
        t_grid = torch.cos(torch.rand(n_t)*2*np.pi)*(t_max/2) + (t_max/2)
        x_grid = torch.linspace(0, 2*np.pi, n_x)
        ic_x_grid = torch.linspace(0, 2*np.pi, n_ic)
        
        if prev_t_grid is not None:
            print(t_grid == prev_t_grid)
        
        # Zero gradients
        model.zero_grad()
        
        # Compute loss
        loss, _, _ = compute_pde_loss(
            model,
            t_grid,
            x_grid,
            ic_x_grid,
            ic_weight=ic_weight,
            c=c,
            temporal_eps=temporal_eps,
        )
        
        # Backward
        loss.backward()
        
        # Collect gradients
        batch_gradients = []
        for param in model.parameters():
            if param.grad is not None:
                batch_gradients.append(param.grad.view(-1).clone())
        batch_gradients = torch.cat(batch_gradients)
        gradients.append(batch_gradients)
        print(torch.all(gradients[-1] == torch.zeros_like(gradients[-1])))
        print(torch.norm(gradients[-1]))
        
        prev_t_grid = t_grid
        
    gradients = torch.stack(gradients)
    # print(gradients[0] == gradients[-1])
    return gradients

def compute_gradient_stats(gradients, prefix=""):
    """
    Compute gradient statistics for PDE problem
    """
    # Compute cosine similarities between consecutive gradients
    cosine_sims = []
    for i in range(1, len(gradients)):
        cosine_sim = torch.nn.functional.cosine_similarity(
            gradients[i-1].unsqueeze(0), 
            gradients[i].unsqueeze(0)
        ).item()
        cosine_sims.append(cosine_sim)
    cosine_sims = torch.tensor(cosine_sims)
    
    return {
        "gradient_cosine_mean": torch.mean(cosine_sims).item(),
        "gradient_cosine_variance": torch.var(cosine_sims, unbiased=True).item(),
        "gradient_l2_norm": torch.norm(gradients, p=2, dim=1).mean().item()
    }

def compute_current_gradient_stats(model, _helper_get_training_grids, n_batches=16, **kwargs):
    """
    Main function to compute gradient statistics for PDE problem
    """
    gradients = get_current_gradients(model, n_batches, _helper_get_training_grids, **kwargs)
    stats = compute_gradient_stats(gradients)
    return stats


# %%
def train_advection(model, n_epochs=1000, lr=1e-3, ic_weight=1.0, c=80, plot_every=100, t_max=1, temporal_eps=1):
    """
    Train model to solve the advection equation
    """
    # Get evaluation grids
    n_eval = 100
    t_eval = torch.linspace(0, t_max, n_eval)
    x_eval = torch.linspace(0, 2*np.pi, n_eval)
    
    # Define lambda for get training grid
    def _helper_get_training_grids():
        return get_training_grids(
            None, #model,
            n_t=2*c+1,
            n_x=2*c,
            n_ic=2*c,
            t_max=t_max,
        )
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = AdaptiveStepLR(
        optimizer,
        step_size=50,  # Check every 50 epochs
        gamma=0.9,     # Halve/double the learning rate
        threshold=1e-6,
        gradient_variance_metric_lambda=0.9,
        gradient_variance_threshold=0.9,
        no_increase_length=100
    )
    early_stopping = EarlyStopping(patience=150, min_delta=1e-6)
    
    # Training history
    history = {
        'loss': [], 
        'pde_residual': [], 
        'ic_residual': [], 
        'l2_error': [], 
        'learning_rates': [],
        'grad_cosine_mean': [],
        'grad_cosine_var': [],
        'grad_l2_norm': [],
    }
    
    print("Starting training...")
    
    time_start = time()
    
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        
        # Get training grids
        t_grid, x_grid, ic_x_grid = _helper_get_training_grids()
        
        # Compute loss
        loss, pde_residual, ic_residual = compute_pde_loss(
            model,
            t_grid,
            x_grid,
            ic_x_grid,
            ic_weight=ic_weight,
            c=c,
            temporal_eps=temporal_eps,
        )
        
        # Backward and optimize
        loss.backward()
        
        # Compute gradient statistics before optimizer step
        gradient_stats = compute_current_gradient_stats(
            model,
            _helper_get_training_grids=_helper_get_training_grids,
            n_batches=16,  # Can adjust this
            ic_weight=ic_weight,
            c=c,
            temporal_eps=temporal_eps
        )
        
        # Update scheduler with gradient statistics
        scheduler.update_gradient_variance_metric(gradient_stats["gradient_cosine_mean"])
        grad_var_metric = gradient_stats["gradient_cosine_mean"]
        
        optimizer.step()
        scheduler.step()
        
        # Evaluate
        eval_results = evaluate_model(model, t_eval, x_eval, c)
        
        # Update history
        history['loss'].append(loss.item())
        history['pde_residual'].append(torch.mean(pde_residual**2).item())
        history['ic_residual'].append(torch.mean(ic_residual**2).item())
        history['l2_error'].append(eval_results['l2_error'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['grad_cosine_mean'].append(gradient_stats["gradient_cosine_mean"])
        history['grad_cosine_var'].append(gradient_stats["gradient_cosine_variance"])
        history['grad_l2_norm'].append(gradient_stats["gradient_l2_norm"])
        
        # Early stopping check
        early_stopping(eval_results['l2_error'], model)
        
        # if early_stopping.early_stop:
        #     print(f"\nEarly stopping triggered at epoch {epoch}")
        #     break
            
        # Print and plot progress
        if (epoch + 1) % plot_every == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"Loss: {loss.item():.2e}, L2 Error: {eval_results['l2_error']:.2e}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Gradient variance metric: {grad_var_metric:.2e}")
            
            # Plot current solution
            plt.figure(figsize=(15, 5))
            
            # Plot predicted solution
            plt.subplot(131)
            plt.imshow(eval_results['u_pred'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Predicted Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot exact solution
            plt.subplot(132)
            plt.imshow(eval_results['u_exact'].T, aspect='auto', 
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
            plt.colorbar(label='u')
            plt.title('Exact Solution')
            plt.xlabel('t')
            plt.ylabel('x')
            
            # Plot error
            plt.subplot(133)
            plt.imshow(torch.abs(eval_results['u_pred'] - eval_results['u_exact']).T, 
                      aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
            plt.colorbar(label='|error|')
            plt.title('Absolute Error')
            plt.xlabel('t')
            plt.ylabel('x')
            
            plt.tight_layout()
            plt.show()
    
    # Load best model
    model.load_state_dict(early_stopping.best_state)
    time_elapsed = time() - time_start
    
    # Final plotting - now with gradient variance metric
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.semilogy(history['loss'])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(222)
    plt.semilogy(history['l2_error'])
    plt.title('L2 Error vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')
    
    plt.subplot(223)
    plt.semilogy(history['learning_rates'])
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.subplot(224)
    plt.plot(history['grad_variance_metric'], label='Instant')
    plt.plot(history['grad_variance_ema'], label='EMA')
    plt.axhline(y=scheduler.gradient_variance_threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Gradient Variance Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Final solution evaluation and plot
    final_eval = evaluate_model(model, t_eval, x_eval, c)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(final_eval['u_pred'].T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Final Predicted Solution')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.subplot(132)
    plt.imshow(final_eval['u_exact'].T, aspect='auto', 
              extent=[0, 1, 0, 2*np.pi], cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Exact Solution')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.subplot(133)
    plt.imshow(torch.abs(final_eval['u_pred'] - final_eval['u_exact']).T, 
              aspect='auto', extent=[0, 1, 0, 2*np.pi], cmap='magma')
    plt.colorbar(label='|error|')
    plt.title('Final Absolute Error')
    plt.xlabel('t')
    plt.ylabel('x')
    
    plt.tight_layout()
    plt.show()

    print(f"\nTraining completed in {time_elapsed:.2f} seconds")
    print(f"Best L2 error: {min(history['l2_error']):.2e}")

    return history


# %%
import torch
import numpy as np
from time import time

# Set random seed for reproducibility
torch.manual_seed(42)

# Spectral model
Ns = [81, 80]  # 21 (odd) Chebyshev points in time, 20 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

# MLP model
mlp_model = MLP(n_dim=2, hidden_dim=64, activation=torch.tanh)

# Train spectral model
print("Training Spectral Model...")
history_spectral = train_advection(
    spectral_model,
    n_epochs=1000,
    lr=0.01,
    ic_weight=10.0,
    c=80,
    plot_every=100,
)

# Train MLP model
print("\nTraining MLP Model...")
history_mlp = train_advection(
    mlp_model,
    n_epochs=1000,
    lr=0.01,
    ic_weight=10.0,
    c=80,
    plot_every=100
)

# Compare final L2 errors
print("\nFinal Results:")
print(f"Spectral Model - Best L2 error: {min(history_spectral['l2_error']):.2e}")
print(f"MLP Model - Best L2 error: {min(history_mlp['l2_error']):.2e}")

# Plot comparison of convergence
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.semilogy(history_spectral['time'][1:], history_spectral['loss'], label='Spectral')
plt.semilogy(history_mlp['time'][1:], history_mlp['loss'], label='MLP')
plt.title('Loss vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Loss')
plt.legend()

plt.subplot(132)
plt.semilogy(history_spectral['time'][1:], history_spectral['l2_error'], label='Spectral')
plt.semilogy(history_mlp['time'][1:], history_mlp['l2_error'], label='MLP')
plt.title('L2 Error vs Time')
plt.xlabel('Time (s)')
plt.ylabel('L2 Error')
plt.legend()

plt.subplot(133)
plt.semilogy(history_spectral['time'][1:], history_spectral['learning_rates'], label='Spectral')
plt.semilogy(history_mlp['time'][1:], history_mlp['learning_rates'], label='MLP')
plt.title('Learning Rate vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()

# %%
import torch
import numpy as np
from time import time

# Set random seed for reproducibility
torch.manual_seed(42)

# Spectral model
Ns = [11, 80]  # 21 (odd) Chebyshev points in time, 20 (even) Fourier points in space
bases = ["chebyshev", "fourier"]  # Chebyshev in time, Fourier in space
domains = [(0, 1/100), (0, 2*np.pi)]  # t ∈ [0,1], x ∈ [0,2π]
spectral_model = SpectralInterpolationND(Ns, bases, domains)

# MLP model
mlp_model = MLP(n_dim=2, hidden_dim=64, activation=torch.tanh)

# Train spectral model
print("Training Spectral Model...")
history_spectral = train_advection(
    spectral_model,
    n_epochs=5000,
    lr=1e-3,
    ic_weight=100,
    c=80,
    plot_every=100,
    t_max=1/100,
    temporal_eps=0.1,
)

# %%
