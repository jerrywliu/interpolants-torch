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

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.set_default_dtype(torch.float64)

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


# %% [markdown]
# ## Models

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
                self.nodes_standard[dim] = torch.linspace(0, 2*np.pi, self.Ns[dim]+1)[:-1]
                # Compute FFT frequencies
                self.k[dim] = torch.fft.fftfreq(self.Ns[dim]) * self.Ns[dim]
                self.cheb_weights[dim] = None
                
            # Set up domain mapping functions for this dimension
            domain = self.domains[dim]
            if self.bases[dim] == "chebyshev":
                self._to_standard[dim] = lambda x, d=dim: 2 * (x - self.domains[d][0]) / self.domain_lengths[d] - 1
                self._from_standard[dim] = lambda x, d=dim: self.domains[d][0] + (x + 1) * self.domain_lengths[d] / 2
            else:  # fourier
                self._to_standard[dim] = lambda x, d=dim: 2*np.pi * (x - self.domains[d][0]) / self.domain_lengths[d]
                self._from_standard[dim] = lambda x, d=dim: self.domains[d][0] + self.domain_lengths[d] * x / (2*np.pi)
            
            # Map standard nodes to physical domain
            self.nodes[dim] = self._from_standard[dim](self.nodes_standard[dim])
        
        # Set up diff matrices cache
        self._diff_matrices = [{} for _ in range(self.n_dim)]
        
        # Create mesh grid of nodes
        mesh_args = [self.nodes[d] for d in range(self.n_dim)]
        self.mesh = torch.meshgrid(*mesh_args, indexing='ij')
        
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
                    D[i,j] = c[i]/c[j] * (-1)**(i+j) / (nodes[i] - nodes[j])

        # Fill diagonal using negative sum trick
        D.diagonal().copy_(-torch.sum(D, dim=1))

        # Scale for domain transformation
        D = D * (2.0/domain_length)

        return D

    def _compute_fourier_derivative_matrix(self, nodes, domain_length):
        """
        Compute the differentiation matrix for 1D equispaced Fourier
        """
        N = len(nodes)
        D = torch.zeros((N, N), dtype=nodes.dtype, device=nodes.device)

        # Create index matrices
        i, j = torch.meshgrid(torch.arange(N, dtype=nodes.dtype, device=nodes.device), 
                            torch.arange(N, dtype=nodes.dtype, device=nodes.device), 
                            indexing='ij')

        # Compute off-diagonal elements using cotangent formula
        mask = i != j
        diff = (i[mask] - j[mask]) * (-1)**(i[mask] - j[mask])
        D[mask] = 0.5 * torch.tan(torch.pi * diff / N).reciprocal()

        # Diagonal elements are 0 for periodic functions
        D.diagonal().zero_()

        # Scale for domain transformation
        D = D * (2*np.pi/domain_length)

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
        assert len(k) == self.n_dim, f"Expected {self.n_dim} derivative orders, got {len(k)}"

        # Get 1D matrices for each dimension
        matrices = []
        for dim in range(self.n_dim):
            if k[dim] == 0:
                # Identity matrix for this dimension
                matrices.append(torch.eye(self.Ns[dim], 
                                       dtype=self.nodes_standard[dim].dtype,
                                       device=self.nodes_standard[dim].device))
            else:
                # Compute/get cached derivative matrix
                if k[dim] not in self._diff_matrices[dim]:
                    if 1 not in self._diff_matrices[dim]:
                        if self.bases[dim] == "chebyshev":
                            D = self._compute_cheb_derivative_matrix(self.nodes_standard[dim], self.domain_lengths[dim])
                        else:  # fourier
                            D = self._compute_fourier_derivative_matrix(self.nodes_standard[dim], self.domain_lengths[dim])
                        self._diff_matrices[dim][1] = D

                    # Compose for higher derivatives
                    Dk = self._diff_matrices[dim][1]
                    for _ in range(k[dim]-1):
                        Dk = Dk @ self._diff_matrices[dim][1]
                    self._diff_matrices[dim][k[dim]] = Dk

                matrices.append(self._diff_matrices[dim][k[dim]])

        # Compute Kronecker product
        D = matrices[0]
        for dim in range(1, self.n_dim):
            D = torch.kron(D, matrices[dim])

        return D
    
    def _cheb_interpolate_1d(self, x_eval, values, nodes_std, to_std, weights, eps=1e-14):
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
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        values_expanded = values.unsqueeze(0)  # (1, B2, B, N)
        nodes_expanded = nodes_std.reshape(1, 1, 1, -1)
        weights_expanded = weights.reshape(1, 1, 1, -1)

        # Compute distances - result is (B1, B2, B, N)
        d_x = x_eval_expanded - nodes_expanded

        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=-1, keepdim=True).values

        d_x = torch.where(
            small_diff_max,
            torch.zeros_like(d_x),
            1.0 / d_x
        )
        d_x[small_diff] = 1

        # Compute weighted sum along last axis
        f_eval_num = torch.sum(values_expanded * d_x * weights_expanded, dim=-1)  # (B1, B2, B)
        f_eval_denom = torch.sum(d_x * weights_expanded, dim=-1)  # (B1, B2, B)

        return f_eval_num / f_eval_denom

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
    
    def interpolate(self, x_eval, values):
        """
        Interpolate values at arbitrary points x_eval

        Args:
            x_eval: Tensor of shape (..., n_dim) containing coordinates to evaluate at
            values: Tensor of shape (*batch, *grid_dims) containing values to interpolate

        Returns:
            Tensor of shape (..., *batch) containing interpolated values
        """
        # Ensure input has correct shape
        assert x_eval.shape[-1] == self.n_dim, f"Expected {self.n_dim} coordinates, got {x_eval.shape[-1]}"

        # Store original batch shapes
        x_eval_batch_shape = x_eval.shape[:-1]  # (...) from (..., n_dim)
        x_eval_batch_shape_prod = int(torch.prod(torch.tensor(x_eval_batch_shape)).item())
        values_batch_shape = values.shape[:-self.n_dim]  # (*batch) from (*batch, *grid_dims)
        values_batch_shape_prod = int(torch.prod(torch.tensor(values_batch_shape)).item())

        # Reshape x_eval to (B1, B) = (prod(x_eval_batch_shape), n_dim)
        x_eval_reshaped = x_eval.reshape(-1, self.n_dim)

        # Reshape values to (B2, ...) = (prod(values_batch_shape), grid_dims)
        result = values.clone().reshape(-1, *values.shape[-self.n_dim:])

        # Interpolate one dimension at a time
        for dim in range(self.n_dim):

            # Get current dimension's coordinates, with shape (prod(x_eval_batch_shape))
            coords = x_eval_reshaped[..., dim]
            if dim == 0:
                # If first dimension, reshape to (B1, B) = (prod(x_eval_batch_shape), 1)
                coords_reshaped = coords.unsqueeze(-1)
            else:
                # Else, reshape to (B1, B) = (1, prod(x_eval_batch_shape))
                coords_reshaped = coords.unsqueeze(0)

            # Move result's target dimension to the end of grid dimensions
            shape = result.shape
            grid_dims = shape[-self.n_dim+dim:] # This gets smaller every iter of the loop, so that the next dim to interpolate over is always grid_dims[0]
            grid_dims_minus_current = grid_dims[1:]
            batch_dims = shape[:-self.n_dim+dim]

            # Permute grid dimensions to put target dim last
            perm = list(range(len(shape)))
            grid_start = len(batch_dims)
            perm.append(perm.pop(grid_start))
            result = result.permute(perm)
            
            # If dim > 0, assume that the last batch dimension is the shared dimension, move it to second last
            if dim > 0:
                perm = list(range(len(result.shape)))
                x_eval_dim = len(batch_dims)-1
                perm.insert(-1, perm.pop(x_eval_dim)) # Move last dim to second-to-last position
                result = result.permute(perm)

            if dim == 0:
                # If first dimension, reshape to (B2, B, N) = (prod(batch_dims + grid_dims[1:]), 1, grid_dims[0])
                result = result.reshape(-1, 1, grid_dims[0])
            else:
                # Else, result is already extended by B1 = prod(x_eval_batch_shape). Reshape to (B2, B, N) = (prod(batch_dims + grid_dims[:] - grid_dims[dim]), prod(x_eval_batch_shape), grid_dims[dim])
                result = result.reshape(-1, x_eval_batch_shape_prod, grid_dims[0])
    
            # Apply 1D interpolation
            if self.bases[dim] == "chebyshev":
                result = self._cheb_interpolate_1d(
                    coords_reshaped,
                    result,
                    self.nodes_standard[dim],
                    self._to_standard[dim],
                    self.cheb_weights[dim]
                )
            else:  # fourier
                result = self._fourier_interpolate_1d(
                    coords_reshaped,
                    result,
                    self._to_standard[dim],
                    self.k[dim]
                )

            # Result is now (B1, B2, B)
            # = (prod(x_eval_batch_shape), prod(batch_dims + grid_dims[1:]), 1) if dim == 0
            # = (1, prod(batch_dims + grid_dims[:] - grid_dims[dim]), prod(x_eval_batch_shape)) if dim > 0

            # Reshape result to restore batch dimensions
            # Then permute result to restore batch dimensions: (batch_dims, x_eval_batch_shape_prod, grid_dims_minus_current)
            if dim == 0:
                # (prod(x_eval_batch_shape), batch_dims, grid_dims[1:])
                result = result.reshape(-1, *batch_dims, *grid_dims_minus_current)
                x_eval_dim = 0
                target_pos = len(batch_dims)
            else:
                # (batch_dims, grid_dims-grid_dims[dim], prod(x_eval_batch_shape))
                result = result.reshape(*batch_dims[:-1], *grid_dims_minus_current, x_eval_batch_shape_prod)
                x_eval_dim = -1
                target_pos = len(batch_dims) - 1
                
            # Move x_eval batch dim after other batch dims
            perm = list(range(len(result.shape)))
            perm.insert(target_pos, perm.pop(x_eval_dim))
            result = result.permute(perm)

        return result
    
    def forward(self, x_eval):
        return self.interpolate(x_eval, self.values)
    
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
            k = (k,) + (0,)*(self.n_dim-1)

        assert len(k) == self.n_dim, f"Expected {self.n_dim} derivative orders, got {len(k)}"

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
        # Compute derivative values at grid points
        dk_nodes = self._derivative_interpolant(k)

        # Interpolate to evaluation points
        return self.interpolate(x_eval, dk_nodes)


# %% [markdown]
# # PINN training

# %% [markdown]
# We look at the Allen-Cahn equation (1 space + 1 time variable, both of which we treat spectrally):
#
# \begin{align*}
# &u_t - 0.0001u_{xx} + 5u^3 - 5u = 0, \quad t \in [0,1], x \in [-1,1], \\
# &u(0,x) = x^2\cos(\pi x), \\
# &u(t,-1) = u(t,1), \\
# &u_x(t,-1) = u_x(t,1)
# \end{align*}
#
# Solution is computed with Exponential Time Differencing.
#
# We use trigonometric polynomials in x to enforce the boundary conditions.

# %% [markdown]
# ## Least squares

# %% [markdown]
# Get ground truth Allen-Cahn solution, solved using Exponential Time Differencing. Effectively solves the spatial exactly and solves in time using RK45.

# %%
import scipy.io
import os

data_dir = "/pscratch/sd/j/jwl50/interpolants-torch/notebooks/data"
save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/notebooks/figures"

def get_allen_cahn_soln():
    data = scipy.io.loadmat(os.path.join(data_dir, "allen_cahn.mat"))
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    return u_ref, t_star, x_star


# %% [markdown]
# ### Sanity check: what's the resolution we need to resolve the GT solution?

# %%
from scipy.interpolate import RegularGridInterpolator

def interpolate_allen_cahn(Nx=800, Nt=21):
    
    # Create interpolant
    interp = SpectralInterpolationND(
        Ns=[Nt, Nx],
        bases=['chebyshev', 'fourier'],
        domains=[(0, 1), (-1, 1)]
    )

    # Get ground truth solution
    u_ref, t_star, x_star = get_allen_cahn_soln()
    TT, XX = np.meshgrid(t_star, x_star, indexing="ij")

    # Sample reference solution at our grid points
    # Create meshgrid for our collocation points
    T_col, X_col = torch.meshgrid(interp.nodes[0], interp.nodes[1], indexing='ij')
    points_col = torch.stack([T_col.flatten(), X_col.flatten()], dim=1)

    # Interpolate reference solution to our collocation points
    u_col = torch.from_numpy(u_ref)
    grid_interpolator = RegularGridInterpolator((t_star, x_star), u_ref, method="cubic")
    u_col_points = grid_interpolator(points_col.numpy())
    u_col = torch.from_numpy(u_col_points).reshape(Nt, Nx)

    # Set these values in our interpolant
    interp.values.data = u_col

    # Create coordinates for interpolation back to reference grid
    points_ref = torch.stack([
        torch.from_numpy(TT).flatten(),
        torch.from_numpy(XX).flatten()
    ], dim=1)

    # Interpolate to reference grid
    u_pred = interp.interpolate(points_ref, interp.values)
    u_pred = u_pred.reshape(len(t_star), len(x_star)).detach().numpy()

    # Plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Interpolated from exact")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, np.abs((u_ref - u_pred)), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Interpolation error (absolute)")
    plt.tight_layout()

    # Save the figure
    fig_path = os.path.join(save_dir, f"ac_interp_nx={nx}_nt={nt}.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.show()

    mean_abs_error = np.abs(u_ref - u_pred).mean()
    print(f"Mean absolute error: {mean_abs_error}")
    return mean_abs_error

# Run sweep
nt = 21
nxs = [100*2**k for k in range(5)]
mean_abs_errors = []
for nx in nxs:
    print(f"Nx = {nx}")
    mean_abs_errors.append(interpolate_allen_cahn(Nx=nx, Nt=nt))
    
plt.plot(nxs, mean_abs_errors)
plt.xlabel("nx")
plt.xscale("log")
plt.ylabel("Mean absolute error")
plt.yscale("log")
plt.title("Interpolating Allen-Cahn solution, nt=21")
fig_path = os.path.join(save_dir, "ac_interp_sweep.pdf")
plt.savefig(fig_path, dpi=300)

# %%
plt.plot(nxs, mean_abs_errors, "-bo")
plt.xlabel("nx")
plt.xscale("log")
plt.ylabel("Mean absolute error")
plt.yscale("log")
plt.title("Interpolating Allen-Cahn solution, nt=21")
fig_path = os.path.join(save_dir, "ac_interp_sweep.pdf")
plt.savefig(fig_path, dpi=300)


# %%
def setup_allen_cahn_ls(Nx=72, Nt=73, u_current=None):
    """
    Set up least squares problem for Allen-Cahn equation
    ut - 0.0001*uxx + 5u³ - 5u = 0
    u(0,x) = x²cos(πx)
    With periodic boundary conditions in x
    """
    # Create interpolant
    interp = SpectralInterpolationND(
        Ns=[Nt, Nx],
        bases=['chebyshev', 'fourier'],  # Chebyshev in t, Fourier in x
        domains=[(0, 1), (-1, 1)]        # t ∈ [0,1], x ∈ [-1,1]
    )
    
    # Get PDE operators
    Dt = interp.derivative_matrix(k=(1,0))   # time derivative
    Dxx = interp.derivative_matrix(k=(0,2))  # second space derivative
    
    # For nonlinear term, evaluate at current iterate
    if u_current is None:
        u_current = torch.zeros((Nt, Nx)).flatten()
    
    # Create diagonal matrix for cubic term: 5u³
    u_cubed = 5 * torch.diag(u_current**3)
    
    # PDE operator: ut - 0.0001*uxx + 5u³ - 5u = 0
    # A = Dt - 0.0001*Dxx + u_cubed - 5*torch.eye(Nt*Nx)
    A = Dt - 0.1*Dxx + u_cubed - 5*torch.eye(Nt*Nx)
           
    # Initial condition: extract t=0 values
    IC = torch.zeros(Nx, Nt, Nx)
    for i in range(Nx):
        IC[i, -1, i] = 1
    IC = IC.reshape(Nx, -1)
    
    # Right hand side
    b = torch.zeros(A.shape[0] + IC.shape[0])
    
    # Initial condition: u(0,x) = x²cos(πx)
    x = interp.nodes[1]
    b[A.shape[0]:A.shape[0]+IC.shape[0]] = x**2 * torch.cos(np.pi*x)
    
    # Full system (Fourier basis handles periodic BCs automatically)
    AA = torch.cat([A, IC], dim=0)
    
    return AA, b, interp


# %%
def visualize_solution(values, interp, u_ref, t_star, x_star, iteration):
    """Helper function to visualize current solution"""
    TT, XX = np.meshgrid(t_star, x_star, indexing="ij")
    points_ref = torch.stack([
        torch.from_numpy(TT).flatten(),
        torch.from_numpy(XX).flatten()
    ], dim=1)
    
    # Interpolate solution
    u_pred = interp.interpolate(points_ref, values)
    u_pred = u_pred.reshape(len(t_star), len(x_star)).detach().numpy()
    
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Exact")
    
    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Predicted (Iteration {iteration})")
    
    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, np.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()
    plt.show()

# Set up and solve system
Nx = 22
Nt = 21
AA, b, interp = setup_allen_cahn_ls(Nx, Nt)

# Get reference solution for visualization
u_ref, t_star, x_star = get_allen_cahn_soln()

# Better initial guess: extend IC across time
x = interp.nodes[1]  # Spatial grid points
u0 = x**2 * torch.cos(np.pi*x)
values = torch.zeros((Nt, Nx))
for i in range(Nt):
    values[i] = u0

# Convert to flattened form for iteration
u = values.flatten()

# The system is nonlinear, so we need to iterate
max_iter = 100
tol = 1e-8
omega = 0.9  # Under-relaxation parameter

for i in range(max_iter):
    # Update system with current solution
    AA_current, b, interp = setup_allen_cahn_ls(Nx, Nt, u)
    
    # Solve linear system
    u_new = torch.linalg.lstsq(AA_current, b, driver="gelsd").solution
    
    # Under-relaxation
    u_next = omega * u_new + (1 - omega) * u
    
    # Check convergence
    rel_error = torch.norm(u_next - u) / torch.norm(u_next)
    print(f"Iteration {i+1}, relative error: {rel_error:.2e}")
    
    # Visualize current solution
    visualize_solution(u_next.reshape(Nt, Nx), interp, u_ref, t_star, x_star, i+1)
    
    if rel_error < tol:
        print("Converged!")
        break
    
    u = u_next

# %% [markdown]
# ### Sanity check: let's start with the interpolated GT solution and see what happens as we apply Picard

# %%
from scipy.interpolate import RegularGridInterpolator

def interpolate_ground_truth(interp, u_ref, t_star, x_star):
    """
    Interpolate ground truth solution to spectral grid points
    
    Args:
        interp: SpectralInterpolationND instance
        u_ref: Reference solution array (t_star × x_star)
        t_star: Time points of reference solution
        x_star: Spatial points of reference solution
    
    Returns:
        torch.Tensor: Solution interpolated to spectral grid points (Nt × Nx)
    """
    # Create meshgrid for collocation points
    T_col, X_col = torch.meshgrid(interp.nodes[0], interp.nodes[1], indexing='ij')
    points_col = torch.stack([T_col.flatten(), X_col.flatten()], dim=1)
    
    # Create interpolator from reference solution
    grid_interpolator = RegularGridInterpolator(
        (t_star, x_star), 
        u_ref, 
        method="cubic"
    )
    
    # Interpolate reference solution to collocation points
    u_col_points = grid_interpolator(points_col.numpy())
    u_col = torch.from_numpy(u_col_points).reshape(interp.Ns[0], interp.Ns[1])
    
    return u_col

# Using the function in our solver:
Nx, Nt = 200, 21
interp = SpectralInterpolationND(
    Ns=[Nt, Nx],
    bases=['chebyshev', 'fourier'],
    domains=[(0, 1), (-1, 1)]
)

# Get ground truth and interpolate to our grid
u_ref, t_star, x_star = get_allen_cahn_soln()
u_init = interpolate_ground_truth(interp, u_ref, t_star, x_star)

# Visualize current solution
visualize_solution(u_init.reshape(Nt, Nx), interp, u_ref, t_star, x_star, 0)

# Use this as initial guess for Picard iteration
u = u_init.flatten()

max_iter = 20
tol = 1e-8
omega = 0.1  # Under-relaxation parameter

for i in range(max_iter):
    AA_current, b, interp = setup_allen_cahn_ls(Nx, Nt, u)
    u_new = torch.linalg.lstsq(AA_current, b, driver="gelsd").solution
    u_next = omega * u_new + (1 - omega) * u
    
    rel_error = torch.norm(u_next - u) / torch.norm(u_next)
    print(f"Iteration {i+1}, relative error: {rel_error:.2e}")
    
    # Visualize current solution
    visualize_solution(u_next.reshape(Nt, Nx), interp, u_ref, t_star, x_star, i+1)
    
    if rel_error < tol:
        print("Converged!")
        break
        
    u = u_next

# %% [markdown]
# ### Sanity check: are we enforcing the initial condition correctly?

# %%
# Create interpolant
Nx, Nt = 72, 73
interp = SpectralInterpolationND(
    Ns=[Nt, Nx],
    bases=['chebyshev', 'fourier'],
    domains=[(0, 1), (-1, 1)]
)

# Create initial condition values at grid points
x = interp.nodes[1]  # Spatial grid points
u0 = x**2 * torch.cos(np.pi*x)

# Set initial condition for all time points
values = torch.zeros((Nt, Nx))
for i in range(Nt):
    values[i] = u0

# Set the values in the interpolant
interp.values.data = values

# Use the provided plotting code
# Get ground truth solution
u_ref, t_star, x_star = get_allen_cahn_soln()
TT, XX = np.meshgrid(t_star, x_star, indexing="ij")

# Create coordinates for interpolation
points_ref = torch.stack([
    torch.from_numpy(TT).flatten(),
    torch.from_numpy(XX).flatten()
], dim=1)

# Interpolate to reference grid
u_pred = interp.interpolate(points_ref, interp.values)
u_pred = u_pred.reshape(len(t_star), len(x_star)).detach().numpy()

# Plot
fig = plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.pcolor(TT, XX, u_ref, cmap="jet", vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Exact")
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.pcolor(TT, XX, u_pred, cmap="jet", vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Predicted")
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.pcolor(TT, XX, np.abs(u_ref - u_pred), cmap="jet", vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel("t")
plt.ylabel("x")
plt.title("Absolute error")
plt.tight_layout()

# Save the figure
fig_path = os.path.join(save_dir, "ac.pdf")
fig.savefig(fig_path, bbox_inches="tight", dpi=300)


# %% [markdown]
# ### Newton's

# %%
def setup_newton_step(Nx=72, Nt=73, u_current=None, epsilon=0.1):
    """
    Set up Newton step for Allen-Cahn equation
    ut - εuxx + 5u³ - 5u = 0
    u(0,x) = x²cos(πx)
    """
    # Create interpolant
    interp = SpectralInterpolationND(
        Ns=[Nt, Nx],
        bases=['chebyshev', 'fourier'],
        domains=[(0, 1), (-1, 1)]
    )
    
    # Get PDE operators
    Dt = interp.derivative_matrix(k=(1,0))   # (Nt*Nx) × (Nt*Nx)
    Dxx = interp.derivative_matrix(k=(0,2))  # (Nt*Nx) × (Nt*Nx)
    
    if u_current is None:
        u_current = torch.zeros(Nt*Nx)
    
    # Compute residual: ut - εuxx + 5u³ - 5u
    # All operators should be (Nt*Nx) × (Nt*Nx), u_current should be (Nt*Nx)
    
    # Split computation to debug
    diff_term = Dt - epsilon*Dxx
    cubic_term = 5*torch.diag(u_current**3)
    linear_term = -5*torch.eye(len(u_current))
    
    # Full operator
    operator = diff_term + cubic_term + linear_term
    residual = operator @ u_current
    
    # Compute Jacobian: Dt - εDxx + 15u² - 5
    jacobian = Dt - epsilon*Dxx + 15*torch.diag(u_current**2) - 5*torch.eye(len(u_current))
    
    # Initial condition constraint
    IC = torch.zeros(Nx, Nt, Nx)
    for i in range(Nx):
        IC[i, -1, i] = 1
    IC = IC.reshape(Nx, -1)
    
    # Initial condition RHS
    x = interp.nodes[1]
    ic_rhs = x**2 * torch.cos(np.pi*x)
    
    # Add IC constraint to system
    full_jacobian = torch.cat([jacobian, IC], dim=0)
    full_residual = torch.cat([residual, IC @ u_current - ic_rhs])
    
    return full_jacobian, -full_residual, interp

# Newton iteration
Nx = 20
Nt = 21
max_iter = 20
tol = 1e-8
newton_step = 0.1

# Create interpolant
interp = SpectralInterpolationND(
    Ns=[Nt, Nx],
    bases=['chebyshev', 'fourier'],
    domains=[(0, 1), (-1, 1)]
)

# Initial guess using reference solution
u_ref, t_star, x_star = get_allen_cahn_soln()
u_init = interpolate_ground_truth(interp, u_ref, t_star, x_star)
u = u_init.flatten()

for i in range(max_iter):
    # Get Newton step
    J, r, interp = setup_newton_step(Nx, Nt, u)
    
    # Solve system J*du = r
    du = torch.linalg.lstsq(J, r, driver="gelsd").solution
    
    # Update solution
    u_new = u + newton_step*du
    
    # Check convergence
    rel_error = torch.norm(du) / torch.norm(u_new)
    print(f"Iteration {i+1}, relative error: {rel_error:.2e}, condition number: {torch.linalg.cond(J):.2e}")
    
    # Visualize current solution
    visualize_solution(u_new.reshape(Nt, Nx), interp, u_ref, t_star, x_star, i+1)
    
    if rel_error < tol:
        print("Converged!")
        break
        
    u = u_new


# %% [markdown]
# ## PINN with interpolant

# %%
def compute_derivative_ac(model, x, eval_mode=False):
    """Compute solution and derivatives for Allen-Cahn"""
    if isinstance(model, SpectralInterpolationND):
        u = model(x)
        u_t = model.derivative(x, k=(1,0))  # First time derivative
        u_xx = model.derivative(x, k=(0,2))  # Second space derivative
        
        if eval_mode:
            u = u.detach()
            u_t = u_t.detach()
            u_xx = u_xx.detach()
            
    else:
        raise NotImplementedError("Only implemented for SpectralInterpolationND")
            
    return u, u_t, u_xx

def compute_pde_loss_ac(model, colloc_points, ic_points=None, ic_weight=1.0, epsilon=0.01):
    """
    Compute loss for Allen-Cahn equation
    u_t - εu_xx + 5u³ - 5u = 0
    u(0,x) = x²cos(πx)
    """
    # Compute solution and derivatives at collocation points
    u, u_t, u_xx = compute_derivative_ac(model, colloc_points)
    
    # PDE residual: u_t - εu_xx + 5u³ - 5u = 0
    pde_residual = u_t - epsilon*u_xx + 5*u**3 - 5*u
    pde_loss = torch.mean(pde_residual**2)
    
    # Initial condition
    if ic_points is None:
        # Extract points where t=0
        mask = torch.abs(colloc_points[..., 0]) < 1e-10
        ic_points = colloc_points[mask]
    
    u_ic = model(ic_points)
    x_ic = ic_points[..., 1]  # get x coordinates
    ic_residual = u_ic - x_ic**2 * torch.cos(np.pi*x_ic)
    ic_loss = ic_weight * torch.mean(ic_residual**2)
    
    total_loss = pde_loss + ic_loss
    
    return total_loss, pde_residual, ic_residual

def visualize_solution(u_pred, u_exact, epoch, T_eval=None, X_eval=None):
    """Plot predicted vs exact solution and error"""
    if T_eval is None or X_eval is None:
        n_eval = int(np.sqrt(len(u_pred.flatten())))
        T_eval, X_eval = torch.meshgrid(
            torch.linspace(0, 1, n_eval),
            torch.linspace(-1, 1, n_eval),
            indexing='ij'
        )
    
    # Convert to numpy for plotting
    u_pred = u_pred.detach().numpy()
    u_exact = u_exact.detach().numpy()
    T_eval = T_eval.detach().numpy()
    X_eval = X_eval.detach().numpy()
    error = np.abs(u_pred - u_exact)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Predicted solution
    im1 = ax1.pcolor(T_eval, X_eval, u_pred, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(f'Predicted (epoch {epoch})')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    
    # Exact solution
    im2 = ax2.pcolor(T_eval, X_eval, u_exact, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Exact')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    
    # Error
    im3 = ax3.pcolor(T_eval, X_eval, error, cmap='jet')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Error')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    
    plt.tight_layout()
    plt.show()

def train_allen_cahn(model, n_epochs=1000, lr=1e-3, ic_weight=1.0, epsilon=0.01, plot_every=100):
    """Train model to solve the Allen-Cahn equation"""
    # Use model's built-in grid for collocation points
    t_points = model.nodes[0]  # Chebyshev in time
    x_points = model.nodes[1]  # Fourier in space
    T, X = torch.meshgrid(t_points, x_points, indexing='ij')
    colloc_points = torch.stack([T.flatten(), X.flatten()], dim=1)
    
    # Get reference solution for comparison
    u_ref, t_star, x_star = get_allen_cahn_soln()
    
    # Create evaluation grid
    n_eval = 100
    t_eval = torch.linspace(0, 1, n_eval)
    x_eval = torch.linspace(-1, 1, n_eval)
    T_eval, X_eval = torch.meshgrid(t_eval, x_eval, indexing='ij')
    eval_points = torch.stack([T_eval.flatten(), X_eval.flatten()], dim=1)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'loss': [], 
        'pde_residual': [], 
        'ic_residual': [],
        'l2_error': []
    }
    
    # Training loop
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        
        # Compute loss
        loss, pde_residual, ic_residual = compute_pde_loss_ac(
            model, colloc_points, ic_weight=ic_weight, epsilon=epsilon
        )
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Compute L2 error against reference solution
        with torch.no_grad():
            u_pred = model(eval_points).reshape(n_eval, n_eval)
            # Interpolate reference solution to evaluation points
            u_ref_interp = RegularGridInterpolator(
                (t_star, x_star), 
                u_ref
            )
            u_exact = torch.from_numpy(
                u_ref_interp(torch.stack([T_eval, X_eval], dim=-1).numpy())
            )
            l2_error = torch.mean((u_pred - u_exact)**2).sqrt()
        
        # Record history
        history['loss'].append(loss.item())
        history['pde_residual'].append(torch.mean(pde_residual**2).item())
        history['ic_residual'].append(torch.mean(ic_residual**2).item())
        history['l2_error'].append(l2_error.item())
        
        # Print progress and plot
        if (epoch + 1) % plot_every == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Total Loss: {loss.item():.2e}")
            print(f"PDE Residual: {torch.mean(pde_residual**2).item():.2e}")
            print(f"IC Residual: {torch.mean(ic_residual**2).item():.2e}")
            print(f"L2 Error: {l2_error.item():.2e}")

            # Visualize current solution
            visualize_solution(u_pred, u_exact, epoch+1, T_eval, X_eval)
    
    return history


# %%
# Create spectral interpolant
Nx = 20
Nt = 21
interp = SpectralInterpolationND(
    Ns=[Nt, Nx],
    bases=['chebyshev', 'fourier'],  # Chebyshev in t, Fourier in x
    domains=[(0, 1), (-1, 1)]        # t ∈ [0,1], x ∈ [-1,1]
)

# Training parameters
n_epochs = 100000
lr = 1e-3
ic_weight = 10.0  # Higher weight on initial condition
epsilon = 0.0001     # Start with larger diffusion coefficient
plot_every = 1000  # Plot every 1000 epochs

# Train the model
history = train_allen_cahn(
    model=interp,
    n_epochs=n_epochs,
    lr=lr,
    ic_weight=ic_weight,
    epsilon=epsilon,
    plot_every=plot_every
)

# Plot training history
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['ic_residual'], label='IC Residual')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')

plt.subplot(132)
plt.semilogy(history['l2_error'])
plt.xlabel('Epoch')
plt.ylabel('L2 Error')
plt.title('L2 Error vs Reference')

plt.tight_layout()
plt.show()

# %%
# Create 1D Fourier interpolant
Nx = 200
interp_ic = SpectralInterpolationND(
   Ns=[Nx],                    # Number of points
   bases=['fourier'],          # Fourier basis
   domains=[(-1, 1)]          # Spatial domain
)

# Create initial condition at Fourier points
x = interp_ic.nodes[0]  # Get Fourier points
u0 = x**2 * torch.cos(np.pi*x)  # Evaluate IC at these points
interp_ic.values.data = u0

# Test interpolation
x_test = torch.linspace(-1, 1, 1000)
x_test = x_test.reshape(-1, 1)  # Shape required by interpolant (batch, n_dim)
u0_interp = interp_ic(x_test).squeeze()  # Remove extra dimension

# Exact solution at test points
u0_exact = x_test.squeeze()**2 * torch.cos(np.pi*x_test.squeeze())

# Plot to verify
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Solution plot
ax1.plot(x_test.squeeze().numpy(), u0_interp.detach().numpy(), 'b-', label='Interpolant')
ax1.plot(x.numpy(), u0.numpy(), 'r.', label='Grid points')
ax1.set_xlabel('x')
ax1.set_ylabel('u(0,x)')
ax1.set_title('Initial Condition Interpolation')
ax1.legend()
ax1.grid(True)

# Error plot
error = torch.abs(u0_interp - u0_exact)
ax2.semilogy(x_test.squeeze().numpy(), error.detach().numpy(), 'k-')
ax2.set_xlabel('x')
ax2.set_ylabel('Absolute Error')
ax2.set_title('Interpolation Error')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print max error at grid points
print(f"Max error at grid points: {torch.max(torch.abs(interp_ic(x.reshape(-1,1)).squeeze() - u0)):.2e}")
# Print max error at test points
print(f"Max error at test points: {torch.max(error):.2e}")

# %%
