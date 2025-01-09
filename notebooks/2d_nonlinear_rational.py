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
# # Goal of this notebook: extend the SpectralND class to barycentric rationals with learnable weights.

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
# # Models

# %% [markdown]
# ## SpectralND: polynomial interpolation

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
# ## MLP

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),  # Tanh activation since our function involves exp
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# %% [markdown]
# # Sanity check: interpolation
# Let's start by interpolating a simple (non-periodic) function: $f(t, x) = \exp(t+x)$.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

def test_spectral_interpolation():
    # Define domain
    t_domain = (0, 1)  # time from 0 to 1
    x_domain = (-1, 1) # space from -1 to 1
    
    # Number of points in each dimension for the interpolation
    Nt = 17  # time points
    Nx = 17  # space points
    
    # Create interpolator
    interpolator = SpectralInterpolationND(
        Ns=[Nt, Nx],  # number of points in each dimension
        bases=['chebyshev', 'chebyshev'],  # using Fourier basis for both dimensions
        domains=[t_domain, x_domain]  # domains for each dimension
    )
    
    # Create true solution at grid points
    t_grid, x_grid = interpolator.mesh
    true_values = torch.exp(t_grid + x_grid)
    
    # Set the interpolator's values to the true solution
    interpolator.values.data = true_values
    
    # Create evaluation points
    t_eval = torch.linspace(0, 1, 100)
    x_eval = torch.linspace(-1, 1, 200)
    T_eval, X_eval = torch.meshgrid(t_eval, x_eval, indexing='ij')
    points_eval = torch.stack([T_eval.flatten(), X_eval.flatten()], dim=-1)
    
    # Evaluate interpolation
    u_interp = interpolator(points_eval)
    u_interp = u_interp.reshape(T_eval.shape).detach()
    
    # Compute true solution at evaluation points
    u_true = torch.exp(T_eval + X_eval)
    
    # Compute error
    error = torch.abs(u_interp - u_true)
    max_error = torch.max(error).item()
    print(f"Maximum interpolation error: {max_error:.2e}")
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot interpolated solution
    c1 = ax1.pcolormesh(T_eval.numpy(), X_eval.numpy(), u_interp.numpy(), 
                        shading='auto', cmap='viridis')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Space (x)')
    ax1.set_title('Interpolated exp(t+x)')
    plt.colorbar(c1, ax=ax1)
    
    # Plot true solution
    c2 = ax2.pcolormesh(T_eval.numpy(), X_eval.numpy(), u_true.numpy(), 
                        shading='auto', cmap='viridis')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Space (x)')
    ax2.set_title('True exp(t+x)')
    plt.colorbar(c2, ax=ax2)
    
    # Plot error
    c3 = ax3.pcolormesh(T_eval.numpy(), X_eval.numpy(), error.numpy(), 
                        shading='auto', cmap='viridis')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Space (x)')
    ax3.set_title('Absolute Error')
    plt.colorbar(c3, ax=ax3)
    
    plt.tight_layout()
    return fig

# Run the test
if __name__ == "__main__":
    fig = test_spectral_interpolation()
    plt.show()

# %% [markdown]
# Now let's run gradient descent.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

def train_spectral_interpolation(n_epochs=10000):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create high-resolution data grid (100x100)
    Nt_data, Nx_data = 100, 100
    t_data = torch.linspace(0, 1, Nt_data)
    x_data = torch.linspace(-1, 1, Nx_data)
    T_data, X_data = torch.meshgrid(t_data, x_data, indexing='ij')
    data_points = torch.stack([T_data.flatten(), X_data.flatten()], dim=-1)
    
    # Create true solution at data points
    u_data = torch.exp(T_data + X_data)
    
    # Create spectral interpolator with lower resolution (17x17)
    interpolator = SpectralInterpolationND(
        Ns=[17, 17],  # 17 nodes in each dimension
        bases=['chebyshev', 'chebyshev'],  # Chebyshev basis for both dimensions
        domains=[(0, 1), (-1, 1)]  # Same domains as data
    )
    
    # Initialize randomly around 1 (since exp(t+x) is always positive)
    interpolator.values.data = torch.randn_like(interpolator.values) * 0.1 + 1.0
    
    # Setup optimizer
    # optimizer = torch.optim.Adam([interpolator.values], lr=1e-3)
    optimizer = torch.optim.SGD([interpolator.values], lr=1)
    
    # Training loop
    losses = []
    
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        
        # Evaluate interpolant at data points
        u_pred = interpolator(data_points)
        
        # Compute MSE loss
        loss = torch.mean((u_pred - u_data.flatten())**2)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():1.3e}')
    
    # Create evaluation grid for final visualization
    t_eval = torch.linspace(0, 1, 200)
    x_eval = torch.linspace(-1, 1, 200)
    T_eval, X_eval = torch.meshgrid(t_eval, x_eval, indexing='ij')
    eval_points = torch.stack([T_eval.flatten(), X_eval.flatten()], dim=-1)
    
    # Evaluate final interpolant
    with torch.no_grad():
        u_final = interpolator(eval_points).reshape(T_eval.shape).detach()
        u_true = torch.exp(T_eval + X_eval)
        error = torch.abs(u_final - u_true)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 5))
    
    # Plot 1: Training Loss
    ax1 = fig.add_subplot(141)
    ax1.semilogy(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot 2: Learned Solution
    ax2 = fig.add_subplot(142)
    c2 = ax2.pcolormesh(T_eval.numpy(), X_eval.numpy(), u_final.numpy(), 
                        shading='auto', cmap='viridis')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Space (x)')
    ax2.set_title('Learned Solution')
    plt.colorbar(c2, ax=ax2)
    
    # Plot 3: True Solution
    ax3 = fig.add_subplot(143)
    c3 = ax3.pcolormesh(T_eval.numpy(), X_eval.numpy(), u_true.numpy(), 
                        shading='auto', cmap='viridis')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Space (x)')
    ax3.set_title('True Solution')
    plt.colorbar(c3, ax=ax3)
    
    # Plot 4: Error
    ax4 = fig.add_subplot(144)
    c4 = ax4.pcolormesh(T_eval.numpy(), X_eval.numpy(), error.numpy(), 
                        shading='auto', cmap='viridis')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('Space (x)')
    ax4.set_title('Absolute Error')
    plt.colorbar(c4, ax=ax4)
    
    plt.tight_layout()
    
    return fig, interpolator, losses, error

# Run the training
if __name__ == "__main__":
    fig, interpolator, losses, error = train_spectral_interpolation(n_epochs=100000)
    plt.show()
    
    # Print final loss
    print(f"Final training loss: {losses[-1]:1.3e}")
    print(f"Maximum absolute error: {torch.max(torch.abs(error)).item():1.3e}")


# %% [markdown]
# Compare to accuracy of an MLP.

# %%
def train_mlp_interpolation(n_epochs=10000):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create high-resolution data grid (100x100)
    Nt_data, Nx_data = 100, 100
    t_data = torch.linspace(0, 1, Nt_data)
    x_data = torch.linspace(-1, 1, Nx_data)
    T_data, X_data = torch.meshgrid(t_data, x_data, indexing='ij')
    data_points = torch.stack([T_data.flatten(), X_data.flatten()], dim=-1)
    
    # Create true solution at data points
    u_data = torch.exp(T_data + X_data).flatten()
    
    # Create MLP
    model = SimpleMLP(hidden_size=64)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    losses = []
    
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        
        # Forward pass
        u_pred = model(data_points)
        
        # Compute MSE loss
        loss = torch.mean((u_pred - u_data)**2)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():1.3e}')
    
    # Create evaluation grid for final visualization
    t_eval = torch.linspace(0, 1, 200)
    x_eval = torch.linspace(-1, 1, 200)
    T_eval, X_eval = torch.meshgrid(t_eval, x_eval, indexing='ij')
    eval_points = torch.stack([T_eval.flatten(), X_eval.flatten()], dim=-1)
    
    # Evaluate final model
    model.eval()
    with torch.no_grad():
        u_final = model(eval_points).reshape(T_eval.shape)
        u_true = torch.exp(T_eval + X_eval)
        error = torch.abs(u_final - u_true)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 5))
    
    # Plot 1: Training Loss
    ax1 = fig.add_subplot(141)
    ax1.semilogy(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot 2: Learned Solution
    ax2 = fig.add_subplot(142)
    c2 = ax2.pcolormesh(T_eval.numpy(), X_eval.numpy(), u_final.numpy(), 
                        shading='auto', cmap='viridis')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Space (x)')
    ax2.set_title('Learned Solution (MLP)')
    plt.colorbar(c2, ax=ax2)
    
    # Plot 3: True Solution
    ax3 = fig.add_subplot(143)
    c3 = ax3.pcolormesh(T_eval.numpy(), X_eval.numpy(), u_true.numpy(), 
                        shading='auto', cmap='viridis')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Space (x)')
    ax3.set_title('True Solution')
    plt.colorbar(c3, ax=ax3)
    
    # Plot 4: Error
    ax4 = fig.add_subplot(144)
    c4 = ax4.pcolormesh(T_eval.numpy(), X_eval.numpy(), error.numpy(), 
                        shading='auto', cmap='viridis')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('Space (x)')
    ax4.set_title('Absolute Error')
    plt.colorbar(c4, ax=ax4)
    
    plt.tight_layout()
    
    return fig, model, losses

# Run the training
if __name__ == "__main__":
    fig, model, losses = train_mlp_interpolation(n_epochs=10000)
    plt.show()
    
    # Print final loss and error
    print(f"Final training loss: {losses[-1]:1.3e}")
    print(f"Maximum absolute error: {torch.max(torch.abs(error)).item():1.3e}")

# %% [markdown]
# # Sanity check: interpolation on Allen-Cahn

# %%
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_interpolant(Nt=21, Nx=100, num_epochs=1000, lr=1e-3):
    # Create the interpolant
    interp = SpectralInterpolationND(
        Ns=[Nt, Nx],
        bases=['chebyshev', 'fourier'],
        domains=[(0, 1), (-1, 1)]
    )
    # interp = SpectralInterpolationND(
    #     Ns=[Nx, Nt],
    #     bases=['fourier', 'chebyshev'],
    #     domains=[(-1, 1), (0, 1)]
    # )
    
    # Get reference solution
    u_ref, t_star, x_star = get_allen_cahn_soln()
    
    # Create evaluation points tensor
    TT, XX = np.meshgrid(t_star, x_star, indexing='ij')
    points = torch.tensor(np.stack([TT.flatten(), XX.flatten()], axis=1), dtype=torch.float32)
    
    # Convert reference solution to tensor
    u_ref_tensor = torch.tensor(u_ref.flatten(), dtype=torch.float32)
    
    # Setup optimizer
    optimizer = optim.Adam([interp.values], lr=lr)
    
    # Training loop
    losses = []
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        # Forward pass
        u_pred = interp(points)
        
        # Compute loss
        loss = torch.mean((u_pred - u_ref_tensor)**2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss history
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_yscale('log')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot final solution
    u_pred_final = interp(points).detach().numpy().reshape(TT.shape)
    
    im2 = ax2.pcolormesh(TT, XX, u_pred_final, cmap='jet', vmin=-1, vmax=1)
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Space (x)')
    ax2.set_title('Interpolated Solution')
    plt.colorbar(im2, ax=ax2)
    
    # Plot error
    error = np.abs(u_ref - u_pred_final)
    im3 = ax3.pcolormesh(TT, XX, error, cmap='jet')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Space (x)')
    ax3.set_title('Absolute Error')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    
    # Print final metrics
    mean_abs_error = np.mean(np.abs(u_ref - u_pred_final))
    max_abs_error = np.max(np.abs(u_ref - u_pred_final))
    print(f"\nFinal Metrics:")
    print(f"Mean Absolute Error: {mean_abs_error:.6f}")
    print(f"Max Absolute Error: {max_abs_error:.6f}")
    
    return interp, losses, u_pred_final


# %% [markdown]
# ## Sanity check: interpolation
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
#
# We start by seeing how well a polynomial can interpolate the true solution.

# %%
import scipy.io
import os

data_dir = "/pscratch/sd/j/jwl50/interpolants-torch/notebooks/data"
save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/notebooks/figures"

def get_allen_cahn_soln(d="1e-4"):
    data = scipy.io.loadmat(os.path.join(data_dir, f"allen_cahn_d={d}.mat"))
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    return u_ref, t_star, x_star


# %%
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def plot_allen_cahn_solution(d="1e-4"):
    # Load the solution
    u_ref, t_star, x_star = get_allen_cahn_soln(d=d)
    
    # Create meshgrid for plotting
    T, X = np.meshgrid(t_star, x_star, indexing="ij")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 2D color plot
    c1 = ax1.pcolormesh(T, X, u_ref, cmap='jet', vmin=-1, vmax=1)
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Space (x)')
    ax1.set_title('Allen-Cahn Solution')
    fig.colorbar(c1, ax=ax1)
    
    # Plot solution at different time points
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for t in times:
        t_idx = np.abs(t_star - t).argmin()
        ax2.plot(x_star, u_ref[t_idx, :], label=f't = {t:.1f}')
    
    ax2.set_xlabel('Space (x)')
    ax2.set_ylabel('u(x,t)')
    ax2.set_title('Solution Profiles at Different Times')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Print solution information
    print(f"Solution shape: {u_ref.shape}")
    print(f"Time points: {len(t_star)} (from {t_star.min():.2f} to {t_star.max():.2f})")
    print(f"Spatial points: {len(x_star)} (from {x_star.min():.2f} to {x_star.max():.2f})")
    
    return fig

# Create and display the plot
fig = plot_allen_cahn_solution()
plt.show()

# %% [markdown]
# ## Fit an interpolant using gradient descent

# %%
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_interpolant(Nt=21, Nx=100, num_epochs=1000, lr=1e-3):
    # Create the interpolant
    interp = SpectralInterpolationND(
        Ns=[Nt, Nx],
        bases=['chebyshev', 'fourier'],
        domains=[(0, 1), (-1, 1)]
    )
    # interp = SpectralInterpolationND(
    #     Ns=[Nx, Nt],
    #     bases=['fourier', 'chebyshev'],
    #     domains=[(-1, 1), (0, 1)]
    # )
    
    # Get reference solution
    u_ref, t_star, x_star = get_allen_cahn_soln()
    
    # Create evaluation points tensor
    TT, XX = np.meshgrid(t_star, x_star, indexing='ij')
    points = torch.tensor(np.stack([TT.flatten(), XX.flatten()], axis=1), dtype=torch.float32)
    
    # Convert reference solution to tensor
    u_ref_tensor = torch.tensor(u_ref.flatten(), dtype=torch.float32)
    
    # Setup optimizer
    optimizer = optim.Adam([interp.values], lr=lr)
    
    # Training loop
    losses = []
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        # Forward pass
        u_pred = interp(points)
        
        # Compute loss
        loss = torch.mean((u_pred - u_ref_tensor)**2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss history
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_yscale('log')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot final solution
    u_pred_final = interp(points).detach().numpy().reshape(TT.shape)
    
    im2 = ax2.pcolormesh(TT, XX, u_pred_final, cmap='jet', vmin=-1, vmax=1)
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Space (x)')
    ax2.set_title('Interpolated Solution')
    plt.colorbar(im2, ax=ax2)
    
    # Plot error
    error = np.abs(u_ref - u_pred_final)
    im3 = ax3.pcolormesh(TT, XX, error, cmap='jet')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Space (x)')
    ax3.set_title('Absolute Error')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    
    # Print final metrics
    mean_abs_error = np.mean(np.abs(u_ref - u_pred_final))
    max_abs_error = np.max(np.abs(u_ref - u_pred_final))
    print(f"\nFinal Metrics:")
    print(f"Mean Absolute Error: {mean_abs_error:.6f}")
    print(f"Max Absolute Error: {max_abs_error:.6f}")
    
    return interp, losses, u_pred_final


# %%
interp, losses, u_pred = train_interpolant(Nt=21, Nx=100)
plt.show()

# %% [markdown]
# ## The naive interpolation, where we 1-D interpolate $N^D$ times, was way too slow.
# Let's try with N-D FFT/DCT approaches for Fourier and Chebyshev.

# %% [markdown]
# ## Fourier

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_2d_fft_interpolation():
    # Create a smooth periodic test function
    def test_function(x, y):
        return torch.sin(2*np.pi*x) * torch.cos(2*np.pi*y) + 0.5*torch.cos(4*np.pi*x) * torch.sin(2*np.pi*y)

    # Add efficient 2D Fourier interpolation method to SpectralInterpolationND
    def _fourier_interpolate_2d(self, x_eval, values):
        """
        Efficient 2D Fourier interpolation using direct 2D FFT
        
        Args:
            x_eval: Points to evaluate at, shape (N, 2)
            values: Values at grid points, shape (N1, N2)
            
        Returns:
            Interpolated values at x_eval points
        """
        assert len(self.Ns) == 2 and all(b == 'fourier' for b in self.bases), "Only for 2D Fourier"
        
        # Compute 2D FFT of values
        coeffs = torch.fft.fftn(values)
        
        # Get frequencies for both dimensions
        k1 = self.k[0]  # frequencies for first dimension
        k2 = self.k[1]  # frequencies for second dimension
        
        # Map evaluation points to standard domain [0, 2π]
        x_std = self._to_standard[0](x_eval[:, 0])
        y_std = self._to_standard[1](x_eval[:, 1])
        
        # Initialize result
        result = torch.zeros(len(x_eval), dtype=torch.complex64)
        
        # Direct evaluation of 2D Fourier series
        for i, (kx, coeff_row) in enumerate(zip(k1, coeffs)):
            for j, (ky, coeff) in enumerate(zip(k2, coeff_row)):
                phase = kx * x_std[:, None] + ky * y_std[:, None]
                result += coeff * torch.exp(1j * phase).squeeze()
                
        return torch.real(result) / (self.Ns[0] * self.Ns[1])
    
    # Add method to SpectralInterpolationND class
    SpectralInterpolationND._fourier_interpolate_2d = _fourier_interpolate_2d
    
    # Test the implementation
    Nx = Ny = 32
    interp = SpectralInterpolationND(
        Ns=[Nx, Ny],
        bases=['fourier', 'fourier'],
        domains=[(0, 1), (0, 1)]
    )
    
    # Create grid points and evaluate test function
    x = interp.nodes[0]
    y = interp.nodes[1]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    values = test_function(X, Y)
    interp.values.data = values
    
    # Create evaluation points
    x_eval = torch.rand(1000, 2)  # Random points in [0,1] × [0,1]
    
    # Compute exact and interpolated values
    exact = test_function(x_eval[:, 0], x_eval[:, 1])
    interp_old = interp.interpolate(x_eval, values)
    
    # Override interpolate method temporarily for testing
    original_interpolate = interp.interpolate
    interp.interpolate = lambda x, v: interp._fourier_interpolate_2d(x, v)
    interp_new = interp.interpolate(x_eval, values)
    
    # Reset interpolate method
    interp.interpolate = original_interpolate
    
    # Compute errors
    error_old = torch.abs(exact - interp_old).mean()
    error_new = torch.abs(exact - interp_new).mean()
    
    print(f"Mean absolute error (old method): {error_old:.3e}")
    print(f"Mean absolute error (new method): {error_new:.3e}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original function
    im1 = axes[0,0].pcolormesh(X.numpy(), Y.numpy(), values.numpy(), shading='auto')
    axes[0,0].set_title("Original Function")
    plt.colorbar(im1, ax=axes[0,0])
    
    # Plot interpolated values (new method)
    scatter1 = axes[0,1].scatter(x_eval[:, 0], x_eval[:, 1], c=interp_new, cmap='viridis')
    axes[0,1].set_title("Interpolated Values (New Method)")
    plt.colorbar(scatter1, ax=axes[0,1])
    
    # Plot exact values at evaluation points
    scatter2 = axes[1,0].scatter(x_eval[:, 0], x_eval[:, 1], c=exact, cmap='viridis')
    axes[1,0].set_title("Exact Values")
    plt.colorbar(scatter2, ax=axes[1,0])
    
    # Plot absolute error
    scatter3 = axes[1,1].scatter(x_eval[:, 0], x_eval[:, 1], c=torch.abs(exact - interp_new), cmap='viridis')
    axes[1,1].set_title("Absolute Error")
    plt.colorbar(scatter3, ax=axes[1,1])
    
    plt.tight_layout()
    return fig

# Run the test
fig = test_2d_fft_interpolation()
plt.show()

# %%
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def train_periodic_interpolant(Nx=32, Ny=32, num_epochs=1000, lr=1e-2):
    # Define test function
    def test_function(x, y):
        return torch.sin(2*np.pi*x) * torch.cos(2*np.pi*y) + 0.5*torch.cos(4*np.pi*x) * torch.sin(2*np.pi*y)
    
    # Create the interpolant
    interp = SpectralInterpolationND(
        Ns=[Nx, Ny],
        bases=['fourier', 'fourier'],
        domains=[(0, 1), (0, 1)]
    )
    
    # Create reference grid for exact solution
    t = torch.linspace(0, 1, 100)
    x = torch.linspace(0, 1, 100)
    TT, XX = torch.meshgrid(t, x, indexing='ij')
    points_ref = torch.stack([
        TT.flatten(),
        XX.flatten()
    ], dim=1)
    
    # Get exact solution on reference grid
    u_exact = test_function(points_ref[:, 0], points_ref[:, 1])
    
    # Setup optimizer
    optimizer = optim.Adam([interp.values], lr=lr)
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        u_pred = interp(points_ref)
        
        # Compute loss
        loss = torch.mean((u_pred - u_exact)**2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():1.3e}')
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot exact solution
    u_exact_grid = u_exact.reshape(100, 100)
    im1 = ax1.pcolormesh(TT.numpy(), XX.numpy(), u_exact_grid, cmap='viridis')
    ax1.set_title('Exact Solution')
    plt.colorbar(im1, ax=ax1)
    
    # Plot interpolated solution
    u_pred_grid = u_pred.detach().reshape(100, 100)
    im2 = ax2.pcolormesh(TT.numpy(), XX.numpy(), u_pred_grid, cmap='viridis')
    ax2.set_title(f'Interpolated Solution (Nx={Nx}, Ny={Ny})')
    plt.colorbar(im2, ax=ax2)
    
    # Plot absolute error
    error = torch.abs(u_exact_grid - u_pred_grid.detach())
    im3 = ax3.pcolormesh(TT.numpy(), XX.numpy(), error, cmap='viridis')
    ax3.set_title('Absolute Error')
    plt.colorbar(im3, ax=ax3)
    
    # Plot loss history
    ax4.plot(losses)
    ax4.set_yscale('log')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MSE Loss')
    ax4.set_title('Training Loss')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Print final metrics
    final_loss = losses[-1]
    max_error = error.max()
    mean_error = error.mean()
    print(f"\nFinal Metrics:")
    print(f"MSE Loss: {final_loss:1.3e}")
    print(f"Max Absolute Error: {max_error:1.3e}")
    print(f"Mean Absolute Error: {mean_error:1.3e}")
    
    return interp, losses, error

# Run training with different resolutions
resolutions = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
for Nx, Ny in resolutions:
    print(f"\nTraining with resolution Nx={Nx}, Ny={Ny}")
    interp, losses, error = train_periodic_interpolant(Nx=Nx, Ny=Ny)
    plt.show()

# %% [markdown]
# ## Chebyshev

# %%
import torch
import numpy as np
from scipy.fft import dct
import matplotlib.pyplot as plt

def test_2d_chebyshev_interpolation():
    # Create a smooth (non-periodic) test function
    def test_function(x, y):
        # Map from [-1,1] to [0,1] for comparison with Fourier case
        x_mapped = (x + 1)/2
        y_mapped = (y + 1)/2
        return torch.sin(2*np.pi*x_mapped) * torch.cos(2*np.pi*y_mapped) + \
               0.5*torch.cos(4*np.pi*x_mapped) * torch.sin(2*np.pi*y_mapped)

    def _chebyshev_interpolate_2d(self, x_eval, values):
        """
        Efficient 2D Chebyshev interpolation using DCT and Clenshaw algorithm
        
        Args:
            x_eval: Points to evaluate at, shape (N, 2)
            values: Values at grid points, shape (N1, N2)
        """
        assert len(self.Ns) == 2 and all(b == 'chebyshev' for b in self.bases), \
               "Only for 2D Chebyshev"
        
        # Map evaluation points to standard domain [-1,1]
        x_std = self._to_standard[0](x_eval[:, 0])
        y_std = self._to_standard[1](x_eval[:, 1])
        
        # Get DCT of values using scipy and convert back to torch
        values_np = values.detach().numpy()
        coeffs_np = dct(dct(values_np, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')
        coeffs = torch.from_numpy(coeffs_np).float()
        
        # Implement 2D Clenshaw algorithm
        def clenshaw_evaluate_2d(x, y, coeffs):
            N, M = coeffs.shape
            
            # Ensure points are in [-1,1]
            x = torch.clamp(x, -1, 1)
            y = torch.clamp(y, -1, 1)
            
            def clenshaw_1d(t, coeffs):
                """
                1D Clenshaw algorithm for vector input t and coefficients
                t: (...) tensor of points
                coeffs: (N,) or (N, ...) tensor of coefficients
                """
                if coeffs.dim() == 1:
                    coeffs = coeffs.view(-1, 1)
                n = coeffs.shape[0]
                
                b_current = torch.zeros_like(t)
                b_prev = torch.zeros_like(t)
                b_prev2 = torch.zeros_like(t)
                
                for k in range(n-1, 0, -1):
                    b_prev2 = b_prev
                    b_prev = b_current
                    b_current = 2 * t * b_prev - b_prev2 + coeffs[k]
                
                return coeffs[0] + t * b_current - b_prev
            
            # Apply Clenshaw along first dimension for each y value
            result = torch.zeros((len(x), M), dtype=coeffs.dtype, device=coeffs.device)
            for j in range(M):
                result[:, j] = clenshaw_1d(x, coeffs[:, j])
            
            # Apply Clenshaw along second dimension for each result
            final_result = torch.zeros(len(y), dtype=coeffs.dtype, device=coeffs.device)
            for i in range(len(x)):
                final_result = clenshaw_1d(y, result[i, :])
            
            return final_result
        
        # Evaluate using Clenshaw
        result = clenshaw_evaluate_2d(x_std, y_std, coeffs)
        return result / (self.Ns[0] * self.Ns[1])
    
    # Add method to SpectralInterpolationND class
    SpectralInterpolationND._chebyshev_interpolate_2d = _chebyshev_interpolate_2d
    
    # Test the implementation
    Nx = Ny = 31
    interp = SpectralInterpolationND(
        Ns=[Nx, Ny],
        bases=['chebyshev', 'chebyshev'],
        domains=[(-1, 1), (-1, 1)]
    )
    
    # Create grid points and evaluate test function
    x = interp.nodes[0]
    y = interp.nodes[1]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    values = test_function(X, Y)
    interp.values.data = values
    
    # Create evaluation points
    x_eval = torch.rand(1000, 2) * 2 - 1  # Random points in [-1,1] × [-1,1]
    
    # Compute exact and interpolated values
    exact = test_function(x_eval[:, 0], x_eval[:, 1])
    interp_old = interp.interpolate(x_eval, values)
    
    # Override interpolate method temporarily for testing
    original_interpolate = interp.interpolate
    interp.interpolate = lambda x, v: interp._chebyshev_interpolate_2d(x, v)
    interp_new = interp.interpolate(x_eval, values)
    
    # Reset interpolate method
    interp.interpolate = original_interpolate
    
    # Compute errors
    error_old = torch.abs(exact - interp_old).mean()
    error_new = torch.abs(exact - interp_new).mean()
    
    print(f"Mean absolute error (old method): {error_old:1.3e}")
    print(f"Mean absolute error (new method): {error_new:1.3e}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original function
    im1 = axes[0,0].pcolormesh(X.numpy(), Y.numpy(), values.numpy(), shading='auto')
    axes[0,0].set_title("Original Function")
    plt.colorbar(im1, ax=axes[0,0])
    
    # Plot interpolated values (new method)
    scatter1 = axes[0,1].scatter(x_eval[:, 0], x_eval[:, 1], c=interp_new, cmap='viridis')
    axes[0,1].set_title("Interpolated Values (New Method)")
    plt.colorbar(scatter1, ax=axes[0,1])
    
    # Plot exact values at evaluation points
    scatter2 = axes[1,0].scatter(x_eval[:, 0], x_eval[:, 1], c=exact, cmap='viridis')
    axes[1,0].set_title("Exact Values")
    plt.colorbar(scatter2, ax=axes[1,0])
    
    # Plot absolute error
    scatter3 = axes[1,1].scatter(x_eval[:, 0], x_eval[:, 1], 
                                c=torch.abs(exact - interp_new), cmap='viridis')
    axes[1,1].set_title("Absolute Error")
    plt.colorbar(scatter3, ax=axes[1,1])
    
    plt.tight_layout()
    return fig

# Run the test
fig = test_2d_chebyshev_interpolation()
plt.show()

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_chebyshev_via_fourier():
    # Create a non-periodic test function on [0,1]×[0,1]
    def test_function(x, y):
        return torch.exp(x+y)
    
    # Transform function to periodic domain
    def transformed_function(theta, phi):
        # Map back to [0,1]×[0,1]
        x = torch.cos(theta)/2 + 0.5
        y = torch.cos(phi)/2 + 0.5
        return test_function(x, y)
    
    # Create interpolant for transformed function
    Nx = Ny = 32
    interp = SpectralInterpolationND(
        Ns=[Nx, Ny],
        bases=['fourier', 'fourier'],
        domains=[(0, 2*np.pi), (0, 2*np.pi)]
    )
    
    # Sample transformed function at grid points
    theta = interp.nodes[0]
    phi = interp.nodes[1]
    THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')
    values = transformed_function(THETA, PHI)
    interp.values.data = values
    
    # Create evaluation points in original domain
    x_eval = torch.linspace(0, 1, 100)
    y_eval = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x_eval, y_eval, indexing='ij')
    
    # Transform evaluation points
    theta_eval = torch.arccos(2*X.flatten() - 1)
    phi_eval = torch.arccos(2*Y.flatten() - 1)
    points_eval = torch.stack([theta_eval, phi_eval], dim=1)
    
    # Interpolate
    interp_values = interp(points_eval)
    
    # Compute exact values for comparison
    exact_values = test_function(X.flatten(), Y.flatten())
    
    # Reshape results
    interp_grid = interp_values.reshape(100, 100)
    exact_grid = exact_values.reshape(100, 100)
    error = torch.abs(interp_grid - exact_grid)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original function
    im1 = axes[0,0].pcolormesh(X.numpy(), Y.numpy(), exact_grid.detach().numpy(), 
                              shading='auto')
    axes[0,0].set_title("Original Function")
    plt.colorbar(im1, ax=axes[0,0])
    
    # Transformed function
    im2 = axes[0,1].pcolormesh(THETA.numpy(), PHI.numpy(), values.detach().numpy(), 
                              shading='auto')
    axes[0,1].set_title("Transformed Function")
    plt.colorbar(im2, ax=axes[0,1])
    
    # Interpolated result
    im3 = axes[1,0].pcolormesh(X.numpy(), Y.numpy(), interp_grid.detach().numpy(), 
                              shading='auto')
    axes[1,0].set_title("Interpolated Result")
    plt.colorbar(im3, ax=axes[1,0])
    
    # Error
    im4 = axes[1,1].pcolormesh(X.numpy(), Y.numpy(), error.detach().numpy(), 
                              shading='auto')
    axes[1,1].set_title("Absolute Error")
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    
    # Print error metrics
    mean_error = error.mean().item()
    max_error = error.max().item()
    print(f"Mean absolute error: {mean_error:1.3e}")
    print(f"Maximum absolute error: {max_error:1.3e}")
    
    return fig

# Run the test
fig = test_chebyshev_via_fourier()
plt.show()

# %%
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def train_transformed_interpolant(Nx=32, Ny=32, num_epochs=1000, lr=1e-2):
    # Define original and transformed test functions
    def test_function(x, y):
        return torch.exp(x + y)
    
    def transformed_function(theta, phi):
        x = torch.cos(theta)/2 + 0.5
        y = torch.cos(phi)/2 + 0.5
        return test_function(x, y)
    
    # Create the interpolant
    interp = SpectralInterpolationND(
        Ns=[Nx, Ny],
        bases=['fourier', 'fourier'],
        domains=[(0, 2*np.pi), (0, 2*np.pi)]
    )
    
    # Create reference grid for training
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Transform training points to θ-φ space
    theta_train = torch.arccos(2*X.flatten() - 1)
    phi_train = torch.arccos(2*Y.flatten() - 1)
    points_train = torch.stack([theta_train, phi_train], dim=1)
    
    # Get exact values at training points
    exact_values = test_function(X.flatten(), Y.flatten())
    
    # Setup optimizer
    optimizer = optim.Adam([interp.values], lr=lr)
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred_values = interp(points_train)
        
        # Compute loss
        loss = torch.mean((pred_values - exact_values)**2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():1.3e}')
    
    # Final evaluation
    pred_final = interp(points_train)
    error = torch.abs(pred_final - exact_values)
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original function
    exact_grid = exact_values.reshape(100, 100)
    im1 = ax1.pcolormesh(X.numpy(), Y.numpy(), exact_grid.detach().numpy(), shading='auto')
    ax1.set_title('Original Function')
    plt.colorbar(im1, ax=ax1)
    
    # Interpolated result
    pred_grid = pred_final.reshape(100, 100)
    im2 = ax2.pcolormesh(X.numpy(), Y.numpy(), pred_grid.detach().numpy(), shading='auto')
    ax2.set_title('Interpolated Function')
    plt.colorbar(im2, ax=ax2)
    
    # Error
    error_grid = error.reshape(100, 100)
    im3 = ax3.pcolormesh(X.numpy(), Y.numpy(), error_grid.detach().numpy(), shading='auto')
    ax3.set_title('Absolute Error')
    plt.colorbar(im3, ax=ax3)
    
    # Loss history
    ax4.plot(losses)
    ax4.set_yscale('log')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MSE Loss')
    ax4.set_title('Training Loss')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Print final metrics
    final_loss = losses[-1]
    max_error = error.max().item()
    mean_error = error.mean().item()
    print(f"\nFinal Metrics:")
    print(f"MSE Loss: {final_loss:1.3e}")
    print(f"Max Absolute Error: {max_error:1.3e}")
    print(f"Mean Absolute Error: {mean_error:1.3e}")
    
    return interp, losses, error

# Try different resolutions
for N in [16, 32, 64]:
    print(f"\nTraining with resolution N={N}x{N}")
    interp, losses, error = train_transformed_interpolant(Nx=N, Ny=N, num_epochs=10000)
    plt.show()

# %%
