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
# ## Datasets

# %%
# InterpolationDataset: for sanity-checking that the networks can interpolate the GT solution function
class InterpolationDataset(Dataset):
    def __init__(self, num_points, domain=(-1, 1), function=torch.sin, sampling='equispaced', noise_std=0):
        """
        Args:
            num_points: Number of points to sample
            domain: Tuple of (min, max) for the domain
            function: Function to apply to x values
            sampling: String indicating sampling method ('equispaced', 'chebyshev', or 'random')
        """
        self.num_points = num_points
        self.domain = domain
        self.function = function
        self.sampling = sampling
        self.noise_std = noise_std
        
        # Sample
        self.resample()
    
    def _generate_points(self):
        valid_methods = {'equispaced', 'chebyshev', 'random'}
        if self.sampling not in valid_methods:
            raise ValueError(f"Sampling method must be one of {valid_methods}")
        if self.sampling == 'equispaced':
            x = torch.linspace(self.domain[0], self.domain[1], self.num_points)
        elif self.sampling == 'chebyshev':
            j = torch.arange(self.num_points)
            x = torch.cos((2*j + 1) * torch.pi / (2*self.num_points))
            x = self._scale_to_domain(x, self.domain)
        else:  # random
            x = torch.rand(self.num_points) * (self.domain[1] - self.domain[0]) + self.domain[0]
        return x
    
    def _scale_to_domain(self, x, domain):
        """Scale points from [-1, 1] to specified domain"""
        return (domain[1] - domain[0]) * (x + 1) / 2 + domain[0]
    
    def resample(self):
        """Regenerate x points and corresponding y values"""
        self.x = self._generate_points()
        self.y = self.function(self.x)
        self.y += torch.randn_like(self.y) * self.noise_std
        return self
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# %% [markdown]
# ## Models

# %%
class MLP2D(nn.Module):
    def __init__(self, hidden_dim=32, activation=torch.tanh):
        """
        2-layer MLP that maps R -> R
        
        Args:
            hidden_dim: Dimension of hidden layer
            activation: Activation function to use (default: ReLU)
        """
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.activation = activation
        
    def forward(self, x):
        # Ensure input is 2D tensor (batch_size, 2)
        assert len(x.shape) >= 2
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x # Return (batch_size, 2)


# %%
class ChebyshevInterpolationModel(nn.Module):
    def __init__(self, num_points, domain_min=-1, domain_max=1):
        """
        Model parameterized by values at Chebyshev points of the second kind
        
        Args:
            num_points: Number of points (N where N is number of points)
            domain_min: Left endpoint of physical domain
            domain_max: Right endpoint of physical domain
        """
        super().__init__()
        
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.domain_length = domain_max - domain_min
        
        # Generate Chebyshev points of the second kind in [-1, 1]
        i = torch.linspace(0, 1, num_points)
        self.nodes_standard = torch.cos(torch.pi * i)  # [-1, 1]
        # Map to physical domain
        self.nodes = self._from_standard(self.nodes_standard)
        
        # Learnable values at these points
        self.values = nn.Parameter(torch.zeros(num_points))
        
        # Precompute barycentric weights for Chebyshev points
        self.weights = torch.zeros(num_points)
        self.weights[::2] = 1
        self.weights[1::2] = -1
        self.weights[0] = 0.5
        self.weights[-1] = 0.5
        
        # Cache for differentiation matrices
        self._diff_matrices = {}
        
    def _to_standard(self, x):
        """Map from physical domain to [-1, 1]"""
        return 2 * (x - self.domain_min) / self.domain_length - 1
    
    def _from_standard(self, x):
        """Map from [-1, 1] to physical domain"""
        return self.domain_min + (x + 1) * self.domain_length / 2
    
    def _cheb_interpolate(self, x_eval, values, eps=1e-14):
        """
        Interpolate values using barycentric formula
        """
        # Map evaluation points to standard domain
        x_eval_standard = self._to_standard(x_eval)
        
        # Compute difference matrix (N, N_eval)
        d_x = x_eval_standard.unsqueeze(0) - self.nodes_standard.unsqueeze(1)
        
        # Handle numerical instability for small differences
        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=0).values
        
        # If small_diff, set the column to 0 and the entry to 1
        d_x = torch.where(
            small_diff_max[None, :],
            torch.zeros_like(d_x),
            1.0 / d_x
        )
        d_x[small_diff] = 1
        
        # Interpolate
        f_eval_num = torch.einsum("...n,nm,n->...m", values, d_x, self.weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, self.weights)
        
        return f_eval_num / f_eval_denom

    def _compute_derivative_matrix(self, nodes):
        """
        Compute the spectral differentiation matrix for Chebyshev points
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
        D = D * (2.0/self.domain_length)
        
        return D
        
    def derivative_matrix(self, k=1):
        """
        Get k-th derivative matrix (cached for efficiency)
        """
        if k == 0:
            n = len(self.nodes_standard)
            return torch.eye(n, dtype=self.nodes_standard.dtype, device=self.nodes_standard.device)
            
        if k not in self._diff_matrices:
            # Compute first derivative matrix if not cached
            if 1 not in self._diff_matrices:
                self._diff_matrices[1] = self._compute_derivative_matrix(self.nodes_standard)
            
            # Compute k-th derivative matrix by composition
            Dk = self._diff_matrices[1]
            for i in range(k-1):
                Dk = Dk @ self._diff_matrices[1]
            self._diff_matrices[k] = Dk
            
        return self._diff_matrices[k]
        
    def derivative(self, x_eval, k=1):
        """
        Compute k-th derivative of interpolant at x_eval points
        """
        if k == 0:
            return self(x_eval)
            
        # Get k-th derivative matrix
        Dk = self.derivative_matrix(k)
        
        # Compute derivative values at nodes (differentiable w.r.t self.values)
        dk_nodes = Dk @ self.values
        
        # Interpolate to evaluation points
        return self._cheb_interpolate(x_eval, dk_nodes)
    
    def forward(self, x_eval):
        return self._cheb_interpolate(x_eval, self.values)


# %%
# This version produces the differentiation matrix explicitly
class FourierInterpolationModel(nn.Module):
    def __init__(self, num_points, domain_min=-1, domain_max=1):
        """
        Model parameterized by values at equispaced points with periodic boundary conditions
        
        Args:
            num_points: Number of points (N where N is number of points)
            domain_min: Left endpoint of physical domain
            domain_max: Right endpoint of physical domain
        """
        super().__init__()
        
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.domain_length = domain_max - domain_min
        
        # Generate equispaced points in [0, 2π) 
        self.nodes_standard = torch.linspace(0, 2*np.pi, num_points+1)[:-1]
        # Map to physical domain
        self.nodes = self._from_standard(self.nodes_standard)
        
        # Learnable values at these points
        self.values = nn.Parameter(torch.zeros(num_points))
        
        # Precompute wavenumbers
        self.k = torch.fft.fftfreq(num_points) * num_points
        
        # Cache for differentiation matrices
        self._diff_matrices = {}
        
    def _to_standard(self, x):
        """Map from physical domain to [0, 2π)"""
        return 2*np.pi * (x - self.domain_min) / self.domain_length
    
    def _from_standard(self, x):
        """Map from [0, 2π) to physical domain"""
        return self.domain_min + self.domain_length * x / (2*np.pi)
    
    def _compute_derivative_matrix(self, nodes):
        """
        Compute the Fourier differentiation matrix using cotangent formula
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
        D = D * (2*np.pi/self.domain_length)
        
        return D
    
    def derivative_matrix(self, k=1):
        """
        Get k-th derivative matrix (cached for efficiency)
        """
        if k == 0:
            n = len(self.nodes)
            return torch.eye(n, dtype=self.nodes.dtype, device=self.nodes.device)
            
        if k not in self._diff_matrices:
            # Compute first derivative matrix if not cached
            if 1 not in self._diff_matrices:
                self._diff_matrices[1] = self._compute_derivative_matrix(self.nodes)
            
            # Compute k-th derivative matrix by composition
            Dk = self._diff_matrices[1]
            for i in range(k-1):
                Dk = Dk @ self._diff_matrices[1]
            self._diff_matrices[k] = Dk
            
        return self._diff_matrices[k]
    
    def _fourier_interpolate(self, x_eval, values):
        """
        Interpolate values using FFT
        """
        N = len(values)
        
        # Get Fourier coefficients
        coeffs = torch.fft.fft(values)
        
        # Map x_eval to standard domain [0,2π]
        x_eval_standard = self._to_standard(x_eval)
        
        # Evaluate Fourier series
        x_matrix = x_eval_standard.unsqueeze(1) * self.k.unsqueeze(0)
        fourier_matrix = torch.exp(1j * x_matrix)
        result = torch.real(fourier_matrix @ coeffs) / N
        
        return result
        
    def derivative(self, x_eval, k=1):
        """
        Compute k-th derivative at x_eval points using differentiation matrix
        """
        if k == 0:
            return self(x_eval)
            
        # Get k-th derivative matrix
        Dk = self.derivative_matrix(k)
        
        # Compute derivative values at nodes
        dk_nodes = Dk @ self.values
        
        # Interpolate to evaluation points
        return self._fourier_interpolate(x_eval, dk_nodes)
    
    def forward(self, x_eval):
        return self._fourier_interpolate(x_eval, self.values)


# %% [markdown]
# ## Sanity check: create model and set to interpolate exp(sin(x)). Are the derivatives accurate?

# %% [markdown]
# ### Cheb (non-periodic)

# %%
n_points = 41
domain_min = 3
domain_max = 5
# exp(sin(pi*x))
true_funcs = [
    lambda x: torch.exp(torch.sin(torch.pi * x)),
    lambda x: torch.exp(torch.sin(torch.pi * x)) * torch.pi * torch.cos(torch.pi * x),
    lambda x: torch.exp(torch.sin(torch.pi * x)) * (torch.pi**2) * (torch.cos(torch.pi * x)**2 - torch.sin(torch.pi * x)),
    lambda x: torch.exp(torch.sin(torch.pi * x)) * (torch.pi**3) * (torch.cos(torch.pi * x)**3 - 3*torch.cos(torch.pi * x)*torch.sin(torch.pi * x) - torch.cos(torch.pi * x))
]

model = ChebyshevInterpolationModel(n_points, domain_min=domain_min, domain_max=domain_max)
model.values.data = true_funcs[0](model.nodes)

# Test points - include both endpoints for visualization
x_eval = torch.linspace(domain_min, domain_max, 200)

# Compute derivatives
derivs = [model.derivative(x_eval, k) for k in range(4)]
true_derivs = [f(x_eval) for f in true_funcs]

# Plot errors
plt.figure(figsize=(15, 5))
for k in range(4):
    error = torch.abs(derivs[k] - true_derivs[k])
    plt.semilogy(x_eval, error.detach().cpu().numpy(), label=f'{k}-th derivative')
plt.grid(True)
plt.legend()
plt.title('Derivative Errors (Chebyshev)')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.show()

# Print max errors
print("\nMaximum errors:")
for k in range(4):
    error = torch.max(torch.abs(derivs[k] - true_derivs[k]))
    print(f"{k}-th derivative: {error:.2e}")

# %% [markdown]
# ### Fourier (periodic)

# %%
n_points = 40
domain_min = 3
domain_max = 5
test_function = lambda x : torch.exp(torch.sin(torch.pi * x))

model = FourierInterpolationModel(n_points, domain_min=domain_min, domain_max=domain_max)
model.values.data = test_function(model.nodes)

# Create fine grid for evaluation
x_eval = torch.linspace(domain_min, domain_max, 200)
y_eval = model(x_eval)

# Plot results
plt.figure(figsize=(12, 6))

# Plot interpolation
plt.plot(x_eval, y_eval.detach(), 'b-', label='Interpolation')

# Plot true function
y_true = test_function(x_eval)
plt.plot(x_eval, y_true, 'r--', label='True sin(πx)')

# Plot nodes
plt.plot(model.nodes, model.values.detach(), 'ko', label='Nodes')

plt.grid(True)
plt.legend()
plt.title('Trigonometric Interpolation Test')

# Plot error
plt.figure(figsize=(12, 4))
error = torch.abs(y_eval - y_true)
plt.semilogy(x_eval, error.detach(), label='Error')
plt.grid(True)
plt.legend()
plt.title('Interpolation Error')

plt.show()

# Print maximum error
print(f"Maximum error: {error.max().item():.2e}")

# %%
n_points = 40
domain_min = 3
domain_max = 5
# exp(sin(pi*x))
true_funcs = [
    lambda x: torch.exp(torch.sin(torch.pi * x)),
    lambda x: torch.exp(torch.sin(torch.pi * x)) * torch.pi * torch.cos(torch.pi * x),
    lambda x: torch.exp(torch.sin(torch.pi * x)) * (torch.pi**2) * (torch.cos(torch.pi * x)**2 - torch.sin(torch.pi * x)),
    lambda x: torch.exp(torch.sin(torch.pi * x)) * (torch.pi**3) * (torch.cos(torch.pi * x)**3 - 3*torch.cos(torch.pi * x)*torch.sin(torch.pi * x) - torch.cos(torch.pi * x))
]

model = FourierInterpolationModel(n_points, domain_min=domain_min, domain_max=domain_max)
model.values.data = true_funcs[0](model.nodes)

# Test points - include both endpoints for visualization
x_eval = torch.linspace(domain_min, domain_max, 200)

# Compute derivatives
derivs = [model.derivative(x_eval, k) for k in range(4)]
true_derivs = [f(x_eval) for f in true_funcs]

# Plot errors
plt.figure(figsize=(15, 5))
for k in range(4):
    error = torch.abs(derivs[k] - true_derivs[k])
    plt.semilogy(x_eval, error.detach().cpu().numpy(), label=f'{k}-th derivative')
plt.grid(True)
plt.legend()
plt.title('Derivative Errors (Fourier)')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.show()

# Print max errors
print("\nMaximum errors:")
for k in range(4):
    error = torch.max(torch.abs(derivs[k] - true_derivs[k]))
    print(f"{k}-th derivative: {error:.2e}")


# %% [markdown]
# ## ND interpolation

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
# ### Test ND interpolation

# %%
# Define a smooth test function
def test_function(x, y):
    return torch.exp(-x**2) * torch.cos(np.pi * y)

# Create interpolator
interp = SpectralInterpolationND(
    Ns=[33, 35],
    bases=['chebyshev', 'chebyshev'],
    domains=[(-1, 1), (-1, 1)]
)

# Set values at grid points
x_grid, y_grid = interp.mesh
interp.values.data = test_function(x_grid, y_grid)
print(f"self.values: {interp.values.shape}")

# Create a fine grid for visualization
n_vis = 100
x_vis = torch.linspace(-1, 1, n_vis)
y_vis = torch.linspace(-1, 1, n_vis)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
points_vis = torch.stack([X_vis.flatten(), Y_vis.flatten()], dim=1)
print(f"points_vis: {points_vis.shape}")

# Compute interpolated and exact values
interpolated = interp(points_vis)
print(f"interpolated: {interpolated.shape}")
interpolated = interpolated.reshape(n_vis, n_vis)
exact = test_function(X_vis, Y_vis)
error = torch.abs(interpolated - exact)

# Plotting
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), exact.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot interpolated solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_vis.numpy(), Y_vis.numpy(), interpolated.detach().numpy(), 
                       cmap='viridis')
ax2.set_title('Interpolated')

# Plot error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), error.detach().numpy(), 
                       cmap='magma')
ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print error statistics
print(f"Maximum absolute error: {error.max().item():.2e}")
print(f"Average absolute error: {error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")

# Test random points
n_test = 1000
test_points = torch.rand(n_test, 2)
test_points[:, 0] = 2 * test_points[:, 1] - 1  # x in [-1,1]
test_points[:, 1] = 2 * test_points[:, 1] - 1  # y in [-1,1]

interpolated_random = interp(test_points)
exact_random = test_function(test_points[:, 0], test_points[:, 1])
error_random = torch.abs(interpolated_random - exact_random)

print("\nRandom points test:")
print(f"Maximum absolute error: {error_random.max().item():.2e}")
print(f"Average absolute error: {error_random.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error_random**2).mean()).item():.2e}")


# %%
# Define a smooth test function
def test_function(x, y):
    return torch.sin(2 * np.pi * x) * torch.cos(np.pi * y)

# Create interpolator
interp = SpectralInterpolationND(
    Ns=[32, 33],
    bases=['fourier', 'chebyshev'],
    domains=[(0, 1), (-1, 1)]
)

# Set values at grid points
x_grid, y_grid = interp.mesh
interp.values.data = test_function(x_grid, y_grid)
print(f"self.values: {interp.values.shape}")

# Create a fine grid for visualization
n_vis = 100
x_vis = torch.linspace(0, 1, n_vis)
y_vis = torch.linspace(-1, 1, n_vis)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
points_vis = torch.stack([X_vis.flatten(), Y_vis.flatten()], dim=1)
print(f"points_vis: {points_vis.shape}")

# Compute interpolated and exact values
interpolated = interp(points_vis)
print(f"interpolated: {interpolated.shape}")
interpolated = interpolated.reshape(n_vis, n_vis)
exact = test_function(X_vis, Y_vis)
error = torch.abs(interpolated - exact)

# Plotting
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), exact.numpy(), 
                       cmap='viridis')
ax1.set_title('Exact')

# Plot interpolated solution
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X_vis.numpy(), Y_vis.numpy(), interpolated.detach().numpy(), 
                       cmap='viridis')
ax2.set_title('Interpolated')

# Plot error
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), error.detach().numpy(), 
                       cmap='magma')
ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# Print error statistics
print(f"Maximum absolute error: {error.max().item():.2e}")
print(f"Average absolute error: {error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error**2).mean()).item():.2e}")

# Test random points
n_test = 1000
test_points = torch.rand(n_test, 2)
test_points[:, 0] = test_points[:, 0]  # x in [0,1]
test_points[:, 1] = 2 * test_points[:, 1] - 1  # y in [-1,1]

interpolated_random = interp(test_points)
exact_random = test_function(test_points[:, 0], test_points[:, 1])
error_random = torch.abs(interpolated_random - exact_random)

print("\nRandom points test:")
print(f"Maximum absolute error: {error_random.max().item():.2e}")
print(f"Average absolute error: {error_random.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error_random**2).mean()).item():.2e}")


# %% [markdown]
# ## Test 2D spectral differentiation

# %%
# Define test function and its derivatives
def test_function(x, y):
    return torch.sin(2 * np.pi * x) * torch.cos(np.pi * y)

def d_dx(x, y):
    return 2 * np.pi * torch.cos(2 * np.pi * x) * torch.cos(np.pi * y)

def d_dy(x, y):
    return -np.pi * torch.sin(2 * np.pi * x) * torch.sin(np.pi * y)

def d_dxdy(x, y):
    return -2 * np.pi**2 * torch.cos(2 * np.pi * x) * torch.sin(np.pi * y)

# Create interpolator
interp = SpectralInterpolationND(
    Ns=[32, 33],
    bases=['fourier', 'chebyshev'],
    domains=[(0, 1), (-1, 1)]
)

# Set values at grid points
x_grid, y_grid = interp.mesh
interp.values.data = test_function(x_grid, y_grid)

# Create visualization grid
n_vis = 100
x_vis = torch.linspace(0, 1, n_vis)
y_vis = torch.linspace(-1, 1, n_vis)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
points_vis = torch.stack([X_vis.flatten(), Y_vis.flatten()], dim=1)

# Compute derivatives
dx_interp = interp.derivative(points_vis, k=(1,0)).reshape(n_vis, n_vis)
dy_interp = interp.derivative(points_vis, k=(0,1)).reshape(n_vis, n_vis)
dxy_interp = interp.derivative(points_vis, k=(1,1)).reshape(n_vis, n_vis)

# Compute exact derivatives
dx_exact = d_dx(X_vis, Y_vis)
dy_exact = d_dy(X_vis, Y_vis)
dxy_exact = d_dxdy(X_vis, Y_vis)

# Compute errors
dx_error = torch.abs(dx_interp - dx_exact)
dy_error = torch.abs(dy_interp - dy_exact)
dxy_error = torch.abs(dxy_interp - dxy_exact)

# Plotting
fig = plt.figure(figsize=(15, 15))

# Plot d/dx
ax1 = fig.add_subplot(331, projection='3d')
surf1 = ax1.plot_surface(X_vis.numpy(), Y_vis.numpy(), dx_exact.numpy(), cmap='viridis')
ax1.set_title('∂/∂x Exact')

ax2 = fig.add_subplot(332, projection='3d')
surf2 = ax2.plot_surface(X_vis.numpy(), Y_vis.numpy(), dx_interp.detach().numpy(), cmap='viridis')
ax2.set_title('∂/∂x Interpolated')

ax3 = fig.add_subplot(333, projection='3d')
surf3 = ax3.plot_surface(X_vis.numpy(), Y_vis.numpy(), dx_error.detach().numpy(), cmap='magma')
ax3.set_title('∂/∂x Error')

# Plot d/dy
ax4 = fig.add_subplot(334, projection='3d')
surf4 = ax4.plot_surface(X_vis.numpy(), Y_vis.numpy(), dy_exact.numpy(), cmap='viridis')
ax4.set_title('∂/∂y Exact')

ax5 = fig.add_subplot(335, projection='3d')
surf5 = ax5.plot_surface(X_vis.numpy(), Y_vis.numpy(), dy_interp.detach().numpy(), cmap='viridis')
ax5.set_title('∂/∂y Interpolated')

ax6 = fig.add_subplot(336, projection='3d')
surf6 = ax6.plot_surface(X_vis.numpy(), Y_vis.numpy(), dy_error.detach().numpy(), cmap='magma')
ax6.set_title('∂/∂y Error')

# Plot d²/dxdy
ax7 = fig.add_subplot(337, projection='3d')
surf7 = ax7.plot_surface(X_vis.numpy(), Y_vis.numpy(), dxy_exact.numpy(), cmap='viridis')
ax7.set_title('∂²/∂x∂y Exact')

ax8 = fig.add_subplot(338, projection='3d')
surf8 = ax8.plot_surface(X_vis.numpy(), Y_vis.numpy(), dxy_interp.detach().numpy(), cmap='viridis')
ax8.set_title('∂²/∂x∂y Interpolated')

ax9 = fig.add_subplot(339, projection='3d')
surf9 = ax9.plot_surface(X_vis.numpy(), Y_vis.numpy(), dxy_error.detach().numpy(), cmap='magma')
ax9.set_title('∂²/∂x∂y Error')

plt.tight_layout()
plt.show()

# Print error statistics
print("∂/∂x errors:")
print(f"Maximum absolute error: {dx_error.max().item():.2e}")
print(f"Average absolute error: {dx_error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((dx_error**2).mean()).item():.2e}")

print("\n∂/∂y errors:")
print(f"Maximum absolute error: {dy_error.max().item():.2e}")
print(f"Average absolute error: {dy_error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((dy_error**2).mean()).item():.2e}")

print("\n∂²/∂x∂y errors:")
print(f"Maximum absolute error: {dxy_error.max().item():.2e}")
print(f"Average absolute error: {dxy_error.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((dxy_error**2).mean()).item():.2e}")


# %% [markdown]
# ## Test interpolation on 1D through 4D

# %%
# Define a smooth 1D test function
def test_function_1d(x):
    return torch.sin(2 * np.pi * x) * torch.exp(-x**2)

interp = SpectralInterpolationND(
    Ns=[31],  # Different numbers to catch any indexing issues
    bases=['chebyshev'],
    domains=[(-1, 1)]
)

# Set values at grid points
x_grid, = interp.mesh
interp.values.data = test_function_1d(x_grid)

# Test random points
n_test = 10000
test_points = torch.rand(n_test, 1)
test_points[:, 0] = 2 * test_points[:, 0] - 1  # x in [0,1]

interpolated_random = interp(test_points)
exact_random = test_function_1d(test_points[:, 0])
error_random = torch.abs(interpolated_random - exact_random)

print("\nRandom points test:")
print(f"Maximum absolute error: {error_random.max().item():.2e}")
print(f"Average absolute error: {error_random.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error_random**2).mean()).item():.2e}")


# %%
# Define a smooth 3D test function
def test_function_2d(x, y):
    return torch.sin(2 * np.pi * x) * torch.exp(-y**2)

interp = SpectralInterpolationND(
    Ns=[31, 33],  # Different numbers to catch any indexing issues
    bases=['chebyshev', 'chebyshev'],
    domains=[(-1, 1), (-1, 1)]
)

# Set values at grid points
x_grid, y_grid = interp.mesh
interp.values.data = test_function_2d(x_grid, y_grid)

# Test random points
n_test = 2
test_points = torch.rand(n_test, 2)
test_points[:, 0] = 2 * test_points[:, 0] - 1  # x in [0,1]
test_points[:, 1] = 2 * test_points[:, 1] - 1  # y in [-1,1]

interpolated_random = interp(test_points)
exact_random = test_function_2d(test_points[:, 0], test_points[:, 1])
error_random = torch.abs(interpolated_random - exact_random)

print("\nRandom points test:")
print(f"Maximum absolute error: {error_random.max().item():.2e}")
print(f"Average absolute error: {error_random.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error_random**2).mean()).item():.2e}")


# %%
# Define a smooth 3D test function
def test_function_3d(x, y, z):
    return torch.sin(2 * np.pi * x) * torch.cos(np.pi * y) * torch.exp(-z**2)

interp = SpectralInterpolationND(
    Ns=[31, 33, 35],  # Different numbers to catch any indexing issues
    bases=['chebyshev', 'chebyshev', 'chebyshev'],
    domains=[(-1, 1), (-1, 1), (-1, 1)]
)

# Set values at grid points
x_grid, y_grid, z_grid = interp.mesh
interp.values.data = test_function_3d(x_grid, y_grid, z_grid)

# Test random points
n_test = 10000
test_points = torch.rand(n_test, 3)
test_points[:, 0] = 2 * test_points[:, 0] - 1  # x in [0,1]
test_points[:, 1] = 2 * test_points[:, 1] - 1  # y in [-1,1]
test_points[:, 2] = 2 * test_points[:, 2] - 1  # z in [-1,1]

interpolated_random = interp(test_points)
exact_random = test_function_3d(test_points[:, 0], test_points[:, 1], test_points[:, 2])
error_random = torch.abs(interpolated_random - exact_random)

print("\nRandom points test:")
print(f"Maximum absolute error: {error_random.max().item():.2e}")
print(f"Average absolute error: {error_random.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error_random**2).mean()).item():.2e}")


# %%
# Define a smooth 4D test function
def test_function_4d(x, y, z, w):
    return torch.sin(2 * np.pi * x) * torch.cos(np.pi * y) * torch.exp(-z**2) * torch.exp(-w**2)

interp = SpectralInterpolationND(
    Ns=[30, 31, 33, 35],  # Different numbers to catch any indexing issues
    bases=['fourier', 'chebyshev', 'chebyshev', 'chebyshev'],
    domains=[(0, 1), (-1, 1), (-1, 1), (-1, 1)]
)

# Set values at grid points
x_grid, y_grid, z_grid, w_grid = interp.mesh
interp.values.data = test_function_4d(x_grid, y_grid, z_grid, w_grid)

# Test random points
n_test = 1000
test_points = torch.rand(n_test, 4)
test_points[:, 0] = test_points[:, 0]  # x in [0,1]
test_points[:, 1] = 2 * test_points[:, 1] - 1  # y in [-1,1]
test_points[:, 2] = 2 * test_points[:, 2] - 1  # z in [-1,1]
test_points[:, 3] = 2 * test_points[:, 3] - 1  # w in [-1,1]

interpolated_random = interp(test_points)
exact_random = test_function_4d(test_points[:, 0], test_points[:, 1], test_points[:, 2], test_points[:, 3])
error_random = torch.abs(interpolated_random - exact_random)

print("\nRandom points test:")
print(f"Maximum absolute error: {error_random.max().item():.2e}")
print(f"Average absolute error: {error_random.mean().item():.2e}")
print(f"RMS error: {torch.sqrt((error_random**2).mean()).item():.2e}")


# %% [markdown]
# # PINN training

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

# %%
def compute_derivative(model, x, eval_mode=False):
    """
    Compute solution and its derivatives at points x
    
    Args:
        model: SpectralInterpolationND or neural network model
        x: points of shape (..., 2) where last dimension is (t,x)
        eval_mode: if True, detach gradients for evaluation
    
    Returns:
        u: solution at x points
        u_t: time derivative
        u_x: space derivative
    """
    if isinstance(model, SpectralInterpolationND):
        u = model(x)
        # Get time derivative (k=(1,0) means first derivative in t, zero in x)
        u_t = model.derivative(x, k=(1,0))  
        # Get space derivative (k=(0,1) means zero derivative in t, first in x)
        u_x = model.derivative(x, k=(0,1))
        
        if eval_mode:
            u = u.detach()
            u_t = u_t.detach()
            u_x = u_x.detach()
            
    else:
        # For MLP, compute gradients manually
        x_clone = x.clone().requires_grad_(True)
        u = model(x_clone)
        
        # Compute gradients with respect to t and x
        grads = torch.autograd.grad(u.sum(), x_clone, 
                                  create_graph=not eval_mode)[0]
        u_t = grads[..., 0]  # gradient with respect to t
        u_x = grads[..., 1]  # gradient with respect to x
        
        if eval_mode:
            u = u.detach()
            u_t = u_t.detach()
            u_x = u_x.detach()
            
    return u, u_t, u_x


def compute_pde_loss(model, colloc_points, ic_points=None, ic_weight=1.0, c=80):
    """
    Compute loss for advection equation u_t + c*u_x = 0
    
    Args:
        model: SpectralInterpolationND or neural network model
        colloc_points: points of shape (n_points, 2) for PDE residual
        ic_points: points of shape (n_ic_points, 2) for initial condition.
                  If None, assumes these are included in colloc_points
        c: advection speed
        
    Returns:
        total_loss: combined PDE and IC loss
        pde_residual: residual at collocation points
        ic_residual: residual at initial condition points
    """
    # Compute solution and derivatives at collocation points
    u, u_t, u_x = compute_derivative(model, colloc_points)
    
    # PDE residual: u_t + c*u_x = 0
    pde_residual = u_t + c*u_x
    pde_loss = torch.mean(pde_residual**2)
    
    # Initial condition: u(0,x) = sin(x)
    if ic_points is None:
        # Extract points where t=0
        mask = torch.abs(colloc_points[..., 0]) < 1e-10
        ic_points = colloc_points[mask]
    
    u_ic = model(ic_points)
    x_ic = ic_points[..., 1]  # get x coordinates
    ic_residual = u_ic - torch.sin(x_ic)
    ic_loss = ic_weight * torch.mean(ic_residual**2)
    
    total_loss = pde_loss + ic_loss
    
    return total_loss, pde_residual, ic_residual


# Simple iterative approach
def running_min(lst):
    result = []
    current_min = float('inf')
    for x in lst:
        current_min = min(current_min, x)
        result.append(current_min)
    return result

def compute_relative_l2_error(pred, true):
    """
    Compute relative L2 error between predicted and true values.
    
    Args:
        pred: predicted values (torch.Tensor)
        true: true values (torch.Tensor)
        
    Returns:
        float: relative L2 error
    """
    return (torch.sqrt(torch.mean((pred - true)**2)) / 
            torch.sqrt(torch.mean(true**2))).item()


def train_advection(model, n_epochs=1000, lr=1e-3, ic_weight=1.0, c=80, plot_every=100):
    """
    Train model to solve the advection equation
    u_t + c*u_x = 0, u(0,x) = sin(x)
    
    Args:
        model: SpectralInterpolationND model
        n_epochs: number of training epochs
        lr: learning rate
        c: advection speed
    """
    # Use model's built-in grid for collocation points
    t_points = model.nodes[0]  # Chebyshev in time
    x_points = model.nodes[1]  # Fourier in space
    T, X = torch.meshgrid(t_points, x_points, indexing='ij')
    colloc_points = torch.stack([T.flatten(), X.flatten()], dim=1)
    
    # Exact solution for comparisons
    def exact_solution(t, x):
        return torch.sin(x - c*t)
    
    # Create evaluation grid (finer than training grid)
    n_eval = 100
    t_eval = torch.linspace(0, 1, n_eval)
    x_eval = torch.linspace(0, 2*np.pi, n_eval)
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
        loss, pde_residual, ic_residual = compute_pde_loss(
            model, colloc_points, ic_weight=ic_weight, c=c
        )
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Compute L2 error against exact solution
        with torch.no_grad():
            u_pred = model(eval_points)
            u_exact = exact_solution(eval_points[:, 0], eval_points[:, 1])
            l2_error = torch.mean((u_pred - u_exact)**2).sqrt()
        
        # Record history
        history['loss'].append(loss.item())
        history['pde_residual'].append(torch.mean(pde_residual**2).item())
        history['ic_residual'].append(torch.mean(ic_residual**2).item())
        history['l2_error'].append(l2_error.item())
        
        # Print progress
        if (epoch + 1) % plot_every == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Total Loss: {loss.item():.2e}")
            print(f"PDE Residual: {torch.mean(pde_residual**2).item():.2e}")
            print(f"IC Residual: {torch.mean(ic_residual**2).item():.2e}")
            print(f"L2 Error: {l2_error.item():.2e}")
            
            # Visualize current solution
            plot_solution(model, eval_points, exact_solution, epoch+1)
    
    return history

def plot_solution(model, eval_points, exact_solution, epoch):
    """Plot current solution vs exact solution"""
    with torch.no_grad():
        u_pred = model(eval_points)
        u_exact = exact_solution(eval_points[:, 0], eval_points[:, 1])
        
        n_eval = int(torch.sqrt(torch.tensor(len(eval_points))))
        u_pred = u_pred.reshape(n_eval, n_eval)
        u_exact = u_exact.reshape(n_eval, n_eval)
        error = torch.abs(u_pred - u_exact)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Predicted solution
        im1 = ax1.imshow(u_pred.T, origin='lower', extent=[0, 1, 0, 2*np.pi],
                        aspect='auto', cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(f'Predicted (epoch {epoch})')
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        
        # Exact solution
        im2 = ax2.imshow(u_exact.T, origin='lower', extent=[0, 1, 0, 2*np.pi],
                        aspect='auto', cmap='viridis')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Exact')
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')
        
        # Error
        im3 = ax3.imshow(error.T, origin='lower', extent=[0, 1, 0, 2*np.pi],
                        aspect='auto', cmap='magma')
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('Error')
        ax3.set_xlabel('t')
        ax3.set_ylabel('x')
        
        plt.tight_layout()
        plt.show()



# %% jupyter={"outputs_hidden": true}
# Example usage:
model = SpectralInterpolationND(
    Ns=[21, 22],
    bases=['chebyshev', 'fourier'],
    # Ns=[11, 11],
    # bases=['chebyshev', 'chebyshev'],
    domains=[(0, 1), (0, 2*np.pi)]
)

history = train_advection(model, n_epochs=100000, lr=1e-3, ic_weight=10, c=8, plot_every=1000)

# %%
# Plot training history
plt.figure(figsize=(10, 5))
plt.semilogy(running_min(history['loss']), label='Total Loss')
plt.semilogy(running_min(history['ic_residual']), label='IC Residual')
plt.semilogy(running_min(history['pde_residual']), label='PDE Residual')
plt.semilogy(running_min(history['l2_error']), label='L2 Error')
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# %%
# Example usage:
model = SpectralInterpolationND(
    Ns=[41, 42],
    bases=['chebyshev', 'fourier'],
    # Ns=[11, 11],
    # bases=['chebyshev', 'chebyshev'],
    domains=[(0, 1), (0, 2*np.pi)]
)

history = train_advection(model, n_epochs=100000, lr=1e-3, ic_weight=1, c=80, plot_every=100)


# %% [markdown]
# ## Least squares

# %%
def setup_advection_ls(Nx=32, Nt=32, c=80):
    """
    Set up least squares problem for advection equation
    ut + c*ux = 0, u(0,x) = sin(x)
    """
    # Create interpolant
    interp = SpectralInterpolationND(
        Ns=[Nt, Nx],
        bases=['chebyshev', 'fourier'],
        domains=[(0, 1), (0, 2*np.pi)]
    )
    
    # Get PDE operator
    Dt = interp.derivative_matrix(k=(1,0))  # time derivative
    Dx = interp.derivative_matrix(k=(0,1))  # space derivative
    A = Dt + c*Dx  # PDE operator
    
    # Initial condition: extract t=0 values (first Nt points)
    # Our values are stored in row-major order (t varies fastest)
    IC = torch.zeros(Nx, Nt*Nx)
    for i in range(Nx):
        IC[i, Nx*(Nt-1) + i] = 1  # Select point where t=0 for each x
    
    # Right hand side
    b = torch.zeros(A.shape[0] + IC.shape[0])
    b[A.shape[0]:] = torch.sin(interp.nodes[1])  # sin(x) at Fourier points
    
    # Full system
    AA = torch.cat([A, IC], dim=0)
    
    return AA, b

# Setup least squares problem
Nx = 72
Nt = 73
c = 80

AA, b = setup_advection_ls(Nx, Nt, c)
print(f"Design matrix shape: {AA.shape}")
print(f"Target vector shape: {b.shape}")
print(f"Condition number of the design matrix: {torch.linalg.cond(AA)}")

# Create interpolant for visualization
interp = SpectralInterpolationND(
    Ns=[Nt, Nx],
    bases=['chebyshev', 'fourier'],
    domains=[(0, 1), (0, 2*np.pi)]
)

# Solve least squares
u = torch.linalg.lstsq(AA, b, driver="gelsd").solution

# Reshape solution to grid
u = u.reshape(Nt, Nx)

# Create fine grid for visualization
n_vis = 100
t_vis = torch.linspace(0, 1, n_vis)
x_vis = torch.linspace(0, 2*np.pi, n_vis)
T_vis, X_vis = torch.meshgrid(t_vis, x_vis, indexing='ij')
points_vis = torch.stack([T_vis.flatten(), X_vis.flatten()], dim=1)

# Interpolate solution to fine grid
u_interp = interp.interpolate(points_vis, u).reshape(n_vis, n_vis)

# Compute exact solution
def exact_solution(t, x):
    return torch.sin(x - c*t)
u_exact = exact_solution(T_vis, X_vis)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot numerical solution
im1 = ax1.imshow(u_interp.detach(), origin='lower', 
                 extent=[0, 2*np.pi, 0, 1], aspect='auto')
ax1.set_title('Numerical Solution')
plt.colorbar(im1, ax=ax1)

# Plot exact solution
im2 = ax2.imshow(u_exact.detach(), origin='lower', 
                 extent=[0, 2*np.pi, 0, 1], aspect='auto')
ax2.set_title('Exact Solution')
plt.colorbar(im2, ax=ax2)

# Plot error
error = torch.abs(u_interp - u_exact)
im3 = ax3.imshow(error.detach(), origin='lower', 
                 extent=[0, 2*np.pi, 0, 1], aspect='auto')
ax3.set_title('Error')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()

# Print error statistics
print(f"Maximum error: {error.max():.2e}")
print(f"Average error: {error.mean():.2e}")
print(f"RMS error: {torch.sqrt((error**2).mean()):.2e}")

# %%
