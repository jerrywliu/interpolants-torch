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
class MLP(nn.Module):
    def __init__(self, hidden_dim=32, activation=torch.tanh):
        """
        2-layer MLP that maps R -> R
        
        Args:
            hidden_dim: Dimension of hidden layer
            activation: Activation function to use (default: ReLU)
        """
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = activation
        
    def forward(self, x):
        # Ensure input is 2D tensor [batch_size, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)  # Return [batch_size]


# %%
class LagrangeInterpolationModel(nn.Module):
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
        n = len(nodes)
        D = torch.zeros((n, n), dtype=nodes.dtype, device=nodes.device)
        
        # Compute weights for endpoints
        c = torch.ones(n, dtype=nodes.dtype, device=nodes.device)
        c[0] = 2
        c[-1] = 2
        
        # Compute off-diagonal entries
        for i in range(n):
            for j in range(n):
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
# This implementation uses FFT to compute derivatives
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
    
    def _to_standard(self, x):
        """Map from physical domain to [0, 2π)"""
        return 2*np.pi * (x - self.domain_min) / self.domain_length
    
    def _from_standard(self, x):
        """Map from [0, 2π) to physical domain"""
        return self.domain_min + self.domain_length * x / (2*np.pi)
    
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
        Compute k-th derivative using FFT
        """
        if k == 0:
            return self(x_eval)
            
        N = len(self.values)
        
        # Get Fourier coefficients
        coeffs = torch.fft.fft(self.values)
        
        # Map x_eval to standard domain [0,2π]
        x_eval_standard = self._to_standard(x_eval)
        
        # Multiply by (ik)^k in Fourier space and scale for domain transformation
        # Each derivative brings down a factor of 2π/L from the chain rule
        scale = (2*np.pi/self.domain_length) ** k
        dk_coeffs = coeffs * (1j * self.k) ** k * scale
        
        # Evaluate derivative
        x_matrix = x_eval_standard.unsqueeze(1) * self.k.unsqueeze(0)
        fourier_matrix = torch.exp(1j * x_matrix)
        result = torch.real(fourier_matrix @ dk_coeffs) / N
        
        return result
    
    def forward(self, x_eval):
        return self._fourier_interpolate(x_eval, self.values)


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

model = LagrangeInterpolationModel(n_points, domain_min=domain_min, domain_max=domain_max)
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
# # PINN training

# %%
def compute_derivative(model, x, eval_mode=False):
    if isinstance(model, LagrangeInterpolationModel) or isinstance(model, FourierInterpolationModel):
        u = model(x)
        du = model.derivative(x, k=1)
    else:
        # For MLP, compute gradient manually
        x_clone = x.clone().requires_grad_(True)
        u = model(x_clone)
        # During eval, we don't need create_graph
        du = torch.autograd.grad(u.sum(), x_clone, 
                               create_graph=not eval_mode)[0]
        if eval_mode:
            u = u.detach()
            du = du.detach()
    return u, du

def compute_pde_loss(model, colloc_points, boundary_weight=1.0, u0=1.0):
    """
    Compute loss for ODE u' = 2π*cos(πx)*u with u(0) = 1
    """
    u, du = compute_derivative(model, colloc_points)
        
    # PDE residual: u' = 2π*cos(πx)*u
    pde_residual = du - 2*np.pi*torch.cos(np.pi*colloc_points)*u
        
    pde_loss = torch.mean(pde_residual**2)
    
    # Boundary condition: u(0) = 1
    bc_point = torch.tensor([0.0], dtype=torch.float64)
    bc_residual = model(bc_point) - u0
    bc_loss = boundary_weight * bc_residual**2
    
    return pde_loss + bc_loss, pde_residual, bc_residual

# Simple iterative approach
def running_min(lst):
    result = []
    current_min = float('inf')
    for x in lst:
        current_min = min(current_min, x)
        result.append(current_min)
    return result

def train_pinn(model, n_colloc=100, n_epochs=1000, lr=1e-3, boundary_weight=1.0, 
                colloc_sampling='equispaced', u0=0):
    """
    Train model to solve the ODE
    
    Args:
        model: neural network model
        n_colloc: number of collocation points
        n_epochs: number of training epochs
        lr: learning rate
        boundary_weight: weight for boundary condition term
        colloc_sampling: sampling strategy for collocation points
            - 'equispaced': evenly spaced points
            - 'chebyshev': Chebyshev points of second kind
            - 'random': uniformly random points, not resampled
            - 'random_resample': uniformly random points, resampled each epoch
    """
    def get_colloc_points():
        if colloc_sampling == 'equispaced':
            points = torch.linspace(-1, 1, n_colloc, dtype=torch.float64)
        elif colloc_sampling == 'chebyshev':
            i = torch.linspace(0, 1, n_colloc, dtype=torch.float64)
            points = torch.cos(torch.pi * i)
        elif colloc_sampling == 'random' or colloc_sampling == 'random_resample':
            # Random points in [-1, 1]
            points = 2 * torch.rand(n_colloc, dtype=torch.float64) - 1
            # Sort points for better visualization
            points, _ = torch.sort(points)
        elif colloc_sampling == 'cheb_random' or colloc_sampling == 'cheb_random_resample':
            # Random points equispaced along the circle, then projected to [-1, 1]
            i = torch.rand(n_colloc, dtype=torch.float64)
            points = torch.cos(torch.pi * i)
            # Sort points for better visualization
            points, _ = torch.sort(points)
        else:
            raise ValueError(f"Unknown sampling method: {colloc_sampling}")
        return points
    
    # Initial collocation points
    colloc_points = get_colloc_points()

    # Eval points
    x_eval = torch.linspace(-1, 1, 200, dtype=torch.float64)

    # True soln
    lambda_u = lambda x: torch.exp(2*torch.sin(np.pi*x))
    lambda_du = lambda x: 2*np.pi*torch.cos(np.pi*x)*torch.exp(2*torch.sin(np.pi*x))

    u_eval = lambda_u(x_eval)
    du_eval = lambda_du(x_eval)
    
    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Training history
    history = {'loss': [], 'pde_residual': [], 'bc_residual': [], 'colloc_points': colloc_points, 'u_error_max': [], 'du_error_max': []}
    
    # Training loop
    for epoch in tqdm(range(n_epochs)):

        # Eval
        u_pred_eval, du_pred_eval = compute_derivative(model, x_eval, eval_mode=True)
        
        u_error = torch.abs(u_pred_eval - u_eval)
        du_error = torch.abs(du_pred_eval - du_eval)
        history['u_error_max'].append(torch.max(u_error).item())
        history['du_error_max'].append(torch.max(du_error).item())
        
        # Resample points if using random sampling
        if 'resample' in colloc_sampling:
            colloc_points = get_colloc_points()
            
        optimizer.zero_grad()
        
        # Compute loss
        loss, pde_residual, bc_residual = compute_pde_loss(
            model, colloc_points, boundary_weight, u0
        )
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Record history
        history['loss'].append(loss.item())
        history['pde_residual'].append(torch.mean(pde_residual**2).item())
        history['bc_residual'].append(bc_residual.item()**2)
        
        # Print progress
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.2e}")
    
    return history

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

def plot_solution(model, history, lambda_u, lambda_du):
    """Plot the learned solution and compare with true solution"""

    # Plot points
    x = torch.linspace(-1, 1, 200, dtype=torch.float64)

    # Collocation points
    colloc_points = history['colloc_points']
    
    # True solution
    true_u = lambda_u(x)
    true_du = lambda_du(x)

    # Model prediction
    u, du = compute_derivative(model, x, eval_mode=True)
    u_colloc, du_colloc = compute_derivative(model, colloc_points, eval_mode=True)
    
    plt.figure(figsize=(15, 5))
    
    # Solution
    plt.subplot(131)
    plt.plot(x, u.detach(), label='Learned')
    plt.plot(x, true_u.detach(), '--', label='True')
    plt.scatter(colloc_points, u_colloc.detach(), color='red')
    plt.grid(True)
    plt.legend()
    plt.title('Solution')
    
    # Derivative
    plt.subplot(132)
    if du is not None:
        plt.plot(x, du.detach(), label="Learned u'")
    plt.plot(x, true_du.detach(), '--', label="True u'")
    plt.scatter(colloc_points, du_colloc.detach(), color='red')
    plt.grid(True)
    plt.legend()
    plt.title("Derivative")
    
    # Error
    plt.subplot(133)
    error = torch.abs(u - true_u).detach()
    if du is not None:
        deriv_error = torch.abs(du - true_du).detach()
    if du is not None:
        error_colloc = torch.abs(du_colloc - 2*np.pi*torch.cos(np.pi*colloc_points)*u_colloc).detach()
        plt.semilogy(colloc_points, error_colloc, label='PDE Error')
        plt.scatter(colloc_points, error_colloc, color='red')
    plt.grid(True)
    plt.legend()
    plt.title('Error')
    
    plt.tight_layout()
    plt.show()

    # Print maximum errors
    print(f"Maximum solution error: {torch.max(error):.2e}")
    if du is not None:
        print(f"Maximum derivative error: {torch.max(deriv_error):.2e}")

    # Print relative l2 errors
    print(f"Relative L2 error for solution: {compute_relative_l2_error(u.detach(), true_u.detach()):.2e}")
    if du is not None:
        print(f"Relative L2 error for derivative: {compute_relative_l2_error(du.detach(), true_du.detach()):.2e}")
    # Relative l2 errors at collocation points
    print(f"Relative L2 error at colloc points for solution: {compute_relative_l2_error(u_colloc.detach(), lambda_u(colloc_points).detach()):.2e}")
    if du is not None:
        print(f"Relative L2 error at colloc points for derivative: {compute_relative_l2_error(du_colloc.detach(), lambda_du(colloc_points).detach()):.2e}")


# %% [markdown]
# Equation:
# $$u' = 2\pi \cos(2\pi x) u$$
# Solution:
# $$u = C \exp(\sin(2\pi x))$$
# We set $u(0) = 1$ to get $C = 1$.

# %%
torch.manual_seed(0)

# Experiment params
n_points = 40
domain_min = -1
domain_max = 1
n_colloc = 65
n_epochs = 100000
colloc_sampling = 'equispaced'
u0 = 1

# u' = 2*pi*cos(2*pi*x)*u, u(0) = 1 has solution u(x) = exp(sin(2*pi*x)), which is periodic on [-1, 1]
lambda_u = lambda x: torch.exp(2*torch.sin(np.pi*x))
lambda_du = lambda x: 2*np.pi*torch.cos(np.pi*x)*torch.exp(2*torch.sin(np.pi*x))

# Create model
model = FourierInterpolationModel(n_points, domain_min=domain_min, domain_max=domain_max)

# Train
history = train_pinn(
    model, 
    n_colloc=n_colloc, 
    n_epochs=n_epochs,
    lr=1e-3,
    boundary_weight=1.0,
    colloc_sampling=colloc_sampling,
    u0=1,
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.semilogy(running_min(history['loss']), label='Total Loss')
plt.semilogy(running_min(history['bc_residual']), label='BC Residual')
plt.semilogy(running_min(history['pde_residual']), label='PDE Residual')
plt.semilogy(running_min(history['u_error_max']), label='$||u-\hat{u}||_\infty$')
plt.semilogy(running_min(history['du_error_max']), label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %%
