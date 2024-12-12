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
    def __init__(self, num_points):
        """
        Model parameterized by values at Chebyshev points of the second kind
        
        Args:
            num_points: Number of points (N where N is number of points)
        """
        super().__init__()
        
        # Generate Chebyshev points of the second kind
        i = torch.linspace(0, 1, num_points)
        self.nodes = torch.cos(torch.pi * i)  # [-1, 1]
        
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
    
    def _cheb_interpolate(self, x_eval, values, eps=1e-14):
        """
        Interpolate values from nodes to x_eval using barycentric formula
        
        Args:
            x_eval: points to evaluate at (N_eval)
            values: values at self.nodes to interpolate from
            eps: tolerance for detecting exact node matches
        Returns:
            interpolated values at x_eval
        """
        # Compute difference matrix (N, N_eval)
        d_x = x_eval.unsqueeze(0) - self.nodes.unsqueeze(1)
        
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
        Compute the spectral differentiation matrix for Chebyshev points of the second kind
        using the stable barycentric formula.
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


# %% [markdown]
# ## Sanity check: create model and set to interpolate sin(x). Are the derivatives accurate?

# %%
# Sanity check: create model and set to interpolate sin(x). Are the derivatives accurate?
n_points = 21
model = LagrangeInterpolationModel(n_points)
model.values.data = torch.sin(model.nodes)

# Test points
x_eval = torch.linspace(-1, 1, 200)

# Compute derivatives
derivs = [model.derivative(x_eval, k) for k in range(4)]

# True derivatives
true_funcs = [
    torch.sin,
    torch.cos,
    lambda x: -torch.sin(x),
    lambda x: -torch.cos(x)
]
true_derivs = [f(x_eval) for f in true_funcs]

# Plot errors
plt.figure(figsize=(15, 5))
for k in range(4):
    error = torch.abs(derivs[k] - true_derivs[k])
    plt.semilogy(x_eval, error.detach().cpu().numpy(), label=f'{k}-th derivative')

plt.grid(True)
plt.legend()
plt.title('Derivative Errors')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.show()

# Print max errors
print("\nMaximum errors:")
for k in range(4):
    error = torch.max(torch.abs(derivs[k] - true_derivs[k]))
    print(f"{k}-th derivative: {error:.2e}")


# %% [markdown]
# ## PINN training utils

# %%
def compute_derivative(model, x, eval_mode=False):
    if isinstance(model, LagrangeInterpolationModel):
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

# PINN training utils
def compute_pde_loss(model, colloc_points, boundary_weight=1.0, u0=0):
    """
    Compute loss for ODE u' = u + x with u(0) = 0
    """
    u, du = compute_derivative(model, colloc_points)
        
    # PDE residual: u' = u + x
    pde_residual = du - u - colloc_points
        
    pde_loss = torch.mean(pde_residual**2)
    
    # Boundary condition: u(0) = u0
    bc_point = torch.tensor([0.0], dtype=torch.float64)
    bc_residual = model(bc_point)-u0
    bc_loss = boundary_weight * bc_residual**2
    
    return pde_loss + bc_loss, pde_residual, bc_residual

# # Torch implementation
# def running_min(x):
#     return torch.cummin(x, dim=0)[0]

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
    lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
    lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

    u_eval = lambda_u(x_eval)
    du_eval = lambda_du(x_eval)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
    plt.plot(x, u, label='Learned')
    plt.plot(x, true_u, '--', label='True')
    plt.scatter(colloc_points, u_colloc, color='red')
    plt.grid(True)
    plt.legend()
    plt.title('Solution')
    
    # Derivative
    plt.subplot(132)
    if du is not None:
        plt.plot(x, du, label="Learned u'")
    plt.plot(x, true_du, '--', label="True u'")
    plt.scatter(colloc_points, du_colloc, color='red')
    plt.grid(True)
    plt.legend()
    plt.title("Derivative")
    
    # Error
    plt.subplot(133)
    error = torch.abs(u - true_u)
    if du is not None:
        deriv_error = torch.abs(du - true_du)
        # plt.semilogy(x, error, label='Solution Error')
        # plt.semilogy(x, deriv_error, label='Derivative Error')
    if du is not None:
        error = torch.abs(du_colloc - u_colloc - colloc_points)
        plt.semilogy(colloc_points, error, label='PDE Error')
        plt.scatter(colloc_points, error, color='red')
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
    print(f"Relative L2 error for solution: {compute_relative_l2_error(u, true_u):.2e}")
    if du is not None:
        print(f"Relative L2 error for derivative: {compute_relative_l2_error(du, true_du):.2e}")
    # Relative l2 errors at collocation points
    print(f"Relative L2 error at colloc points for solution: {compute_relative_l2_error(u_colloc, lambda_u(colloc_points)):.2e}")
    if du is not None:
        print(f"Relative L2 error at colloc points for derivative: {compute_relative_l2_error(du_colloc, lambda_du(colloc_points)):.2e}")


# %% [markdown]
# ## Experiments

# %% [markdown]
# N=11, N_colloc=65, model=interpolant. Change the collocation.
# - Fixed: uniform, cheb.
# - Sampling once: uniform, cheb.
# - Resampling: uniform, cheb.

# %%
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'equispaced'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'chebyshev'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'random'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'cheb_random'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'random_resample'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'cheb_random_resample'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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

# %% [markdown]
# N_colloc=65, model=MLP. Change the collocation.
# - Resampling: uniform, cheb.

# %%
torch.manual_seed(0)

# Experiment params
hidden_dim = 64
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'random_resample'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = MLP(hidden_dim=hidden_dim, activation=torch.tanh)

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
torch.manual_seed(0)

# Experiment params
hidden_dim = 64
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'cheb_random_resample'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = MLP(hidden_dim=hidden_dim, activation=torch.tanh)

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
torch.manual_seed(0)

# Experiment params
hidden_dim = 64
n_colloc = 65
n_epochs = 30000
colloc_sampling = 'chebyshev'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = MLP(hidden_dim=hidden_dim, activation=torch.tanh)

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

# %%

# %% [markdown]
# ### N=21, N_colloc=21, model=interpolant, sampling=cheb

# %%
torch.manual_seed(0)

# Experiment params
n_points = 21
n_colloc = 21
n_epochs = 90000
colloc_sampling = 'chebyshev'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=41, N_colloc=41, model=interpolant, sampling=cheb

# %%
torch.manual_seed(0)

# Experiment params
n_points = 41
n_colloc = 61
n_epochs = 90000
colloc_sampling = 'chebyshev'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

# Train
history = train_pinn(
    model, 
    n_colloc=n_colloc, 
    n_epochs=n_epochs,
    lr=1e-2,
    boundary_weight=1.0,
    colloc_sampling=colloc_sampling,
    u0=1,
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=11, N_colloc=11, model=interpolant, sampling=random_resample

# %%
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 11
n_epochs = 90000
colloc_sampling = 'random_resample'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=11, N_colloc=11, model=interpolant, sampling=cheb_random_resample

# %%
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 11
n_epochs = 90000
colloc_sampling = 'cheb_random_resample'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=11, N_colloc=11, model=interpolant, sampling=cheb_random

# %%
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 11
n_epochs = 90000
colloc_sampling = 'cheb_random'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=11, N_colloc=41, model=interpolant, sampling=cheb_random

# %%
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 41
n_epochs = 90000
colloc_sampling = 'cheb_random'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=11, N_colloc=41, model=interpolant, sampling=cheb_random_resample

# %%
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 41
n_epochs = 90000
colloc_sampling = 'cheb_random_resample'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

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
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=11, N_colloc=41, model=interpolant, sampling=cheb_random, boundary_weight=0.1

# %%
torch.manual_seed(0)

# Experiment params
n_points = 11
n_colloc = 41
n_epochs = 90000
colloc_sampling = 'cheb_random'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = LagrangeInterpolationModel(n_points)

# Train
history = train_pinn(
    model, 
    n_colloc=n_colloc, 
    n_epochs=n_epochs,
    lr=1e-3,
    boundary_weight=0.1,
    colloc_sampling=colloc_sampling,
    u0=1,
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %% [markdown]
# ### N=11, N_colloc=41, model=MLP, sampling=cheb_random, boundary_weight=0.1

# %%
torch.manual_seed(0)

# Experiment params
hidden_dim = 64
n_colloc = 41
n_epochs = 90000
colloc_sampling = 'cheb_random'
u0 = 1

# u' = u + x, u(0) = u0 has solution u(x) = (u0+1)e^x - x - 1
lambda_u = lambda x : (u0+1)*torch.exp(x) - x - 1
lambda_du = lambda x : (u0+1)*torch.exp(x) - 1

# Create model
model = MLP(hidden_dim=hidden_dim, activation=torch.tanh)

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
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['u_error_max'], label='$||u-\hat{u}||_\infty$')
plt.semilogy(history['du_error_max'], label="$||u'-\hat{u'}||_\infty$")
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model, history, lambda_u, lambda_du)

# %%
