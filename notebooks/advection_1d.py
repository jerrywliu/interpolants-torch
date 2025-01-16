# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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
BASE_DIR = "/scratch/interpolants-torch"
sys.path.append(BASE_DIR)

# %%
from src.models.nd_interpolant import SpectralInterpolationND
from src.experiments.pdes.simple.advection import Advection

# %% [markdown]
# # Problem setup

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
n_t = 4*c+1
n_x = 4*c
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
def plot_derivative_comparison(u_x_spectral, u_x_autograd, t_grid, x_grid, save_path=None):
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
        extent=[x_grid[0].item(), x_grid[-1].item(), 
                t_grid[0].item(), t_grid[-1].item()],
        origin='lower',
        aspect='auto'
    )
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Spectral Derivative')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    
    # Plot autograd derivatives
    im2 = ax2.imshow(
        u_x_autograd.detach(),
        extent=[x_grid[0].item(), x_grid[-1].item(), 
                t_grid[0].item(), t_grid[-1].item()],
        origin='lower',
        aspect='auto'
    )
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Autograd Derivative')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    
    # Plot error on log scale
    error = torch.abs(u_x_spectral - u_x_autograd)
    im3 = ax3.imshow(
        error.detach(),
        extent=[x_grid[0].item(), x_grid[-1].item(), 
                t_grid[0].item(), t_grid[-1].item()],
        origin='lower',
        aspect='auto',
        norm='log'
    )
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Log Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    
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
    Ns=[33, 32],
    bases=["chebyshev", "fourier"],
    domains=[(0, 1), (0, 2*torch.pi)]
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
