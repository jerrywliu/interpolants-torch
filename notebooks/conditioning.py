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
plt.rcParams['text.usetex'] = True

# %%
import os, sys

BASE_DIR = "/pscratch/sd/j/jwl50/interpolants-torch"
sys.path.append(BASE_DIR)

# %%
from src.models.interpolant_nd import SpectralInterpolationND
from src.experiments.interpolation.simple_fcns.sine_1d import Sine1DTarget

# %%
# Problem setup
target = Sine1DTarget()

# Polynomial interpolation
n_x = 20
bases = ["fourier"]
domains = target.domain
model = SpectralInterpolationND(
    Ns=[n_x],
    bases=bases,
    domains=domains,
)

# Interpolate the data
model.values.data = target.get_function(model.nodes[0])

# %%
# Check that the model interpolates the function
n_eval_points = 200
x_eval = torch.linspace(model.domains[0][0], model.domains[0][1], n_eval_points)
f_eval_pred = model([x_eval]).detach()
f_eval_true = target.get_function(x_eval)
plt.semilogy(x_eval, torch.abs(f_eval_pred - f_eval_true))
print(f"Average error: {torch.mean(torch.abs(f_eval_pred - f_eval_true))}")

# %%
# Get the differentiation matrix
D_spectral = model.derivative_matrix([1])
print(f"Shape of spectral D matrix: {D_spectral.shape}")
U, S, Vh = torch.linalg.svd(D_spectral, full_matrices=False)
print(f"Condition number of spectral D matrix: {torch.linalg.cond(D_spectral)}")
print(f"sigma_max / sigma_min: {S[0] / S[-3]}")

# %% jupyter={"source_hidden": true}
# Compare to finite difference matrix

import numpy as np

def finite_difference_matrix(n, dx=1.0, order=1, accuracy=2):
    """
    Generate an nxn finite difference matrix for approximating derivatives.
    
    Parameters:
    -----------
    n : int
        Size of the matrix (number of points)
    dx : float, optional
        Spacing between points (default=1.0)
    order : int, optional
        Order of the derivative to approximate (default=1)
    accuracy : int, optional
        Order of accuracy (2 for central difference, 1 for forward/backward)
        Must be even for central difference (default=2)
        
    Returns:
    --------
    D : numpy.ndarray
        nxn finite difference matrix
    """
    if accuracy < 1:
        raise ValueError("Accuracy order must be positive")
    if order < 1:
        raise ValueError("Derivative order must be positive")
        
    # Initialize the matrix
    D = np.zeros((n, n))
    
    # Compute stencil coefficients using central difference
    if accuracy % 2 == 0:  # Central difference
        half_width = accuracy // 2
        # Generate points for finite difference stencil
        x = np.arange(-half_width, half_width + 1)
        # Create Vandermonde matrix
        A = np.vander(x, accuracy + 1, increasing=True)
        # Create target vector (all zeros except for factorial of derivative order)
        b = np.zeros(accuracy + 1)
        b[order] = np.math.factorial(order)
        # Solve for coefficients
        coeffs = np.linalg.solve(A, b)
        
        # Fill the matrix using periodic boundary conditions
        for i in range(n):
            for j, coeff in enumerate(coeffs):
                idx = (i - half_width + j) % n  # Periodic boundary conditions
                D[i, idx] = coeff / (dx ** order)
                
    else:  # Forward difference (less accurate but handles boundaries better)
        # Similar process but using forward difference stencil
        x = np.arange(0, accuracy + 1)
        A = np.vander(x, accuracy + 1, increasing=True)
        b = np.zeros(accuracy + 1)
        b[order] = np.math.factorial(order)
        coeffs = np.linalg.solve(A, b)
        
        # Fill the matrix using forward differences at boundaries
        for i in range(n):
            for j, coeff in enumerate(coeffs):
                if i + j < n:  # Only fill within bounds
                    D[i, i + j] = coeff / (dx ** order)
                
    return D


# %%
D_FD = torch.tensor(finite_difference_matrix(n_x, dx=(target.domain[0][1] - target.domain[0][0])/n_x, order=1, accuracy=2))
print(f"Shape of FD D matrix: {D_FD.shape}")
U, S, Vh = torch.linalg.svd(D_FD, full_matrices=False)
print(f"Condition number of FD D matrix: {torch.linalg.cond(D_FD)}")
print(f"sigma_max / sigma_min: {S[0] / S[-3]}")

# %%
