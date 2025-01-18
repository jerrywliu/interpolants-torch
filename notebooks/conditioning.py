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

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
import os, sys

BASE_DIR = "/pscratch/sd/j/jwl50/interpolants-torch"
sys.path.append(BASE_DIR)

# %%
from src.models.interpolant_nd import SpectralInterpolationND
from src.experiments.interpolation.pde_solns.advection import AdvectionTarget
from src.experiments.pdes.simple.advection import Advection

# %%
# Problem setup
c = 40
t_final = 1
u_0 = lambda x: torch.sin(x)
pde = Advection(c=c, t_final=t_final, u_0=u_0)
save_dir = "/pscratch/sd/j/jwl50/interpolants-torch/notebooks/figures/advection/"

# Eval
n_eval = 200
t_eval = torch.linspace(pde.domain[0][0], pde.domain[0][1], n_eval)
x_eval = torch.linspace(pde.domain[1][0], pde.domain[1][1], n_eval + 1)[:-1]

# Baseline: least squares
print("Fitting model with least squares...")
n_t_ls = c + 1
n_x_ls = c
bases_ls = ["chebyshev", "fourier"]
model_ls = SpectralInterpolationND(
    Ns=[n_t_ls, n_x_ls],
    bases=bases_ls,
    domains=pde.domain,
)
A, b = pde.get_least_squares(model_ls)
model_ls = pde.fit_least_squares(model_ls)
pde.plot_solution(
    [t_eval, x_eval],
    model_ls.interpolate([t_eval, x_eval]).detach(),
    save_path=os.path.join(save_dir, "advection_ls_solution.png"),
)

# %%
x_true = torch.linalg.lstsq(A, b, driver="gelsd").solution
x_true.shape

# %%
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
print(f"Condition number: {S[0]/S[-1]}")

# %%
import torch
from torch.optim import Adam, LBFGS
import time

class PreconditionedLinearSystem:
    """
    Preconditioned linear system solver using SVD-based preconditioning.
    Solves Ax = b by parameterizing x = My where M is based on SVD.
    """
    def __init__(self, A, b, precondition_method='svd'):
        """
        Initialize the preconditioned system.
        
        Parameters:
        -----------
        A : torch.Tensor
            Coefficient matrix of shape (m, n) where:
            m = number of equations (rows)
            n = number of variables (columns)
        b : torch.Tensor
            Target vector of shape (m,) or (m, 1)
        precondition_method : str
            Type of preconditioning ('svd' or 'none')
        """
        self.A = A
        self.m, self.n = A.shape
        self.b = b.view(self.m, 1) if b.dim() == 1 else b
        
        if precondition_method == 'jacobi':
            # For rectangular system, use diagonal of A^T A
            ATA_diag = torch.sum(A**2, dim=0)  # Diagonal of A^T A
            # Create diagonal preconditioner
            D_inv = torch.where(
                ATA_diag > 1e-10,
                1.0 / torch.sqrt(ATA_diag),  # Use sqrt since we apply M on both sides
                torch.zeros_like(ATA_diag)
            )
            self.M = torch.diag(D_inv)
            print(f"Jacobi scaling range: {torch.min(D_inv[D_inv > 0]):.2e} to {torch.max(D_inv):.2e}")
            
        elif precondition_method == 'svd':
            # Compute SVD of A (m x n)
            # U will be (m x m)
            # S will be (min(m,n),)
            # V will be (n x n)
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # Use economy SVD
            self.U = U
            self.S = S
            self.V = Vh.T  # Convert V^H to V
            
            # Create preconditioner using more conservative approach
            n = A.shape[1]  # Number of columns (variables)
            S_trunc = S[:n]
            V_trunc = self.V[:, :n]
            
            # Print original spectrum info
            print(f"Original condition number: {S_trunc[0]/S_trunc[-1]:.2e}")
            print(f"Singular values range: {S_trunc[0]:.2e} to {S_trunc[-1]:.2e}")
            
            # Use a more conservative threshold based on largest singular value
            max_s = S_trunc[0]
            threshold = max_s * 1e-4  # Only keep 4 orders of magnitude
            
            # Create inverse singular values with stronger regularization
            S_inv = torch.where(
                S_trunc > threshold,
                1.0 / (S_trunc + threshold),  # More stable regularization
                torch.zeros_like(S_trunc)
            )
            
            # Print diagnostic info
            n_used = torch.sum(S_trunc > threshold).item()
            print(f"Using {n_used}/{n} singular values")
            
            # Check preconditioned system
            self.M = V_trunc @ torch.diag(S_inv)
            AM = A @ self.M
            U_AM, S_AM, _ = torch.linalg.svd(AM)
            print(f"Preconditioned matrix condition number: {S_AM[0]/S_AM[-1]:.2e}")
            print(f"Preconditioned singular values range: {S_AM[0]:.2e} to {S_AM[-1]:.2e}")
            
            # Scale M to try to keep singular values close to 1
            scale = 1.0 / S_AM[0]
            self.M = scale * self.M
            
            # Print condition number and other diagnostics
            print(f"Original condition number: {S[0]/S[-1]:.2e}")
            print(f"Singular values range: {S[0]:.2e} to {S[-1]:.2e}")
            
            # Analyze the preconditioned system matrix AM
            AM = A @ self.M
            U_AM, S_AM, _ = torch.linalg.svd(AM)
            print(f"Preconditioned matrix condition number: {S_AM[0]/S_AM[-1]:.2e}")
            print(f"Preconditioned singular values range: {S_AM[0]:.2e} to {S_AM[-1]:.2e}")
        else:
            self.M = torch.eye(self.n, device=A.device)
    
    def compute_x(self, y):
        """Transform y to x using preconditioner."""
        return self.M @ y
    
    def loss_fn(self, y):
        """Compute loss for the preconditioned system."""
        x = self.compute_x(y)
        return torch.norm(self.A @ x - self.b) ** 2

def solve_preconditioned_system(A, b, method='adam', precondition='svd', 
                              max_iter=1000, tol=1e-6, verbose=True):
    """
    Solve linear system Ax = b using preconditioned gradient descent.
    
    Parameters:
    -----------
    A : torch.Tensor
        Coefficient matrix of shape (n, n)
    b : torch.Tensor
        Target vector of shape (n,) or (n, 1)
    method : str
        Optimization method ('adam' or 'lbfgs')
    precondition : str
        Preconditioning method ('svd' or 'none')
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance for relative residual
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    x : torch.Tensor
        Solution vector
    convergence_info : dict
        Information about convergence
    """
    # Initialize preconditioned system
    system = PreconditionedLinearSystem(A, b, precondition)
    
    # Initialize y (should have same dimension as number of columns in A)
    y = torch.randn(system.n, 1, requires_grad=True, device=A.device)
    
    # Initialize optimizer
    if method.lower() == 'adam':
        # For preconditioned system:
        # - Use larger learning rate
        # - Reduce momentum (beta1) since problem is well-conditioned
        # - Reduce epsilon to allow more aggressive steps
        if precondition == 'svd':
            optimizer = Adam([y], lr=0.1, betas=(0.5, 0.999), eps=1e-8)
        else:
            optimizer = Adam([y], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    else:  # L-BFGS
        optimizer = LBFGS([y], lr=1, max_iter=20, history_size=100,
                         line_search_fn='strong_wolfe')
    
    # Lists to store convergence information
    losses = []
    residuals = []
    times = []
    start_time = time.time()
    
    def closure():
        optimizer.zero_grad()
        loss = system.loss_fn(y)
        loss.backward()
        return loss
    
    # Main optimization loop
    best_y = None
    best_residual = float('inf')
    
    for i in range(max_iter):
        if method.lower() == 'adam':
            optimizer.zero_grad()
            loss = system.loss_fn(y)
            loss.backward()
            optimizer.step()
        else:
            loss = optimizer.step(closure)
            
        # Calculate relative residual
        with torch.no_grad():
            x = system.compute_x(y)
            current_residual = torch.norm(A @ x - b) / torch.norm(b)
            if current_residual < best_residual:
                best_residual = current_residual.item()
                best_y = y.clone().detach()
            
            losses.append(loss.item())
            residuals.append(current_residual.item())
            times.append(time.time() - start_time)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Relative Residual = {current_residual:.2e}")
            
            # Check convergence
            if current_residual < tol:
                if verbose:
                    print(f"Converged at iteration {i+1}")
                break
    
    # Compute final solution
    x = system.compute_x(best_y)
    
    convergence_info = {
        'losses': losses,
        'residuals': residuals,
        'times': times,
        'final_residual': best_residual,
        'iterations': len(losses)
    }
    
    return x, convergence_info


# %%
# Solve with and without preconditioning
methods = ['adam']
preconditions = ['none', 'jacobi']

for method in methods:
    for precond in preconditions:
        print(f"\nSolving with {method.upper()} ({precond} preconditioning):")
        x_sol, info = solve_preconditioned_system(
            A, b, method=method, precondition=precond, max_iter=10000)
        
        print(f"Final Residual: {info['final_residual']:.2e}")
        print(f"Time: {info['times'][-1]:.2f} seconds")
        # print(f"Error: {torch.norm(x_sol - x_true) / torch.norm(x_true):.2e}")

# Plot convergence if matplotlib is available
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

for method in methods:
    for precond in preconditions:
        _, info = solve_preconditioned_system(
            A, b, method=method, precondition=precond, verbose=False)
        label = f"{method.upper()} ({precond})"
        
        plt.subplot(121)
        plt.semilogy(info['residuals'], label=label)
        
        plt.subplot(122)
        plt.semilogy(info['times'], info['residuals'], label=label)

plt.subplot(121)
plt.xlabel('Iteration')
plt.ylabel('Relative Residual')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.xlabel('Time (seconds)')
plt.ylabel('Relative Residual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
U, S, Vh = torch.linalg.svd(A, full_matrices=True)

# %%
plt.semilogy(S)

# %%
S

# %%
# Create an ill-conditioned test problem
n = 40**2
U = torch.randn(n, n)
U, _ = torch.linalg.qr(U)  # Make U orthogonal
V = torch.randn(n, n)
V, _ = torch.linalg.qr(V)  # Make V orthogonal
s = torch.logspace(0, 4, n)  # Singular values from 1 to 10^4
A_new = U @ torch.diag(s) @ V.T
x_true = torch.randn(n, 1)
b_new = A_new @ x_true

# Solve with and without preconditioning
methods = ['adam', 'lbfgs']
preconditions = ['none', 'svd']

for method in methods:
    for precond in preconditions:
        print(f"\nSolving with {method.upper()} ({precond} preconditioning):")
        x_sol, info = solve_preconditioned_system(
            A_new, b_new, method=method, precondition=precond)
        
        print(f"Final Residual: {info['final_residual']:.2e}")
        print(f"Time: {info['times'][-1]:.2f} seconds")
        print(f"Error: {torch.norm(x_sol - x_true) / torch.norm(x_true):.2e}")

# Plot convergence if matplotlib is available
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

for method in methods:
    for precond in preconditions:
        _, info = solve_preconditioned_system(
            A_new, b_new, method=method, precondition=precond, verbose=False)
        label = f"{method.upper()} ({precond})"
        
        plt.subplot(121)
        plt.semilogy(info['residuals'], label=label)
        
        plt.subplot(122)
        plt.semilogy(info['times'], info['residuals'], label=label)

plt.subplot(121)
plt.xlabel('Iteration')
plt.ylabel('Relative Residual')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.xlabel('Time (seconds)')
plt.ylabel('Relative Residual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
