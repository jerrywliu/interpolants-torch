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
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float64)


# %% [markdown]
# # Chebyshev polynomials

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
# ### How well can polynomials fit a function with near discontinuity?

# %%
from dataclasses import dataclass

@dataclass
class LogisticConfig:
    alpha: float = 50.0  # steepness
    transition_point: float = 0.5  # where the transition happens

def steep_transition(x, config: LogisticConfig):
    """Logistic transition function from example"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-config.alpha * (config.transition_point - torch.abs(x))))

def test_polynomial_interpolation(num_points_list=None, config=LogisticConfig()):
    if num_points_list is None:
        # Generate a range of degrees for convergence study
        num_points_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    """Test polynomial interpolation for different numbers of points"""
    # Dense grid for evaluation
    x_eval = torch.linspace(-1, 1, 1000)
    y_true = steep_transition(x_eval, config)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot true function
    ax1.plot(x_eval, y_true, 'k-', label='True', linewidth=2)
    ax2.set_yscale('log')
    
    # Store errors for convergence plot
    errors = []
    
    # Use colormap for many degrees
    colors = plt.cm.viridis(np.linspace(0, 1, len(num_points_list)))
    
    for n, color in zip(num_points_list, colors):
        # Create and evaluate interpolant
        model = LagrangeInterpolationModel(n)
        
        # Set values at nodes directly
        with torch.no_grad():
            model.values.copy_(steep_transition(model.nodes, config))
        
        # Evaluate interpolant
        y_interp = model(x_eval).detach()
        
        # Compute error
        error = torch.abs(y_interp - y_true)
        errors.append(error)
        
        # Plot results
        ax1.plot(x_eval, y_interp, '--', color=color, label=f'N={n}', alpha=0.7)
        ax2.plot(x_eval, error, '-', color=color, label=f'N={n}')
        
        # Plot nodes
        ax1.plot(model.nodes, model.values.detach(), '.', color=color, markersize=8)
        
        # Print max error
        print(f"N={n}, Max Error: {error.max():.2e}")
    
    ax1.set_title("Function Comparison")
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(-1.5, 1.5)
    
    ax2.set_title("Interpolation Error (log scale)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(1e-16, 1e0)
    
    # Add convergence plot
    degrees = torch.tensor(num_points_list) - 1  # number of points - 1 = polynomial degree
    max_errors = torch.tensor([error.max() for n, error in zip(num_points_list, errors)])
    
    ax3.loglog(degrees, max_errors, 'ko-', label='Max Error')
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Maximum Error')
    ax3.set_title('Convergence with Polynomial Degree')
    ax3.grid(True)
    
    # # Add theoretical reference lines
    # ref_x = torch.tensor([degrees.min(), degrees.max()])
    # # Add 1/N reference line
    # ref_y = max_errors[0] * (ref_x[0] / ref_x)
    # ax3.loglog(ref_x, ref_y, 'r--', label='O(1/N)')
    # # Add exp(-N) reference line
    # ref_y = max_errors[0] * torch.exp(-(ref_x - ref_x[0]))
    # ax3.loglog(ref_x, ref_y, 'b--', label='O(exp(-N))')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    return degrees, max_errors

test_polynomial_interpolation(num_points_list=[10, 20, 40, 80, 160])

# %% [markdown]
# # Rational interpolation

# %% [markdown]
# ### First, let's write the 1D rational interpolation code, which starts with Chebyshev nodes, then adds additional poles in complex conjugate pairs.

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_cheb_nodes(n):
    """Get Chebyshev points of the second kind"""
    i = torch.linspace(0, 1, n)
    return torch.cos(torch.pi * i)

def compute_barycentric_weights(nodes, pole_real=None, pole_imag=None):
    """
    Compute barycentric weights maintaining gradients through poles.
    
    Args:
        nodes: tensor of nodes
        pole_real: tensor of real parts of poles
        pole_imag: tensor of imaginary parts of poles (positive values, will add conjugates)
    """
    n = len(nodes)
    log_weights = torch.zeros_like(nodes)
    signs = torch.ones_like(nodes)
    
    for j in range(n):
        # Compute standard node differences
        diffs = nodes[j] - nodes
        diffs = diffs[torch.arange(n) != j]
        log_weights[j] = -torch.sum(torch.log(torch.abs(diffs)))
        signs[j] = (-1)**(torch.sum(diffs < 0))
        
        # Add pole contributions maintaining gradients
        if pole_real is not None and pole_imag is not None:
            for r, i in zip(pole_real, pole_imag):
                # Add contribution from pole and its conjugate
                pole_diff = torch.sqrt((nodes[j] - r)**2 + i**2)  # distance to p
                log_weights[j] += torch.log(pole_diff)  # p contribution
                # Conjugate is same distance so just multiply by 2
                log_weights[j] += torch.log(pole_diff)  # p̄ contribution
    
    # Convert back from log space
    log_weights = log_weights - log_weights.max()
    weights = signs * torch.exp(log_weights)
    weights = weights / torch.abs(weights[0])
    
    return weights

class RationalInterpolationModel(nn.Module):
    def __init__(self, num_points, num_poles=2):
        """
        Rational interpolation with Chebyshev backbone and complex conjugate poles
        
        Args:
            num_points: Number of Chebyshev points
            num_poles: Number of pole pairs (must be even for conjugate pairs)
        """
        super().__init__()
        assert num_poles % 2 == 0, "num_poles must be even for conjugate pairs"
        
        # Chebyshev backbone
        self.nodes = get_cheb_nodes(num_points)
        
        # Initialize poles as conjugate pairs
        num_pairs = num_poles // 2
        # Initialize real parts uniformly in [-1, 1]
        self.pole_real = nn.Parameter(2 * torch.rand(num_pairs) - 1)
        # Initialize imaginary parts small and positive
        self.pole_imag = nn.Parameter(0.1 * torch.ones(num_pairs))
        
        # Values at nodes
        self.values = nn.Parameter(torch.zeros(num_points))
        
        # Barycentric weights
        self.update_weights()
        
        # Cache for differentiation matrices
        self._diff_matrices = {}
    
    def get_poles(self):
        """Return complex poles as (real, imag) pairs maintaining gradients"""
        poles = []
        for r, i in zip(self.pole_real, self.pole_imag):
            # Create conjugate pair keeping gradient connection
            poles.extend([(r, i), (r, -i)])
        return torch.tensor(poles)
    
    def update_weights(self):
        """Update barycentric weights based on current pole locations"""
        poles = self.get_poles()
        self.weights = compute_barycentric_weights(self.nodes, self.pole_real, self.pole_imag)
        
    def _compute_derivative_matrix(self, nodes):
        """
        Compute the spectral differentiation matrix for the given nodes
        using the dynamic barycentric weights.
        """
        n = len(nodes)
        D = torch.zeros((n, n), dtype=nodes.dtype, device=nodes.device)

        # Compute dynamic weights
        weights = compute_barycentric_weights(nodes, self.pole_real, self.pole_imag)

        # Compute off-diagonal entries
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i,j] = weights[j] / weights[i] / (nodes[i] - nodes[j])

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

    def _interpolate(self, x_eval, values):
        """
        Interpolate values from nodes to x_eval using barycentric formula
        
        Args:
            x_eval: points to evaluate at (N_eval)
            values: values at self.nodes to interpolate from
        Returns:
            interpolated values at x_eval
        """
        # Compute weights directly for interpolation
        weights = compute_barycentric_weights(self.nodes, self.pole_real, self.pole_imag)
        
        # Compute difference matrix
        d_x = x_eval.unsqueeze(0) - self.nodes.unsqueeze(1)
        
        # Handle numerical instability
        small_diff = torch.abs(d_x) < 1e-14
        small_diff_max = torch.max(small_diff, dim=0).values
        d_x = torch.where(
            small_diff_max[None, :],
            torch.zeros_like(d_x),
            1.0 / d_x
        )
        d_x[small_diff] = 1
        
        # Compute barycentric terms using weights
        f_eval_num = torch.einsum("...n,nm,n->...m", values, d_x, weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, weights)
        
        return f_eval_num / f_eval_denom

    def forward(self, x_eval):
        """Evaluate rational interpolant at x_eval points"""
        return self._interpolate(x_eval, self.values)

    def derivative(self, x_eval, k=1):
        """
        Compute k-th derivative of rational interpolant at x_eval points
        """
        if k == 0:
            return self(x_eval)
            
        # Get k-th derivative matrix
        Dk = self.derivative_matrix(k)
        
        # Compute derivative values at nodes (differentiable w.r.t self.values)
        dk_nodes = Dk @ self.values
        
        # Interpolate to evaluation points using the derivative values
        return self._interpolate(x_eval, dk_nodes)
    
def plot_results(x_eval, y_true, y_standard, y_rational, rational_model):
    """Plot comparison and pole locations"""
    plt.figure(figsize=(15, 5))
    
    # Function comparison
    plt.subplot(131)
    plt.plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
    plt.plot(x_eval, y_standard, 'b--', label='Standard')
    plt.plot(x_eval, y_rational, 'r:', label='Rational')
    plt.grid(True)
    plt.legend()
    plt.title('Function Comparison')
    
    # Complex plane with poles
    plt.subplot(132)
    poles = rational_model.get_poles()
    pole_real, pole_imag = poles[:, 0], poles[:, 1]
    
    # Plot real axis
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.plot([-1, 1], [0, 0], 'k-', linewidth=2, label='[-1,1]')
    
    # Plot poles
    plt.plot(pole_real, pole_imag, 'r+', markersize=10, label='Poles')
    plt.plot(pole_real, -pole_imag, 'r+', markersize=10)  # conjugates
    
    plt.grid(True)
    plt.axis('equal')  # maintain aspect ratio
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Pole Locations')
    plt.legend()
    
    # Error comparison
    plt.subplot(133)
    err_standard = torch.abs(y_standard - y_true)
    err_rational = torch.abs(y_rational - y_true)
    plt.semilogy(x_eval, err_standard, 'b-', label='Standard Error')
    plt.semilogy(x_eval, err_rational, 'r-', label='Rational Error')
    for pole_real_component in pole_real:
        plt.axvline(pole_real_component, color="black", linestyle="--")
    plt.grid(True)
    plt.legend()
    plt.title('Error Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # Print max errors
    print(f"Standard max error: {err_standard.max():.2e}")
    print(f"Rational max error: {err_rational.max():.2e}")

def test_rational_interp():
    """Test rational interpolation on cos(x)"""
    # Create models
    n_points = 20
    standard_model = LagrangeInterpolationModel(n_points)
    rational_model = RationalInterpolationModel(n_points, num_poles=4)
    
    # Evaluation points
    x_eval = torch.linspace(-1, 1, 1000)
    y_true = torch.cos(x_eval * torch.pi)
    
    # Set values at nodes for both models
    with torch.no_grad():
        standard_model.values.copy_(torch.cos(standard_model.nodes * torch.pi))
        rational_model.values.copy_(torch.cos(rational_model.nodes * torch.pi))
        
    # Initialize real parts
    rational_model.pole_real.data = torch.tensor([-0.5, 0.5])
    # Initialize imaginary parts
    rational_model.pole_imag.data = torch.tensor([0.7, 0.7])
    
    # Evaluate models
    y_standard = standard_model(x_eval).detach()
    y_rational = rational_model(x_eval).detach()
    
    # Plot results
    plot_results(x_eval, y_true, y_standard, y_rational, rational_model)

if __name__ == "__main__":
    test_rational_interp()

# %% [markdown]
# # Let's try learning the poles for the steep function.

# %%
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def steep_transition(x, alpha=50.0):
    """Logistic transition function"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))

def train_interpolation(standard_model, rational_model, num_epochs=1000, batch_size=128, freeze_poles=500):
    """Train both models on the steep transition function"""
    # Generate training data with more points near transition
    x_uniform = torch.linspace(-1, 1, 1000)
    x_dense = torch.linspace(-0.6, 0.4, 1000)  # More points near transition
    x_train = torch.cat([x_uniform, x_dense])
    y_train = steep_transition(x_train)
    
    # Create data loader
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizers
    opt_standard = optim.Adam(standard_model.parameters(), lr=1e-3)
    opt_rational = optim.Adam(rational_model.parameters(), lr=1e-3)
    
    # Training history
    standard_losses = []
    rational_losses = []
    
    for epoch in tqdm(range(num_epochs)):
        standard_epoch_loss = 0.0
        rational_epoch_loss = 0.0
        
        if epoch == freeze_poles:
            rational_model.pole_real.requires_grad = False
            rational_model.pole_imag.requires_grad = False
        
        for x_batch, y_batch in loader:
            # Train standard model
            opt_standard.zero_grad()
            y_pred_standard = standard_model(x_batch)
            loss_standard = torch.nn.functional.mse_loss(y_pred_standard, y_batch)
            loss_standard.backward()
            opt_standard.step()
            
            # Train rational model
            opt_rational.zero_grad()
            y_pred_rational = rational_model(x_batch)
            loss_rational = torch.nn.functional.mse_loss(y_pred_rational, y_batch)
            loss_rational.backward()
            opt_rational.step()
            
            standard_epoch_loss += loss_standard.item()
            rational_epoch_loss += loss_rational.item()
        
        # Record average epoch losses
        standard_losses.append(standard_epoch_loss / len(loader))
        rational_losses.append(rational_epoch_loss / len(loader))
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Standard Loss: {standard_losses[-1]:.6f}")
            print(f"Rational Loss: {rational_losses[-1]:.6f}")
    
    return standard_losses, rational_losses

def plot_training_results(x_eval, standard_model, rational_model, losses):
    """Plot training results and final approximation"""
    standard_losses, rational_losses = losses
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(131)
    plt.semilogy(standard_losses, 'b-', label='Standard', alpha=0.7)
    plt.semilogy(rational_losses, 'r-', label='Rational', alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    # Plot pole locations
    plt.subplot(132)
    poles = rational_model.get_poles()
    pole_real, pole_imag = poles[:, 0], poles[:, 1]
    
    # Plot real axis
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.plot([-1, 1], [0, 0], 'k-', linewidth=2, label='[-1,1]')
    
    # Plot poles
    plt.plot(pole_real, pole_imag, 'r+', markersize=10, label='Poles')
    plt.plot(pole_real, -pole_imag, 'r+', markersize=10)
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Final Pole Locations')
    plt.legend()
    
    # Plot function comparison
    plt.subplot(133)
    y_true = steep_transition(x_eval)
    with torch.no_grad():
        y_standard = standard_model(x_eval)
        y_rational = rational_model(x_eval)
    
    plt.plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
    plt.plot(x_eval, y_standard, 'b--', label='Standard')
    plt.plot(x_eval, y_rational, 'r:', label='Rational')
    
    for pole_real_component in pole_real:
        plt.axvline(pole_real_component, color="black", linestyle="--")
    
    # Plot nodes
    plt.plot(standard_model.nodes, standard_model.values.detach(), 
             'b.', alpha=0.5, markersize=8, label='Standard nodes')
    plt.plot(rational_model.nodes, rational_model.values.detach(), 
             'r.', alpha=0.5, markersize=8, label='Rational nodes')
    
    plt.grid(True)
    plt.legend()
    plt.title('Final Approximation')
    
    plt.tight_layout()
    plt.show()

# Create models
n_points = 20
standard_model = LagrangeInterpolationModel(n_points)
rational_model = RationalInterpolationModel(n_points, num_poles=4)

# Train models
losses = train_interpolation(standard_model, rational_model, num_epochs=2000, freeze_poles=1000)

# Evaluate and plot results
x_eval = torch.linspace(-1, 1, 1000)
plot_training_results(x_eval, standard_model, rational_model, losses)

# %%
rational_model.pole_imag

# %% [markdown]
# ### Sanity check: least squares for rational approximation + convert to nodes/poles.

# %%
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class RationalCoeffs:
    """Rational function in coefficient form"""
    num_coeffs: np.ndarray  # coefficients of numerator [c₀, c₁, ..., cₘ]
    den_coeffs: np.ndarray  # coefficients of denominator [1, d₁, ..., dₘ]

@dataclass
class RationalNodes:
    """Rational function in nodes/poles form"""
    nodes: np.ndarray       # interpolation nodes
    values: np.ndarray      # values at nodes
    poles: np.ndarray       # complex poles (includes conjugate pairs)

def fit_rational_ls(func, degree: int, domain: Tuple[float, float] = (-1, 1), 
                   num_samples: int = 2000) -> Tuple[RationalCoeffs, RationalNodes]:
    """
    Fit a rational function to func using least squares.
    
    Args:
        func: callable to approximate
        degree: maximum degree of numerator and denominator
        domain: interval to fit on
        num_samples: number of sample points for least squares
    
    Returns:
        Tuple of (coefficient form, nodes/poles form)
    """
    # Sample the function
    x_samp = np.linspace(domain[0], domain[1], num_samples)
    f_samp = func(x_samp)
    
    # Build design matrices for least squares
    Xpow = np.column_stack([x_samp**k for k in range(degree+1)])
    Xpow_d = np.column_stack([x_samp**k for k in range(1, degree+1)])
    A_right = -(f_samp.reshape(-1,1)) * Xpow_d
    A = np.hstack([Xpow, A_right])
    b = f_samp.copy()
    
    # Solve least squares
    sol, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    
    # Extract coefficients
    c_coeffs = sol[0 : (degree+1)]          # c₀..cₘ
    d_coeffs = np.concatenate(([1.0],       # d₀=1
                             sol[degree+1 : (2*degree+1)]))  # d₁..dₘ
    
    # Get poles by finding roots of denominator polynomial
    poles = np.roots(d_coeffs[::-1])  # np.roots expects coeffs in ascending order
    
    # Get interpolation nodes (using Chebyshev points)
    k = np.arange(degree+1)
    nodes = np.cos((2*k + 1)*np.pi / (2*(degree+1)))
    # Transform from [-1,1] to domain
    a, b = domain
    nodes = 0.5*(b - a)*nodes + 0.5*(b + a)
    
    # Get values at nodes
    values = func(nodes)
    
    return (RationalCoeffs(c_coeffs, d_coeffs), 
            RationalNodes(nodes, values, poles))

def evaluate_rational(x: np.ndarray, coeffs: RationalCoeffs) -> np.ndarray:
    """Evaluate rational function in coefficient form"""
    degree = len(coeffs.num_coeffs) - 1
    x = np.asarray(x)
    
    # Evaluate numerator and denominator
    xpow = np.column_stack([x**k for k in range(degree+1)])
    num = xpow.dot(coeffs.num_coeffs)
    den = xpow.dot(coeffs.den_coeffs)
    
    return num / den

def chebyshev_approx(func, degree, domain=[-1,1]):
    """Fit Chebyshev polynomial approximation"""
    # Get Chebyshev nodes
    k = np.arange(degree + 1)
    nodes = np.cos((2*k + 1)*np.pi / (2*(degree+1)))
    
    # Transform from [-1,1] to domain
    a, b = domain
    x_nodes = 0.5*(b - a)*nodes + 0.5*(b + a)
    y_nodes = func(x_nodes)
    
    # Get coefficients - using numpy's polynomial package
    coeffs = np.polynomial.chebyshev.chebfit(x_nodes, y_nodes, degree)
    return coeffs, nodes, y_nodes

def evaluate_chebyshev(x, coeffs, domain=[-1,1]):
    """Evaluate Chebyshev polynomial"""
    return np.polynomial.chebyshev.chebval(x, coeffs)

def test_rational_fit(degree=10):
    """Compare Chebyshev and rational approximations"""
    # Define test function
    def f(x, alpha=50.0):
        return -1.0 + 2.0 / (1.0 + np.exp(-alpha * (0.5 - np.abs(x))))
    
    # Chebyshev fit
    cheb_coeffs, cheb_nodes, cheb_values = chebyshev_approx(f, 2*degree)
    
    # Rational fit
    rat_coeffs, rat_nodes = fit_rational_ls(f, degree)
    
    # Evaluate on a grid
    x_grid = np.linspace(-1, 1, 1000)
    f_true = f(x_grid)
    f_cheb = evaluate_chebyshev(x_grid, cheb_coeffs)
    f_rat = evaluate_rational(x_grid, rat_coeffs)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Function comparison
    plt.subplot(221)
    plt.plot(x_grid, f_true, 'k-', label='True', alpha=0.7)
    plt.plot(x_grid, f_cheb, 'b--', label='Chebyshev')
    plt.plot(x_grid, f_rat, 'r:', label='Rational')
    plt.grid(True)
    plt.legend()
    plt.title('Function Comparison')
    
    # Nodes comparison
    plt.subplot(222)
    plt.plot(x_grid, f_true, 'k-', label='True', alpha=0.3)
    plt.plot(cheb_nodes, cheb_values, 'b.', 
             label='Chebyshev nodes', markersize=10)
    plt.plot(rat_nodes.nodes, rat_nodes.values, 'r.', 
             label='Rational nodes', markersize=10)
    plt.grid(True)
    plt.legend()
    plt.title('Interpolation Nodes')
    
    # Poles in complex plane
    plt.subplot(223)
    poles = rat_nodes.poles
    plt.plot(poles.real, poles.imag, 'r+', label='Poles', markersize=10)
    plt.axhline(y=0, color='k', alpha=0.3)
    plt.plot([-1, 1], [0, 0], 'k-', linewidth=2, label='[-1,1]')
    plt.grid(True)
    plt.xlim(-1, 1)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Pole Locations')
    plt.legend()
    
    # Error comparison
    plt.subplot(224)
    err_cheb = np.abs(f_true - f_cheb)
    err_rat = np.abs(f_true - f_rat)
    plt.semilogy(x_grid, err_cheb, 'b-', label='Chebyshev error')
    plt.semilogy(x_grid, err_rat, 'r-', label='Rational error')
    plt.grid(True)
    plt.legend()
    plt.title('Error Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Maximum errors:")
    print(f"Chebyshev: {err_cheb.max():.2e}")
    print(f"Rational:  {err_rat.max():.2e}")
    
    print("\nChebyshev coefficients:")
    print(cheb_coeffs)
    print("\nRational numerator coefficients:")
    print(rat_coeffs.num_coeffs)
    print("\nRational denominator coefficients:")
    print(rat_coeffs.den_coeffs)
    print("\nPoles:")
    print(poles)

if __name__ == "__main__":
    test_rational_fit(degree=20)


# %% [markdown]
# ## Check accuracy of derivatives

# %%
class TestFunctions:
    @staticmethod
    def smooth_func(x):
        """Test function: f(x) = cos(πx)"""
        return torch.cos(np.pi * x)
    
    @staticmethod
    def smooth_derivative_1(x):
        """First derivative of smooth function"""
        return -np.pi * torch.sin(np.pi * x)
    
    @staticmethod
    def smooth_derivative_2(x):
        """Second derivative of smooth function"""
        return -(np.pi**2) * torch.cos(np.pi * x)
    
    @staticmethod
    def steep_transition(x, alpha=50.0):
        """Logistic transition function"""
        return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))
    
    @staticmethod
    def steep_derivative_1(x, alpha=50.0):
        """First derivative of steep transition"""
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)
        exp_term = torch.exp(-alpha * (0.5 - abs_x))
        return -2.0 * alpha * exp_term * sign_x / (1.0 + exp_term)**2
    
    @staticmethod
    def steep_derivative_2(x, alpha=50.0):
        """Second derivative of steep transition"""
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)
        exp_term = torch.exp(-alpha * (0.5 - abs_x))
        factor1 = 2.0 * alpha**2 * exp_term * (1.0 - exp_term)
        factor2 = (1.0 + exp_term)**3
        return factor1 / factor2

def compute_convergence(test_func, test_deriv1, test_deriv2, name):
    """Run convergence test for a given test function and its derivatives"""
    # Test parameters
    n_points_range = range(11, 102, 10)  # 11, 21, ..., 101
    n_eval = 1000  # Number of evaluation points
    x_eval = torch.linspace(-1, 1, n_eval)
    
    # True function and derivatives
    y_true = test_func(x_eval)
    dy_true = test_deriv1(x_eval)
    d2y_true = test_deriv2(x_eval)
    
    # Storage for errors
    errors_standard = {'f': [], 'df': [], 'd2f': []}
    errors_rational = {'f': [], 'df': [], 'd2f': []}
    
    # Test each number of points
    for n_points in tqdm(n_points_range, desc=f"Testing {name}"):
        # Create models
        standard_model = LagrangeInterpolationModel(n_points)
        rational_model = RationalInterpolationModel(n_points, num_poles=4)
        
        # Initialize real parts
        rational_model.pole_real.data = torch.tensor([-0.5, 0.5])
        # Initialize imaginary parts
        rational_model.pole_imag.data = torch.tensor([0.07, 0.07])
        
        # Set values at nodes
        with torch.no_grad():
            standard_model.values.copy_(test_func(standard_model.nodes))
            rational_model.values.copy_(test_func(rational_model.nodes))
        
        # Evaluate models and derivatives
        y_standard = standard_model(x_eval).detach()
        dy_standard = standard_model.derivative(x_eval, k=1).detach()
        d2y_standard = standard_model.derivative(x_eval, k=2).detach()
        
        y_rational = rational_model(x_eval).detach()
        dy_rational = rational_model.derivative(x_eval, k=1).detach()
        d2y_rational = rational_model.derivative(x_eval, k=2).detach()
        
        # Compute max errors
        errors_standard['f'].append(torch.max(torch.abs(y_standard - y_true)).item())
        errors_standard['df'].append(torch.max(torch.abs(dy_standard - dy_true)).item())
        errors_standard['d2f'].append(torch.max(torch.abs(d2y_standard - d2y_true)).item())
        
        errors_rational['f'].append(torch.max(torch.abs(y_rational - y_true)).item())
        errors_rational['df'].append(torch.max(torch.abs(dy_rational - dy_true)).item())
        errors_rational['d2f'].append(torch.max(torch.abs(d2y_rational - d2y_true)).item())
    
    return n_points_range, errors_standard, errors_rational

def plot_convergence(n_points_range, errors_standard, errors_rational, title):
    plt.figure(figsize=(15, 5))
    
    # Function values
    plt.subplot(131)
    plt.semilogy(list(n_points_range), errors_standard['f'], 'b-o', label='Standard')
    plt.semilogy(list(n_points_range), errors_rational['f'], 'r-o', label='Rational')
    plt.grid(True)
    plt.xlabel('Number of nodes')
    plt.ylabel('Max error')
    plt.title(f'{title}\nFunction Values')
    plt.legend()
    
    # First derivative
    plt.subplot(132)
    plt.semilogy(list(n_points_range), errors_standard['df'], 'b-o', label='Standard')
    plt.semilogy(list(n_points_range), errors_rational['df'], 'r-o', label='Rational')
    plt.grid(True)
    plt.xlabel('Number of nodes')
    plt.ylabel('Max error')
    plt.title('First Derivative')
    plt.legend()
    
    # Second derivative
    plt.subplot(133)
    plt.semilogy(list(n_points_range), errors_standard['d2f'], 'b-o', label='Standard')
    plt.semilogy(list(n_points_range), errors_rational['d2f'], 'r-o', label='Rational')
    plt.grid(True)
    plt.xlabel('Number of nodes')
    plt.ylabel('Max error')
    plt.title('Second Derivative')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Test smooth function (cosine)
n_points_range, errors_standard, errors_rational = compute_convergence(
    TestFunctions.smooth_func,
    TestFunctions.smooth_derivative_1,
    TestFunctions.smooth_derivative_2,
    "Smooth Function (cos(πx))"
)
plot_convergence(n_points_range, errors_standard, errors_rational, "Smooth Function")

# Test steep transition function
n_points_range, errors_standard_steep, errors_rational_steep = compute_convergence(
    TestFunctions.steep_transition,
    TestFunctions.steep_derivative_1,
    TestFunctions.steep_derivative_2,
    "Steep Transition"
)
plot_convergence(n_points_range, errors_standard_steep, errors_rational_steep, "Steep Transition")

# %% [markdown]
# ## Least squares and rationals

# %%
import torch
import torch.nn as nn
import numpy as np

class PolynomialRationalModel(nn.Module):
    def __init__(self, num_points, numerator_degree, denominator_degree):
        """
        Model using polynomial ratio representation: P(x)/Q(x)
        P(x) = p₀ + p₁x + ... + pₙxⁿ
        Q(x) = 1 + xq(x), where q(x) = q₁x + ... + qₘxᵐ
        
        Args:
            num_points: Number of interpolation points
            numerator_degree: Degree of numerator polynomial (n)
            denominator_degree: Degree of denominator polynomial (m)
        """
        super().__init__()
        
        # Generate Chebyshev points
        i = torch.linspace(0, 1, num_points)
        self.nodes = torch.cos(torch.pi * i)  # [-1, 1]
        
        # Initialize coefficients
        self.p = nn.Parameter(torch.zeros(numerator_degree + 1))
        self.q = nn.Parameter(torch.zeros(denominator_degree))  # q₀ = 1 fixed
        
    def _evaluate_polynomial(self, x, coeffs):
        """Evaluate polynomial with given coefficients at x"""
        result = torch.zeros_like(x)
        for i, c in enumerate(coeffs):
            result = result + c * (x**i)
        return result
    
    def _polynomial_derivative(self, x, coeffs, k=1):
        """Compute k-th derivative of polynomial analytically"""
        if k == 0:
            return self._evaluate_polynomial(x, coeffs)
            
        # Get derivative coefficients
        n = len(coeffs)
        deriv_coeffs = torch.zeros(max(n-k, 1), dtype=coeffs.dtype, device=coeffs.device)
        
        for i in range(k, n):
            # Compute multiplication factor: i * (i-1) * ... * (i-k+1)
            factor = 1
            for j in range(k):
                factor *= (i - j)
            deriv_coeffs[i-k] = coeffs[i] * factor
            
        return self._evaluate_polynomial(x, deriv_coeffs)
    
    def fit_least_squares(self, f_values):
        """Fit coefficients using least squares to match function values at nodes"""
        x_samp = self.nodes.numpy()
        f_samp = f_values.detach().numpy()
        N = len(x_samp)
        n = len(self.p) - 1
        m = len(self.q)
        
        # Build least squares matrix A and right-hand side b
        A = np.zeros((N, n + m + 1))
        b = np.zeros(N)
        
        for k in range(N):
            xk = x_samp[k]
            fk = f_samp[k]
            
            # Columns for pᵢ coefficients
            for r in range(n + 1):
                A[k, r] = xk**r
            
            # Columns for qⱼ coefficients
            for s in range(1, m + 1):
                A[k, n + s] = -fk * (xk**s)
            
            b[k] = fk
        
        # Solve least squares problem
        unknowns, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Update parameters
        with torch.no_grad():
            self.p.copy_(torch.tensor(unknowns[:n + 1], dtype=torch.float64))
            self.q.copy_(torch.tensor(unknowns[n + 1:], dtype=torch.float64))

    def forward(self, x_eval):
        """Evaluate rational approximation at points x_eval"""
        # Evaluate numerator P(x)
        num = self._evaluate_polynomial(x_eval, self.p)
        
        # Evaluate denominator Q(x) = 1 + xq(x)
        q_poly = self._evaluate_polynomial(x_eval, self.q)
        den = 1.0 + x_eval * q_poly
        
        return num / den
    
    def derivative(self, x_eval, k=1):
        """
        Compute k-th derivative of rational function R(x) = P(x)/Q(x)
        where Q(x) = 1 + xq(x)
        """
        if k == 0:
            return self(x_eval)
            
        # Get base polynomial q(x) evaluation
        q_poly = self._evaluate_polynomial(x_eval, self.q)
        
        if k == 1:
            # Get P and P' values
            p_val = self._evaluate_polynomial(x_eval, self.p)
            p_prime = self._polynomial_derivative(x_eval, self.p, 1)
            
            # Get Q and Q' values
            # Q = 1 + xq
            q_val = 1.0 + x_eval * q_poly
            # Q' = q + xq'
            q_prime = q_poly + x_eval * self._polynomial_derivative(x_eval, self.q, 1)
            
            # R' = (P'Q - PQ')/Q²
            return (p_prime * q_val - p_val * q_prime) / (q_val * q_val)
            
        elif k == 2:
            # Get P, P', P'' values
            p_val = self._evaluate_polynomial(x_eval, self.p)
            p_prime = self._polynomial_derivative(x_eval, self.p, 1)
            p_dprime = self._polynomial_derivative(x_eval, self.p, 2)
            
            # Get Q, Q', Q'' values
            # Q = 1 + xq
            q_val = 1.0 + x_eval * q_poly
            # Q' = q + xq'
            q_prime = q_poly + x_eval * self._polynomial_derivative(x_eval, self.q, 1)
            # Q'' = 2q' + xq''
            q_dprime = (2.0 * self._polynomial_derivative(x_eval, self.q, 1) + 
                       x_eval * self._polynomial_derivative(x_eval, self.q, 2))
            
            # R'' = (P''Q - PQ'' - 2P'Q' + 2P(Q')²/Q) / Q²
            return (p_dprime / q_val) - (p_val * q_dprime / (q_val * q_val)) - \
                   2 * (p_prime * q_prime / (q_val * q_val)) + \
                   2 * p_val * q_prime * q_prime / (q_val* q_val * q_val)
        else:
            raise NotImplementedError("Higher derivatives not yet implemented")

"""Test polynomial rational approximation"""
# Create model
n_points = 20
model = PolynomialRationalModel(n_points, numerator_degree=4, denominator_degree=4)

# Test function: exp(x)
x_eval = torch.linspace(-1, 1, 1000)
y_true = torch.exp(x_eval)

# Fit model
with torch.no_grad():
    f_nodes = torch.exp(model.nodes)
    model.fit_least_squares(f_nodes)

# Evaluate
y_approx = model(x_eval)
max_error = torch.max(torch.abs(y_approx - y_true))
print(f"Max approximation error: {max_error:.2e}")

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(x_eval, y_true, 'k-', label='True')
plt.plot(x_eval, y_approx.detach(), 'r--', label='Rational')
plt.plot(model.nodes, f_nodes, 'bo', label='Nodes')
plt.legend()
plt.title('Function Comparison')
plt.grid(True)

plt.subplot(122)
plt.semilogy(x_eval, torch.abs(y_approx.detach() - y_true), 'b-')
plt.title(f'Error (max = {max_error:.2e})')
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def steep_transition(x, alpha=50.0):
    """Logistic transition function"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))

def steep_derivative_1(x, alpha=50.0):
    """First derivative of steep transition"""
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    exp_term = torch.exp(-alpha * (0.5 - abs_x))
    return -2.0 * alpha * exp_term * sign_x / (1.0 + exp_term)**2

def steep_derivative_2(x, alpha=50.0):
    """Second derivative of steep transition"""
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    exp_term = torch.exp(-alpha * (0.5 - abs_x))
    factor1 = 2.0 * alpha**2 * exp_term * (1.0 - exp_term)
    factor2 = (1.0 + exp_term)**3
    return factor1 / factor2

def compute_convergence():
    """Test convergence for both smooth and steep functions"""
    # Test parameters
    n_points_range = range(11, 202, 20)  # 11, 21, ..., 101
    n_eval = 1000  # Number of evaluation points
    x_eval = torch.linspace(-1, 1, n_eval)
    
    # Test functions
    test_functions = {
        'Smooth': {
            'f': lambda x: torch.cos(np.pi * x),
            'df': lambda x: -np.pi * torch.sin(np.pi * x),
            'd2f': lambda x: -(np.pi**2) * torch.cos(np.pi * x)
        },
        'Steep': {
            'f': steep_transition,
            'df': steep_derivative_1,
            'd2f': steep_derivative_2
        }
    }
    
    # Storage for errors
    results = {
        'Smooth': {'standard': {'f': [], 'df': [], 'd2f': []},
                  'rational': {'f': [], 'df': [], 'd2f': []}},
        'Steep': {'standard': {'f': [], 'df': [], 'd2f': []},
                 'rational': {'f': [], 'df': [], 'd2f': []}}
    }
    
    # Test each function type
    for func_name, funcs in test_functions.items():
        print(f"\nTesting {func_name} function:")
        
        # True function and derivatives
        y_true = funcs['f'](x_eval)
        dy_true = funcs['df'](x_eval)
        d2y_true = funcs['d2f'](x_eval)
        
        # Test each number of points
        for n_points in tqdm(n_points_range):
            # Create models
            standard_model = LagrangeInterpolationModel(n_points)
            rational_model = PolynomialRationalModel(n_points, 
                                                   numerator_degree=n_points//2-1,
                                                   denominator_degree=n_points//2)
            
            # Set values at nodes
            with torch.no_grad():
                standard_model.values.copy_(funcs['f'](standard_model.nodes))
                rational_model.fit_least_squares(funcs['f'](rational_model.nodes))
            
            # Evaluate models and derivatives
            y_standard = standard_model(x_eval).detach()
            dy_standard = standard_model.derivative(x_eval, k=1).detach()
            d2y_standard = standard_model.derivative(x_eval, k=2).detach()
            
            y_rational = rational_model(x_eval).detach()
            dy_rational = rational_model.derivative(x_eval, k=1).detach()
            d2y_rational = rational_model.derivative(x_eval, k=2).detach()
            
            # Compute max errors
            results[func_name]['standard']['f'].append(
                torch.max(torch.abs(y_standard - y_true)).item())
            results[func_name]['standard']['df'].append(
                torch.max(torch.abs(dy_standard - dy_true)).item())
            results[func_name]['standard']['d2f'].append(
                torch.max(torch.abs(d2y_standard - d2y_true)).item())
            
            results[func_name]['rational']['f'].append(
                torch.max(torch.abs(y_rational - y_true)).item())
            results[func_name]['rational']['df'].append(
                torch.max(torch.abs(dy_rational - dy_true)).item())
            results[func_name]['rational']['d2f'].append(
                torch.max(torch.abs(d2y_rational - d2y_true)).item())
    
    return n_points_range, results

def plot_convergence(n_points_range, results):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    titles = ['Function Values', 'First Derivative', 'Second Derivative']
    keys = ['f', 'df', 'd2f']
    
    for i, (func_name, func_results) in enumerate(results.items()):
        for j, (title, key) in enumerate(zip(titles, keys)):
            ax = axs[i, j]
            
            # Plot standard and rational errors
            ax.semilogy(list(n_points_range), func_results['standard'][key], 
                       'b-o', label='Standard')
            ax.semilogy(list(n_points_range), func_results['rational'][key], 
                       'r-o', label='Rational')
            
            ax.grid(True)
            ax.set_xlabel('Number of nodes')
            ax.set_ylabel('Max error')
            if i == 0:
                ax.set_title(f'Smooth: {title}')
            else:
                ax.set_title(f'Steep: {title}')
            ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run convergence test
    n_points_range, results = compute_convergence()
    
    # Plot results
    plot_convergence(n_points_range, results)

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def steep_transition(x, alpha=50.0):
    """Logistic transition function"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))

def steep_derivative_1(x, alpha=50.0):
    """First derivative of steep transition"""
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    exp_term = torch.exp(-alpha * (0.5 - abs_x))
    return -2.0 * alpha * exp_term * sign_x / (1.0 + exp_term)**2

def steep_derivative_2(x, alpha=50.0):
    """Second derivative of steep transition"""
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    exp_term = torch.exp(-alpha * (0.5 - abs_x))
    factor1 = 2.0 * alpha**2 * exp_term * (1.0 - exp_term)
    factor2 = (1.0 + exp_term)**3
    return factor1 / factor2

def compute_convergence():
    """Test convergence for both smooth and steep functions"""
    # Test parameters
    n_points_range = range(11, 52, 5)  # 11, 21, ..., 101
    n_eval = 1000  # Number of evaluation points
    x_eval = torch.linspace(-1, 1, n_eval)
    
    # Test functions
    test_functions = {
        'Smooth': {
            'f': lambda x: torch.cos(np.pi * x),
            'df': lambda x: -np.pi * torch.sin(np.pi * x),
            'd2f': lambda x: -(np.pi**2) * torch.cos(np.pi * x)
        },
        'Steep': {
            'f': steep_transition,
            'df': steep_derivative_1,
            'd2f': steep_derivative_2
        }
    }
    
    # Storage for errors
    results = {
        'Smooth': {'standard': {'f': [], 'df': [], 'd2f': []},
                  'rational': {'f': [], 'df': [], 'd2f': []}},
        'Steep': {'standard': {'f': [], 'df': [], 'd2f': []},
                 'rational': {'f': [], 'df': [], 'd2f': []}}
    }
    
    # Test each function type
    for func_name, funcs in test_functions.items():
        print(f"\nTesting {func_name} function:")
        
        # True function and derivatives
        y_true = funcs['f'](x_eval)
        dy_true = funcs['df'](x_eval)
        d2y_true = funcs['d2f'](x_eval)
        
        # Test each number of points
        for n_points in tqdm(n_points_range):
            # Create models
            standard_model = LagrangeInterpolationModel(n_points)
            rational_model = RationalInterpolationModel(n_points, num_poles=4)
            
            # Set values at nodes
            with torch.no_grad():
                standard_model.values.copy_(funcs['f'](standard_model.nodes))
                
            # Fit rational model
            rational_model.fit(funcs['f'])
            
            # Evaluate models and derivatives
            y_standard = standard_model(x_eval).detach()
            dy_standard = standard_model.derivative(x_eval, k=1).detach()
            d2y_standard = standard_model.derivative(x_eval, k=2).detach()
            
            y_rational = rational_model(x_eval).detach()
            dy_rational = rational_model.derivative(x_eval, k=1).detach()
            d2y_rational = rational_model.derivative(x_eval, k=2).detach()
            
            # Compute max errors
            results[func_name]['standard']['f'].append(
                torch.max(torch.abs(y_standard - y_true)).item())
            results[func_name]['standard']['df'].append(
                torch.max(torch.abs(dy_standard - dy_true)).item())
            results[func_name]['standard']['d2f'].append(
                torch.max(torch.abs(d2y_standard - d2y_true)).item())
            
            results[func_name]['rational']['f'].append(
                torch.max(torch.abs(y_rational - y_true)).item())
            results[func_name]['rational']['df'].append(
                torch.max(torch.abs(dy_rational - dy_true)).item())
            results[func_name]['rational']['d2f'].append(
                torch.max(torch.abs(d2y_rational - d2y_true)).item())
    
    return n_points_range, results

def plot_convergence(n_points_range, results):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    titles = ['Function Values', 'First Derivative', 'Second Derivative']
    keys = ['f', 'df', 'd2f']
    
    for i, (func_name, func_results) in enumerate(results.items()):
        for j, (title, key) in enumerate(zip(titles, keys)):
            ax = axs[i, j]
            
            # Plot standard and rational errors
            ax.semilogy(list(n_points_range), func_results['standard'][key], 
                       'b-o', label='Standard')
            ax.semilogy(list(n_points_range), func_results['rational'][key], 
                       'r-o', label='Rational')
            
            ax.grid(True)
            ax.set_xlabel('Number of nodes')
            ax.set_ylabel('Max error')
            if i == 0:
                ax.set_title(f'Smooth: {title}')
            else:
                ax.set_title(f'Steep: {title}')
            ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run convergence test
    n_points_range, results = compute_convergence()
    
    # Plot results
    plot_convergence(n_points_range, results)

# %% [markdown]
# ## Learn a rational interpolant for the logistic function using gradient descent

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def steep_transition(x, alpha=50.0):
    """Logistic transition function"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))

def compare_methods():
    # Create dense evaluation points
    x_eval = torch.linspace(-1, 1, 1000)
    y_true = steep_transition(x_eval)
    
    # Create models
    n_points = 51  # Odd number to include x=0
    numerator_degree = 21
    denominator_degree = 21
    
    # Initialize both models
    grad_model = PolynomialRationalModel(n_points, numerator_degree, denominator_degree)
    ls_model = PolynomialRationalModel(n_points, numerator_degree, denominator_degree)
    
    # Fit least squares model
    with torch.no_grad():
        ls_model.fit_least_squares(steep_transition(ls_model.nodes))
    
    # Train gradient model
    optimizer = optim.Adam(grad_model.parameters(), lr=1e-3)
    
    # Training loop
    n_epochs = 100000
    losses = []
    
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = grad_model(x_eval)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        
        # Add regularization to prevent poles in [-1,1]
        # x_dense = torch.linspace(-1, 1, 200)
        # q_poly = grad_model._evaluate_polynomial(x_dense, grad_model.q)
        # denom = 1.0 + x_dense * q_poly
        # reg_loss = 0.1 * torch.mean(1.0 / (denom * denom))
        
        # total_loss = loss + reg_loss
        total_loss = loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.3e}')
    
    # Evaluate both models
    with torch.no_grad():
        y_grad = grad_model(x_eval)
        y_ls = ls_model(x_eval)
        max_error_grad = torch.max(torch.abs(y_grad - y_true))
        max_error_ls = torch.max(torch.abs(y_ls - y_true))
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Function comparison
    axs[0,0].plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
    axs[0,0].plot(x_eval, y_grad.detach(), 'r--', label='Gradient')
    axs[0,0].plot(x_eval, y_ls.detach(), 'b--', label='Least Squares')
    axs[0,0].plot(grad_model.nodes, steep_transition(grad_model.nodes), 'ko', label='Nodes', markersize=3)
    axs[0,0].grid(True)
    axs[0,0].legend()
    axs[0,0].set_title('Function Comparison')
    
    # Error comparison
    axs[0,1].semilogy(x_eval, torch.abs(y_grad.detach() - y_true), 'r-', label=f'Gradient (max={max_error_grad:.2e})')
    axs[0,1].semilogy(x_eval, torch.abs(y_ls.detach() - y_true), 'b-', label=f'LS (max={max_error_ls:.2e})')
    axs[0,1].grid(True)
    axs[0,1].legend()
    axs[0,1].set_title('Error Comparison')
    
    # Loss history
    axs[1,0].semilogy(losses)
    axs[1,0].grid(True)
    axs[1,0].set_title('Loss History (Gradient)')
    
    # Compare derivatives
    dy_true = steep_derivative_1(x_eval)
    dy_grad = grad_model.derivative(x_eval, k=1).detach()
    dy_ls = ls_model.derivative(x_eval, k=1).detach()
    
    axs[1,1].plot(x_eval, dy_true, 'k-', label='True', alpha=0.5)
    axs[1,1].plot(x_eval, dy_grad, 'r--', label='Gradient')
    axs[1,1].plot(x_eval, dy_ls, 'b--', label='Least Squares')
    axs[1,1].grid(True)
    axs[1,1].legend()
    axs[1,1].set_title('First Derivative Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nMaximum Errors:")
    print(f"Gradient method: {max_error_grad:.2e}")
    print(f"Least squares:   {max_error_ls:.2e}")
    
    # Print final coefficients
    print("\nGradient Method Coefficients:")
    print("Numerator:", grad_model.p.detach().numpy())
    print("Denominator:", grad_model.q.detach().numpy())
    print("\nLeast Squares Coefficients:")
    print("Numerator:", ls_model.p.detach().numpy())
    print("Denominator:", ls_model.q.detach().numpy())
    
    return grad_model, ls_model, losses

if __name__ == "__main__":
    grad_model, ls_model, losses = compare_methods()

# %% [markdown]
# Let's try with L-BFGS too.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def steep_transition(x, alpha=50.0):
    """Logistic transition function"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))

def train_rational_comparison():
    """Compare LBFGS and Adam training"""
    # Create dense evaluation points
    x_eval = torch.linspace(-1, 1, 1000)
    y_true = steep_transition(x_eval)
    
    # Create two identical models
    n_points = 51
    numerator_degree = 21
    denominator_degree = 4
    
    model_lbfgs = PolynomialRationalModel(n_points, numerator_degree, denominator_degree)
    model_adam = PolynomialRationalModel(n_points, numerator_degree, denominator_degree)
    
    # Copy initial parameters
    with torch.no_grad():
        for p_adam, p_lbfgs in zip(model_adam.parameters(), model_lbfgs.parameters()):
            p_adam.copy_(p_lbfgs)
    
    # Optimizers
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=1e-3)
    optimizer_lbfgs = optim.LBFGS(model_lbfgs.parameters(), max_iter=20,
                                 history_size=100, line_search_fn="strong_wolfe")
    
    # Training loop for Adam
    n_epochs_adam = 30000
    losses_adam = []
    
    print("Training with Adam...")
    for epoch in tqdm(range(n_epochs_adam)):
        optimizer_adam.zero_grad()
        
        # Forward pass
        y_pred = model_adam(x_eval)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        optimizer_adam.step()
        
        losses_adam.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f'Adam Epoch {epoch}, Loss: {loss.item():.3e}')
    
    # Training loop for L-BFGS
    n_epochs_lbfgs = 500
    losses_lbfgs = []
    
    print("\nTraining with L-BFGS...")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        y_pred = model_lbfgs(x_eval)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        
        loss.backward()
        return loss
    
    for epoch in tqdm(range(n_epochs_lbfgs)):
        loss = optimizer_lbfgs.step(closure)
        losses_lbfgs.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'L-BFGS Epoch {epoch}, Loss: {loss.item():.3e}')
    
    # Evaluate final models
    with torch.no_grad():
        y_adam = model_adam(x_eval)
        y_lbfgs = model_lbfgs(x_eval)
        
        error_adam = torch.abs(y_adam - y_true)
        error_lbfgs = torch.abs(y_lbfgs - y_true)
        
        max_error_adam = torch.max(error_adam)
        max_error_lbfgs = torch.max(error_lbfgs)
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Function comparison
    axs[0,0].plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
    axs[0,0].plot(x_eval, y_adam.detach(), 'r--', label=f'Adam (err={max_error_adam:.2e})')
    axs[0,0].plot(x_eval, y_lbfgs.detach(), 'b--', label=f'L-BFGS (err={max_error_lbfgs:.2e})')
    axs[0,0].grid(True)
    axs[0,0].legend()
    axs[0,0].set_title('Function Comparison')
    
    # Error comparison
    axs[0,1].semilogy(x_eval, error_adam, 'r-', label='Adam')
    axs[0,1].semilogy(x_eval, error_lbfgs, 'b-', label='L-BFGS')
    axs[0,1].grid(True)
    axs[0,1].legend()
    axs[0,1].set_title('Error Distribution')
    
    # Loss history - full
    axs[1,0].semilogy(losses_adam, 'r-', label='Adam')
    axs[1,0].semilogy(np.linspace(0, len(losses_adam), len(losses_lbfgs)), 
                      losses_lbfgs, 'b-', label='L-BFGS')
    axs[1,0].grid(True)
    axs[1,0].legend()
    axs[1,0].set_title('Loss History (Full)')
    
    # Loss history - zoomed
    axs[1,1].semilogy(losses_adam[-1000:], 'r-', label='Adam')
    axs[1,1].semilogy(np.linspace(len(losses_adam)-1000, len(losses_adam), len(losses_lbfgs)), 
                      losses_lbfgs, 'b-', label='L-BFGS')
    axs[1,1].grid(True)
    axs[1,1].legend()
    axs[1,1].set_title('Loss History (Zoomed)')
    
    plt.tight_layout()
    plt.show()
    
    return model_adam, model_lbfgs, (losses_adam, losses_lbfgs)

if __name__ == "__main__":
    model_adam, model_lbfgs, losses = train_rational_comparison()


# %% [markdown]
# What's the condition number of the resulting least squares problem?

# %%
def analyze_ls_convergence(model, f_values, threshold=1e-10):
    """
    Analyze convergence properties of the least squares problem for gradient descent
    
    Args:
        model: PolynomialRationalModel instance
        f_values: function values at nodes
        threshold: threshold for considering singular values as zero
    Returns:
        effective_cond: ratio of max to min non-zero singular values
        convergence_rate: theoretical convergence rate (1 - 1/effective_cond)
    """
    x_samp = model.nodes.numpy()
    f_samp = f_values.detach().numpy()
    N = len(x_samp)
    n = len(model.p) - 1
    m = len(model.q)
    
    # Build least squares matrix A
    A = np.zeros((N, n + m + 1))
    
    for k in range(N):
        xk = x_samp[k]
        fk = f_samp[k]
        
        # Columns for pᵢ coefficients
        for r in range(n + 1):
            A[k, r] = xk**r
        
        # Columns for qⱼ coefficients
        for s in range(1, m + 1):
            A[k, n + s] = -fk * (xk**s)
    
    # Form normal equations matrix A^T A since we're doing gradient descent on normal equations
    ATA = A.T @ A
    
    # Compute eigenvalues of A^T A (squares of singular values of A)
    eigs = np.linalg.eigvalsh(ATA)
    
    # Find non-zero eigenvalues
    nonzero_eigs = eigs[eigs > threshold]
    
    if len(nonzero_eigs) < len(eigs):
        print(f"\nFound {len(eigs) - len(nonzero_eigs)} (near) zero singular values")
        print("Matrix is effectively rank deficient")
    
    # Compute effective condition number
    effective_cond = np.sqrt(nonzero_eigs[-1] / nonzero_eigs[0])
    
    # Theoretical convergence rate for gradient descent
    convergence_rate = 1 - 1/effective_cond**2
    
    print(f"\nLeast Squares Matrix Shape: {A.shape}")
    print(f"Matrix Rank (numerical): {len(nonzero_eigs)}")
    print(f"Full Singular Value Spectrum: {np.sqrt(eigs)}")
    print(f"Effective Condition Number: {effective_cond:.2e}")
    print(f"Theoretical Convergence Rate: {convergence_rate:.4f}")
    print(f"Expected iterations for 1e-6 error: {np.log(1e-6)/np.log(convergence_rate):.0f}")
    
    # Plot singular value spectrum
    plt.figure(figsize=(10, 5))
    plt.semilogy(np.sqrt(eigs), 'bo-')
    plt.axhline(threshold, color='r', linestyle='--', label=f'Zero threshold ({threshold:.0e})')
    plt.grid(True)
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Value Spectrum')
    plt.legend()
    plt.show()
    
    return effective_cond, convergence_rate, A

def analyze_convergence_vs_degree():
    """Analyze how convergence properties vary with polynomial degree"""
    degrees = range(5, 31, 5)
    effective_conds = []
    convergence_rates = []
    
    x_eval = torch.linspace(-1, 1, 1000)
    y_true = steep_transition(x_eval)
    
    for deg in degrees:
        n_points = 2*deg + 1
        model = PolynomialRationalModel(n_points, deg, deg)
        f_values = steep_transition(model.nodes)
        
        eff_cond, conv_rate, _ = analyze_ls_convergence(model, f_values)
        effective_conds.append(eff_cond)
        convergence_rates.append(conv_rate)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.semilogy(degrees, effective_conds, 'b-o')
    ax1.grid(True)
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Effective Condition Number')
    ax1.set_title('Effective Condition Number vs Degree')
    
    ax2.plot(degrees, convergence_rates, 'r-o')
    ax2.grid(True)
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('Convergence Rate')
    ax2.set_title('GD Convergence Rate vs Degree')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test for a specific model
    n_points = 51
    numerator_degree = 21
    denominator_degree = 4
    model = PolynomialRationalModel(n_points, numerator_degree, denominator_degree)
    f_values = steep_transition(model.nodes)
    
    eff_cond, conv_rate, A = analyze_ls_convergence(model, f_values)
    
    # Analyze trend with degree
    # analyze_convergence_vs_degree()

# %%
import torch
import matplotlib.pyplot as plt

def test_quadratic():
    """Compare polynomial and rational interpolation of f(x) = x²"""
    # Create models
    n_points = 41  # Should be plenty for a quadratic
    standard_model = LagrangeInterpolationModel(n_points)
    rational_model = PolynomialRationalModel(n_points, 
                                           numerator_degree=16,  # Slightly higher degree
                                           denominator_degree=15)
    
    # Test points
    x_eval = torch.linspace(-1, 1, 1000)
    
    # True function and derivatives
    y_true = x_eval**2
    dy_true = 2*x_eval
    d2y_true = 2*torch.ones_like(x_eval)
    
    # Fit models
    with torch.no_grad():
        f_nodes_poly = standard_model.nodes**2
        standard_model.values.copy_(f_nodes_poly)
        f_nodes_rat = rational_model.nodes**2
        rational_model.fit_least_squares(f_nodes_rat)
    
    # Evaluate models
    y_standard = standard_model(x_eval).detach()
    dy_standard = standard_model.derivative(x_eval, k=1).detach()
    d2y_standard = standard_model.derivative(x_eval, k=2).detach()
    
    y_rational = rational_model(x_eval).detach()
    dy_rational = rational_model.derivative(x_eval, k=1).detach()
    d2y_rational = rational_model.derivative(x_eval, k=2).detach()
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Function values
    axs[0].plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
    axs[0].plot(x_eval, y_standard, 'b--', label='Polynomial')
    axs[0].plot(x_eval, y_rational, 'r:', label='Rational')
    axs[0].plot(standard_model.nodes, f_nodes_poly, 'ko', label='Nodes')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title('Function: f(x) = x²')
    
    # First derivative
    axs[1].plot(x_eval, dy_true, 'k-', label='True', alpha=0.5)
    axs[1].plot(x_eval, dy_standard, 'b--', label='Polynomial')
    axs[1].plot(x_eval, dy_rational, 'r:', label='Rational')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title('First Derivative: f\'(x) = 2x')
    
    # Second derivative
    axs[2].plot(x_eval, d2y_true, 'k-', label='True', alpha=0.5)
    axs[2].plot(x_eval, d2y_standard, 'b--', label='Polynomial')
    axs[2].plot(x_eval, d2y_rational, 'r:', label='Rational')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_title('Second Derivative: f\'\'(x) = 2')
    
    plt.tight_layout()
    plt.show()
    
    # Print max errors
    print("\nMaximum Errors:")
    print("Polynomial:")
    print(f"  Function: {torch.max(torch.abs(y_standard - y_true)):.2e}")
    print(f"  First derivative: {torch.max(torch.abs(dy_standard - dy_true)):.2e}")
    print(f"  Second derivative: {torch.max(torch.abs(d2y_standard - d2y_true)):.2e}")
    print("\nRational:")
    print(f"  Function: {torch.max(torch.abs(y_rational - y_true)):.2e}")
    print(f"  First derivative: {torch.max(torch.abs(dy_rational - dy_true)):.2e}")
    print(f"  Second derivative: {torch.max(torch.abs(d2y_rational - d2y_true)):.2e}")

if __name__ == "__main__":
    test_quadratic()

# %% jupyter={"outputs_hidden": true}
import torch
import matplotlib.pyplot as plt

def plot_progression():
    """Plot rational interpolants with increasing degrees"""
    # Starting degrees
    # num_degree_start = 2  # Start with quadratic numerator
    # den_degree_start = 1  # Start with linear denominator
    num_degree_start = 0
    den_degree_start = 20
    num_increments = 21   # How many degree increments to try
    
    # Test points
    x_eval = torch.linspace(-1, 1, 1000)
    
    # True function and derivatives
    y_true = x_eval**2
    dy_true = 2*x_eval
    d2y_true = 2*torch.ones_like(x_eval)
    
    for i in range(num_increments):
        # num_degree = num_degree_start + i
        # den_degree = den_degree_start + i
        num_degree = num_degree_start + den_degree_start - i
        den_degree = i
        # n_points = max(num_degree, den_degree)  # Add some extra points for stability
        n_points = num_degree + den_degree
        
        # Create model
        rational_model = PolynomialRationalModel(n_points, 
                                               numerator_degree=num_degree,
                                               denominator_degree=den_degree)
        
        # Fit model
        with torch.no_grad():
            f_nodes = rational_model.nodes**2
            rational_model.fit_least_squares(f_nodes)
        
        # Evaluate model
        y_rational = rational_model(x_eval).detach()
        dy_rational = rational_model.derivative(x_eval, k=1).detach()
        d2y_rational = rational_model.derivative(x_eval, k=2).detach()
        
        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Rational Interpolant: P_{num_degree}/Q_{den_degree}')
        
        # Function values
        axs[0].plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
        axs[0].plot(x_eval, y_rational, 'r-', label='Rational')
        axs[0].plot(rational_model.nodes, f_nodes, 'ko', label='Nodes', markersize=4)
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_title('Function: f(x) = x²')
        
        # First derivative
        axs[1].plot(x_eval, dy_true, 'k-', label='True', alpha=0.5)
        axs[1].plot(x_eval, dy_rational, 'r-', label='Rational')
        axs[1].grid(True)
        axs[1].legend()
        axs[1].set_title('First Derivative: f\'(x) = 2x')
        
        # Second derivative
        axs[2].plot(x_eval, d2y_true, 'k-', label='True', alpha=0.5)
        axs[2].plot(x_eval, d2y_rational, 'r-', label='Rational')
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title('Second Derivative: f\'\'(x) = 2')
        
        # Print errors
        print(f"\nDegrees: P_{num_degree}/Q_{den_degree}")
        print(f"  Function error: {torch.max(torch.abs(y_rational - y_true)):.2e}")
        print(f"  First derivative error: {torch.max(torch.abs(dy_rational - dy_true)):.2e}")
        print(f"  Second derivative error: {torch.max(torch.abs(d2y_rational - d2y_true)):.2e}")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_progression()

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

def steep_transition(x, alpha=50.0):
    """Logistic transition function"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))

def steep_derivative_1(x, alpha=50.0):
    """First derivative of steep transition"""
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    exp_term = torch.exp(-alpha * (0.5 - abs_x))
    return -2.0 * alpha * exp_term * sign_x / (1.0 + exp_term)**2

def steep_derivative_2(x, alpha=50.0):
    """Second derivative of steep transition"""
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    exp_term = torch.exp(-alpha * (0.5 - abs_x))
    factor1 = 2.0 * alpha**2 * exp_term * (1.0 - exp_term)
    factor2 = (1.0 + exp_term)**3
    return factor1 / factor2

def plot_progression():
    """Plot rational interpolants with increasing degrees"""
    # Starting degrees
    num_degree_start = 0
    den_degree_start = 20
    num_increments = 21
    
    # Test points
    x_eval = torch.linspace(-1, 1, 1000)
    
    # True function and derivatives
    y_true = steep_transition(x_eval)
    dy_true = steep_derivative_1(x_eval)
    d2y_true = steep_derivative_2(x_eval)
    
    for i in range(num_increments):
        num_degree = num_degree_start + den_degree_start - i
        den_degree = i
        n_points = num_degree + den_degree + 1  # n+m+1 points
        
        # Create model
        rational_model = PolynomialRationalModel(n_points, 
                                               numerator_degree=num_degree,
                                               denominator_degree=den_degree)
        
        # Fit model
        with torch.no_grad():
            f_nodes = steep_transition(rational_model.nodes)
            rational_model.fit_least_squares(f_nodes)
        
        # Evaluate model
        y_rational = rational_model(x_eval).detach()
        dy_rational = rational_model.derivative(x_eval, k=1).detach()
        d2y_rational = rational_model.derivative(x_eval, k=2).detach()
        
        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Rational Interpolant: P_{num_degree}/Q_{den_degree}')
        
        # Function values
        axs[0].plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
        axs[0].plot(x_eval, y_rational, 'r-', label='Rational')
        axs[0].plot(rational_model.nodes, f_nodes, 'ko', label='Nodes', markersize=4)
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_title('Function: Logistic')
        axs[0].set_ylim(-1.5, 1.5)
        
        # First derivative
        axs[1].plot(x_eval, dy_true, 'k-', label='True', alpha=0.5)
        axs[1].plot(x_eval, dy_rational, 'r-', label='Rational')
        axs[1].grid(True)
        axs[1].legend()
        axs[1].set_title('First Derivative')
        axs[1].set_ylim(-100, 100)
        
        # Second derivative
        axs[2].plot(x_eval, d2y_true, 'k-', label='True', alpha=0.5)
        axs[2].plot(x_eval, d2y_rational, 'r-', label='Rational')
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title('Second Derivative')
        axs[2].set_ylim(-1000, 1000)
        
        # Print errors
        print(f"\nDegrees: P_{num_degree}/Q_{den_degree}")
        print(f"  Function error: {torch.max(torch.abs(y_rational - y_true)):.2e}")
        print(f"  First derivative error: {torch.max(torch.abs(dy_rational - dy_true)):.2e}")
        print(f"  Second derivative error: {torch.max(torch.abs(d2y_rational - d2y_true)):.2e}")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_progression()

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_progression():
    """Plot rational interpolants with increasing degrees for cos(πx)"""
    # Starting degrees
    num_degree_start = 2
    den_degree_start = 1
    num_increments = 20
    
    # Test points
    x_eval = torch.linspace(-1, 1, 1000)
    
    # True function and derivatives
    y_true = torch.cos(np.pi * x_eval)
    dy_true = -np.pi * torch.sin(np.pi * x_eval)
    d2y_true = -(np.pi**2) * torch.cos(np.pi * x_eval)
    
    for i in range(num_increments):
        num_degree = num_degree_start + i
        den_degree = den_degree_start + i
        n_points = max(num_degree, den_degree) + 5  # Add some extra points for stability
        
        # Create model
        rational_model = PolynomialRationalModel(n_points, 
                                               numerator_degree=num_degree,
                                               denominator_degree=den_degree)
        
        # Fit model
        with torch.no_grad():
            f_nodes = torch.cos(np.pi * rational_model.nodes)
            rational_model.fit_least_squares(f_nodes)
        
        # Evaluate model
        y_rational = rational_model(x_eval).detach()
        dy_rational = rational_model.derivative(x_eval, k=1).detach()
        d2y_rational = rational_model.derivative(x_eval, k=2).detach()
        
        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Rational Interpolant: P_{num_degree}/Q_{den_degree}')
        
        # Function values
        axs[0].plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
        axs[0].plot(x_eval, y_rational, 'r-', label='Rational')
        axs[0].plot(rational_model.nodes, f_nodes, 'ko', label='Nodes', markersize=4)
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_title('Function: f(x) = cos(πx)')
        axs[0].set_ylim(-1.5, 1.5)
        
        # First derivative
        axs[1].plot(x_eval, dy_true, 'k-', label='True', alpha=0.5)
        axs[1].plot(x_eval, dy_rational, 'r-', label='Rational')
        axs[1].grid(True)
        axs[1].legend()
        axs[1].set_title('First Derivative: f\'(x) = -πsin(πx)')
        axs[1].set_ylim(-5, 5)
        
        # Second derivative
        axs[2].plot(x_eval, d2y_true, 'k-', label='True', alpha=0.5)
        axs[2].plot(x_eval, d2y_rational, 'r-', label='Rational')
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title('Second Derivative: f\'\'(x) = -π²cos(πx)')
        axs[2].set_ylim(-15, 15)
        
        # Print errors
        print(f"\nDegrees: P_{num_degree}/Q_{den_degree}")
        print(f"  Function error: {torch.max(torch.abs(y_rational - y_true)):.2e}")
        print(f"  First derivative error: {torch.max(torch.abs(dy_rational - dy_true)):.2e}")
        print(f"  Second derivative error: {torch.max(torch.abs(d2y_rational - d2y_true)):.2e}")
        
        plt.tight_layout()
        plt.show()
        
        # Also print coefficients
        print("\nCoefficients:")
        print("P(x):", rational_model.p.detach().numpy())
        print("q(x):", rational_model.q.detach().numpy())

if __name__ == "__main__":
    plot_progression()


# %% [markdown]
# ## TODO Fit rationals using Prony's method

# %%
class PronyRationalModel(PolynomialRationalModel):
    def __init__(self, num_points, numerator_degree, denominator_degree):
        super().__init__(num_points, numerator_degree, denominator_degree)
        # Use equispaced nodes instead of Chebyshev
        self.nodes = torch.linspace(0, 1, num_points)
    
    def fit_prony(self, f_values):
        """
        Fit rational function R(x) = P(x)/Q(x) using Prony's method
        
        Args:
            f_values: Function values at equispaced nodes
        """
        x_samp = self.nodes.numpy()
        f_samp = f_values.detach().numpy()
        N = len(x_samp)
        n = len(self.p) - 1  # numerator degree
        m = len(self.q)      # denominator degree
        
        # For R(x) = P(x)/Q(x), we have f(x)Q(x) = P(x) at sample points
        # Q(x) = 1 + q₁x + ... + qₘxᵐ
        # P(x) = p₀ + p₁x + ... + pₙxⁿ
        
        # Step 1: Form system for denominator coefficients
        # At each point: f(x)Q(x) = P(x)
        A = np.zeros((N-n-1, m))
        b = np.zeros(N-n-1)
        
        for i in range(N-n-1):
            for j in range(m):
                # Coefficient of qⱼ in equation i
                A[i,j] = f_samp[i+j] * x_samp[i+j]**j
            # Right hand side
            b[i] = -f_samp[i+m]
            
        # Solve for denominator coefficients
        q_coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Step 2: Solve for numerator coefficients
        # Form Vandermonde-like matrix for P(x)
        V = np.zeros((N, n+1))
        for i in range(N):
            for j in range(n+1):
                V[i,j] = x_samp[i]**j
                
        # Compute Q(x) at sample points
        Q = np.ones(N)
        for i in range(N):
            for j in range(m):
                Q[i] += q_coeffs[j] * x_samp[i]**(j+1)
                
        # Solve for P coefficients: V p = f Q
        p_coeffs = np.linalg.lstsq(V, f_samp * Q, rcond=None)[0]
        
        # Update parameters
        with torch.no_grad():
            self.p.copy_(torch.tensor(p_coeffs, dtype=torch.float64))
            self.q.copy_(torch.tensor(q_coeffs, dtype=torch.float64))

def test_prony():
    """Test both standard and Prony-based rational interpolation"""
    # Parameters
    n_points = 20
    num_degree = 4
    den_degree = 4
    
    # Create models
    standard_model = PolynomialRationalModel(n_points, num_degree, den_degree)
    prony_model = PronyRationalModel(n_points, num_degree, den_degree)
    
    # Test points
    x_eval = torch.linspace(-1, 1, 1000)
    
    # Test function: cos(πx)
    y_true = torch.cos(np.pi * x_eval)
    dy_true = -np.pi * torch.sin(np.pi * x_eval)
    d2y_true = -(np.pi**2) * torch.cos(np.pi * x_eval)
    
    # Fit models
    with torch.no_grad():
        standard_model.fit_least_squares(torch.cos(np.pi * standard_model.nodes))
        prony_model.fit_prony(torch.cos(np.pi * prony_model.nodes))
    
    # Evaluate
    y_standard = standard_model(x_eval).detach()
    dy_standard = standard_model.derivative(x_eval, k=1).detach()
    d2y_standard = standard_model.derivative(x_eval, k=2).detach()
    
    y_prony = prony_model(x_eval).detach()
    dy_prony = prony_model.derivative(x_eval, k=1).detach()
    d2y_prony = prony_model.derivative(x_eval, k=2).detach()
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Function values
    axs[0].plot(x_eval, y_true, 'k-', label='True', alpha=0.5)
    axs[0].plot(x_eval, y_standard, 'b--', label='Standard')
    axs[0].plot(x_eval, y_prony, 'r:', label='Prony')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title('Function Values')
    
    # First derivative
    axs[1].plot(x_eval, dy_true, 'k-', label='True', alpha=0.5)
    axs[1].plot(x_eval, dy_standard, 'b--', label='Standard')
    axs[1].plot(x_eval, dy_prony, 'r:', label='Prony')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title('First Derivative')
    
    # Second derivative
    axs[2].plot(x_eval, d2y_true, 'k-', label='True', alpha=0.5)
    axs[2].plot(x_eval, d2y_standard, 'b--', label='Standard')
    axs[2].plot(x_eval, d2y_prony, 'r:', label='Prony')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_title('Second Derivative')
    
    plt.tight_layout()
    plt.show()
    
    # Print errors
    print("\nStandard Method Errors:")
    print(f"  Function: {torch.max(torch.abs(y_standard - y_true)):.2e}")
    print(f"  First derivative: {torch.max(torch.abs(dy_standard - dy_true)):.2e}")
    print(f"  Second derivative: {torch.max(torch.abs(d2y_standard - d2y_true)):.2e}")
    
    print("\nProny Method Errors:")
    print(f"  Function: {torch.max(torch.abs(y_prony - y_true)):.2e}")
    print(f"  First derivative: {torch.max(torch.abs(dy_prony - dy_true)):.2e}")
    print(f"  Second derivative: {torch.max(torch.abs(d2y_prony - d2y_true)):.2e}")

if __name__ == "__main__":
    test_prony()
