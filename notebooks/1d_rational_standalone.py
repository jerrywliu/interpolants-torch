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
        
    def forward(self, x_eval):
        """Evaluate rational interpolant with gradient-aware weights"""
        # Get poles maintaining gradient connection
        poles = self.get_poles()
        # Compute weights directly in forward pass
        weights = compute_barycentric_weights(self.nodes, self.pole_real, self.pole_imag)
        
        # Standard barycentric interpolation
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
        
        # Compute barycentric terms using newly computed weights
        f_eval_num = torch.einsum("...n,nm,n->...m", self.values, d_x, weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, weights)
        
        return f_eval_num / f_eval_denom
    
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
    
    # Evaluate models
    y_standard = standard_model(x_eval).detach()
    y_rational = rational_model(x_eval).detach()
    
    # Plot results
    plot_results(x_eval, y_true, y_standard, y_rational, rational_model)

if __name__ == "__main__":
    test_rational_interp()

# %% [markdown]
# ### Let's try learning the poles for the steep function.

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

def test_training():
    # Create models
    n_points = 60
    standard_model = LagrangeInterpolationModel(n_points)
    rational_model = RationalInterpolationModel(n_points, num_poles=4)
    
    # Train models
    losses = train_interpolation(standard_model, rational_model, num_epochs=2000, freeze_poles=1000)
    
    # Evaluate and plot results
    x_eval = torch.linspace(-1, 1, 1000)
    plot_training_results(x_eval, standard_model, rational_model, losses)

if __name__ == "__main__":
    test_training()

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

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RationalInterpolationModel(nn.Module):
    def __init__(self, num_points, num_poles=4):
        """
        Model with both Chebyshev interpolation and rational poles
        
        Args:
            num_points: Number of Chebyshev points
            num_poles: Number of pole pairs to add near discontinuity
        """
        super().__init__()
        
        # Generate Chebyshev points of the second kind
        i = torch.linspace(0, 1, num_points)
        self.nodes = torch.cos(torch.pi * i)  # [-1, 1]
        
        # Learnable values at these points
        self.values = nn.Parameter(torch.zeros(num_points))
        
        # Learnable pole locations near ±0.5 (where the transition happens)
        # We'll place them in conjugate pairs for stability
        self.pole_offsets = nn.Parameter(0.1 * torch.randn(num_poles))
        
        # Standard Chebyshev weights
        self.weights = torch.zeros(num_points)
        self.weights[::2] = 1
        self.weights[1::2] = -1
        self.weights[0] = 0.5
        self.weights[-1] = 0.5
        
    def _get_poles(self):
        """Get complex pole locations in conjugate pairs"""
        # Place poles near ±0.5 with learned offsets
        real_parts = 0.5 * torch.ones_like(self.pole_offsets)
        imag_parts = 0.1 * torch.sigmoid(self.pole_offsets)  # Keep imaginary parts small but > 0
        
        # Create conjugate pairs at both +0.5 and -0.5
        poles = []
        for r, i in zip(real_parts, imag_parts):
            poles.extend([
                r + 1j*i, r - 1j*i,    # Conjugate pair near +0.5
                -r + 1j*i, -r - 1j*i,  # Conjugate pair near -0.5
            ])
        return torch.tensor(poles)
    
    def forward(self, x_eval):
        """
        Evaluate the rational interpolant at x_eval points
        """
        # First get standard barycentric interpolant
        d_x = x_eval.unsqueeze(0) - self.nodes.unsqueeze(1)
        
        # Handle numerical instability for small differences
        small_diff = torch.abs(d_x) < 1e-14
        small_diff_max = torch.max(small_diff, dim=0).values
        
        d_x = torch.where(
            small_diff_max[None, :],
            torch.zeros_like(d_x),
            1.0 / d_x
        )
        d_x[small_diff] = 1
        
        # Get pole locations
        poles = self._get_poles()
        
        # Add pole factors to denominator
        pole_terms = torch.prod(
            (x_eval.unsqueeze(-1) - poles.real[None, :].to(x_eval.device))**2 + 
            poles.imag[None, :].to(x_eval.device)**2,
            dim=-1
        )
        
        # Combine with barycentric formula
        f_eval_num = torch.einsum("...n,nm,n->...m", self.values, d_x, self.weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, self.weights) * pole_terms
        
        return f_eval_num / f_eval_denom

def steep_transition(x, alpha=50.0):
    """Logistic transition function from example"""
    return -1.0 + 2.0 / (1.0 + torch.exp(-alpha * (0.5 - torch.abs(x))))

# Training code
def train_model(model, alpha=50.0, num_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        # Sample points more densely near the transition
        x_uniform = torch.linspace(-1, 1, 100)
        x_dense = torch.linspace(-0.6, 0.6, 100)
        x_train = torch.cat([x_uniform, x_dense])
        y_train = steep_transition(x_train, alpha)
        
        y_pred = model(x_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Visualization
def plot_comparison(standard_model, rational_model, alpha=50.0):
    x_eval = torch.linspace(-1, 1, 1000)
    y_true = steep_transition(x_eval, alpha)
    
    with torch.no_grad():
        y_standard = standard_model(x_eval)
        y_rational = rational_model(x_eval)
    
    plt.figure(figsize=(12, 5))
    
    # Function values
    plt.subplot(121)
    plt.plot(x_eval, y_true, 'k-', label='True')
    plt.plot(x_eval, y_standard, '--', label='Standard Chebyshev')
    plt.plot(x_eval, y_rational, ':', label='Rational')
    plt.legend()
    plt.title('Function Comparison')
    plt.grid(True)
    
    # Error
    plt.subplot(122)
    plt.semilogy(x_eval, torch.abs(y_true - y_standard), '--', label='Standard Error')
    plt.semilogy(x_eval, torch.abs(y_true - y_rational), ':', label='Rational Error')
    plt.legend()
    plt.title('Error Comparison (log scale)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create models
    n_points = 30
    standard_model = LagrangeInterpolationModel(n_points)
    rational_model = RationalInterpolationModel(n_points, num_poles=4)
    
    # Train both models
    train_model(standard_model)
    train_model(rational_model)
    
    # Compare results
    plot_comparison(standard_model, rational_model)

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
