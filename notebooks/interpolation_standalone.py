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
# # Dataset
# Our goal is to do simple interpolation.
# We have an unknown function $f : \mathbb{R} \to \mathbb{R}$.
# We're given samples $\{(x_i, f(x_i))\}_i$, and our goal is to recover the function.

# %%
class FunctionDataset(Dataset):
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


# %%
# Create dataset and dataloader
num_points = 100
dataset = FunctionDataset(num_points, sampling='random')
dataloader = DataLoader(dataset, batch_size=num_points, shuffle=False)

# %%
# Example usage:
for x, y in dataloader:
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    # You can access first few points:
    print("\nFirst few points:")
    for i in range(min(5, len(x))):
        print(f"x: {x[i]:.4f}, sin(x): {y[i]:.4f}")


# %%
def plot_function_samples(dataset, title=None, num_plot_points=1000):
    """
    Plot the sampled points and the true continuous function.
    
    Args:
        dataset: FunctionDataset instance
        title: Optional title for the plot
        num_plot_points: Number of points for plotting the continuous function
    """
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot true continuous function
    x_continuous = torch.linspace(dataset.domain[0], dataset.domain[1], num_plot_points)
    y_continuous = dataset.function(x_continuous)
    plt.plot(x_continuous, y_continuous, 'b-', label='True function', alpha=0.5)
    
    # Plot sampled points
    plt.plot(dataset.x, dataset.y, 'ro', label=f'Samples ({dataset.sampling})', 
             markersize=4, alpha=0.6)
    
    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    if title:
        plt.title(title)
    else:
        plt.title(f'Function Samples using {dataset.sampling} points')
    plt.legend()
    plt.show()


# %%
plot_function_samples(dataset)


# %% [markdown]
# # Models
# We look at two models:
# 1. A standard 2-layer $MLP : \mathbb{R} \to \mathbb{R}$, with a single hidden layer.
# 2. We parameterize a function on [-1, 1] via the Lagrange interpolating polynomial on $n$ Chebyshev nodes, where the values on those nodes is _learned_.

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
        
    def forward(self, x_eval, eps=1e-14):
        """
        Evaluate Lagrange interpolant at points x using barycentric formula
        Args:
            x_eval: points to evaluate at (N_eval)
        Returns:
            interpolated values (..., N_eval)
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
        d_x[small_diff] = 1  # assumes no column has multiple small_diff
        
        # Interpolate
        f_eval_num = torch.einsum("...n,nm,n->...m", self.values, d_x, self.weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, self.weights)
        
        return f_eval_num / f_eval_denom


# %%
def test_lagrange_model():
    """Run sanity checks on LagrangeInterpolationModel"""
    
    # Test 1: Exact interpolation at nodes
    print("Test 1: Checking exact interpolation at nodes...")
    model = LagrangeInterpolationModel(num_points=5)
    
    # Set some arbitrary values at the nodes
    model.values.data = torch.tensor([1.0, -0.5, 2.0, 0.8, -1.2])
    
    # Evaluate at nodes
    y_nodes = model(model.nodes)
    print(model.nodes.shape)
    print(y_nodes.shape)
    print(y_nodes)
    max_error = torch.max(torch.abs(y_nodes - model.values))
    print(f"Maximum error at nodes: {max_error:.2e}")
    assert max_error < 1e-10, "Failed exact interpolation at nodes"
    
    # Test 2: Reproduce sin(x)
    print("\nTest 2: Checking reproduction of sin(x)...")
    model = LagrangeInterpolationModel(num_points=11)
    model.values.data = torch.sin(model.nodes)
    
    # Evaluate at fine grid
    x_test = torch.linspace(-1, 1, 100)
    y_test = model(x_test)
    y_true = torch.sin(x_test)
    max_error = torch.max(torch.abs(y_test - y_true))
    print(f"Maximum error approximating sin(x): {max_error:.2e}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_true, 'b-', label='True sin(x)', alpha=0.7)
    plt.plot(x_test, y_test.detach().cpu().numpy(), 'r--', label='Interpolant', alpha=0.7)
    plt.plot(model.nodes.detach().cpu().numpy(), model.values.detach().cpu().numpy(), 'ko', label='Nodes', markersize=4)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Sin(x) Reproduction Test')
    plt.show()
    
    # Test 3: Linear precision
    print("\nTest 3: Checking linear precision...")
    model = LagrangeInterpolationModel(num_points=21)
    
    # Set values to a linear function f(x) = 2x + 1
    model.values.data = 2 * model.nodes + 1
    
    # Test at random points
    x_random = torch.rand(100) * 2 - 1  # Random points in [-1, 1]
    y_random = model(x_random)
    y_true = 2 * x_random + 1
    max_error = torch.max(torch.abs(y_random - y_true))
    print(f"Maximum error on linear function: {max_error:.2e}")
    assert max_error < 1e-6, "Failed linear precision test"
    
test_lagrange_model()


# %% [markdown]
# # Training

# %%
def train_models(mlp_model, lagrange_model, train_dataset, test_dataset, 
                num_epochs=1000, learning_rate=1e-3, batch_size=32):
    """
    Train both models and compare their performance
    """
    # Setup dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Setup optimizers
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
    lagrange_optimizer = optim.Adam(lagrange_model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'mlp_train_loss': [],
        'lagrange_train_loss': [],
        'mlp_test_loss': [],
        'lagrange_test_loss': []
    }
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        mlp_epoch_loss = 0.0
        lagrange_epoch_loss = 0.0
        
        # Training
        for x, y in train_loader:
            # Train MLP
            mlp_optimizer.zero_grad()
            mlp_out = mlp_model(x)
            mlp_loss = criterion(mlp_out, y)
            mlp_loss.backward()
            mlp_optimizer.step()
            
            # Train Lagrange
            lagrange_optimizer.zero_grad()
            lagrange_out = lagrange_model(x)
            lagrange_loss = criterion(lagrange_out, y)
            lagrange_loss.backward()
            lagrange_optimizer.step()
            
            mlp_epoch_loss += mlp_loss.item()
            lagrange_epoch_loss += lagrange_loss.item()
        
        # Compute average training loss
        mlp_epoch_loss /= len(train_loader)
        lagrange_epoch_loss /= len(train_loader)
        
        # Test loss
        with torch.no_grad():
            x_test, y_test = next(iter(test_loader))
            mlp_test_loss = criterion(mlp_model(x_test), y_test).item()
            lagrange_test_loss = criterion(lagrange_model(x_test), y_test).item()
        
        # Record history
        history['mlp_train_loss'].append(mlp_epoch_loss)
        history['lagrange_train_loss'].append(lagrange_epoch_loss)
        history['mlp_test_loss'].append(mlp_test_loss)
        history['lagrange_test_loss'].append(lagrange_test_loss)
        
        # Print progress
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"MLP - Train Loss: {mlp_epoch_loss:.8f}, Test Loss: {mlp_test_loss:.8f}")
            print(f"Lagrange - Train Loss: {lagrange_epoch_loss:.8f}, Test Loss: {lagrange_test_loss:.8f}\n")
    
    return history


# %%
def plot_results(history, train_dataset, test_dataset, mlp_model, lagrange_model, function=torch.sin):
    """Plot training curves, final fits, and residuals"""
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.semilogy(history['mlp_train_loss'], label='MLP Train')
    plt.semilogy(history['mlp_test_loss'], label='MLP Test')
    plt.semilogy(history['lagrange_train_loss'], label='Lagrange Train')
    plt.semilogy(history['lagrange_test_loss'], label='Lagrange Test')
    plt.grid(True)
    plt.legend()
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Final fits
    plt.subplot(2, 2, 2)
    with torch.no_grad():
        # Get fine grid for smooth plots
        x_fine = torch.linspace(-1, 1, 200)
        y_mlp = mlp_model(x_fine)
        y_lagrange = lagrange_model(x_fine)
        y_true = function(x_fine)  # Assuming sin(x) is the true function
        
        # Plot fits
        plt.plot(x_fine, y_mlp, label='MLP', alpha=0.7)
        plt.plot(x_fine, y_lagrange, label='Lagrange', alpha=0.7)
        plt.plot(x_fine, y_true, '--', label='True', alpha=0.7)
        
        # Plot training points
        x_train, y_train = train_dataset[:]
        plt.scatter(x_train, y_train, c='b', label='Train', alpha=0.5, s=20)
        
        # Plot test points
        x_test, y_test = test_dataset[:]
        plt.scatter(x_test, y_test, c='r', label='Test', alpha=0.5, s=20)
    
    plt.grid(True)
    plt.legend()
    plt.title('Final Fits')
    plt.xlabel('x')
    plt.ylabel('y')
        
    # MLP Residual plot
    plt.subplot(2, 2, 3)
    mlp_residual = torch.abs(y_mlp - y_true)
    plt.semilogy(x_fine, mlp_residual, label='MLP', alpha=0.7)
    plt.axhline(y=1e-16, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.title('MLP Absolute Error')
    plt.xlabel('x')
    plt.ylabel('|Error|')
    
    # Lagrange Residual plot
    plt.subplot(2, 2, 4)
    lagrange_residual = torch.abs(y_lagrange - y_true)
    plt.semilogy(x_fine, lagrange_residual, label='Lagrange', alpha=0.7)
    plt.axhline(y=1e-16, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.title('Lagrange Absolute Error')
    plt.xlabel('x')
    plt.ylabel('|Error|')
    
    # Print max errors
    print(f"Maximum absolute errors:")
    print(f"MLP: {torch.max(torch.abs(mlp_residual)):.2e}")
    print(f"Lagrange: {torch.max(torch.abs(lagrange_residual)):.2e}")
    
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Train on noiseless data $(x_i, sin(x_i))$
# Observation: optimization problem becomes better conditioned as N_train > N_cheb.
#
# With proper choice of N_cheb (based on smoothness of $f$), convergence of Lagrange interpolant is much faster than MLP.
#
# Why the linear convergence? Optimizing the values of the Lagrange poylnomial at the Cheb nodes is equivalent to gradient descent on least squares (TODO write this out)

# %% jupyter={"outputs_hidden": true}
torch.manual_seed(0)

# Create datasets
N_train = 30
N_test = 100
N_cheb = 11

train_dataset = FunctionDataset(N_train, sampling='random')
test_dataset = FunctionDataset(N_test, sampling='equispaced')

# Create models
mlp_model = MLP(hidden_dim=32, activation=torch.tanh)
lagrange_model = LagrangeInterpolationModel(N_cheb)

# Train models
history = train_models(
    mlp_model, 
    lagrange_model, 
    train_dataset, 
    test_dataset,
    num_epochs=50000,
    learning_rate=1e-2,
    batch_size=32
)

# Plot results
plot_results(history, train_dataset, test_dataset, mlp_model, lagrange_model)

# %% [markdown]
# ## Train on noisy data $(x_i, sin(x_i)+\varepsilon_i)$
# With too few samples, interpolant overfits to noise and completely blows up...

# %% jupyter={"outputs_hidden": true}
torch.manual_seed(0)

# Create datasets
N_train = 30
N_test = 100
N_cheb = 11

train_dataset = FunctionDataset(N_train, sampling='random', noise_std=0.1)
test_dataset = FunctionDataset(N_test, sampling='equispaced')

# Create models
mlp_model = MLP(hidden_dim=32, activation=torch.tanh)
lagrange_model = LagrangeInterpolationModel(N_cheb)

# Train models
history = train_models(
    mlp_model, 
    lagrange_model, 
    train_dataset, 
    test_dataset,
    num_epochs=10000,
    learning_rate=1e-2,
    batch_size=32
)

# Plot results
plot_results(history, train_dataset, test_dataset, mlp_model, lagrange_model)

# %% [markdown]
# ... but with enough samples, interpolant works.

# %% jupyter={"outputs_hidden": true}
torch.manual_seed(0)

# Create datasets
N_train = 100
N_test = 1000
N_cheb = 11

train_dataset = FunctionDataset(N_train, sampling='random', noise_std=0.1)
test_dataset = FunctionDataset(N_test, sampling='equispaced')

# Create models
mlp_model = MLP(hidden_dim=32, activation=torch.tanh)
lagrange_model = LagrangeInterpolationModel(N_cheb)

# Train models
history = train_models(
    mlp_model, 
    lagrange_model, 
    train_dataset, 
    test_dataset,
    num_epochs=10000,
    learning_rate=1e-2,
    batch_size=32
)

# Plot results
plot_results(history, train_dataset, test_dataset, mlp_model, lagrange_model)


# %% [markdown]
# # PINNs
# Now, we're given an initial condition ($u(0) = 0$), and a PDE ($u' = u + t$).
# From this, we want to learn $u(t)$ over the full [-1, 1] such that $u$ satisfies both.
#
# As above, we try parameterizing $u$ in two ways: standard MLP, and interpolating on Chebyshev nodes.

# %%
def compute_derivative_matrix(nodes):
    """Compute the spectral differentiation matrix for Chebyshev points"""
    n = len(nodes)
    D = torch.zeros((n, n), dtype=nodes.dtype, device=nodes.device)
    
    # Helper for endpoint formulas 
    i = torch.arange(n)
    c = torch.ones(n)
    c[0] = 2
    c[-1] = 2
    
    # Interior formulas
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = c[i]/c[j] * (-1)**(i+j) / (nodes[i] - nodes[j])
    
    # Diagonal entries
    D.diagonal().copy_(-torch.sum(D, dim=1))
    
    return D

def test_derivative_approximation(ns=[5, 10, 20, 40]):
    """Test derivative approximation for different numbers of points"""
    # Evaluation points for error computation
    x_eval = torch.linspace(-1, 1, 1000)
    errors = []
    
    plt.figure(figsize=(15, 10))
    
    # Function and true derivative
    f = lambda x: torch.sin(x)
    df = lambda x: torch.cos(x)
    
    # Plot for each n
    for n in ns:
        # Generate Chebyshev points
        j = torch.linspace(0, 1, n)
        nodes = torch.cos(torch.pi * j)  # [-1, 1]
        
        # Compute differentiation matrix
        D = compute_derivative_matrix(nodes)
        
        # Compute function values and derivative at nodes
        f_nodes = f(nodes)
        df_nodes = D @ f_nodes
        
        # Create interpolant for derivative values
        lagrange_model = LagrangeInterpolationModel(n)
        lagrange_model.nodes = nodes
        lagrange_model.values.data = df_nodes
        
        # Evaluate at fine grid
        df_approx = lagrange_model(x_eval)
        df_true = df(x_eval)
        error = torch.abs(df_approx - df_true)
        max_error = torch.max(error)
        errors.append(max_error.item())
        
        # Plot absolute error
        plt.subplot(2, 2, 1)
        plt.semilogy(x_eval, error.detach().cpu().numpy(), label=f'n={n}')
        
        # Plot true vs approximated derivative
        plt.subplot(2, 2, 2)
        if n == ns[-1]:  # only plot for largest n
            plt.plot(x_eval, df_true, '--', label='True', alpha=0.7)
            plt.plot(x_eval, df_approx.detach().cpu().numpy(), label=f'Approx (n={n})', alpha=0.7)
    
    # Plot convergence
    plt.subplot(2, 2, 3)
    plt.loglog(ns, errors, 'o-')
    plt.grid(True)
    plt.xlabel('Number of points (n)')
    plt.ylabel('Maximum error')
    plt.title('Convergence of derivative approximation')
    
    # Finish error plot
    plt.subplot(2, 2, 1)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('Absolute error')
    plt.title('Error in derivative approximation')
    plt.legend()
    
    # Finish derivative plot
    plt.subplot(2, 2, 2)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('Derivative')
    plt.title('True vs Approximated Derivative')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print errors
    print("\nMaximum errors:")
    for n, err in zip(ns, errors):
        print(f"n = {n}: {err:.2e}")

test_derivative_approximation(ns=[5, 11, 21, 41])


# %% [markdown]
# ## Redefine the models, now with the ability to compute derivatives w.r.t. spatial dimension.

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
        
        Args:
            x_eval: points to evaluate at
            k: which derivative (k=0 returns function values)
        Returns:
            k-th derivative values at x_eval
        """
        if k == 0:
            return self(x_eval)
            
        # Compute k-th derivative values at nodes
        dk_nodes = self.derivative_matrix(k) @ self.values
        
        # Create interpolant for these derivative values
        deriv_model = LagrangeInterpolationModel(len(self.nodes))
        deriv_model.nodes = self.nodes
        deriv_model.values.data = dk_nodes
        
        # Evaluate at desired points
        return deriv_model(x_eval)
        
    def forward(self, x_eval, eps=1e-14):
        """
        Evaluate Lagrange interpolant at points x using barycentric formula
        Args:
            x_eval: points to evaluate at (N_eval)
        Returns:
            interpolated values (..., N_eval)
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
        d_x[small_diff] = 1  # assumes no column has multiple small_diff
        
        # Interpolate
        f_eval_num = torch.einsum("...n,nm,n->...m", self.values, d_x, self.weights)
        f_eval_denom = torch.einsum("nm,n->m", d_x, self.weights)
        
        return f_eval_num / f_eval_denom


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
    
    # derivative_matrix and compute_derivative_matrix remain the same


# %%
# Create model and set to interpolate sin(x)
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


# %%
def compute_pde_loss(model, colloc_points, boundary_weight=1.0):
    """
    Compute loss for ODE u' = u + x with u(0) = 0
    """
    if isinstance(model, LagrangeInterpolationModel):
        u = model(colloc_points)
        du = model.derivative(colloc_points, k=1)
    else:
        # For MLP, compute gradient manually
        x = colloc_points.clone().requires_grad_(True)
        u = model(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        
    # PDE residual: u' = u + x
    if isinstance(model, LagrangeInterpolationModel):
        pde_residual = du - u - colloc_points
    else:
        pde_residual = du - u - x
        
    pde_loss = torch.mean(pde_residual**2)
    
    # Boundary condition: u(0) = 0
    bc_point = torch.tensor([0.0], dtype=torch.float64)
    bc_residual = model(bc_point)
    bc_loss = boundary_weight * bc_residual**2
    
    return pde_loss + bc_loss, pde_residual, bc_residual

def train_pinn(model, n_colloc=100, n_epochs=1000, lr=1e-3, boundary_weight=1.0, 
                colloc_sampling='equispaced'):
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
        else:
            raise ValueError(f"Unknown sampling method: {colloc_sampling}")
        return points
    
    # Initial collocation points
    colloc_points = get_colloc_points()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {'loss': [], 'pde_residual': [], 'bc_residual': []}
    
    # Training loop
    for epoch in tqdm(range(n_epochs)):
        # Resample points if using random sampling
        if colloc_sampling == 'random_resample':
            colloc_points = get_colloc_points()
            
        optimizer.zero_grad()
        
        # Compute loss
        loss, pde_residual, bc_residual = compute_pde_loss(
            model, colloc_points, boundary_weight
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

def plot_solution(model):
    """Plot the learned solution and compare with true solution"""
    x = torch.linspace(-1, 1, 200, dtype=torch.float64)
    
    # True solution
    true_u = torch.exp(x) - x - 1
    true_du = torch.exp(x) - 1  # derivative of true solution
    
    with torch.no_grad():
        u = model(x)
        du = model.derivative(x, k=1) if isinstance(model, LagrangeInterpolationModel) else None
    
    plt.figure(figsize=(15, 5))
    
    # Solution
    plt.subplot(131)
    plt.plot(x, u, label='Learned')
    plt.plot(x, true_u, '--', label='True')
    plt.grid(True)
    plt.legend()
    plt.title('Solution')
    
    # Derivative
    plt.subplot(132)
    if du is not None:
        plt.plot(x, du, label="Learned u'")
    plt.plot(x, true_du, '--', label="True u'")
    plt.grid(True)
    plt.legend()
    plt.title("Derivative")
    
    # Error
    plt.subplot(133)
    error = torch.abs(u - true_u)
    if du is not None:
        deriv_error = torch.abs(du - true_du)
        plt.semilogy(x, error, label='Solution Error')
        plt.semilogy(x, deriv_error, label='Derivative Error')
    plt.grid(True)
    plt.legend()
    plt.title('Error')
    
    plt.tight_layout()
    plt.show()
    
    # Print maximum errors
    print(f"Maximum solution error: {torch.max(error):.2e}")
    if du is not None:
        print(f"Maximum derivative error: {torch.max(deriv_error):.2e}")


# %% [markdown]
# ## Sanity check: how well do MLP and Lagrange interpolant learn the solution function in a regression setting?
# Answer: pretty well. The solution is smooth.

# %% jupyter={"outputs_hidden": true}
torch.manual_seed(0)

# Create datasets
N_train = 30
N_test = 100
N_cheb = 11

function = lambda x : torch.exp(x) - x - 1

train_dataset = FunctionDataset(N_train, sampling='random', function=function)
test_dataset = FunctionDataset(N_test, sampling='equispaced', function=function)

# Create models
mlp_model = MLP(hidden_dim=32, activation=torch.tanh)
lagrange_model = LagrangeInterpolationModel(N_cheb)

# Train models
history = train_models(
    mlp_model, 
    lagrange_model, 
    train_dataset, 
    test_dataset,
    num_epochs=25000,
    learning_rate=1e-2,
    batch_size=32
)

# Plot results
plot_results(history, train_dataset, test_dataset, mlp_model, lagrange_model, function=function)

# %% [markdown]
# ## How well does the Lagrange interpolant learn the function in a PINN setting?

# %%
torch.manual_seed(0)

# Create model
n_points = 41
model = LagrangeInterpolationModel(n_points)

# Train
history = train_pinn(
    model, 
    n_colloc=21, 
    n_epochs=50000,
    lr=1e-3,
    boundary_weight=1.0,
    colloc_sampling='random',
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model)

# %% [markdown]
# ## What about the MLP?

# %%
torch.manual_seed(0)

# Create model
model = MLP(hidden_dim=64)
# Train
history = train_pinn(
    model, 
    n_colloc=100, 
    n_epochs=50000,
    lr=1e-4,
    boundary_weight=10,
    colloc_sampling='random',
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.semilogy(history['loss'], label='Total Loss')
plt.semilogy(history['pde_residual'], label='PDE Residual')
plt.semilogy(history['bc_residual'], label='BC Residual')
plt.grid(True)
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot solution comparison
plot_solution(model)


# %% [markdown]
# Training is a lot harder for the PINN setting. For this problem
# $$u' = u + t$$
# $$u(0) = 0$$
# it makes sense that gradient descent on randomly sampled points (enforcing PDE loss at each step) would be slow, since the only ground truth signal we have about $u$ is its value at $0$, and we have to use the local PDE constraint to propagate that info across the full domain.

# %% [markdown]
# For this 1D problem, this differential equation also reduces to a least squares problem on the Cheb nodes, but somehow gradient descent on the interpolating polynomial doesn't work well.
# Likely because the problem's really ill-conditioned: in effect you'd need to invert the Cheb differentiation matrix to solve. I also ran into this when trying to solve the in-context ODE problems from the workshop paper with gradient descent.

# %%
def cheb_diff_operator(N):
    """
    Compute differentiation operator in Chebyshev basis.
    Returns matrix D such that if f(x) = sum(c_k T_k(x)),
    then f'(x) = sum((Dc)_k T_k(x))
    """
    D = torch.zeros((N+1, N+1), dtype=torch.float64)
    
    # Fill in the differentiation matrix for Chebyshev coefficients
    for k in range(N):
        for j in range(k+1, N+1, 2):  # Only odd j-k contribute
            D[k,j] = 2*j
    
    return D

def modified_cond(A):
    """
    Compute modified condition number ignoring smallest singular value
    Returns sigma_1 / sigma_{n-1}
    """
    # Get singular values (sorted in descending order)
    _, sigma, _ = torch.linalg.svd(A)
    # Return ratio of largest to second-smallest
    return sigma[0] / sigma[-2]

def analyze_differentiation(Ns=[4, 10, 20, 40]):
    """Complete analysis of differentiation operators"""
    results = {
        'N': Ns,
        'cond_cheb': [],
        'cond_nodal': [],
        'sigma_cheb': [],
        'sigma_nodal': []
    }
    
    for N in Ns:
        # Get points and matrices
        x = torch.cos(torch.pi * torch.arange(N+1) / N)
        V = cheb_vandermonde(x, N)
        D_cheb = cheb_diff_operator(N)
        D_nodal = cheb_diff_matrix(N)[0]
        
        # Store condition numbers
        results['cond_cheb'].append(modified_cond(D_cheb))
        results['cond_nodal'].append(torch.linalg.cond(D_nodal))
        
        # Store spectra
        _, sigma_cheb, _ = torch.linalg.svd(D_cheb)
        _, sigma_nodal, _ = torch.linalg.svd(D_nodal)
        results['sigma_cheb'].append(sigma_cheb)
        results['sigma_nodal'].append(sigma_nodal)
        
        print(f"\nFor N = {N}:")
        print(f"Condition numbers:")
        print(f"  D (Chebyshev basis, modified): {results['cond_cheb'][-1]:.2e}")
        print(f"  D (nodal basis): {results['cond_nodal'][-1]:.2e}")
        print(f"  V (Vandermonde): {torch.linalg.cond(V):.2e}")
        print(f"Expected O(N²): {N**2:.1f}")
        
        # Test accuracy on sin(x)
        if N == 20:  # Detailed analysis for N=20
            f = torch.sin(x)
            
            # Chebyshev basis
            f_coeffs = torch.linalg.solve(V, f)
            df_coeffs = D_cheb @ f_coeffs
            df_cheb = V @ df_coeffs
            
            # Nodal basis
            df_nodal = D_nodal @ f
            df_true = torch.cos(x)
            
            plt.figure(figsize=(20, 10))
            
            # Compare derivatives
            plt.subplot(231)
            plt.plot(x, df_cheb, 'o', label='Chebyshev')
            plt.plot(x, df_nodal, 'x', label='Nodal')
            plt.plot(x, df_true, '-', label='True')
            plt.grid(True)
            plt.legend()
            plt.title('sin\'(x)')
            
            # Plot errors
            plt.subplot(232)
            plt.semilogy(x, torch.abs(df_cheb - df_true), 'o-', label='Chebyshev')
            plt.semilogy(x, torch.abs(df_nodal - df_true), 'x-', label='Nodal')
            plt.grid(True)
            plt.legend()
            plt.title('Error')
            
            # Show coefficient decay
            plt.subplot(233)
            plt.semilogy(torch.abs(f_coeffs), 'o-', label='f')
            plt.semilogy(torch.abs(df_coeffs), 'x-', label='f\'')
            plt.grid(True)
            plt.legend()
            plt.title('Coefficient Decay')
            
            # Compare matrices
            plt.subplot(234)
            plt.imshow(D_cheb, cmap='RdBu')
            plt.colorbar()
            plt.title('D (Chebyshev basis)')
            
            # Spectra
            plt.subplot(235)
            plt.semilogy(results['sigma_cheb'][-1], 'o-', label='Chebyshev')
            plt.semilogy(results['sigma_nodal'][-1], 'x-', label='Nodal')
            plt.grid(True)
            plt.legend()
            plt.title('Singular Values')
            
            # Condition number growth
            plt.subplot(236)
            plt.loglog(Ns[:len(results['cond_cheb'])], results['cond_cheb'], 'o-', label='Chebyshev (modified)')
            plt.loglog(Ns[:len(results['cond_nodal'])], results['cond_nodal'], 'x-', label='Nodal')
            plt.loglog(Ns[:len(results['cond_cheb'])], [n**2 for n in Ns[:len(results['cond_cheb'])]], 
                      '--', label='O(N²)')
            plt.grid(True)
            plt.legend()
            plt.title('Condition Number Growth')
            
            plt.tight_layout()
            plt.show()

# Run analysis
analyze_differentiation([4, 10, 20, 40, 80])

# %%
import torch

def cheb_diff_matrix(N):
    """
    Compute standard square differentiation matrix on N second-kind points
    with improved numerical stability
    """
    # Compute points
    j = torch.arange(N, dtype=torch.float64)
    theta = j * torch.pi / (N-1)
    x = torch.cos(theta)
    
    # Initialize matrix
    D = torch.zeros((N, N), dtype=torch.float64)
    
    # Explicitly handle endpoints
    D[0, 0] = -(2*(N-1)**2 + 1) / 6
    D[-1, -1] = (2*(N-1)**2 + 1) / 6
    
    if N > 2:
        # Interior formulas using sine functions to avoid cancellation
        for i in range(1, N-1):
            for j in range(N):
                if i != j:
                    D[i,j] = (-1)**(i+j) * torch.sin(theta[i]) * torch.sin(theta[j]) / torch.sin(theta[i] - theta[j])
            
            # Fill diagonal using explicit formula rather than sum
            D[i,i] = -x[i] / (2*(1 - x[i]**2))
    
    return D, x

def interpolation_matrix(x_from, x_to, eps=1e-14):
    """
    Compute barycentric interpolation matrix using stable formulas
    """
    n_from = len(x_from)
    n_to = len(x_to)
    
    # Compute barycentric weights for source points
    # Use logarithms to avoid overflow in product
    w = torch.ones(n_from, dtype=torch.float64)
    for j in range(n_from):
        for k in range(n_from):
            if k != j:
                w[j] *= (x_from[j] - x_from[k])
    w = 1.0 / w
    w[0] *= 0.5
    w[-1] *= 0.5
    
    # Initialize matrix
    P = torch.zeros((n_to, n_from), dtype=torch.float64)
    
    # Compute entries
    for i in range(n_to):
        # Handle exact matches first
        exact_match = torch.abs(x_to[i] - x_from) < eps
        if exact_match.any():
            idx = torch.where(exact_match)[0][0]
            P[i, idx] = 1.0
        else:
            # Compute denominator first to check scale
            denom = torch.sum(w / (x_to[i] - x_from))
            P[i,:] = (w / (x_to[i] - x_from)) / denom
    
    return P

def rectangular_diff_matrix(N):
    """
    Compute rectangular differentiation matrix via two-step process
    with improved numerical stability
    """
    # Step 1: Compute square differentiation matrix
    D, x = cheb_diff_matrix(N)
    
    # Step 2: Compute first-kind points for output
    k = torch.arange(N-1, dtype=torch.float64)
    y = torch.cos((2*k + 1) * torch.pi / (2*(N-1)))
    
    # Get downsampling matrix and compose
    P = interpolation_matrix(x, y)
    
    return P @ D, x, y

def test_derivatives():
    """Test derivatives with error monitoring"""
    ns = [5, 10, 20, 40]
    test_funcs = [
        (torch.sin, torch.cos, "sin(x)"),
        (torch.cos, lambda x: -torch.sin(x), "cos(x)"),
        (lambda x: x**2, lambda x: 2*x, "x^2"),
        (lambda x: x**3, lambda x: 3*x**2, "x^3")
    ]
    
    for n in ns:
        print(f"\nN = {n}:")
        D, x, y = rectangular_diff_matrix(n)
        
        for f, df, name in test_funcs:
            try:
                # Compute numerical derivative
                f_nodes = f(x)
                df_numeric = D @ f_nodes
                
                # Compare with true derivative
                df_true = df(y)
                error = torch.max(torch.abs(df_numeric - df_true))
                print(f"Max error for {name}: {error:.2e}")
            except:
                print(f"Overflow in {name}")
        
        # Print condition number and matrix norms
        print(f"Condition number: {torch.linalg.cond(D):.2e}")
        print(f"Matrix norm: {torch.linalg.norm(D):.2e}")

test_derivatives()

# Show example matrix for N=5
D, x, y = rectangular_diff_matrix(5)
print("\nExample 4x5 differentiation matrix:")
print(D)

# Test exact differentiation of linear function
f = x  # should give exact derivative = 1
df_numeric = D @ f
df_true = torch.ones_like(y)
error = torch.max(torch.abs(df_numeric - df_true))
print(f"\nLinear function differentiation error: {error:.2e}")

# %%
for n in [5, 11, 21, 41]:
    print(n)
    D, x = cheb_diff_matrix(n)
    _, sigma, _ = torch.linalg.svd(D-torch.eye(n))
    print(sigma)
    print(sigma[0] / sigma[-1])

# %%
import torch
import math
import matplotlib.pyplot as plt

def fd_matrix(N):
    """
    Compute square finite difference matrix on N equispaced points.
    
    Args:
        N: number of points
    Returns:
        D: N x N first derivative matrix
        x: grid points
    """
    # Grid points on [-1, 1]
    x = torch.linspace(-1, 1, N, dtype=torch.float64)
    h = x[1] - x[0]
    
    # Create matrix
    D = torch.zeros((N, N), dtype=torch.float64)
    
    # Interior points: centered difference
    for i in range(1, N-1):
        D[i, i-1] = -1/(2*h)
        D[i, i+1] = 1/(2*h)
    
    # Boundary points: one-sided differences
    D[0, 0:2] = torch.tensor([-1/h, 1/h])
    D[-1, -2:] = torch.tensor([-1/h, 1/h])
    
    return D, x

def cheb_diff_matrix(N):
    """
    Compute square Chebyshev differentiation matrix.
    
    Args:
        N: number of points
    Returns:
        D: N x N differentiation matrix
        x: Chebyshev points
    """
    # Chebyshev points
    i = torch.arange(N, dtype=torch.float64)
    x = torch.cos(i * math.pi / (N-1))
    
    # Weights for barycentric formula
    c = torch.ones(N, dtype=torch.float64)
    c[0] = 2
    c[-1] = 2
    
    # Initialize matrix
    D = torch.zeros((N, N), dtype=torch.float64)
    
    # Fill off-diagonal entries
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i,j] = (c[i]/c[j]) * (-1)**(i+j) / (x[i] - x[j])
    
    # Fill diagonal using negative sum
    D.diagonal().copy_(-torch.sum(D, dim=1))
    
    return D, x

def compare_methods():
    """Compare FD and spectral differentiation"""
    Ns = [5, 10, 20, 40, 80]
    conds_fd = []
    conds_spec = []
    
    for N in Ns:
        # Get matrices
        D_fd, x_fd = fd_matrix(N)
        D_spec, x_spec = cheb_diff_matrix(N)
        
        # Get singular values
        _, s_fd, _ = torch.linalg.svd(D_fd)
        _, s_spec, _ = torch.linalg.svd(D_spec)
        
        # Compute condition numbers using largest / second-smallest singular value
        conds_fd.append((s_fd[0] / s_fd[-2]).item())
        conds_spec.append((s_spec[0] / s_spec[-2]).item())
        
        # Test on sin(x) for largest N
        if N == Ns[-1]:
            # FD error
            f_fd = torch.sin(x_fd)
            df_fd = D_fd @ f_fd
            df_true_fd = torch.cos(x_fd)
            error_fd = torch.max(torch.abs(df_fd - df_true_fd))
            
            # Spectral error
            f_spec = torch.sin(x_spec)
            df_spec = D_spec @ f_spec
            df_true_spec = torch.cos(x_spec)
            error_spec = torch.max(torch.abs(df_spec - df_true_spec))
            
            print(f"\nErrors for N = {N}:")
            print(f"FD error: {error_fd:.2e}")
            print(f"Spectral error: {error_spec:.2e}")
    
    # Plot condition numbers
    plt.figure(figsize=(10, 5))
    plt.loglog(Ns, conds_fd, 'o-', label='Finite Difference')
    plt.loglog(Ns, conds_spec, 'o-', label='Spectral')
    plt.loglog(Ns, [n**2 for n in Ns], '--', label='O(N²)', alpha=0.5)
    plt.grid(True)
    plt.xlabel('N')
    plt.ylabel('Condition Number')
    plt.legend()
    plt.title('Condition Numbers of Square Differentiation Matrices')
    
    # Print table
    print("\nCondition Numbers:")
    print("N\tFinite Diff\tSpectral")
    for i, N in enumerate(Ns):
        print(f"{N}\t{conds_fd[i]:.2e}\t{conds_spec[i]:.2e}")
    
    plt.show()

if __name__ == "__main__":
    compare_methods()

# %%
