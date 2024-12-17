import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_dim=1, hidden_dim=32, activation=torch.tanh):
        """
        2-layer MLP that maps (B, n_dim) -> (B, 1)
        
        Args:
            hidden_dim: Dimension of hidden layer
            activation: Activation function to use (default: ReLU)
        """
        super().__init__()
        self.n_dim = n_dim
        self.activation = activation
        self.fc1 = nn.Linear(self.n_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    """
    Forward pass of the network
    Args:
        x: Input tensor of shape (B, n_dim)
    Returns:
        Output tensor of shape (B, 1)
    """
    def forward(self, x):
        assert x.shape[-1] == self.n_dim, f"Input shape must be (..., {self.n_dim})"
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x