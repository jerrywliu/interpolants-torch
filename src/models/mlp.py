import torch
import torch.nn as nn
from typing import List


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

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the network
        Args:
            x: List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at
        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) - interpolated values
        """
        # Form the meshgrid of points
        out_shape = [sample.shape[0] for sample in x]
        x_mesh = torch.meshgrid(*x, indexing="ij")
        x_mesh = torch.stack(x_mesh, dim=-1)
        x_mesh = x_mesh.reshape(-1, self.n_dim)
        x_mesh = self.activation(self.fc1(x_mesh))
        x_mesh = self.fc2(x_mesh)
        return x_mesh.reshape(out_shape)
