import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    def __init__(
        self,
        n_dim: int = 1,
        n_layers: int = 2,
        hidden_dim: int = 32,
        activation: torch.nn.Module = torch.tanh,
        device: str = "cpu",
    ):
        """
        2-layer MLP that maps (B, n_dim) -> (B, 1)

        Args:
            hidden_dim: Dimension of hidden layer
            activation: Activation function to use (default: ReLU)
        """
        super().__init__()
        self.device = torch.device(device)
        self.n_dim = n_dim
        self.activation = activation
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.fc = nn.ModuleList([
            nn.Linear(self.n_dim, self.hidden_dim, device=device)
        ]
            + [
                nn.Linear(self.hidden_dim, self.hidden_dim, device=device)
                for _ in range(self.n_layers - 2)
            ]
            + [nn.Linear(self.hidden_dim, 1, device=device)]
        )
        self.device = device

    def make_grid(self, x: List[torch.Tensor]):
        # Form the meshgrid of points
        x_mesh = torch.meshgrid(
            # *[x_elem.to(dtype=torch.float64, device=self.device) for x_elem in x],
            *x,
            indexing="ij",
        )
        x_mesh = torch.stack(x_mesh, dim=-1)
        return x_mesh

    def forward_grid(self, x_mesh: torch.Tensor):
        # Form the meshgrid of points
        out_shape = x_mesh.shape[:-1]
        x_mesh = x_mesh.reshape(-1, self.n_dim)
        for i in range(self.n_layers):
            x_mesh = self.activation(self.fc[i](x_mesh))
        return x_mesh.reshape(out_shape)

    def interpolate(self, x: List[torch.Tensor]):
        return self.forward(x)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the network
        Args:
            x: List of tensors of shapes (m1,), (m2,), ..., (m_ndim,) - points to evaluate at
        Returns:
            Tensor of shape (m1, m2, ..., m_ndim) - interpolated values
        """
        x_mesh = self.make_grid(x)
        return self.forward_grid(x_mesh)
