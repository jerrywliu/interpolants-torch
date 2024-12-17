import torch

"""
Compute the Chebyshev derivative matrix for a set of nodes
Args:
    nodes: Tensor of shape (N,) containing the Chebyshev nodes
    domain_length: Length of the domain (default: 2.0)
Returns:
    D: Tensor of shape (N, N) containing the derivative matrix
"""
def _compute_cheb_derivative_matrix(nodes, domain_length=2.0):
    N = len(nodes)
    D = torch.zeros((N, N), dtype=nodes.dtype, device=nodes.device)

    # Compute weights for endpoints
    c = torch.ones(N, dtype=nodes.dtype, device=nodes.device)
    c[0] = 2
    c[-1] = 2

    # Compute off-diagonal entries
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i,j] = c[i]/c[j] * (-1)**(i+j) / (nodes[i] - nodes[j])

    # Fill diagonal using negative sum trick
    D.diagonal().copy_(-torch.sum(D, dim=1))

    # Scale for domain transformation
    D = D * (2.0/domain_length)

    return D

"""
Compute the Fourier derivative matrix for a set of nodes
Args:
    nodes: Tensor of shape (N,) containing the Fourier nodes
    domain_length: Length of the domain (default: 2*torch.pi)
Returns:
    D: Tensor of shape (N, N) containing the derivative matrix
"""
def _compute_fourier_derivative_matrix(nodes, domain_length=2*torch.pi):
    N = len(nodes)
    D = torch.zeros((N, N), dtype=nodes.dtype, device=nodes.device)

    # Create index matrices
    i, j = torch.meshgrid(torch.arange(N, dtype=nodes.dtype, device=nodes.device), 
                        torch.arange(N, dtype=nodes.dtype, device=nodes.device), 
                        indexing='ij')

    # Compute off-diagonal elements using cotangent formula
    mask = i != j
    diff = (i[mask] - j[mask]) * (-1)**(i[mask] - j[mask])
    D[mask] = 0.5 * torch.tan(torch.pi * diff / N).reciprocal()

    # Diagonal elements are 0 for periodic functions
    D.diagonal().zero_()

    # Scale for domain transformation
    D = D * (2*torch.pi/domain_length)

    return D