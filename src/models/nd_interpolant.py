import torch
import torch.nn as nn

class SpectralInterpolationND(nn.Module):
    def __init__(self, Ns, bases, domains):
        """
        ND interpolation using spectral methods
        
        Args:
            Ns: List of number of points per direction
            bases: List of bases per direction, either 'fourier' or 'chebyshev'
            domains: List of tuples of (min, max) per direction
        """
        super().__init__()
        
        # Store domain information
        assert len(Ns) == len(bases) == len(domains)
        self.n_dim = len(Ns)
        
        self.Ns = Ns
        self.bases = bases
        self.domains = domains
        self.domain_lengths = [domain[1] - domain[0] for domain in domains]
        
        # Set up nodes and weight/frequencies for interpolation (as needed)
        self.nodes = [None for _ in range(self.n_dim)]
        self.nodes_standard = [None for _ in range(self.n_dim)]
        self._from_standard = [None for _ in range(self.n_dim)]
        self._to_standard = [None for _ in range(self.n_dim)]
        self.cheb_weights = [None for _ in range(self.n_dim)]
        self.k = [None for _ in range(self.n_dim)]
        
        for dim in range(self.n_dim):
            if self.bases[dim] == "chebyshev":
                i = torch.linspace(0, 1, self.Ns[dim])
                self.nodes_standard[dim] = torch.cos(torch.pi * i)
                # Compute barycentric weights for Chebyshev
                N = self.Ns[dim]
                weights = torch.ones(N)
                weights[0] *= 0.5
                weights[-1] *= 0.5
                weights[1::2] = -1
                self.cheb_weights[dim] = weights
                
                self.k[dim] = None
            else:
                self.nodes_standard[dim] = torch.linspace(0, 2*np.pi, self.Ns[dim]+1)[:-1]
                # Compute FFT frequencies
                self.k[dim] = torch.fft.fftfreq(self.Ns[dim]) * self.Ns[dim]
                self.cheb_weights[dim] = None
                
            # Set up domain mapping functions for this dimension
            domain = self.domains[dim]
            if self.bases[dim] == "chebyshev":
                self._to_standard[dim] = lambda x, d=dim: 2 * (x - self.domains[d][0]) / self.domain_lengths[d] - 1
                self._from_standard[dim] = lambda x, d=dim: self.domains[d][0] + (x + 1) * self.domain_lengths[d] / 2
            else:  # fourier
                self._to_standard[dim] = lambda x, d=dim: 2*np.pi * (x - self.domains[d][0]) / self.domain_lengths[d]
                self._from_standard[dim] = lambda x, d=dim: self.domains[d][0] + self.domain_lengths[d] * x / (2*np.pi)
            
            # Map standard nodes to physical domain
            self.nodes[dim] = self._from_standard[dim](self.nodes_standard[dim])
        
        # Set up diff matrices cache
        self._diff_matrices = [{} for _ in range(self.n_dim)]
        
        # Create mesh grid of nodes
        mesh_args = [self.nodes[d] for d in range(self.n_dim)]
        self.mesh = torch.meshgrid(*mesh_args, indexing='ij')
        
        # Learnable values at node points
        self.values = nn.Parameter(torch.zeros(self.Ns))
        
    def _compute_cheb_derivative_matrix(self, nodes, domain_length):
        """
        Compute the differentiation matrix for 1D Chebyshev points
        """
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

    def _compute_fourier_derivative_matrix(self, nodes, domain_length):
        """
        Compute the differentiation matrix for 1D equispaced Fourier
        """
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
        D = D * (2*np.pi/domain_length)

        return D
        
    def derivative_matrix(self, k):
        """
        Get mixed derivative matrix D^k where k is a tuple of derivative orders

        Args:
            k: List/tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Matrix operator for the mixed derivative
        """
        assert len(k) == self.n_dim, f"Expected {self.n_dim} derivative orders, got {len(k)}"

        # Get 1D matrices for each dimension
        matrices = []
        for dim in range(self.n_dim):
            if k[dim] == 0:
                # Identity matrix for this dimension
                matrices.append(torch.eye(self.Ns[dim], 
                                       dtype=self.nodes_standard[dim].dtype,
                                       device=self.nodes_standard[dim].device))
            else:
                # Compute/get cached derivative matrix
                if k[dim] not in self._diff_matrices[dim]:
                    if 1 not in self._diff_matrices[dim]:
                        if self.bases[dim] == "chebyshev":
                            D = self._compute_cheb_derivative_matrix(self.nodes_standard[dim], self.domain_lengths[dim])
                        else:  # fourier
                            D = self._compute_fourier_derivative_matrix(self.nodes_standard[dim], self.domain_lengths[dim])
                        self._diff_matrices[dim][1] = D

                    # Compose for higher derivatives
                    Dk = self._diff_matrices[dim][1]
                    for _ in range(k[dim]-1):
                        Dk = Dk @ self._diff_matrices[dim][1]
                    self._diff_matrices[dim][k[dim]] = Dk

                matrices.append(self._diff_matrices[dim][k[dim]])

        # Compute Kronecker product
        D = matrices[0]
        for dim in range(1, self.n_dim):
            D = torch.kron(D, matrices[dim])

        return D
    
    def _cheb_interpolate_1d(self, x_eval, values, nodes_std, to_std, weights, eps=1e-14):
        """Helper for 1D Chebyshev interpolation along last axis

        Args:
            x_eval: shape (B1, B) - points to evaluate at
            values: shape (B2, B, N) - function values at nodes
            nodes_std: shape (N,) - standard Chebyshev nodes
            to_std: function - maps from physical to standard domain
            weights: shape (N,) - barycentric weights

        Returns:
            shape (B1, B2, B) - interpolated values
        """
        x_eval_standard = to_std(x_eval)  # (B1, B)

        # Reshape inputs for broadcasting:
        # x_eval: (B1, 1, B, 1)
        # values: (1, B2, B, N)
        # nodes: (1, 1, 1, N)
        # weights: (1, 1, 1, N)
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        values_expanded = values.unsqueeze(0)  # (1, B2, B, N)
        nodes_expanded = nodes_std.reshape(1, 1, 1, -1)
        weights_expanded = weights.reshape(1, 1, 1, -1)

        # Compute distances - result is (B1, B2, B, N)
        d_x = x_eval_expanded - nodes_expanded

        small_diff = torch.abs(d_x) < eps
        small_diff_max = torch.max(small_diff, dim=-1, keepdim=True).values

        d_x = torch.where(
            small_diff_max,
            torch.zeros_like(d_x),
            1.0 / d_x
        )
        d_x[small_diff] = 1

        # Compute weighted sum along last axis
        f_eval_num = torch.sum(values_expanded * d_x * weights_expanded, dim=-1)  # (B1, B2, B)
        f_eval_denom = torch.sum(d_x * weights_expanded, dim=-1)  # (B1, B2, B)

        return f_eval_num / f_eval_denom

    def _fourier_interpolate_1d(self, x_eval, values, to_std, k):
        """Helper for 1D Fourier interpolation along last axis

        Args:
            x_eval: shape (B1, B) - points to evaluate at
            values: shape (B2, B, N) - function values at nodes
            to_std: function - maps from physical to standard domain
            k: shape (N,) - frequency modes

        Returns:
            shape (B1, B2, B) - interpolated values
        """
        N = values.shape[-1]
        x_eval_standard = to_std(x_eval)  # (B1, B)

        # Compute FFT along last axis
        coeffs = torch.fft.fft(values, dim=-1)  # (B2, B, N)

        # Reshape inputs for broadcasting:
        # x_eval: (B1, 1, B, 1)
        # coeffs: (1, B2, B, N)
        # k: (1, 1, 1, N)
        x_eval_expanded = x_eval_standard.unsqueeze(1).unsqueeze(-1)  # (B1, 1, B, 1)
        coeffs_expanded = coeffs.unsqueeze(0)  # (1, B2, B, N)
        k_expanded = k.reshape(1, 1, 1, -1)

        # Compute Fourier matrix - result is (B1, 1, B, N)
        x_matrix = x_eval_expanded * k_expanded
        fourier_matrix = torch.exp(1j * x_matrix)

        # Matrix multiply and sum along last axis - result is (B1, B2, B)
        result = torch.sum(fourier_matrix * coeffs_expanded, dim=-1)
        return torch.real(result) / N
    
    def interpolate(self, x_eval, values):
        """
        Interpolate values at arbitrary points x_eval

        Args:
            x_eval: Tensor of shape (..., n_dim) containing coordinates to evaluate at
            values: Tensor of shape (*batch, *grid_dims) containing values to interpolate

        Returns:
            Tensor of shape (..., *batch) containing interpolated values
        """
        # Ensure input has correct shape
        assert x_eval.shape[-1] == self.n_dim, f"Expected {self.n_dim} coordinates, got {x_eval.shape[-1]}"

        # Store original batch shapes
        x_eval_batch_shape = x_eval.shape[:-1]  # (...) from (..., n_dim)
        x_eval_batch_shape_prod = int(torch.prod(torch.tensor(x_eval_batch_shape)).item())
        values_batch_shape = values.shape[:-self.n_dim]  # (*batch) from (*batch, *grid_dims)
        values_batch_shape_prod = int(torch.prod(torch.tensor(values_batch_shape)).item())

        # Reshape x_eval to (B1, B) = (prod(x_eval_batch_shape), n_dim)
        x_eval_reshaped = x_eval.reshape(-1, self.n_dim)

        # Reshape values to (B2, ...) = (prod(values_batch_shape), grid_dims)
        result = values.clone().reshape(-1, *values.shape[-self.n_dim:])

        # Interpolate one dimension at a time
        for dim in range(self.n_dim):

            # Get current dimension's coordinates, with shape (prod(x_eval_batch_shape))
            coords = x_eval_reshaped[..., dim]
            if dim == 0:
                # If first dimension, reshape to (B1, B) = (prod(x_eval_batch_shape), 1)
                coords_reshaped = coords.unsqueeze(-1)
            else:
                # Else, reshape to (B1, B) = (1, prod(x_eval_batch_shape))
                coords_reshaped = coords.unsqueeze(0)

            # Move result's target dimension to the end of grid dimensions
            shape = result.shape
            grid_dims = shape[-self.n_dim+dim:] # This gets smaller every iter of the loop, so that the next dim to interpolate over is always grid_dims[0]
            grid_dims_minus_current = grid_dims[1:]
            batch_dims = shape[:-self.n_dim+dim]

            # Permute grid dimensions to put target dim last
            perm = list(range(len(shape)))
            grid_start = len(batch_dims)
            perm.append(perm.pop(grid_start))
            result = result.permute(perm)
            
            # If dim > 0, assume that the last batch dimension is the shared dimension, move it to second last
            if dim > 0:
                perm = list(range(len(result.shape)))
                x_eval_dim = len(batch_dims)-1
                perm.insert(-1, perm.pop(x_eval_dim)) # Move last dim to second-to-last position
                result = result.permute(perm)

            if dim == 0:
                # If first dimension, reshape to (B2, B, N) = (prod(batch_dims + grid_dims[1:]), 1, grid_dims[0])
                result = result.reshape(-1, 1, grid_dims[0])
            else:
                # Else, result is already extended by B1 = prod(x_eval_batch_shape). Reshape to (B2, B, N) = (prod(batch_dims + grid_dims[:] - grid_dims[dim]), prod(x_eval_batch_shape), grid_dims[dim])
                result = result.reshape(-1, x_eval_batch_shape_prod, grid_dims[0])
    
            # Apply 1D interpolation
            if self.bases[dim] == "chebyshev":
                result = self._cheb_interpolate_1d(
                    coords_reshaped,
                    result,
                    self.nodes_standard[dim],
                    self._to_standard[dim],
                    self.cheb_weights[dim]
                )
            else:  # fourier
                result = self._fourier_interpolate_1d(
                    coords_reshaped,
                    result,
                    self._to_standard[dim],
                    self.k[dim]
                )

            # Result is now (B1, B2, B)
            # = (prod(x_eval_batch_shape), prod(batch_dims + grid_dims[1:]), 1) if dim == 0
            # = (1, prod(batch_dims + grid_dims[:] - grid_dims[dim]), prod(x_eval_batch_shape)) if dim > 0

            # Reshape result to restore batch dimensions
            # Then permute result to restore batch dimensions: (batch_dims, x_eval_batch_shape_prod, grid_dims_minus_current)
            if dim == 0:
                # (prod(x_eval_batch_shape), batch_dims, grid_dims[1:])
                result = result.reshape(-1, *batch_dims, *grid_dims_minus_current)
                x_eval_dim = 0
                target_pos = len(batch_dims)
            else:
                # (batch_dims, grid_dims-grid_dims[dim], prod(x_eval_batch_shape))
                result = result.reshape(*batch_dims[:-1], *grid_dims_minus_current, x_eval_batch_shape_prod)
                x_eval_dim = -1
                target_pos = len(batch_dims) - 1
                
            # Move x_eval batch dim after other batch dims
            perm = list(range(len(result.shape)))
            perm.insert(target_pos, perm.pop(x_eval_dim))
            result = result.permute(perm)

        return result
    
    def forward(self, x_eval):
        return self.interpolate(x_eval, self.values)
    
    def _derivative_interpolant(self, k):
        """
        Compute mixed derivative of interpolant

        Args:
            k: Tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Tensor containing derivative values at grid points
        """
        # Handle the case where k is a single integer (apply to first dimension)
        if isinstance(k, int):
            k = (k,) + (0,)*(self.n_dim-1)

        assert len(k) == self.n_dim, f"Expected {self.n_dim} derivative orders, got {len(k)}"

        # If all derivatives are zero, return values
        if all(ki == 0 for ki in k):
            return self.values

        # Get mixed derivative matrix
        Dk = self.derivative_matrix(k)

        # Compute derivative values at nodes (differentiable w.r.t self.values)
        # Reshape values to a flat vector for matrix multiplication
        values_flat = self.values.reshape(-1)
        dk_nodes = Dk @ values_flat

        # Reshape back to grid shape
        dk_nodes = dk_nodes.reshape(*self.values.shape)

        return dk_nodes

    def derivative(self, x_eval, k):
        """
        Compute mixed derivative of interpolant at arbitrary evaluation points

        Args:
            x_eval: Tensor of shape (..., n_dim) containing coordinates to evaluate at
            k: Tuple of length n_dim specifying derivative order in each dimension
               e.g., (2,0,1) means second derivative in x, none in y, first in z

        Returns:
            Tensor of shape (...) containing derivative values at x_eval points
        """
        # Compute derivative values at grid points
        dk_nodes = self._derivative_interpolant(k)

        # Interpolate to evaluation points
        return self.interpolate(x_eval, dk_nodes)

# Test the ND interpolant
if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    # Define a smooth 3D test function
    def test_function_3d(x, y, z):
        return torch.sin(2 * torch.pi * x) * torch.cos(torch.pi * y) * torch.exp(-z**2)

    interp = SpectralInterpolationND(
        Ns=[32, 33, 35],  # Different numbers to catch any indexing issues
        bases=['fourier', 'chebyshev', 'chebyshev'],
        domains=[(0, 1), (-1, 1), (-1, 1)]
    )

    # Set values at grid points
    x_grid, y_grid, z_grid = interp.mesh
    interp.values.data = test_function_3d(x_grid, y_grid, z_grid)
    print(f"self.values: {interp.values.shape}")

    # Test random points
    n_test = 10000
    test_points = torch.rand(n_test, 3)
    test_points[:, 0] = 2 * test_points[:, 0] - 1  # x in [0,1]
    test_points[:, 1] = 2 * test_points[:, 1] - 1  # y in [-1,1]
    test_points[:, 2] = 2 * test_points[:, 2] - 1  # z in [-1,1]

    interpolated_random = interp(test_points)
    exact_random = test_function_3d(test_points[:, 0], test_points[:, 1], test_points[:, 2])
    error_random = torch.abs(interpolated_random - exact_random)

    print("\nRandom points test:")
    print(f"Maximum absolute error: {error_random.max().item():.2e}")
    print(f"Average absolute error: {error_random.mean().item():.2e}")
    print(f"RMS error: {torch.sqrt((error_random**2).mean()).item():.2e}")