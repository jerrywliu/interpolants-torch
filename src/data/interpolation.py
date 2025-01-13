import torch
from torch.utils.data import Dataset


# InterpolationDataset: for sanity-checking that the networks can interpolate the GT solution function
class InterpolationDataset(Dataset):
    def __init__(
        self,
        num_points,
        domain=(-1, 1),
        function=torch.sin,
        sampling="equispaced",
        noise_std=0,
    ):
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
        valid_methods = {"equispaced", "chebyshev", "random"}
        if self.sampling not in valid_methods:
            raise ValueError(f"Sampling method must be one of {valid_methods}")
        if self.sampling == "equispaced":
            x = torch.linspace(self.domain[0], self.domain[1], self.num_points)
        elif self.sampling == "chebyshev":
            j = torch.arange(self.num_points)
            x = torch.cos((2 * j + 1) * torch.pi / (2 * self.num_points))
            x = self._scale_to_domain(x, self.domain)
        else:  # random
            x = (
                torch.rand(self.num_points) * (self.domain[1] - self.domain[0])
                + self.domain[0]
            )
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
