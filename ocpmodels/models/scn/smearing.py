import torch


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=-5.0, stop=5.0, num_gaussians=50, basis_width_scalar=1.0):
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
