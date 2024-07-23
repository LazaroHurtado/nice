import torch
import torch.nn as nn

from coupling import AdditiveCoupling
from distributions import StandardLogisitcDistribution
from orderings import Parity, FlipOnMiddle

from typing import Union

class Quantizer:
    def quantize(x: torch.Tensor):
        x = torch.floor(x * 256.0)
        x = x.clamp(0.0, 255.0) / 255.0
        return x

    def dequantize(x: torch.Tensor):
        x = x * 255.0
        x = (x + torch.rand_like(x)) / 256.0
        return x

class Nice(nn.Module):
    
    def __init__(self,
                 dim: int,
                 hidden_dim: int = 1000,
                 hidden_layers: int = 5,
                 coupling_layers: int = 4,
                 device: Union[int, str] = 0):
        super().__init__()
        half_dim = dim//2

        self.device = device
        self.prior = StandardLogisitcDistribution(dim, device)
        
        self.partition = Parity(dim)
        self.reorder = FlipOnMiddle(dim)

        self.additive_couplings = nn.ModuleList([
            AdditiveCoupling(half_dim, hidden_dim, hidden_layers, device=device)
            for _ in range(coupling_layers)
        ])
        self.scaling = nn.Parameter(torch.zeros(dim, device=self.device))

    def inverse(self, z: torch.Tensor):
        z = z / self.scaling.exp()
        for layer in self.additive_couplings[::-1]:
            z = self.reorder.original_order(z)
            z = layer.inverse(z)

        x = self.partition.original_order(z)
        return x

    def forward(self, x: torch.Tensor):
        x = self.partition.reorder(x)

        for layer in self.additive_couplings:
            x = layer.forward(x)
            x = self.reorder.reorder(x)
        z = x * self.scaling.exp()

        return z
    
    def sample(self, n: int):
        z = self.prior.sample(n)
        x = self.inverse(z)

        return Quantizer.quantize(x)

    def loss(self, z: torch.Tensor):
        log_prob = self.prior.log_prob(z)
        
        # Since NICE is volume preserving the log of coupling fn determinate is 0
        # so only the scaling matters, which is a diagonal matrix.
        log_det_jacobian = self.scaling.sum()
        mll = -(log_prob + log_det_jacobian).mean()

        return mll
        
