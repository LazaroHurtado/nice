from typing import Union
import torch

from torch.distributions import TransformedDistribution, Uniform, SigmoidTransform, AffineTransform

class StandardLogisitcDistribution:
    def __init__(self, dim: int = 784, device: Union[int, str] = 0):
        self.dist = TransformedDistribution(
            Uniform(torch.zeros(dim, device=device),
                    torch.ones(dim, device=device)),
            [SigmoidTransform().inv,
             AffineTransform(torch.zeros(dim, device=device),
                             torch.ones(dim, device=device))])
        
    def log_prob(self, z: torch.Tensor):
        return self.dist.log_prob(z).sum(dim=-1)
    
    def sample(self, n: int = 1):
        return self.dist.sample((n,))