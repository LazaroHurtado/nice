import torch
import torch.nn as nn

from typing import Union

class AdditiveCoupling(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: Union[int, str] = 0):
        super().__init__()
        
        self.dim = dim
        self.device = device
        
        coupling_fn = self.coupling_layer(dim, hidden_dim)
        for _ in range(hidden_layers-1):
            coupling_fn += self.coupling_layer(hidden_dim, hidden_dim)
        coupling_fn += [nn.Linear(hidden_dim, dim, device=self.device)]
        
        self.coupling_fn = nn.Sequential(*coupling_fn)

        self._initialize_weights()
    
    def coupling_layer(self, in_dim: int, out_dim: int):
        return [
            nn.Linear(in_dim, out_dim, device=self.device),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            ]

    def _initialize_weights(self):
        for fn in self.coupling_fn:
            if isinstance(fn, nn.Linear):
                nn.init.uniform_(fn.weight, a=-0.01, b=0.01)
                nn.init.zeros_(fn.bias)

    def forward(self, x: torch.Tensor):
        z1 = x[:, :self.dim]
        z2 = x[:, self.dim:] + self.coupling_fn(x[:, :self.dim])

        z = torch.cat((z1, z2), dim=-1)
        return z
    
    def inverse(self, z: torch.Tensor):
        x1 = z[:, :self.dim]
        x2 = z[:, self.dim:] - self.coupling_fn(z[:, :self.dim])

        x = torch.cat((x1, x2), dim=-1)
        return x
