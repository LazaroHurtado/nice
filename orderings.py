import torch

from abc import abstractmethod

class Orderings():
    @abstractmethod
    def reorder(self, x: torch.Tensor):
        pass

    @abstractmethod
    def original_order(self, x: torch.Tensor):
        pass

class Parity(Orderings):
    def __init__(self, dim: int):
        tmp = torch.arange(dim)
        self.parity = torch.empty_like(tmp)
        self.parity[:dim//2] = tmp[::2]
        self.parity[dim//2:] = tmp[1::2]

        self.original = self.parity.argsort()

    def reorder(self, x: torch.Tensor):
        return x[:, self.parity]

    def original_order(self, x: torch.Tensor):
        return x[:, self.original]
    
class FlipOnMiddle(Orderings):
    def __init__(self, dim: int):
        self.order = torch.arange(dim).flip(-1)
        self.original = self.order.argsort()

    def reorder(self, x: torch.Tensor):
        return x[:, self.order]

    def original_order(self, x: torch.Tensor):
        return x[:, self.original]