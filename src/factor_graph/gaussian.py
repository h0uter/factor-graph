from typing import List, Optional

import torch


class Gaussian:
    def __init__(
        self,
        dim: int,
        eta: Optional[torch.Tensor] = None,
        lam: Optional[torch.Tensor] = None,
        type: torch.dtype = torch.float,
    ):
        self.dim = dim

        if eta is not None and eta.shape == torch.Size([dim]):
            self.eta = eta.type(type)
        else:
            self.eta = torch.zeros(dim, dtype=type)

        if lam is not None and lam.shape == torch.Size([dim, dim]):
            self.lam = lam.type(type)
        else:
            self.lam = torch.zeros([dim, dim], dtype=type)

    def mean(self) -> torch.Tensor:
        return torch.matmul(torch.inverse(self.lam), self.eta)

    def cov(self) -> torch.Tensor:
        return torch.inverse(self.lam)

    def mean_and_cov(self) -> List[torch.Tensor]:
        cov = self.cov()
        mean = torch.matmul(cov, self.eta)
        return [mean, cov]

    def set_with_cov_form(self, mean: torch.Tensor, cov: torch.Tensor) -> None:
        self.lam = torch.inverse(cov)
        self.eta = self.lam @ mean
