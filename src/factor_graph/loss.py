from typing import Union

import torch

"""
    Defines squared loss functions that correspond to Gaussians.
    Robust losses are implemented by scaling the Gaussian covariance.
"""


class SquaredLoss:
    def __init__(self, dofs: int, diag_cov: Union[float, torch.Tensor]) -> None:
        """
        dofs: dofs of the measurement
        cov: diagonal elements of covariance matrix
        """
        assert diag_cov.shape == torch.Size([dofs])
        mat = torch.zeros(dofs, dofs, dtype=diag_cov.dtype)
        mat[range(dofs), range(dofs)] = diag_cov
        self.cov = mat
        self.effective_cov = mat.clone()

    def get_effective_cov(self, residual: torch.Tensor) -> None:
        """Returns the covariance of the Gaussian (squared loss) that matches the loss at the error value."""
        self.effective_cov = self.cov.clone()

    def robust(self) -> bool:
        return not torch.equal(self.cov, self.effective_cov)


class HuberLoss(SquaredLoss):
    def __init__(
        self, dofs: int, diag_cov: Union[float, torch.Tensor], stds_transition: float
    ) -> None:
        """
        stds_transition: num standard deviations from minimum at which quadratic loss transitions to linear
        """
        SquaredLoss.__init__(self, dofs, diag_cov)
        self.stds_transition = stds_transition

    def get_effective_cov(self, residual: torch.Tensor) -> None:
        mahalanobis_dist = torch.sqrt(residual @ torch.inverse(self.cov) @ residual)
        if mahalanobis_dist > self.stds_transition:
            self.effective_cov = (
                self.cov
                * mahalanobis_dist**2
                / (
                    2 * self.stds_transition * mahalanobis_dist
                    - self.stds_transition**2
                )
            )
        else:
            self.effective_cov = self.cov.clone()


class TukeyLoss(SquaredLoss):
    def __init__(
        self, dofs: int, diag_cov: Union[float, torch.Tensor], stds_transition: float
    ) -> None:
        """
        stds_transition: num standard deviations from minimum at which quadratic loss transitions to constant
        """
        SquaredLoss.__init__(self, dofs, diag_cov)
        self.stds_transition = stds_transition

    def get_effective_cov(self, residual: torch.Tensor) -> None:
        mahalanobis_dist = torch.sqrt(residual @ torch.inverse(self.cov) @ residual)
        if mahalanobis_dist > self.stds_transition:
            self.effective_cov = (
                self.cov * mahalanobis_dist**2 / self.stds_transition**2
            )
        else:
            self.effective_cov = self.cov.clone()
