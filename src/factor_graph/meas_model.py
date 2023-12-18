from typing import Callable

import torch

from factor_graph.loss import SquaredLoss


class MeasModel:
    def __init__(
        self, meas_fn: Callable, jac_fn: Callable, loss: SquaredLoss, *args
    ) -> None:
        self._meas_fn = meas_fn
        self._jac_fn = jac_fn
        self.loss = loss
        self.args = args
        self.linear = True

    def jac_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._jac_fn(x, *self.args)

    def meas_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._meas_fn(x, *self.args)
