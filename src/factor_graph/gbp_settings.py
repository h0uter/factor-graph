from typing import List

import torch

"""
    Defines classes for variable nodes, factor nodes and edges and factor graph.
"""


class GBPSettings:
    def __init__(
        self,
        damping: float = 0.0,
        beta: float = 0.1,
        num_undamped_iters: int = 5,
        min_linear_iters: int = 10,
        dropout: float = 0.0,
        reset_iters_since_relin: List[int] = [],
        type: torch.dtype = torch.float,
    ) -> None:
        # Parameters for damping the eta component of the message
        self.damping = damping
        self.num_undamped_iters = num_undamped_iters  # Number of undamped iterations after relinearisation before damping is set to damping

        self.dropout = dropout

        # Parameters for just in time factor relinearisation
        self.beta = beta  # Threshold absolute distance between linpoint and adjacent belief means for relinearisation.
        self.min_linear_iters = min_linear_iters  # Minimum number of linear iterations before a factor is allowed to realinearise.
        self.reset_iters_since_relin = reset_iters_since_relin

    def get_damping(self, iters_since_relin: int) -> float:
        if iters_since_relin > self.num_undamped_iters:
            return self.damping
        else:
            return 0.0
