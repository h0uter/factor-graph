import matplotlib.pyplot as plt
import torch

from factor_graph.factor_graph import FactorGraph
from factor_graph.gbp_settings import GBPSettings
from factor_graph.loss import SquaredLoss
from factor_graph.meas_model import MeasModel

"""Create Custom factors (measurement models))"""


def smooth_meas_fn(x: torch.Tensor):
    return torch.tensor([x[1] - x[0]])


def smooth_jac_fn(x: torch.Tensor):
    return torch.tensor([[-1.0, 1.0]])


class SmoothingModel(MeasModel):
    """Create model that preferst smoothness over jumps."""

    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, smooth_meas_fn, smooth_jac_fn, loss)
        self.linear = True


"""Set parameters"""
n_varnodes = 10
x_range = 10
n_measurements = 15

gbp_settings = GBPSettings(
    damping=0.1,
    beta=0.01,
    num_undamped_iters=1,
    min_linear_iters=1,
    dropout=0.0,
)

"""Gaussian noise measurement model parameters:"""
prior_cov = torch.tensor([10.0])
data_cov = torch.tensor([0.05])
smooth_cov = torch.tensor([0.1])
data_std = torch.sqrt(data_cov)


"""Create factor graph"""
fg = FactorGraph(gbp_settings)

xs = torch.linspace(0, x_range, n_varnodes).float().unsqueeze(0).T

print(f"xs: {xs}")
# initialize variable nodes. AKA With what resolution are we going to estimate the function?
for i in range(n_varnodes):
    prior_mean = torch.rand(1) * 100
    fg.add_var_node(1, prior_mean, prior_cov)

# add smoothness factors between adjacent nodes
for i in range(n_varnodes - 1):
    fg.add_factor(
        [i, i + 1], torch.tensor([0.0]), SmoothingModel(SquaredLoss(1, smooth_cov))
    )


fg.print(brief=True)

"""Plot beliefs and measurements"""

# Beliefs are initialized to zero
covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt="o", color="C0", label="Beliefs")
plt.legend()
plt.show()


"""Solve with GBP"""
fg.gbp_solve(n_iters=50)

# Plot beliefs and measurements
covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt="o", color="C0", label="Beliefs")
plt.legend()
plt.show()
