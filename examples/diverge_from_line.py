import matplotlib.pyplot as plt
import torch

from factor_graph.factor_graph import FactorGraph
from factor_graph.gbp_settings import GBPSettings
from factor_graph.loss import SquaredLoss
from factor_graph.meas_model import MeasModel

"""Create Custom factors (measurement models))"""


def goal_meas_fn(x: torch.Tensor):
    distance = x[0] - x[1]
    print(f"distance between goals: {distance}")
    R_D = 10
    if distance > R_D:
        return torch.tensor([[0.0]])
    else:
        return torch.tensor([[1 - distance / R_D]])


def goal_jac_fn(x: torch.Tensor):
    distance = x[0] - x[1]
    R_D = 10
    if distance > R_D:
        return torch.tensor([[0.0, 0.0]])
    else:
        return torch.tensor([[-1 / R_D, 1 / R_D]])


class DivergenceModel(MeasModel):
    """Create model that prefers divergence over smoothness."""

    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, goal_meas_fn, goal_jac_fn, loss)
        self.linear = True


"""Set parameters"""
n_varnodes = 4
x_range = 10
# n_measurements = 15

gbp_settings = GBPSettings(
    damping=0.1,
    beta=0.01,
    num_undamped_iters=1,
    min_linear_iters=1,
    dropout=0.0,
)

"""Gaussian noise measurement model parameters:"""
# set this one super high, cause we dont care about the prior
prior_cov = torch.tensor([10.0])

# set this one super low, because we want to convergence to our measurements
smooth_cov = torch.tensor([0.0001])


"""Create factor graph"""
fg = FactorGraph(gbp_settings)

xs = torch.linspace(0, x_range, n_varnodes).float().unsqueeze(0).T

# initialize variable nodes. AKA With what resolution are we going to estimate the function?
for i in range(n_varnodes):
    prior_mean = torch.tensor([50.0])
    fg.add_var_node(1, prior_mean, prior_cov)

# add smoothness factors between adjacent nodes
for i in range(n_varnodes - 1):
    fg.add_factor(
        [i, i + 1], torch.tensor([0.0]), DivergenceModel(SquaredLoss(1, smooth_cov))
    )


fg.print(brief=True)

"""Plot initial beliefs"""

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
