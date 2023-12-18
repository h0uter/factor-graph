import matplotlib.pyplot as plt
import numpy as np
import torch

from factor_graph.factor_graph import FactorGraph
from factor_graph.gbp_settings import GBPSettings
from factor_graph.loss import SquaredLoss
from factor_graph.meas_model import MeasModel

"""Create Custom factors (measurement models))"""


def goal_meas_fn(x: torch.Tensor):
    """height on a position.
    gamma = height"""
    distance = float(np.linalg.norm(x[0] - x[1]))
    R_D = 2
    if distance > R_D:
        return torch.tensor([[0.0]])
    else:
        return torch.tensor([[1 - distance / R_D]])


def goal_jac_fn(x: torch.Tensor):
    distance = float(np.linalg.norm(x[0] - x[1]))
    R_D = 2
    if distance > R_D:
        return torch.tensor([[0.0, 0.0]])
    else:
        return torch.tensor(
            [
                [
                    (x[0] - x[1]) / R_D * np.linalg.norm(x[0] - x[1]),
                    -(x[0] - x[1]) / R_D * np.linalg.norm(x[0] - x[1]),
                ]
            ]
        )


class GoalMeasurementModel(MeasModel):
    """Create a measurement model for the height."""

    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, goal_meas_fn, goal_jac_fn, loss)
        self.linear = True


"""Set parameters"""
n_varnodes = 2
n_measurements = 1

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

"""Create measurements"""

# Plot measurements
meas_x = torch.tensor([1.0])
meas_y = torch.tensor([2.0])
print(f"meas_x: {meas_x} meas_y: {meas_y}")
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()


"""Create factor graph"""
fg = FactorGraph(gbp_settings)

# xs = torch.linspace(0, x_range, n_varnodes).float().unsqueeze(0).T
xs = torch.tensor([[1.1], [0.9]])
print(f"xs: {xs}")

INITIAL_GOAL = 2.0

# initialize variable nodes. AKA With what resolution are we going to estimate the function?
for i in range(n_varnodes):
    fg.add_var_node(1, torch.tensor([INITIAL_GOAL]), prior_cov)

# do goal measurements.
for i in range(n_measurements):
    # add factor between two variable nodes.
    ix2 = 1
    ix1 = 0
    fg.add_factor([ix1, ix2], meas_y[i], GoalMeasurementModel(SquaredLoss(1, data_cov)))

fg.print(brief=True)

"""Plot beliefs and measurements"""

# Beliefs are initialized to zero
covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt="o", color="C0", label="Beliefs")
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()


"""Solve with GBP"""
fg.gbp_solve(n_iters=50)

# Plot beliefs and measurements
covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt="o", color="C0", label="Beliefs")
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()
