import matplotlib.pyplot as plt
import numpy as np
import torch

from factor_graph.gbp import FactorGraph, GBPSettings
from factor_graph.utility_functions import MeasModel, SquaredLoss

"""Create Custom factors"""


def height_meas_fn(x: torch.Tensor, gamma: torch.Tensor):
    J = torch.tensor([[1 - gamma, gamma]])
    return J @ x


def height_jac_fn(x: torch.Tensor, gamma: torch.Tensor):
    return torch.tensor([[1 - gamma, gamma]])


class HeightMeasurementModel(MeasModel):
    def __init__(self, loss: SquaredLoss, gamma: torch.Tensor) -> None:
        MeasModel.__init__(self, height_meas_fn, height_jac_fn, loss, gamma)
        self.linear = True


def smooth_meas_fn(x: torch.Tensor):
    return torch.tensor([x[1] - x[0]])


def smooth_jac_fn(x: torch.Tensor):
    return torch.tensor([[-1.0, 1.0]])


class SmoothingModel(MeasModel):
    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, smooth_meas_fn, smooth_jac_fn, loss)
        self.linear = True


"""Set parameters"""
n_varnodes = 20
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

"""Create measurements"""

# Plot measurements
meas_x = torch.rand(n_measurements) * x_range
meas_y = torch.sin(meas_x) + torch.normal(
    0, torch.full([n_measurements], data_std.item())
)
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()


"""Create factor graph"""
fg = FactorGraph(gbp_settings)

xs = torch.linspace(0, x_range, n_varnodes).float().unsqueeze(0).T
for i in range(n_varnodes):
    fg.add_var_node(1, torch.tensor([0.0]), prior_cov)

for i in range(n_varnodes - 1):
    fg.add_factor(
        [i, i + 1], torch.tensor([0.0]), SmoothingModel(SquaredLoss(1, smooth_cov))
    )

for i in range(n_measurements):
    ix2 = np.argmax(xs > meas_x[i])
    ix1 = ix2 - 1
    gamma = (meas_x[i] - xs[ix1]) / (xs[ix2] - xs[ix1])
    fg.add_factor(
        [ix1, ix2], meas_y[i], HeightMeasurementModel(SquaredLoss(1, data_cov), gamma)
    )

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
