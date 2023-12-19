from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sympy import Matrix, diff, lambdify, sqrt, symbols

from factor_graph.factor_graph import FactorGraph
from factor_graph.gbp_settings import GBPSettings
from factor_graph.loss import SquaredLoss
from factor_graph.meas_model import MeasModel

"""Create Custom factors (measurement models))"""

# TODO: make everything 2 dimensional

R_D = 100


def get_jacobian() -> (
    tuple[
        Callable[[float, float, float, float, float], float],
        Callable[[float, float, float, float, float], float],
        Callable[[float, float, float, float, float], float],
        Callable[[float, float, float, float, float], float],
    ]
):
    # Define vector variables
    x1, x2, y1, y2 = symbols("x1 x2 y1 y2")
    z1, z2 = symbols("z1 z2")
    CONSTANT = symbols("CONSTANT")

    # Define the vectors as matrices
    z1 = Matrix([x1, y1])
    z2 = Matrix([x2, y2])

    # Define the function
    f = Matrix([1 - sqrt((z1 - z2).dot(z1 - z2)) / CONSTANT])

    # Use the jacobian function to compute the Jacobian matrix
    jacobian_matrix = f.jacobian(Matrix([[z1], [z2]]))
    print(f"{jacobian_matrix=}")
    j1_func = lambdify((x1, y1, x2, y2, CONSTANT), jacobian_matrix[0], "numpy")
    j2_func = lambdify((x1, y1, x2, y2, CONSTANT), jacobian_matrix[1], "numpy")
    j3_func = lambdify((x1, y1, x2, y2, CONSTANT), jacobian_matrix[2], "numpy")
    j4_func = lambdify((x1, y1, x2, y2, CONSTANT), jacobian_matrix[3], "numpy")

    return j1_func, j2_func, j3_func, j4_func


def goal_meas_fn(x: torch.Tensor):
    distance = float(np.linalg.norm(x[0:2] - x[2:]))
    print(f"{x=}")
    print(f"distance between goals: {distance}")
    if distance > R_D:
        return torch.tensor([[0.0]])
    else:
        return torch.tensor([[1 - distance / R_D]])


def goal_jac_fn(x: torch.Tensor):
    distance = float(np.linalg.norm(x[0:2] - x[2:]))
    if distance > R_D:
        return torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    else:
        j1_func, j2_func, j3_func, j4_func = get_jacobian()
        x1, y1, x2, y2 = x[0], x[1], x[2], x[3]
        j1 = float(j1_func(x1, y1, x2, y2, R_D))
        j2 = float(j2_func(x1, y1, x2, y2, R_D))
        j3 = float(j3_func(x1, y1, x2, y2, R_D))
        j4 = float(j4_func(x1, y1, x2, y2, R_D))
        print(f"{j1=}, {j2=}, {j3=}, {j4=}")

        return torch.tensor([[j1, j2, j3, j4]])


class DivergenceModel(MeasModel):
    """Create model that prefers divergence over smoothness."""

    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, goal_meas_fn, goal_jac_fn, loss)
        self.linear = True


"""Gaussian noise measurement model parameters:"""
# set this one super high, cause we dont care about the prior
prior_cov = torch.tensor([100.0, 100.0])
# prior_cov = torch.tensor([10.0, 10.0, 10.0, 10.0])

# set this one super low, because we want to convergence to our measurements
# smooth_cov = torch.tensor([0.0001, 0.0001, 0.0001, 0.0001])
# smooth_cov = torch.tensor([0.0001, 0.0001, 0.0001, 0.0001])
# smooth_cov = torch.tensor([0.0001, 0.0001, 0.0001])
# smooth_cov = torch.tensor([0.0001, 0.0001])
smooth_cov = torch.tensor([0.000001])


"""Set parameters"""
n_varnodes = 3
x_range = 10

gbp_settings = GBPSettings(
    damping=0.1,
    beta=0.01,
    num_undamped_iters=1,
    min_linear_iters=1,
    dropout=0.0,
)

"""Create factor graph"""
fg = FactorGraph(gbp_settings)

xs = torch.linspace(0, x_range, n_varnodes).float().unsqueeze(0).T

# initialize variable nodes. AKA With what resolution are we going to estimate the function?
# for i in range(n_varnodes):
prior_mean = torch.tensor([4.0, 51.0])
fg.add_var_node(2, prior_mean, prior_cov)
prior_mean = torch.tensor([4.0, 50.0])
fg.add_var_node(2, prior_mean, prior_cov)
prior_mean = torch.tensor([6.0, 49.0])
fg.add_var_node(2, prior_mean, prior_cov)

factors_to_add = [[0, 1], [1, 2], [2, 0]]
for connection in factors_to_add:
    fg.add_factor(
        connection,
        torch.tensor([0.1]),
        DivergenceModel(SquaredLoss(1, smooth_cov)),
    )


# add smoothness factors between adjacent nodes
# for i in range(n_varnodes - 1):
#     fg.add_factor(
#         [i, i + 1],
#         torch.tensor([0.1]),
#         DivergenceModel(SquaredLoss(1, smooth_cov)),
#     )


fg.print(brief=True)

"""Plot initial beliefs"""

# Beliefs are initialized to zero
covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
print(f"{fg.belief_means()=} and {xs=}")
# plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt="o", color="C0", label="Beliefs")
list_of_means = fg.belief_means()

xss_pre = list_of_means[::2]  # Select elements at even indices
yss_pre = list_of_means[1::2]  # Select elements at odd indices
plt.errorbar(xss_pre, yss_pre, fmt="o", color="red", label="Beliefs Pre optimization")
plt.legend()
plt.show()


"""Solve with GBP"""
fg.gbp_solve(n_iters=50)

# Plot beliefs and measurements
# covs = torch.sqrt(torch.cat(fg.belief_covs()).flatten())
list_of_means = fg.belief_means()

xss = list_of_means[::2]  # Select elements at even indices
yss = list_of_means[1::2]  # Select elements at odd indices
plt.errorbar(xss, yss, fmt="o", color="C0", label="Optimized Beliefs")
plt.errorbar(xss_pre, yss_pre, fmt="o", color="red", label="Beliefs Pre optimization")

# plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt="o", color="C0", label="Beliefs")
plt.legend()
plt.show()
