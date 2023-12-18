from typing import List

import torch

from factor_graph.utility_functions import Gaussian, MeasModel
from factor_graph.variable_node import VariableNode


class Factor:
    def __init__(
        self,
        id: int,
        adj_var_nodes: List[VariableNode],
        measurement: torch.Tensor,
        meas_model: MeasModel,
        type: torch.dtype = torch.float,
        properties: dict = {},
    ) -> None:
        self.factorID = id
        self.properties = properties

        self.adj_var_nodes = adj_var_nodes
        self.dofs = sum([var.dofs for var in adj_var_nodes])
        self.adj_vIDs = [var.variableID for var in adj_var_nodes]
        self.messages = [Gaussian(var.dofs) for var in adj_var_nodes]

        self.factor = Gaussian(self.dofs)
        self.linpoint = torch.zeros(self.dofs, dtype=type)

        self.measurement = measurement
        self.meas_model = meas_model

        # For smarter GBP implementations
        self.iters_since_relin = 0

        self.compute_factor()

    def get_adj_means(self) -> torch.Tensor:
        adj_belief_means = [var.belief.mean() for var in self.adj_var_nodes]
        return torch.cat(adj_belief_means)

    def get_residual(self, eval_point: torch.Tensor = None) -> torch.Tensor:
        """Compute the residual vector."""
        if eval_point is None:
            eval_point = self.get_adj_means()
        return self.meas_model.meas_fn(eval_point) - self.measurement

    def get_energy(self, eval_point: torch.Tensor = None) -> float:
        """Computes the squared error using the appropriate loss function."""
        residual = self.get_residual(eval_point)
        # print("adj_belifes", self.get_adj_means())
        # print("pred and meas", self.meas_model.meas_fn(self.get_adj_means()), self.measurement)
        # print("residual", self.get_residual(), self.meas_model.loss.effective_cov)
        return (
            0.5
            * residual
            @ torch.inverse(self.meas_model.loss.effective_cov)
            @ residual
        )

    def robust(self) -> bool:
        return self.meas_model.loss.robust()

    def compute_factor(self) -> None:
        """
        Compute the factor at current adjacente beliefs using robust.
        If measurement model is linear then factor will always be the same regardless of linearisation point.
        """
        self.linpoint = self.get_adj_means()
        J = self.meas_model.jac_fn(self.linpoint)
        pred_measurement = self.meas_model.meas_fn(self.linpoint)
        self.meas_model.loss.get_effective_cov(pred_measurement - self.measurement)
        effective_lam = torch.inverse(self.meas_model.loss.effective_cov)
        self.factor.lam = J.T @ effective_lam @ J
        self.factor.eta = (
            (J.T @ effective_lam)
            @ (J @ self.linpoint + self.measurement - pred_measurement)
        ).flatten()
        self.iters_since_relin = 0

    def robustify_loss(self) -> None:
        """
        Rescale the variance of the noise in the Gaussian measurement model if necessary and update the factor
        correspondingly.
        """
        old_effective_cov = self.meas_model.loss.effective_cov[0, 0]
        self.meas_model.loss.get_effective_cov(self.get_residual())
        self.factor.eta *= old_effective_cov / self.meas_model.loss.effective_cov[0, 0]
        self.factor.lam *= old_effective_cov / self.meas_model.loss.effective_cov[0, 0]

    def compute_messages(self, damping: float = 0.0) -> None:
        """Compute all outgoing messages from the factor."""
        messages_eta, messages_lam = [], []

        start_dim = 0
        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = (
                self.factor.eta.clone().double(),
                self.factor.lam.clone().double(),
            )

            # Take product of factor with incoming messages
            start = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    eta_factor[start : start + var_dofs] += (
                        self.adj_var_nodes[var].belief.eta - self.messages[var].eta
                    )
                    lam_factor[start : start + var_dofs, start : start + var_dofs] += (
                        self.adj_var_nodes[var].belief.lam - self.messages[var].lam
                    )
                start += self.adj_var_nodes[var].dofs

            # Divide up parameters of distribution
            mess_dofs = self.adj_var_nodes[v].dofs
            eo = eta_factor[start_dim : start_dim + mess_dofs]
            eno = torch.cat(
                (eta_factor[:start_dim], eta_factor[start_dim + mess_dofs :])
            )

            loo = lam_factor[
                start_dim : start_dim + mess_dofs, start_dim : start_dim + mess_dofs
            ]
            lono = torch.cat(
                (
                    lam_factor[start_dim : start_dim + mess_dofs, :start_dim],
                    lam_factor[
                        start_dim : start_dim + mess_dofs, start_dim + mess_dofs :
                    ],
                ),
                dim=1,
            )
            lnoo = torch.cat(
                (
                    lam_factor[:start_dim, start_dim : start_dim + mess_dofs],
                    lam_factor[
                        start_dim + mess_dofs :, start_dim : start_dim + mess_dofs
                    ],
                ),
                dim=0,
            )
            lnono = torch.cat(
                (
                    torch.cat(
                        (
                            lam_factor[:start_dim, :start_dim],
                            lam_factor[:start_dim, start_dim + mess_dofs :],
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            lam_factor[start_dim + mess_dofs :, :start_dim],
                            lam_factor[
                                start_dim + mess_dofs :, start_dim + mess_dofs :
                            ],
                        ),
                        dim=1,
                    ),
                ),
                dim=0,
            )

            new_message_lam = loo - lono @ torch.inverse(lnono) @ lnoo
            new_message_eta = eo - lono @ torch.inverse(lnono) @ eno
            messages_eta.append(
                (1 - damping) * new_message_eta + damping * self.messages[v].eta
            )
            messages_lam.append(
                (1 - damping) * new_message_lam + damping * self.messages[v].lam
            )
            start_dim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]
