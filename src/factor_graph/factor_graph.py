import random
from typing import List, Optional, Union

import torch

from factor_graph.factor import Factor
from factor_graph.gbp_settings import GBPSettings
from factor_graph.utility_functions import Gaussian, MeasModel
from factor_graph.variable_node import VariableNode


class FactorGraph:
    def __init__(self, gbp_settings: GBPSettings = GBPSettings()) -> None:
        self.var_nodes = []
        self.factors: list[Factor] = []
        self.gbp_settings = gbp_settings

    def add_var_node(
        self,
        dofs: int,
        prior_mean: Optional[torch.Tensor] = None,
        prior_diag_cov: Optional[Union[float, torch.Tensor]] = None,
        properties: dict = {},
    ) -> None:
        variableID = len(self.var_nodes)
        self.var_nodes.append(VariableNode(variableID, dofs, properties=properties))
        if prior_mean is not None and prior_diag_cov is not None:
            prior_cov = torch.zeros(dofs, dofs, dtype=prior_diag_cov.dtype)
            prior_cov[range(dofs), range(dofs)] = prior_diag_cov
            self.var_nodes[-1].prior.set_with_cov_form(prior_mean, prior_cov)
            self.var_nodes[-1].update_belief()

    def add_factor(
        self,
        adj_var_ids: List[int],
        measurement: torch.Tensor,
        meas_model: MeasModel,
        properties: dict = {},
    ) -> None:
        factorID = len(self.factors)
        adj_var_nodes = [self.var_nodes[i] for i in adj_var_ids]
        self.factors.append(
            Factor(
                factorID, adj_var_nodes, measurement, meas_model, properties=properties
            )
        )
        for var in adj_var_nodes:
            var.adj_factors.append(self.factors[-1])

    def update_all_beliefs(self) -> None:
        for var_node in self.var_nodes:
            var_node.update_belief()

    def compute_all_messages(self, apply_dropout: bool = True) -> None:
        for factor in self.factors:
            if (
                apply_dropout
                and random.random() > self.gbp_settings.dropout
                or not apply_dropout
            ):
                damping = self.gbp_settings.get_damping(factor.iters_since_relin)
                factor.compute_messages(damping)

    def linearise_all_factors(self) -> None:
        for factor in self.factors:
            factor.compute_factor()

    def robustify_all_factors(self) -> None:
        for factor in self.factors:
            factor.robustify_loss()

    def jit_linearisation(self) -> None:
        """
        Check for all factors that the current estimate is close to the linearisation point.
        If not, relinearise the factor distribution.
        Relinearisation is only allowed at a maximum frequency of once every min_linear_iters iterations.
        """
        for factor in self.factors:
            if not factor.meas_model.linear:
                adj_belief_means = factor.get_adj_means()
                factor.iters_since_relin += 1
                if (
                    torch.norm(factor.linpoint - adj_belief_means)
                    > self.gbp_settings.beta
                    and factor.iters_since_relin >= self.gbp_settings.min_linear_iters
                ):
                    factor.compute_factor()

    def synchronous_iteration(self) -> None:
        self.robustify_all_factors()
        self.jit_linearisation()  # For linear factors, no compute is done
        self.compute_all_messages()
        self.update_all_beliefs()

    def gbp_solve(
        self,
        n_iters: Optional[int] = 20,
        converged_threshold: Optional[float] = 1e-6,
        include_priors: bool = True,
    ) -> None:
        energy_log = [self.energy()]
        # print(f"\nInitial Energy {energy_log[0]:.5f}")
        print(f"\nInitial Energy {energy_log[0]}")
        i = 0
        count = 0
        not_converged = True
        while not_converged and i < n_iters:
            self.synchronous_iteration()
            if i in self.gbp_settings.reset_iters_since_relin:
                for f in self.factors:
                    f.iters_since_relin = 1

            energy_log.append(self.energy(include_priors=include_priors))
            print(
                f"Iter {i+1}  --- "
                # f"Energy {energy_log[-1]:.5f} --- "
                f"Energy {energy_log[-1]} --- "
                # f"Belief means: {self.belief_means().numpy()} --- "
                # f"Robust factors: {[factor.meas_model.loss.robust() for factor in self.factors]}"
                # f"Relins: {sum([(factor.iters_since_relin==0 and not factor.meas_model.linear) for factor in self.factors])}"
            )
            i += 1
            if abs(energy_log[-2] - energy_log[-1]) < converged_threshold:
                count += 1
                if count == 3:
                    not_converged = False
            else:
                count = 0

    def energy(
        self, eval_point: torch.Tensor = None, include_priors: bool = True
    ) -> float:
        """Computes the sum of all of the squared errors in the graph using the appropriate local loss function."""
        if eval_point is None:
            energy = sum([factor.get_energy() for factor in self.factors])
        else:
            var_dofs = torch.tensor([v.dofs for v in self.var_nodes])
            var_ix = torch.cat([torch.tensor([0]), torch.cumsum(var_dofs, dim=0)[:-1]])
            energy = 0.0
            for f in self.factors:
                local_eval_point = torch.cat(
                    [
                        eval_point[var_ix[v.variableID] : var_ix[v.variableID] + v.dofs]
                        for v in f.adj_var_nodes
                    ]
                )
                energy += f.get_energy(local_eval_point)

        if include_priors:
            prior_energy = sum([var.get_prior_energy() for var in self.var_nodes])
            energy += prior_energy

        return energy

    def get_joint_dim(self) -> int:
        return sum([var.dofs for var in self.var_nodes])

    def get_joint(self) -> Gaussian:
        """
        Get the joint distribution over all variables in the information form
        If nonlinear factors, it is taken at the current linearisation point.
        """
        dim = self.get_joint_dim()
        joint = Gaussian(dim)

        # Priors
        var_ix = [0] * len(self.var_nodes)
        counter = 0
        for var in self.var_nodes:
            var_ix[var.variableID] = int(counter)
            joint.eta[counter : counter + var.dofs] += var.prior.eta
            joint.lam[
                counter : counter + var.dofs, counter : counter + var.dofs
            ] += var.prior.lam
            counter += var.dofs

        # Other factors
        for factor in self.factors:
            factor_ix = 0
            for adj_var_node in factor.adj_var_nodes:
                vID = adj_var_node.variableID
                # Diagonal contribution of factor
                joint.eta[
                    var_ix[vID] : var_ix[vID] + adj_var_node.dofs
                ] += factor.factor.eta[factor_ix : factor_ix + adj_var_node.dofs]
                joint.lam[
                    var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                    var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                ] += factor.factor.lam[
                    factor_ix : factor_ix + adj_var_node.dofs,
                    factor_ix : factor_ix + adj_var_node.dofs,
                ]
                other_factor_ix = 0
                for other_adj_var_node in factor.adj_var_nodes:
                    if other_adj_var_node.variableID > adj_var_node.variableID:
                        other_vID = other_adj_var_node.variableID
                        # Off diagonal contributions of factor
                        joint.lam[
                            var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                            var_ix[other_vID] : var_ix[other_vID]
                            + other_adj_var_node.dofs,
                        ] += factor.factor.lam[
                            factor_ix : factor_ix + adj_var_node.dofs,
                            other_factor_ix : other_factor_ix + other_adj_var_node.dofs,
                        ]
                        joint.lam[
                            var_ix[other_vID] : var_ix[other_vID]
                            + other_adj_var_node.dofs,
                            var_ix[vID] : var_ix[vID] + adj_var_node.dofs,
                        ] += factor.factor.lam[
                            other_factor_ix : other_factor_ix + other_adj_var_node.dofs,
                            factor_ix : factor_ix + adj_var_node.dofs,
                        ]
                    other_factor_ix += other_adj_var_node.dofs
                factor_ix += adj_var_node.dofs

        return joint

    def MAP(self) -> torch.Tensor:
        return self.get_joint().mean()

    def dist_from_MAP(self) -> torch.Tensor:
        return torch.norm(self.get_joint().mean() - self.belief_means())

    def belief_means(self) -> torch.Tensor:
        """Get an array containing all current estimates of belief means."""
        return torch.cat([var.belief.mean() for var in self.var_nodes])

    def belief_covs(self) -> List[torch.Tensor]:
        """Get a list containing all current estimates of belief covariances."""
        covs = [var.belief.cov() for var in self.var_nodes]
        return covs

    def get_gradient(self, include_priors: bool = True) -> torch.Tensor:
        """Return gradient wrt the total energy."""
        dim = self.get_joint_dim()
        grad = torch.zeros(dim)
        var_dofs = torch.tensor([v.dofs for v in self.var_nodes])
        var_ix = torch.cat([torch.tensor([0]), torch.cumsum(var_dofs, dim=0)[:-1]])

        if include_priors:
            for v in self.var_nodes:
                grad[var_ix[v.variableID] : var_ix[v.variableID] + v.dofs] += (
                    v.belief.mean() - v.prior.mean()
                ) @ v.prior.cov()

        for f in self.factors:
            r = f.get_residual()
            jac = f.meas_model.jac_fn(f.linpoint)  # jacobian wrt residual
            local_grad = (
                r @ torch.inverse(f.meas_model.loss.effective_cov) @ jac
            ).flatten()

            factor_ix = 0
            for adj_var_node in f.adj_var_nodes:
                vID = adj_var_node.variableID
                grad[var_ix[vID] : var_ix[vID] + adj_var_node.dofs] += local_grad[
                    factor_ix : factor_ix + adj_var_node.dofs
                ]
                factor_ix += adj_var_node.dofs
        return grad

    def gradient_descent_step(self, lr: float = 1e-3) -> None:
        grad = self.get_gradient()
        i = 0
        for v in self.var_nodes:
            v.belief.eta = v.belief.lam @ (v.belief.mean() - lr * grad[i : i + v.dofs])
            i += v.dofs
        self.linearise_all_factors()

    def lm_step(self, lambda_lm: float, a: float = 1.5, b: float = 3) -> bool:
        """Very close to an LM step, except we always accept update even if it increases the energy.
        As to compute the energy if we were to do the update, we would need to relinearise all factors.
        Returns lambda parameters for LM.
        If lambda_lm = 0, then it is Gauss-Newton.
        """
        current_x = self.belief_means()
        initial_energy = self.energy()

        joint = self.get_joint()
        A = joint.lam + lambda_lm * torch.eye(len(joint.eta))
        b_mat = -self.get_gradient()
        delta_x = torch.inverse(A) @ b_mat

        i = 0  # apply update
        for v in self.var_nodes:
            v.belief.eta = v.belief.lam @ (v.belief.mean() + delta_x[i : i + v.dofs])
            i += v.dofs
        self.linearise_all_factors()
        new_energy = self.energy()

        if lambda_lm == 0.0:  # Gauss-Newton
            return lambda_lm
        if new_energy < initial_energy:  # accept update
            lambda_lm /= a
            return lambda_lm
        else:  # undo update
            i = 0  # apply update
            for v in self.var_nodes:
                v.belief.eta = v.belief.lam @ (
                    v.belief.mean() - delta_x[i : i + v.dofs]
                )
                i += v.dofs
            self.linearise_all_factors()
            lambda_lm = min(lambda_lm * b, 1e5)
            return lambda_lm

    def print(self, brief=False) -> None:
        print("\nFactor Graph:")
        print(f"# Variable nodes: {len(self.var_nodes)}")
        if not brief:
            for i, var in enumerate(self.var_nodes):
                print(
                    f"Variable {i}: connects to factors {[f.factorID for f in var.adj_factors]}"
                )
                print(f"    dofs: {var.dofs}")
                print(f"    prior mean: {var.prior.mean().numpy()}")
                print(
                    f"    prior covariance: diagonal sigma {torch.diag(var.prior.cov()).numpy()}"
                )

        print(f"# Factors: {len(self.factors)}")

        if not brief:
            for i, factor in enumerate(self.factors):
                if factor.meas_model.linear:
                    print("Linear", end=" ")
                else:
                    print("Nonlinear", end=" ")
                print(f"Factor {i}: connects to variables {factor.adj_vIDs}")
                print(
                    f"    measurement model: {type(factor.meas_model).__name__},"
                    f" {type(factor.meas_model.loss).__name__},"
                    f" diagonal sigma {torch.diag(factor.meas_model.loss.effective_cov).detach().numpy()}"
                )
                print(f"    measurement: {factor.measurement.numpy()}")

        print("\n")
