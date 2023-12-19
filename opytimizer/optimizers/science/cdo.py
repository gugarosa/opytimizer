"""Chernobyl Disaster Optimizer.
"""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.core import Optimizer
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class CDO(Optimizer):
    """An CDO class, inherited from Optimizer.

    This is the designed class to define CDO-related
    variables and methods.

    References:
        H. Abedinpourshotorban et al.
        Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
        Swarm and Evolutionary Computation (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(CDO, self).__init__()


        self.build(params)

        logger.info("Class overrided.")

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.gamma_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.gamma_fit = c.FLOAT_MAX

        self.beta_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.beta_fit = c.FLOAT_MAX

        self.alpha_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.alpha_fit = c.FLOAT_MAX

    def update(self, space: Space, function: Function, iteration: int, n_iterations: int) -> None:
        """Wraps Chernobyl Disaster Optimizer over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        for agent in space.agents:

            fit = function(agent.position)

            if fit < self.alpha_fit:
                self.alpha_fit = fit
                self.alpha_pos = copy.deepcopy(agent.position)

            if fit < self.alpha_fit and fit < self.beta_fit:
                self.beta_fit = fit
                self.beta_pos = copy.deepcopy(agent.position)

            if fit < self.alpha_fit and fit < self.beta_fit and fit < self.gamma_fit:
                self.gamma_fit = fit
                self.gamma_pos = copy.deepcopy(agent.position)

        ws = 3 - 3 * iteration/n_iterations
        s_gamma = np.log10(r.generate_uniform_random_number(1, 300000))
        s_beta = np.log10(r.generate_uniform_random_number(1, 270000))
        s_alpha = np.log10(r.generate_uniform_random_number(1, 16000))

        for agent in space.agents:

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_gamma = np.pi * r1 * r1 / s_gamma - ws * r2
            a_gamma = r3 * r3 * np.pi
            grad_gamma = np.abs(a_gamma * self.gamma_pos - agent.position)
            v_gamma = agent.position - rho_gamma * grad_gamma

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_beta = np.pi * r1 * r1 / (0.5 * s_beta) - ws * r2
            a_beta = r3 * r3 * np.pi
            grad_beta = np.abs(a_beta * self.beta_pos - agent.position)
            v_beta = 0.5 * (agent.position - rho_beta * grad_beta)

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_alpha = np.pi * r1 * r1 / (0.25 * s_alpha) - ws * r2
            a_alpha = r3 * r3 * np.pi
            grad_alpha = np.abs(a_alpha * self.alpha_pos - agent.position)
            v_alpha = 0.25 * (agent.position - rho_alpha * grad_alpha)

            agent.position = (v_alpha + v_beta + v_gamma) / 3
