# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Chernobyl Disaster Optimizer.

References:
    H. A. Shehadeh. Chernobyl disaster optimizer (CDO): a novel meta-heuristic method for global optimization.
    Neural Computing and Applications (2023). https://doi.org/10.1007/s00521-023-08261-1

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class CDO(Optimizer):
    """Implement Chernobyl Disaster Optimizer.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize radiation-inspired population movement.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            This optimizer has no algorithm-specific configuration keys.
            Compilation initializes alpha, beta, and gamma position buffers to zero
            and their fitness values to the largest supported floating-point value.

        """

        super(CDO, self).__init__()

        self.build(params)

    def compile(self, space: Space) -> None:
        self.gamma_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.gamma_fit = c.FLOAT_MAX

        self.beta_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.beta_fit = c.FLOAT_MAX

        self.alpha_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.alpha_fit = c.FLOAT_MAX

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
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

        ws = 3 - 3 * iteration / n_iterations
        s_gamma = np.log10(np.random.uniform(1, 300000, 1))
        s_beta = np.log10(np.random.uniform(1, 270000, 1))
        s_alpha = np.log10(np.random.uniform(1, 16000, 1))

        for agent in space.agents:

            r1 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))
            r2 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))
            r3 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

            rho_gamma = np.pi * r1 * r1 / s_gamma - ws * r2
            a_gamma = r3 * r3 * np.pi
            grad_gamma = np.abs(a_gamma * self.gamma_pos - agent.position)
            v_gamma = agent.position - rho_gamma * grad_gamma

            r1 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))
            r2 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))
            r3 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

            rho_beta = np.pi * r1 * r1 / (0.5 * s_beta) - ws * r2
            a_beta = r3 * r3 * np.pi
            grad_beta = np.abs(a_beta * self.beta_pos - agent.position)
            v_beta = 0.5 * (agent.position - rho_beta * grad_beta)

            r1 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))
            r2 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))
            r3 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

            rho_alpha = np.pi * r1 * r1 / (0.25 * s_alpha) - ws * r2
            a_alpha = r3 * r3 * np.pi
            grad_alpha = np.abs(a_alpha * self.alpha_pos - agent.position)
            v_alpha = 0.25 * (agent.position - rho_alpha * grad_alpha)

            agent.position = (v_alpha + v_beta + v_gamma) / 3
