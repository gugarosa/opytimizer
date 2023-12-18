"""Walrus Optimization Algorithm.
"""

import copy
import time
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class WAOA(Optimizer):
    """A WAOA class, inherited from Optimizer.

    This is the designed class to define WAOA-related
    variables and methods.

    References:
        P. Trojovský and M. Dehghani. A new bio-inspired metaheuristic algorithm for
        solving optimization problems based on walruses behavior. Scientific Reports (2023).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params (str): Contains key-value parameters to the meta-heuristics.
        """

        logger.info("Overriding class: Optimizer -> WAOA")

        super(WAOA, self).__init__()

        self.build(params)

        logger.info("Class overrided.")

    def evaluate(self, space: Space) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.

        """
        print('evaluating...')
        for agent in space.agents:
            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space, function: Function, iteration: int) -> None:
        """Wraps Walrus Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.

        """

        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            r1 = r.generate_integer_random_number(
                1, 3, size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            a.position = agent.position + r2 * (
                space.best_agent.position - r1 * agent.position
            )

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            k = r.generate_integer_random_number(0, space.n_agents, i)

            if space.agents[k].fit < agent.fit:

                r3 = r.generate_integer_random_number(
                    1, 3, size=(space.n_variables, space.n_dimensions)
                )
                r4 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )

                a.position = agent.position + r4 * (
                    space.agents[k].position - r3 * agent.position
                )

            else:

                r5 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )

                a.position = agent.position + r5 * (
                    agent.position - space.agents[k].position
                )

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            r6 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            lb = (agent.lb / (iteration + 1)).reshape(-1, 1)
            ub = (agent.ub / (iteration + 1)).reshape(-1, 1)

            a.position = agent.position + (lb + (ub - r6 * lb))

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)