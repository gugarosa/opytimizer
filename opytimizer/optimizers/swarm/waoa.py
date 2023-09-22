"""Walrus Optimization Algorithm.
"""

import copy
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

    This is the designed class to dife WAOA-related
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

        logger.info("Overriding class: Optimizer -> SSA")

        super(WAOA, self).__init__()

        self.build(params)

        logger.info("Class overrided.")

    def update(self, space: Space, function: Function, iteration: int) -> None:
        """Wraps Walrus Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.

        """

        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            for j in range(space.n_variables):

                r1 = r.generate_integer_random_number(1, 2)
                r2 = r.generate_uniform_random_number()

                a.position[j] = agent.position[j] + r2 * (
                    space.best_agent.position[j] - r1 * agent.position[j]
                )

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            k = r.generate_integer_random_number(0, space.n_agents, i)

            if space.agents[k].fit < agent.fit:

                for j in range(space.n_variables):

                    r1 = r.generate_integer_random_number(1, 2)
                    r2 = r.generate_uniform_random_number()

                    a.position[j] = agent.position[j] + r2 * (
                        space.agents[k].position[j] - r1 * agent.position[j]
                    )

            else:

                for j in range(space.n_variables):

                    r2 = r.generate_uniform_random_number()

                    a.position[j] = agent.position[j] + r2 * (
                        agent.position[j] - space.agents[k].position[j]
                    )

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            for j in range(space.n_variables):

                r2 = r.generate_uniform_random_number()

                a.position[j] = agent.position[j] + (
                    (agent.ub[j] / (iteration + 1))
                    - r2 * (agent.lb[j] / (iteration + 1))
                )

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
