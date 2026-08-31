"""Walrus Optimization Algorithm."""

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space


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

        super(WAOA, self).__init__()

        self.build(params)

    def evaluate(self, space: Space) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.

        """
        for agent in space.agents:
            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space, function: Callable, iteration: int) -> None:
        """Wraps Walrus Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.

        """

        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            r1 = np.random.randint(1, 3, (space.n_variables, space.n_dimensions))
            r2 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

            a.position = agent.position + r2 * (
                space.best_agent.position - r1 * agent.position
            )

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            k = r.integer(0, space.n_agents, exclude=i, size=None)

            if space.agents[k].fit < agent.fit:

                r3 = np.random.randint(1, 3, (space.n_variables, space.n_dimensions))
                r4 = np.random.uniform(
                    0.0, 1.0, (space.n_variables, space.n_dimensions)
                )

                a.position = agent.position + r4 * (
                    space.agents[k].position - r3 * agent.position
                )

            else:

                r5 = np.random.uniform(
                    0.0, 1.0, (space.n_variables, space.n_dimensions)
                )

                a.position = agent.position + r5 * (
                    agent.position - space.agents[k].position
                )

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            r6 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

            lb = (agent.lb / (iteration + 1)).reshape(-1, 1)
            ub = (agent.ub / (iteration + 1)).reshape(-1, 1)

            a.position = agent.position + (lb + (ub - r6 * lb))

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
