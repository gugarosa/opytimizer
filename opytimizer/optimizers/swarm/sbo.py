"""Satin Bowerbird Optimizer."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SBO(Optimizer):
    """A SBO class, inherited from Optimizer.

    This is the designed class to define SBO-related
    variables and methods.

    References:
        S. H. S. Moosavi and V. K. Bardsiri.
        Satin bowerbird optimizer: a new optimization algorithm to optimize ANFIS
        for software development effort estimation.
        Engineering Applications of Artificial Intelligence (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the mp_mutation-heuristics.

        """

        super(SBO, self).__init__()

        self.alpha = 0.9
        self.p_mutation = 0.05
        self.z = 0.02

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.sigma = [self.z * (ub - lb) for lb, ub in zip(space.lb, space.ub)]

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Satin Bowerbird Optimizer over all agents and variables (eq. 1-7).

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        fitness = [
            1 / (1 + agent.fit) if agent.fit >= 0 else 1 + np.abs(agent.fit)
            for agent in space.agents
        ]
        total_fitness = np.sum(fitness)
        probs = [fit / total_fitness for fit in fitness]

        for agent in space.agents:
            for j in range(agent.n_variables):
                s = np.random.choice(len(space.agents), 1, p=probs, replace=False)[0]

                lambda_k = self.alpha / (1 + probs[s])

                agent.position[j] += lambda_k * (
                    (space.agents[s].position[j] + space.best_agent.position[j]) / 2
                    - agent.position[j]
                )

                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 < self.p_mutation:
                    agent.position[j] += self.sigma[j] * np.random.normal(0.0, 1.0, 1)
            agent.clip_by_bound()

            agent.fit = function(agent.position)
