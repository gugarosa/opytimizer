"""Water Cycle Algorithm."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class WCA(Optimizer):
    """A WCA class, inherited from Optimizer.

    This is the designed class to define WCA-related
    variables and methods.

    References:
        H. Eskandar.
        Water cycle algorithm – A novel metaheuristic optimization method for
        solving constrained engineering optimization problems.
        Computers & Structures (2012).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(WCA, self).__init__()

        self.nsr = 2
        self.d_max = 0.1

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.flows = np.zeros(self.nsr, dtype=int)

    def _flow_intensity(self, agents: List[Agent]) -> None:
        """Calculates the intensity of each possible flow (eq. 6).

        Args:
            agents: List of agents.

        """

        cost = np.sum([agents[i].fit for i in range(self.nsr)])

        for i in range(self.nsr):
            self.flows[i] = np.floor(
                np.fabs(agents[i].fit / cost) * (len(agents) - self.nsr)
            )

    def _raining_process(self, agents: List[Agent], best_agent: Agent) -> None:
        """Performs the raining process (eq. 11-12).

        Args:
            agents: List of agents.
            best_agent: Global best agent.

        """

        for i in range(0, self.nsr):
            for j in range(self.nsr, self.flows[i] + self.nsr):
                distance = np.linalg.norm(best_agent.position - agents[j].position)
                if distance < self.d_max:
                    if i == 0:
                        # Updates position (eq. 12)
                        r1 = np.random.normal(1, agents[j].n_variables, 1)
                        agents[j].position = best_agent.position + np.sqrt(0.1) * r1
                    else:
                        # Updates position (eq. 11)
                        agents[j].fill_with_uniform()

    def _update_stream(self, agents: List[Agent], function: Callable) -> None:
        """Updates every stream position (eq. 8).

        Args:
            agents: List of agents.
            function: A callable that will be used as the objective function.

        """

        n_flows = 0

        for i in range(0, self.nsr):
            n_flows += self.flows[i]

            for j in range((self.nsr + n_flows - self.flows[i]), self.nsr + n_flows):
                r1 = np.random.uniform(0.0, 1.0, 1)
                agents[j].position += r1 * 2 * (agents[i].position - agents[j].position)
                agents[j].clip_by_bound()

                agents[j].fit = function(agents[j].position)

    def _update_river(
        self, agents: List[Agent], best_agent: Agent, function: Callable
    ) -> None:
        """Updates every river position (eq. 9).

        Args:
            agents: List of agents.
            best_agent: Global best agent.
            function: A callable that will be used as the objective function.

        """

        for i in range(1, self.nsr):
            r1 = np.random.uniform(0.0, 1.0, 1)
            agents[i].position += r1 * 2 * (best_agent.position - agents[i].position)
            agents[i].clip_by_bound()

            agents[i].fit = function(agents[i].position)

    def update(self, space: Space, function: Callable, n_iterations: int) -> None:
        """Wraps Water Cycle Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            n_iterations: Maximum number of iterations.

        """

        self._flow_intensity(space.agents)
        self._update_stream(space.agents, function)
        self._update_river(space.agents, space.best_agent, function)

        for i in range(1, self.nsr):
            for j in range(self.nsr, len(space.agents)):
                if space.agents[j].fit < space.agents[i].fit:
                    space.agents[i], space.agents[j] = space.agents[j], space.agents[i]

        for i in range(1, self.nsr):
            if space.agents[i].fit < space.agents[0].fit:
                space.agents[i], space.agents[0] = space.agents[0], space.agents[i]

        # Performs the raining process (eq. 12)
        self._raining_process(space.agents, space.best_agent)

        self.d_max -= self.d_max / n_iterations
