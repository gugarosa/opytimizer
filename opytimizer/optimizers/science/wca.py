"""Water Cycle Algorithm.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Overriding class: Optimizer -> WCA.")

        super(WCA, self).__init__()

        self.nsr = 2
        self.d_max = 0.1

        self.build(params)

        logger.info("Class overrided.")

    @property
    def nsr(self) -> float:
        """Number of rivers summed with a single sea."""

        return self._nsr

    @nsr.setter
    def nsr(self, nsr: float) -> None:
        if not isinstance(nsr, int):
            raise e.TypeError("`nsr` should be an integer")
        if nsr < 1:
            raise e.ValueError("`nsr` should be > 1")

        self._nsr = nsr

    @property
    def d_max(self) -> float:
        """Maximum evaporation condition."""

        return self._d_max

    @d_max.setter
    def d_max(self, d_max: float) -> None:
        if not isinstance(d_max, (float, int)):
            raise e.TypeError("`d_max` should be a float or integer")
        if d_max < 0:
            raise e.ValueError("`d_max` should be >= 0")

        self._d_max = d_max

    @property
    def flows(self) -> np.ndarray:
        """Array of flows."""

        return self._flows

    @flows.setter
    def flows(self, flows: np.ndarray) -> None:
        if not isinstance(flows, np.ndarray):
            raise e.TypeError("`flows` should be a numpy array")

        self._flows = flows

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
                        r1 = r.generate_gaussian_random_number(1, agents[j].n_variables)
                        agents[j].position = best_agent.position + np.sqrt(0.1) * r1
                    else:
                        # Updates position (eq. 11)
                        agents[j].fill_with_uniform()

    def _update_stream(self, agents: List[Agent], function: Function) -> None:
        """Updates every stream position (eq. 8).

        Args:
            agents: List of agents.
            function: A Function object that will be used as the objective function.

        """

        n_flows = 0

        for i in range(0, self.nsr):
            n_flows += self.flows[i]

            for j in range((self.nsr + n_flows - self.flows[i]), self.nsr + n_flows):
                r1 = r.generate_uniform_random_number()
                agents[j].position += r1 * 2 * (agents[i].position - agents[j].position)
                agents[j].clip_by_bound()

                agents[j].fit = function(agents[j].position)

    def _update_river(
        self, agents: List[Agent], best_agent: Agent, function: Function
    ) -> None:
        """Updates every river position (eq. 9).

        Args:
            agents: List of agents.
            best_agent: Global best agent.
            function: A Function object that will be used as the objective function.

        """

        for i in range(1, self.nsr):
            r1 = r.generate_uniform_random_number()
            agents[i].position += r1 * 2 * (best_agent.position - agents[i].position)
            agents[i].clip_by_bound()

            agents[i].fit = function(agents[i].position)

    def update(self, space: Space, function: Function, n_iterations: int) -> None:
        """Wraps Water Cycle Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
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
