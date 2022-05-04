"""Search space.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Agent
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Space:
    """A Space class for agents, variables and methods
    related to the search space.

    """

    def __init__(
        self,
        n_agents: Optional[int] = 1,
        n_variables: Optional[int] = 1,
        n_dimensions: Optional[int] = 1,
        lower_bound: Optional[Union[float, List, Tuple, np.ndarray]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, np.ndarray]] = 1.0,
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Dimension of search space.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.

        """

        # Number of agents
        self.n_agents = n_agents

        # Number of variables
        self.n_variables = n_variables

        # Number of dimensions
        self.n_dimensions = n_dimensions

        # Lower bounds
        self.lb = np.asarray(lower_bound)

        # Upper bounds
        self.ub = np.asarray(upper_bound)

        # Agents
        self.agents = []

        # Best agent
        self.best_agent = Agent(n_variables, n_dimensions, lower_bound, upper_bound)

        # Indicates whether the space is built or not
        self.built = False

    @property
    def n_agents(self) -> int:
        """Number of agents."""

        return self._n_agents

    @n_agents.setter
    def n_agents(self, n_agents: int) -> None:
        if not isinstance(n_agents, int):
            raise e.TypeError("`n_agents` should be an integer")
        if n_agents <= 0:
            raise e.ValueError("`n_agents` should be > 0")

        self._n_agents = n_agents

    @property
    def n_variables(self) -> int:
        """Number of decision variables."""

        return self._n_variables

    @n_variables.setter
    def n_variables(self, n_variables: int) -> None:
        if not isinstance(n_variables, int):
            raise e.TypeError("`n_variables` should be an integer")
        if n_variables <= 0:
            raise e.ValueError("`n_variables` should be > 0")

        self._n_variables = n_variables

    @property
    def n_dimensions(self) -> int:
        """Number of search space dimensions."""

        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions: int) -> None:
        if not isinstance(n_dimensions, int):
            raise e.TypeError("`n_dimensions` should be an integer")
        if n_dimensions <= 0:
            raise e.ValueError("`n_dimensions` should be > 0")

        self._n_dimensions = n_dimensions

    @property
    def lb(self) -> np.ndarray:
        """Minimum possible values."""

        return self._lb

    @lb.setter
    def lb(self, lb: np.ndarray) -> None:
        if not isinstance(lb, np.ndarray):
            raise e.TypeError("`lb` should be a numpy array")
        if not lb.shape:
            lb = np.expand_dims(lb, -1)
        if lb.shape[0] != self.n_variables:
            raise e.SizeError("`lb` should be the same size as `n_variables`")

        self._lb = lb

    @property
    def ub(self) -> np.ndarray:
        """Maximum possible values."""

        return self._ub

    @ub.setter
    def ub(self, ub: np.ndarray) -> None:
        if not isinstance(ub, np.ndarray):
            raise e.TypeError("`ub` should be a numpy array")
        if not ub.shape:
            ub = np.expand_dims(ub, -1)
        if not ub.shape or ub.shape[0] != self.n_variables:
            raise e.SizeError("`ub` should be the same size as `n_variables`")

        self._ub = ub

    @property
    def agents(self) -> List[Agent]:
        """list: Agents that belongs to the space."""

        return self._agents

    @agents.setter
    def agents(self, agents: List[Agent]) -> None:
        if not isinstance(agents, list):
            raise e.TypeError("`agents` should be a list")

        self._agents = agents

    @property
    def best_agent(self) -> Agent:
        """Agent: Best agent."""

        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent: Agent) -> None:
        if not isinstance(best_agent, Agent):
            raise e.TypeError("`best_agent` should be an Agent")

        self._best_agent = best_agent

    @property
    def built(self) -> bool:
        """Indicates whether the space is built."""

        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        if not isinstance(built, bool):
            raise e.TypeError("`built` should be a boolean")

        self._built = built

    def _create_agents(self) -> None:
        """Creates a list of agents."""

        self.agents = [
            Agent(self.n_variables, self.n_dimensions, self.lb, self.ub)
            for _ in range(self.n_agents)
        ]

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent.

        As each child has a different procedure of initialization,
        you will need to implement it directly on its class.

        """

        pass

    def build(self) -> None:
        """Builds the object by creating and initializing the agents."""

        self._create_agents()
        self._initialize_agents()

        # If no errors were shown, we can declare the space as `built`
        self.built = True

        logger.debug(
            "Agents: %d | Size: (%d, %d) | "
            "Lower Bound: %s | Upper Bound: %s | Built: %s.",
            self.n_agents,
            self.n_variables,
            self.n_dimensions,
            self.lb,
            self.ub,
            self.built,
        )

    def clip_by_bound(self) -> None:
        """Clips the agents' decision variables to the bounds limits."""

        for agent in self.agents:
            agent.clip_by_bound()
