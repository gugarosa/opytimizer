"""Cuckoo Search.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class CS(Optimizer):
    """A CS class, inherited from Optimizer.

    This is the designed class to define CS-related
    variables and methods.

    References:
        X.-S. Yang and D. Suash. Cuckoo search via Lévy flights.
        World Congress on Nature & Biologically Inspired Computing (2009).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> CS.")

        # Overrides its parent class with the receiving params
        super(CS, self).__init__()

        # Step size
        self.alpha = 1

        # Lévy distribution parameter
        self.beta = 1.5

        # Probability of replacing worst nests
        self.p = 0.2

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Step size."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Lévy distribution parameter."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta <= 0 or beta > 2:
            raise e.ValueError("`beta` should be between 0 and 2")

        self._beta = beta

    @property
    def p(self) -> float:
        """Probability of replacing worst nests."""

        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise e.TypeError("`p` should be a float or integer")
        if p < 0 or p > 1:
            raise e.ValueError("`p` should be between 0 and 1")

        self._p = p

    def _generate_new_nests(
        self, agents: List[Agent], best_agent: Agent
    ) -> List[Agent]:
        """Generate new nests (eq. 1).

        Args:
            agents: List of agents.
            best_agent: Global best agent.

        Returns:
            (List[Agent]): A new list of agents which can be seen as new nests.

        """

        # Makes a temporary copy of current agents
        new_agents = copy.deepcopy(agents)

        # Then, we iterate for every agent
        for new_agent in new_agents:
            # Calculates the Lévy distribution
            step = d.generate_levy_distribution(self.beta, new_agent.n_variables)

            # Expanding its dimension to perform entrywise multiplication
            step = np.expand_dims(step, axis=1)

            # Calculates the difference vector between local and best positions
            # Alpha controls the intensity of the step size
            step_size = self.alpha * step * (new_agent.position - best_agent.position)

            # Generates a random normal distribution
            g = r.generate_gaussian_random_number(size=new_agent.n_variables)

            # Expanding its dimension to perform entrywise multiplication
            g = np.expand_dims(g, axis=1)

            # Acutally performs the random walk / flight
            new_agent.position += step_size * g

        return new_agents

    def _generate_abandoned_nests(
        self, agents: List[Agent], prob: float
    ) -> List[Agent]:
        """Generate a fraction of nests to be replaced.

        Args:
            agents: List of agents.
            prob: Probability of replacing worst nests.

        Returns:
            (List[Agent]): A new list of agents which can be seen as the new nests to be replaced.

        """

        # Makes a temporary copy of current agents
        new_agents = copy.deepcopy(agents)

        # Generates a bernoulli distribution array
        # It will be used to replace or not a certain nest
        b = d.generate_bernoulli_distribution(1 - prob, len(agents))

        # Iterates through every new agent
        for j, new_agent in enumerate(new_agents):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # Then, we select two random nests
            k = r.generate_integer_random_number(0, len(agents) - 1)
            l = r.generate_integer_random_number(0, len(agents) - 1, exclude_value=k)

            # Calculates the random walk between these two nests
            step_size = r1 * (agents[k].position - agents[l].position)

            # Finally, we replace the old nest
            # Note it will only be replaced if 'b' is 1
            new_agent.position += step_size * b[j]

        return new_agents

    def _evaluate_nests(
        self, agents: List[Agent], new_agents: List[Agent], function: Function
    ) -> None:
        """Evaluate new nests according to a fitness function.

        Args:
            agents: List of current agents.
            new_agents: List of new agents to be evaluated.
            function: Fitness function used to evaluate.

        """

        # Iterates through each agent and new agent
        for agent, new_agent in zip(agents, new_agents):
            # Checks agent's limits
            new_agent.clip_by_bound()

            # Calculates the new agent fitness
            new_agent.fit = function(new_agent.position)

            # If new agent's fitness is better than agent's
            if new_agent.fit < agent.fit:
                # Replace its position and fitness
                agent.position = copy.deepcopy(new_agent.position)
                agent.fit = copy.deepcopy(new_agent.fit)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Cuckoo Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Generate new nests
        new_agents = self._generate_new_nests(space.agents, space.best_agent)

        # Evaluate new generated nests
        self._evaluate_nests(space.agents, new_agents, function)

        # Generate new nests to be replaced
        new_agents = self._generate_abandoned_nests(space.agents, self.p)

        # Evaluate new generated nests for further replacement
        self._evaluate_nests(space.agents, new_agents, function)
