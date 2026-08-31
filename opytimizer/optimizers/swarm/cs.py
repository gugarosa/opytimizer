"""Cuckoo Search."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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

        super(CS, self).__init__()

        self.alpha = 1.0
        self.beta = 1.5
        self.p = 0.2

        self.build(params)

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

        new_agents = copy.deepcopy(agents)
        for new_agent in new_agents:
            step = d.generate_levy_distribution(self.beta, new_agent.n_variables)
            step = np.expand_dims(step, axis=1)

            # Alpha controls the intensity of the step size
            step_size = self.alpha * step * (new_agent.position - best_agent.position)

            g = np.random.normal(0.0, 1.0, new_agent.n_variables)
            g = np.expand_dims(g, axis=1)

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

        new_agents = copy.deepcopy(agents)

        # It will be used to replace or not a certain nest
        b = np.random.binomial(1, 1 - prob, len(agents))

        for j, new_agent in enumerate(new_agents):
            r1 = np.random.uniform(0.0, 1.0, 1)

            k = np.random.randint(0, len(agents) - 1, None)
            l = r.integer(0, len(agents) - 1, exclude=k, size=None)

            step_size = r1 * (agents[k].position - agents[l].position)
            new_agent.position += step_size * b[j]

        return new_agents

    def _evaluate_nests(
        self, agents: List[Agent], new_agents: List[Agent], function: Callable
    ) -> None:
        """Evaluate new nests according to a fitness function.

        Args:
            agents: List of current agents.
            new_agents: List of new agents to be evaluated.
            function: Fitness function used to evaluate.

        """

        for agent, new_agent in zip(agents, new_agents):
            new_agent.clip_by_bound()

            new_agent.fit = function(new_agent.position)
            if new_agent.fit < agent.fit:
                agent.position = copy.deepcopy(new_agent.position)
                agent.fit = copy.deepcopy(new_agent.fit)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Cuckoo Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        new_agents = self._generate_new_nests(space.agents, space.best_agent)
        self._evaluate_nests(space.agents, new_agents, function)

        new_agents = self._generate_abandoned_nests(space.agents, self.p)
        self._evaluate_nests(space.agents, new_agents, function)
