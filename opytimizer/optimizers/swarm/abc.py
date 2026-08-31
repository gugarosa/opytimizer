"""Artificial Bee Colony."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class ABC(Optimizer):
    """An ABC class, inherited from Optimizer.

    This is the designed class to define ABC-related
    variables and methods.

    References:
        D. Karaboga and B. Basturk.
        A powerful and efficient algorithm for numerical function optimization: Artificial bee colony (ABC) algorithm.
        Journal of Global Optimization (2007).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(ABC, self).__init__()

        self.n_trials = 10

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.trial = np.zeros(space.n_agents)

    def _evaluate_location(
        self, agent: Agent, neighbour: Agent, function: Callable, index: int
    ) -> None:
        """Evaluates a food source location and update its value if possible (eq. 2.2).

        Args:
            agent: An agent.
            neighbour: A neightbour agent.
            function: A function object.
            index: Index of trial.

        """

        r1 = np.random.uniform(-1, 1, 1)

        a = copy.deepcopy(agent)

        # Change its location (eq. 2.2)
        a.position = agent.position + (agent.position - neighbour.position) * r1
        a.clip_by_bound()

        a.fit = function(a.position)
        if a.fit < agent.fit:
            self.trial[index] = 0

            agent.position = copy.deepcopy(a.position)
            agent.fit = copy.deepcopy(a.fit)
        else:
            self.trial[index] += 1

    def _send_employee(self, agents: List[Agent], function: Callable) -> None:
        """Sends employee bees onto food source to evaluate its nectar.

        Args:
            agents: List of agents.
            function: A function object.

        """

        for i, agent in enumerate(agents):
            source = np.random.randint(0, len(agents), None)
            self._evaluate_location(agent, agents[source], function, i)

    def _send_onlooker(self, agents: List[Agent], function: Callable) -> None:
        """Sends onlooker bees to select new food sources (eq. 2.1).

        Args:
            agents: List of agents.
            function: A function object.

        """

        total = sum(agent.fit for agent in agents)

        k = 0
        while k < len(agents):
            for i, agent in enumerate(agents):
                r1 = np.random.uniform(0.0, 1.0, 1)
                probs = (agent.fit / (total + c.EPSILON)) + 0.1

                if r1 < probs:
                    k += 1

                    source = np.random.randint(0, len(agents), None)
                    self._evaluate_location(agent, agents[source], function, i)

    def _send_scout(self, agents: List[Agent], function: Callable) -> None:
        """Sends scout bees to scout for new possible food sources.

        Args:
            agents: List of agents.
            function: A function object.

        """

        max_trial, max_index = np.max(self.trial), np.argmax(self.trial)
        if max_trial > self.n_trials:
            self.trial[max_index] = 0

            a = copy.deepcopy(agents[max_index])
            a.position += np.random.uniform(-1, 1, 1)
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agents[max_index].fit:
                agents[max_index] = copy.deepcopy(a)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Artificial Bee Colony over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        self._send_employee(space.agents, function)
        self._send_onlooker(space.agents, function)
        self._send_scout(space.agents, function)
