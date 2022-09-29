"""Artificial Bee Colony.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Overriding class: Optimizer -> ABC.")

        super(ABC, self).__init__()

        self.n_trials = 10

        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_trials(self) -> int:
        """Number of trial limits."""

        return self._n_trials

    @n_trials.setter
    def n_trials(self, n_trials: int) -> None:
        if not isinstance(n_trials, int):
            raise e.TypeError("`n_trials` should be an integer")
        if n_trials <= 0:
            raise e.ValueError("`n_trials` should be > 0")

        self._n_trials = n_trials

    @property
    def trial(self) -> np.ndarray:
        """Array of trial."""

        return self._trial

    @trial.setter
    def trial(self, trial: np.ndarray) -> None:
        if not isinstance(trial, np.ndarray):
            raise e.TypeError("`trial` should be a numpy array")

        self._trial = trial

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.trial = np.zeros(space.n_agents)

    def _evaluate_location(
        self, agent: Agent, neighbour: Agent, function: Function, index: int
    ) -> None:
        """Evaluates a food source location and update its value if possible (eq. 2.2).

        Args:
            agent: An agent.
            neighbour: A neightbour agent.
            function: A function object.
            index: Index of trial.

        """

        r1 = r.generate_uniform_random_number(-1, 1)

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

    def _send_employee(self, agents: List[Agent], function: Function) -> None:
        """Sends employee bees onto food source to evaluate its nectar.

        Args:
            agents: List of agents.
            function: A function object.

        """

        for i, agent in enumerate(agents):
            source = r.generate_integer_random_number(0, len(agents))
            self._evaluate_location(agent, agents[source], function, i)

    def _send_onlooker(self, agents: List[Agent], function: Function) -> None:
        """Sends onlooker bees to select new food sources (eq. 2.1).

        Args:
            agents: List of agents.
            function: A function object.

        """

        total = sum(agent.fit for agent in agents)

        k = 0
        while k < len(agents):
            for i, agent in enumerate(agents):
                r1 = r.generate_uniform_random_number()
                probs = (agent.fit / (total + c.EPSILON)) + 0.1

                if r1 < probs:
                    k += 1

                    source = r.generate_integer_random_number(0, len(agents))
                    self._evaluate_location(agent, agents[source], function, i)

    def _send_scout(self, agents: List[Agent], function: Function) -> None:
        """Sends scout bees to scout for new possible food sources.

        Args:
            agents: List of agents.
            function: A function object.

        """

        max_trial, max_index = np.max(self.trial), np.argmax(self.trial)
        if max_trial > self.n_trials:
            self.trial[max_index] = 0

            a = copy.deepcopy(agents[max_index])
            a.position += r.generate_uniform_random_number(-1, 1)
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agents[max_index].fit:
                agents[max_index] = copy.deepcopy(a)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Artificial Bee Colony over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        self._send_employee(space.agents, function)
        self._send_onlooker(space.agents, function)
        self._send_scout(space.agents, function)
