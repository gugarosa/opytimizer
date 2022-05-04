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

        # Overrides its parent class with the receiving params
        super(ABC, self).__init__()

        # Number of trial limits
        self.n_trials = 10

        # Builds the class
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

        # Arrays of trials
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

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number(-1, 1)

        # Copies actual food source location
        a = copy.deepcopy(agent)

        # Change its location according to equation 2.2
        a.position = agent.position + (agent.position - neighbour.position) * r1

        # Checks agent's limits
        a.clip_by_bound()

        # Evaluating its fitness
        a.fit = function(a.position)

        # Check if fitness is improved
        if a.fit < agent.fit:
            # If yes, reset the number of trials for this particular food source
            self.trial[index] = 0

            # Copies the new position and fitness
            agent.position = copy.deepcopy(a.position)
            agent.fit = copy.deepcopy(a.fit)

        # If not
        else:
            # We increse the trials counter
            self.trial[index] += 1

    def _send_employee(self, agents: List[Agent], function: Function) -> None:
        """Sends employee bees onto food source to evaluate its nectar.

        Args:
            agents: List of agents.
            function: A function object.

        """

        # Iterate through all food sources
        for i, agent in enumerate(agents):
            # Gathers a random source to be used
            source = r.generate_integer_random_number(0, len(agents))

            # Measuring food source location
            self._evaluate_location(agent, agents[source], function, i)

    def _send_onlooker(self, agents: List[Agent], function: Function) -> None:
        """Sends onlooker bees to select new food sources (eq. 2.1).

        Args:
            agents: List of agents.
            function: A function object.

        """

        # Calculates the fitness somatory
        total = sum(agent.fit for agent in agents)

        # Defines food sources' counter
        k = 0

        # While counter is less than the amount of food sources
        while k < len(agents):
            # We iterate through every agent
            for i, agent in enumerate(agents):
                # Creates a random uniform number
                r1 = r.generate_uniform_random_number()

                # Calculates the food source's probability
                probs = (agent.fit / (total + c.EPSILON)) + 0.1

                # If the random number is smaller than food source's probability
                if r1 < probs:
                    # We need to increment the counter
                    k += 1

                    # Gathers a random source to be used
                    source = r.generate_integer_random_number(0, len(agents))

                    # Evaluate its location
                    self._evaluate_location(agent, agents[source], function, i)

    def _send_scout(self, agents: List[Agent], function: Function) -> None:
        """Sends scout bees to scout for new possible food sources.

        Args:
            agents: List of agents.
            function: A function object.

        """

        # Calculates the maximum trial counter value and index
        max_trial, max_index = np.max(self.trial), np.argmax(self.trial)

        # If maximum trial is bigger than number of possible trials
        if max_trial > self.n_trials:
            # Resets the trial counter
            self.trial[max_index] = 0

            # Copies the current agent
            a = copy.deepcopy(agents[max_index])

            # Updates its position with a random shakeness
            a.position += r.generate_uniform_random_number(-1, 1)

            # Checks agent's limits
            a.clip_by_bound()

            # Recalculates its fitness
            a.fit = function(a.position)

            # If fitness is better
            if a.fit < agents[max_index].fit:
                # We copy the temporary agent to the current one
                agents[max_index] = copy.deepcopy(a)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Artificial Bee Colony over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Sends employee bees step
        self._send_employee(space.agents, function)

        # Sends onlooker bees step
        self._send_onlooker(space.agents, function)

        # Sends scout bees step
        self._send_scout(space.agents, function)
