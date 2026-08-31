"""Jellyfish Search-based algorithms."""

from typing import Any, Dict, List, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class JS(Optimizer):
    """A JS class, inherited from Optimizer.

    This is the designed class to define JS-related
    variables and methods.

    References:
        J.-S. Chou and D.-N. Truong. A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean.
        Applied Mathematics and Computation (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(JS, self).__init__()

        self.eta = 4.0
        self.beta = 3.0
        self.gamma = 0.1

        self.build(params)

    def _initialize_chaotic_map(self, agents: List[Agent]) -> None:
        """Initializes a set of agents using a logistic chaotic map.

        Args:
            agents: List of agents.

        """

        for i, agent in enumerate(agents):
            if i == 0:
                for j in range(agent.n_variables):
                    agent.position[j] = np.random.uniform(0.0, 1.0, agent.n_dimensions)
            else:
                for j in range(agent.n_variables):
                    # Calculates its position using logistic chaotic map (eq. 18)
                    agent.position[j] = (
                        self.eta
                        * agents[i - 1].position[j]
                        * (1 - agents[i - 1].position[j])
                    )

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self._initialize_chaotic_map(space.agents)

    def _ocean_current(self, agents: List[Agent], best_agent: Agent) -> np.ndarray:
        """Calculates the ocean current (eq. 9).

        Args:
            agents: List of agents.
            best_agent: Best agent.

        Returns:
            (np.ndarray): A trend value for the ocean current.

        """

        r1 = np.random.uniform(0.0, 1.0, 1)
        u = np.mean([agent.position for agent in agents])

        # Calculates the ocean current (eq. 9)
        trend = best_agent.position - self.beta * r1 * u

        return trend

    def _motion_a(self, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Calculates type A motion (eq. 12).

        Args:
            lb: Array of lower bounds.
            ub: Array of upper bounds.

        Returns:
            (np.ndarray): A type A motion array.

        """

        r1 = np.random.uniform(0.0, 1.0, 1)
        motion = self.gamma * r1 * (np.expand_dims(ub, -1) - np.expand_dims(lb, -1))

        return motion

    def _motion_b(self, agent_i: Agent, agent_j: Agent) -> np.ndarray:
        """Calculates type B motion (eq. 15).

        Args:
            agent_i: Current agent to be updated.
            agent_j: Selected agent.

        Returns:
            (np.ndarray): A type B motion array.

        """

        r1 = np.random.uniform(0.0, 1.0, 1)

        if agent_i.fit >= agent_j.fit:
            # Determines its direction (eq. 15 - top)
            d = agent_j.position - agent_i.position
        else:
            # Determines its direction (eq. 15 - bottom)
            d = agent_i.position - agent_j.position

        motion = r1 * d

        return motion

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Jellyfish Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        for agent in space.agents:
            r1 = np.random.uniform(0.0, 1.0, 1)

            # Calculates the time control mechanism (eq. 17)
            c = np.fabs((1 - iteration / n_iterations) * (2 * r1 - 1))

            if c >= 0.5:
                # Calculates the ocean current (eq. 9)
                trend = self._ocean_current(space.agents, space.best_agent)

                # Updates the location of current jellyfish (eq. 11)
                r2 = np.random.uniform(0.0, 1.0, 1)
                agent.position += r2 * trend
            else:
                r2 = np.random.uniform(0.0, 1.0, 1)
                if r2 > (1 - c):
                    # Update jellyfish's location with type A motion (eq. 12)
                    agent.position += self._motion_a(agent.lb, agent.ub)
                else:
                    # Updates jellyfish's location with type B motion (eq. 16)
                    j = np.random.randint(0, len(space.agents), None)
                    agent.position += self._motion_b(agent, space.agents[j])
            agent.clip_by_bound()


class NBJS(JS):
    """An NBJS class, inherited from JS.

    This is the designed class to define NBJS-related
    variables and methods.

    References:
        Publication pending.

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(NBJS, self).__init__(params)

    def _motion_a(self, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Calculates type A motion.

        Args:
            lb: Array of lower bounds.
            ub: Array of upper bounds.

        Returns:
            (np.ndarray): A type A motion array.

        """

        r1 = np.random.uniform(0.0, 1.0, 1)
        motion = self.gamma * r1

        return motion
