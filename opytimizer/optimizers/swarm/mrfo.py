"""Manta Ray Foraging Optimization.
"""

import copy
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


class MRFO(Optimizer):
    """An MRFO class, inherited from Optimizer.

    This is the designed class to define MRFO-related
    variables and methods.

    References:
        W. Zhao, Z. Zhang and L. Wang.
        Manta Ray Foraging Optimization: An effective bio-inspired optimizer for engineering applications.
        Engineering Applications of Artificial Intelligence (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> MRFO.")

        # Overrides its parent class with the receiving params
        super(MRFO, self).__init__()

        # Somersault foraging
        self.S = 2.0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def S(self) -> float:
        """Somersault foraging."""

        return self._S

    @S.setter
    def S(self, S: float) -> None:
        if not isinstance(S, (float, int)):
            raise e.TypeError("`S` should be a float or integer")
        if S < 0:
            raise e.ValueError("`S` should be >= 0")

        self._S = S

    def _cyclone_foraging(
        self,
        agents: List[Agent],
        best_position: np.ndarray,
        i: int,
        iteration: int,
        n_iterations: int,
    ) -> np.ndarray:
        """Performs the cyclone foraging procedure (eq. 3-7).

        Args:
            agents: List of agents.
            best_position: Global best position.
            i: Index of current manta ray.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        Returns:
            (np.ndarray): A new cyclone foraging.

        """

        # Generates uniform random numbers
        r1 = r.generate_uniform_random_number()
        r2 = r.generate_uniform_random_number()
        r3 = r.generate_uniform_random_number()

        # Calculates the beta constant
        beta = (
            2
            * np.exp(r1 * (n_iterations - iteration + 1) / n_iterations)
            * np.sin(2 * np.pi * r1)
        )

        # Check if current iteration proportion is smaller than random generated number
        if iteration / n_iterations < r2:
            # Generates an array for holding the random position
            r_position = np.zeros((agents[i].n_variables, agents[i].n_dimensions))

            # For every decision variable
            for j, (lb, ub) in enumerate(zip(agents[i].lb, agents[i].ub)):
                # Generates uniform random positions
                r_position[j] = r.generate_uniform_random_number(
                    lb, ub, size=agents[i].n_dimensions
                )

            # Checks if the index is equal to zero
            if i == 0:
                cyclone_foraging = (
                    r_position
                    + r3 * (r_position - agents[i].position)
                    + beta * (r_position - agents[i].position)
                )

            # If index is different than zero
            else:
                cyclone_foraging = (
                    r_position
                    + r3 * (agents[i - 1].position - agents[i].position)
                    + beta * (r_position - agents[i].position)
                )

        # If current iteration proportion is bigger than random generated number
        else:
            # Checks if the index is equal to zero
            if i == 0:
                cyclone_foraging = (
                    best_position
                    + r3 * (best_position - agents[i].position)
                    + beta * (best_position - agents[i].position)
                )

            # If index is different than zero
            else:
                cyclone_foraging = (
                    best_position
                    + r3 * (agents[i - 1].position - agents[i].position)
                    + beta * (best_position - agents[i].position)
                )

        return cyclone_foraging

    def _chain_foraging(
        self, agents: List[Agent], best_position: np.ndarray, i: int
    ) -> np.ndarray:
        """Performs the chain foraging procedure (eq. 1-2).

        Args:
            agents: List of agents.
            best_position: Global best position.
            i: Index of current manta ray.

        Returns:
            (np.ndarray): A new chain foraging.

        """

        # Generates uniform random numbers
        r1 = r.generate_uniform_random_number()
        r2 = r.generate_uniform_random_number()

        # Calculates the alpha constant
        alpha = 2 * r1 * np.sqrt(np.abs(np.log(r1)))

        # Checks if the index is equal to zero
        if i == 0:
            # If yes, uses this equation
            chain_foraging = (
                agents[i].position
                + r2 * (best_position - agents[i].position)
                + alpha * (best_position - agents[i].position)
            )

        # If index is different than zero
        else:
            # Uses this equation
            chain_foraging = (
                agents[i].position
                + r2 * (agents[i - 1].position - agents[i].position)
                + alpha * (best_position - agents[i].position)
            )

        return chain_foraging

    def _somersault_foraging(
        self, position: np.ndarray, best_position: np.ndarray
    ) -> np.ndarray:
        """Performs the somersault foraging procedure (eq. 8).

        Args:
            position: Agent's current position.
            best_position: Global best position.

        Returns:
            (np.ndarray): A new somersault foraging.

        """

        # Generates uniform random numbers
        r1 = r.generate_uniform_random_number()
        r2 = r.generate_uniform_random_number()

        # Calculates the somersault foraging
        somersault_foraging = position + self.S * (r1 * best_position - r2 * position)

        return somersault_foraging

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Manta Ray Foraging Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than 1/2
            if r1 < 0.5:
                # Performs the cyclone foraging
                agent.position = self._cyclone_foraging(
                    space.agents, space.best_agent.position, i, iteration, n_iterations
                )

            # If random number is bigger than 1/2
            else:
                # Performs the chain foraging
                agent.position = self._chain_foraging(
                    space.agents, space.best_agent.position, i
                )

            # Clips the agent's limits
            agent.clip_by_bound()

            # Evaluates the agent
            agent.fit = function(agent.position)

            # If new agent's fitness is better than best
            if agent.fit < space.best_agent.fit:
                # Replace the best agent's position and fitness with its copy
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)

        # Iterates through all agents
        for agent in space.agents:
            # Performs the somersault foraging
            agent.position = self._somersault_foraging(
                agent.position, space.best_agent.position
            )
