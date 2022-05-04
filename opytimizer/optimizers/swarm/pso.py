"""Particle Swarm Optimization-based algorithms.
"""

import copy
import time
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


class PSO(Optimizer):
    """A PSO class, inherited from Optimizer.

    This is the designed class to define PSO-related
    variables and methods.

    References:
        J. Kennedy, R. C. Eberhart and Y. Shi. Swarm intelligence.
        Artificial Intelligence (2001).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> PSO.")

        # Overrides its parent class with the receiving params
        super(PSO, self).__init__()

        # Inertia weight
        self.w = 0.7

        # Cognitive constant
        self.c1 = 1.7

        # Social constant
        self.c2 = 1.7

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def w(self) -> float:
        """Inertia weight."""

        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise e.TypeError("`w` should be a float or integer")
        if w < 0:
            raise e.ValueError("`w` should be >= 0")

        self._w = w

    @property
    def c1(self) -> float:
        """Cognitive constant."""

        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        if not isinstance(c1, (float, int)):
            raise e.TypeError("`c1` should be a float or integer")
        if c1 < 0:
            raise e.ValueError("`c1` should be >= 0")

        self._c1 = c1

    @property
    def c2(self) -> float:
        """Social constant."""

        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        if not isinstance(c2, (float, int)):
            raise e.TypeError("`c2` should be a float or integer")
        if c2 < 0:
            raise e.ValueError("`c2` should be >= 0")

        self._c2 = c2

    @property
    def local_position(self) -> np.ndarray:
        """Array of velocities."""

        return self._local_position

    @local_position.setter
    def local_position(self, local_position: np.ndarray) -> None:
        if not isinstance(local_position, np.ndarray):
            raise e.TypeError("`local_position` should be a numpy array")

        self._local_position = local_position

    @property
    def velocity(self) -> np.ndarray:
        """Array of velocities."""

        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        if not isinstance(velocity, np.ndarray):
            raise e.TypeError("`velocity` should be a numpy array")

        self._velocity = velocity

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of local positions and velocities
        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the fitness value of current agent
            fit = function(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the local best position to current's agent position
                self.local_position[i] = copy.deepcopy(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's local best position and fitness to the best agent
                space.best_agent.position = copy.deepcopy(self.local_position[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        """Wraps Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates random numbers
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates agent's velocity (p. 294)
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates agent's position (p. 294)
            agent.position += self.velocity[i]


class AIWPSO(PSO):
    """An AIWPSO class, inherited from PSO.

    This is the designed class to define AIWPSO-related
    variables and methods.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
        A novel particle swarm optimization algorithm with adaptive inertia weight.
        Applied Soft Computing (2011).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> AIWPSO.")

        # Minimum inertia weight
        self.w_min = 0.1

        # Maximum inertia weight
        self.w_max = 0.9

        # Overrides its parent class with the receiving params
        super(AIWPSO, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def w_min(self) -> float:
        """Minimum inertia weight."""

        return self._w_min

    @w_min.setter
    def w_min(self, w_min: float) -> None:
        if not isinstance(w_min, (float, int)):
            raise e.TypeError("`w_min` should be a float or integer")
        if w_min < 0:
            raise e.ValueError("`w_min` should be >= 0")

        self._w_min = w_min

    @property
    def w_max(self) -> float:
        """Maximum inertia weight."""

        return self._w_max

    @w_max.setter
    def w_max(self, w_max: float) -> None:
        if not isinstance(w_max, (float, int)):
            raise e.TypeError("`w_max` should be a float or integer")
        if w_max < 0:
            raise e.ValueError("`w_max` should be >= 0")
        if w_max < self.w_min:
            raise e.ValueError("`w_max` should be >= `w_min`")

        self._w_max = w_max

    @property
    def fitness(self) -> List[float]:
        """List of fitnesses."""

        return self._fitness

    @fitness.setter
    def fitness(self, fitness: List[float]) -> None:
        if not isinstance(fitness, list):
            raise e.TypeError("`fitness` should be a list")

        self._fitness = fitness

    def _compute_success(self, agents: List[Agent]) -> None:
        """Computes the particles' success for updating inertia weight (eq. 16).

        Args:
            agents: List of agents.

        """

        # Initial counter
        p = 0

        # Iterates through every agent
        for i, agent in enumerate(agents):
            # If current agent fitness is smaller than its best
            if agent.fit < self.fitness[i]:
                # Increments the counter
                p += 1

            # Replaces fitness with current agent's fitness
            self.fitness[i] = agent.fit

        # Update inertia weight value
        self.w = (self.w_max - self.w_min) * (p / len(agents)) + self.w_min

    def update(self, space: Space, iteration: int) -> None:
        """Wraps Adaptive Inertia Weight Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.

        """

        # Checks if it is the first iteration
        if iteration == 0:
            # Creates a list of initial fitnesses
            self.fitness = [agent.fit for agent in space.agents]

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates random numbers
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates agent's velocity
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates agent's position
            agent.position += self.velocity[i]

        # Computing particle's success and updating inertia weight
        self._compute_success(space.agents)


class RPSO(PSO):
    """An RPSO class, inherited from Optimizer.

    This is the designed class to define RPSO-related
    variables and methods.

    References:
        M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
        Harnessing Particle Swarm Optimization Through Relativistic Velocity.
        IEEE Congress on Evolutionary Computation (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> RPSO.")

        # Overrides its parent class with the receiving params
        super(RPSO, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def mass(self) -> np.ndarray:
        """Array of masses."""

        return self._mass

    @mass.setter
    def mass(self, mass: np.ndarray) -> None:
        if not isinstance(mass, np.ndarray):
            raise e.TypeError("`mass` should be a numpy array")

        self._mass = mass

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of local positions, velocities and masses
        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.mass = r.generate_uniform_random_number(
            size=(space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space) -> None:
        """Wraps Relativistic Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Calculates the maximum velocity
        max_velocity = np.max(self.velocity)

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates rnadom number
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates current agent velocity (eq. 11)
            gamma = 1 / np.sqrt(1 - (max_velocity**2 / c.LIGHT_SPEED**2))
            self.velocity[i] = (
                self.mass[i] * self.velocity[i] * gamma
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates current agent position
            agent.position += self.velocity[i]


class SAVPSO(PSO):
    """An SAVPSO class, inherited from Optimizer.

    This is the designed class to define SAVPSO-related
    variables and methods.

    References:
        H. Lu and W. Chen.
        Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
        Journal of global optimization (2008).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> SAVPSO.")

        # Overrides its parent class with the receiving params
        super(SAVPSO, self).__init__(params)

        logger.info("Class overrided.")

    def update(self, space: Space) -> None:
        """Wraps Self-adaptive Velocity Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Creates an array of positions
        positions = np.zeros(
            (space.agents[0].position.shape[0], space.agents[0].position.shape[1])
        )

        # For every agent
        for agent in space.agents:
            # Sums up its position
            positions += agent.position

        # Divides by the number of agents
        positions /= len(space.agents)

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates a random index for selecting an agent
            idx = r.generate_integer_random_number(0, len(space.agents))

            # Updates current agent's velocity (eq. 8)
            r1 = r.generate_uniform_random_number()
            self.velocity[i] = (
                self.w
                * np.fabs(self.local_position[idx] - self.local_position[i])
                * np.sign(self.velocity[i])
                + r1 * (self.local_position[i] - agent.position)
                + (1 - r1) * (space.best_agent.position - agent.position)
            )

            # Updates current agent's position
            agent.position += self.velocity[i]

            # For every decision variable
            for j in range(agent.n_variables):
                # Generates a random number
                r4 = r.generate_uniform_random_number(0, 1)

                # If position is greater than upper bound
                if agent.position[j] > agent.ub[j]:
                    # Replaces its value
                    agent.position[j] = positions[j] + 1 * r4 * (
                        agent.ub[j] - positions[j]
                    )

                # If position is smaller than lower bound
                if agent.position[j] < agent.lb[j]:
                    # Replaces its value
                    agent.position[j] = positions[j] + 1 * r4 * (
                        agent.lb[j] - positions[j]
                    )


class VPSO(PSO):
    """A VPSO class, inherited from Optimizer.

    This is the designed class to define VPSO-related
    variables and methods.

    References:
        W.-P. Yang. Vertical particle swarm optimization algorithm and its application in soft-sensor modeling.
        International Conference on Machine Learning and Cybernetics (2007).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: PSO -> VPSO.")

        # Overrides its parent class with the receiving params
        super(VPSO, self).__init__(params)

        logger.info("Class overrided.")

    @property
    def v_velocity(self) -> np.ndarray:
        """Array of vertical velocities."""

        return self._v_velocity

    @v_velocity.setter
    def v_velocity(self, v_velocity: np.ndarray) -> None:
        if not isinstance(v_velocity, np.ndarray):
            raise e.TypeError("`v_velocity` should be a numpy array")

        self._v_velocity = v_velocity

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of local positions, velocities and vertical velocities
        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.v_velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space) -> None:
        """Wraps Vertical Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates uniform random numbers
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates current agent velocity (eq. 3)
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates current agent vertical velocity (eq. 4)
            self.v_velocity[i] -= (
                np.dot(self.velocity[i].T, self.v_velocity[i])
                / (np.dot(self.velocity[i].T, self.velocity[i]) + c.EPSILON)
            ) * self.velocity[i]

            # Updates current agent position (eq. 5)
            r1 = r.generate_uniform_random_number()
            agent.position += r1 * self.velocity[i] + (1 - r1) * self.v_velocity[i]
