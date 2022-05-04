"""Krill Herd.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class KH(Optimizer):
    """A KH class, inherited from Optimizer.

    This is the designed class to define KH-related
    variables and methods.

    References:
        A. Gandomi and A. Alavi. Krill herd: A new bio-inspired optimization algorithm.
        Communications in Nonlinear Science and Numerical Simulation (2012).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(KH, self).__init__()

        # Maximum induced speed
        self.N_max = 0.01

        # Inertia weight of the neighbours' motion
        self.w_n = 0.42

        # Number of neighbours
        self.NN = 5

        # Foraging speed
        self.V_f = 0.02

        # Inertia weight of the foraging motion
        self.w_f = 0.38

        # Maximum diffusion speed
        self.D_max = 0.002

        # Position constant
        self.C_t = 0.5

        # Crossover rate
        self.Cr = 0.2

        # Mutation probability
        self.Mu = 0.05

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def N_max(self) -> float:
        """Maximum induced speed."""

        return self._N_max

    @N_max.setter
    def N_max(self, N_max: float) -> None:
        if not isinstance(N_max, (float, int)):
            raise e.TypeError("`N_max` should be a float or integer")
        if N_max < 0:
            raise e.ValueError("`N_max` should be >= 0")

        self._N_max = N_max

    @property
    def w_n(self) -> float:
        """Inertia weight of the neighbours' motion."""

        return self._w_n

    @w_n.setter
    def w_n(self, w_n: float) -> None:
        if not isinstance(w_n, (float, int)):
            raise e.TypeError("`w_n` should be a float or integer")
        if w_n < 0 or w_n > 1:
            raise e.ValueError("`w_n` should be between 0 and 1")

        self._w_n = w_n

    @property
    def NN(self) -> int:
        """Number of neighbours."""

        return self._NN

    @NN.setter
    def NN(self, NN: int) -> None:
        if not isinstance(NN, int):
            raise e.TypeError("`NN` should be a integer")
        if NN < 0:
            raise e.ValueError("`NN` should be >= 0")

        self._NN = NN

    @property
    def V_f(self) -> float:
        """Foraging speed."""

        return self._V_f

    @V_f.setter
    def V_f(self, V_f: float) -> None:
        if not isinstance(V_f, (float, int)):
            raise e.TypeError("`V_f` should be a float or integer")
        if V_f < 0:
            raise e.ValueError("`V_f` should be >= 0")

        self._V_f = V_f

    @property
    def w_f(self) -> float:
        """Inertia weight of the foraging motion."""

        return self._w_f

    @w_f.setter
    def w_f(self, w_f: float) -> None:
        if not isinstance(w_f, (float, int)):
            raise e.TypeError("`w_f` should be a float or integer")
        if w_f < 0 or w_f > 1:
            raise e.ValueError("`w_f` should be between 0 and 1")

        self._w_f = w_f

    @property
    def D_max(self) -> float:
        """Maximum diffusion speed."""

        return self._D_max

    @D_max.setter
    def D_max(self, D_max: float) -> None:
        if not isinstance(D_max, (float, int)):
            raise e.TypeError("`D_max` should be a float or integer")
        if D_max < 0:
            raise e.ValueError("`D_max` should be >= 0")

        self._D_max = D_max

    @property
    def C_t(self) -> float:
        """Position constant."""

        return self._C_t

    @C_t.setter
    def C_t(self, C_t: float) -> None:
        if not isinstance(C_t, (float, int)):
            raise e.TypeError("`C_t` should be a float or integer")
        if C_t < 0 or C_t > 2:
            raise e.ValueError("`C_t` should be between 0 and 2")

        self._C_t = C_t

    @property
    def Cr(self) -> float:
        """Crossover probability."""

        return self._Cr

    @Cr.setter
    def Cr(self, Cr: float) -> None:
        if not isinstance(Cr, (float, int)):
            raise e.TypeError("`Cr` should be a float or integer")
        if Cr < 0 or Cr > 1:
            raise e.ValueError("`Cr` should be between 0 and 1")

        self._Cr = Cr

    @property
    def Mu(self) -> float:
        """Mutation probability."""

        return self._Mu

    @Mu.setter
    def Mu(self, Mu: float) -> None:
        if not isinstance(Mu, (float, int)):
            raise e.TypeError("`Mu` should be a float or integer")
        if Mu < 0 or Mu > 1:
            raise e.ValueError("`Mu` should be between 0 and 1")

        self._Mu = Mu

    @property
    def motion(self) -> np.ndarray:
        """Array of motions."""

        return self._motion

    @motion.setter
    def motion(self, motion: np.ndarray) -> None:
        if not isinstance(motion, np.ndarray):
            raise e.TypeError("`motion` should be a numpy array")

        self._motion = motion

    @property
    def foraging(self) -> np.ndarray:
        """Array of foragings."""

        return self._foraging

    @foraging.setter
    def foraging(self, foraging: np.ndarray) -> None:
        if not isinstance(foraging, np.ndarray):
            raise e.TypeError("`foraging` should be a numpy array")

        self._foraging = foraging

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of motions and foragings
        self.motion = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))
        self.foraging = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _food_location(self, agents: List[Agent], function: Function) -> Agent:
        """Calculates the food location.

        Args:
            agents: List of agents.
            function: A Function object that will be used as the objective function.

        Returns:
           (Agent): A new food location.

        """

        # Making a deepcopy of an agent
        food = copy.deepcopy(agents[0])

        # Calculates the sum of inverse of agents' fitness * agents' position
        sum_fitness_pos = np.sum(
            [1 / (agent.fit + c.EPSILON) * agent.position for agent in agents], axis=0
        )

        # Calculates the sum of inverse of agents' fitness
        sum_fitness = np.sum([1 / (agent.fit + c.EPSILON) for agent in agents])

        # Calculates the new food's position
        food.position = sum_fitness_pos / sum_fitness

        # Clips the food's position
        food.clip_by_bound()

        # Evaluates the food
        food.fit = function(food.position)

        return food

    def _sensing_distance(self, agents: List[Agent], idx: int) -> Tuple[float, float]:
        """Calculates the sensing distance for an individual krill (eq. 7).

        Args:
            agents: List of agents.
            idx: Selected agent.

        Returns:
            (Tuple[float, float]): The sensing distance for an individual krill.

        """

        # Calculates the euclidean distances between selected krill and other krills
        eucl_distance = [
            g.euclidean_distance(agents[idx].position, agent.position)
            for agent in agents
        ]

        # Calculates the sensing distance
        distance = np.sum(eucl_distance) / (self.NN * len(agents))

        return distance, eucl_distance

    def _get_neighbours(
        self,
        agents: List[Agent],
        idx: int,
        sensing_distance: float,
        eucl_distance: List[float],
    ) -> List[Agent]:
        """Gathers the neighbours based on the sensing distance.

        Args:
            agents: List of agents.
            idx: Selected agent.
            sensing_distance: Sensing distanced used to gather the krill's neighbours.
            eucl_distance: List of euclidean distances.

        Returns:
            (List[Agent]): A list containing the krill's neighbours.

        """

        # Creates a list to hold the neighbours
        neighbours = []

        # Iterates through all agents and euclidean distances
        for i, dist in enumerate(eucl_distance):
            # If selected agent is different from current agent
            # and the sensing distance is greather than its euclidean distance
            if idx != i and sensing_distance > dist:
                # Appends the agent to the neighbours' list
                neighbours.append(agents[i])

        return neighbours

    def _local_alpha(
        self, agent: Agent, worst: Agent, best: Agent, neighbours: List[Agent]
    ) -> float:
        """Calculates the local alpha (eq. 4).

        Args:
            agent: Selected agent.
            worst: Worst agent.
            best: Best agent.
            neighbours: List of neighbours.

        Returns:
            (float): The local alpha.

        """

        # Calculates a list of neighbours' fitness
        fitness = [
            (agent.fit - neighbour.fit) / (worst.fit - best.fit + c.EPSILON)
            for neighbour in neighbours
        ]

        # Calculates a list of krills' position based on neighbours
        position = [
            (neighbour.position - agent.position)
            / (g.euclidean_distance(neighbour.position, agent.position) + c.EPSILON)
            for neighbour in neighbours
        ]

        # Calculates the local alpha
        alpha = np.sum([fit * pos for (fit, pos) in zip(fitness, position)], axis=0)

        return alpha

    def _target_alpha(
        self, agent: Agent, worst: Agent, best: Agent, C_best: float
    ) -> float:
        """Calculates the target alpha (eq. 8).

        Args:
            agent: Selected agent.
            worst: Worst agent.
            best: Best agent.
            C_best: Effectiveness coefficient.

        Returns:
            (float): The target alpha.

        """

        # Calculates a list of neighbours' fitness
        fitness = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

        # Calculates a list of krills' position based on neighbours
        position = (best.position - agent.position) / (
            g.euclidean_distance(best.position, agent.position) + c.EPSILON
        )

        # Calculates the target alpha
        alpha = C_best * fitness * position

        return alpha

    def _neighbour_motion(
        self,
        agents: List[Agent],
        idx: int,
        iteration: int,
        n_iterations: int,
        motion: np.ndarray,
    ) -> np.ndarray:
        """Performs the motion induced by other krill individuals (eq. 2).

        Args:
            agents: List of agents.
            idx: Selected agent.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.
            motion: Array of motions.

        Returns:
            (np.ndarray): The krill's neighbour motion.

        """

        # Calculates the sensing distance (eq. 7)
        sensing_distance, eucl_distance = self._sensing_distance(agents, idx)

        # Gathers the neighbours
        neighbours = self._get_neighbours(agents, idx, sensing_distance, eucl_distance)

        # Calculates the local alpha (eq. 4)
        alpha_l = self._local_alpha(agents[idx], agents[-1], agents[0], neighbours)

        # Calculates the effective coefficient (eq. 9)
        C_best = 2 * (r.generate_uniform_random_number() + iteration / n_iterations)

        # Calculates the target alpha (eq. 8)
        alpha_t = self._target_alpha(agents[idx], agents[-1], agents[0], C_best)

        # Calculates the neighbour motion (eq. 2)
        neighbour_motion = self.N_max * (alpha_l + alpha_t) + self.w_n * motion

        return neighbour_motion

    def _food_beta(
        self, agent: Agent, worst: Agent, best: Agent, food: np.ndarray, C_food: float
    ) -> np.ndarray:
        """Calculates the food attraction (eq. 13).

        Args:
            agent: Selected agent.
            worst: Worst agent.
            best: Best agent.
            food: Food location.
            C_food: Food coefficient.

        Returns:
            (np.ndarray): The food attraction.

        """

        # Calculates the fitness
        fitness = (agent.fit - food.fit) / (worst.fit - best.fit + c.EPSILON)

        # Calculates the positioning
        position = (food.position - agent.position) / (
            g.euclidean_distance(food.position, agent.position) + c.EPSILON
        )

        # Calculates the food attraction
        beta = C_food * fitness * position

        return beta

    def _best_beta(self, agent: Agent, worst: Agent, best: Agent) -> np.ndarray:
        """Calculates the best attraction (eq. 15).

        Args:
            agent: Selected agent.
            worst: Worst agent.
            best: Best agent.

        Returns:
            (np.ndarray): The best attraction.

        """

        # Calculates the fitness
        fitness = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

        # Calculates the positioning
        position = (best.position - agent.position) / (
            g.euclidean_distance(best.position, agent.position) + c.EPSILON
        )

        # Calculates the food attraction
        beta = fitness * position

        return beta

    def _foraging_motion(
        self,
        agents: List[Agent],
        idx: int,
        iteration: int,
        n_iterations: int,
        food: np.ndarray,
        foraging: np.ndarray,
    ) -> np.ndarray:
        """Performs the foraging induced by the food location (eq. 10).

        Args:
            agents: List of agents.
            idx: Selected agent.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.
            food: Food location.
            foraging: Array of foraging motions.

        Returns:
            (np.ndarray): The krill's foraging motion.

        """

        # Calculates the food coefficient (eq. 14)
        C_food = 2 * (1 - iteration / n_iterations)

        # Calculates the food attraction (eq. 13)
        beta_f = self._food_beta(agents[idx], agents[-1], agents[0], food, C_food)

        # Calculates the best attraction (eq. 15)
        beta_b = self._best_beta(agents[idx], agents[-1], agents[0])

        # Calculates the foraging motion (eq. 10)
        foraging_motion = self.V_f * (beta_f + beta_b) + self.w_f * foraging

        return foraging_motion

    def _physical_diffusion(
        self, n_variables: int, n_dimensions: int, iteration: int, n_iterations: int
    ) -> float:
        """Performs the physical diffusion of individual krills (eq. 16-17).

        Args:
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        Returns:
            (float): The physical diffusion.

        """

        # Generates uniform random numbers
        r1 = r.generate_uniform_random_number(-1, 1, size=(n_variables, n_dimensions))

        # Calculates the physical diffusion (eq. 17)
        physical_diffusion = self.D_max * (1 - iteration / n_iterations) * r1

        return physical_diffusion

    def _update_position(
        self,
        agents: List[Agent],
        idx: int,
        iteration: int,
        n_iterations: int,
        food: np.ndarray,
        motion: np.ndarray,
        foraging: np.ndarray,
    ) -> np.ndarray:
        """Updates a single krill position (eq. 18-19).

        Args:
            agents: List of agents.
            idx: Selected agent.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.
            food: Food location.
            motion: Array of motions.
            foraging: Array of foraging motions.

        Returns:
            (np.ndarray): The updated position.

        """

        # Calculates the neighbour motion
        neighbour_motion = self._neighbour_motion(
            agents, idx, iteration, n_iterations, motion
        )

        # Calculates the foraging motion
        foraging_motion = self._foraging_motion(
            agents, idx, iteration, n_iterations, food, foraging
        )

        # Calculates the physical diffusion
        physical_diffusion = self._physical_diffusion(
            agents[idx].n_variables, agents[idx].n_dimensions, iteration, n_iterations
        )

        # Calculates the delta (eq. 19)
        delta_t = self.C_t * np.sum(agents[idx].ub - agents[idx].lb)

        # Updates the current agent's position (eq. 18)
        new_position = agents[idx].position + delta_t * (
            neighbour_motion + foraging_motion + physical_diffusion
        )

        return new_position

    def _crossover(self, agents: List[Agent], idx: int) -> Agent:
        """Performs the crossover between selected agent and a randomly agent (eq. 21).

        Args:
            agents: List of agents.
            idx: Selected agent.

        Returns:
            (Agent): An agent after suffering a crossover operator.

        """

        # Makes a deep copy of an agent
        a = copy.deepcopy(agents[idx])

        # Samples a random integer
        m = r.generate_integer_random_number(0, len(agents), exclude_value=idx)

        # Calculates the current crossover probability
        Cr = self.Cr * (
            (agents[idx].fit - agents[0].fit)
            / (agents[-1].fit - agents[0].fit + c.EPSILON)
        )

        # Iterates through all variables
        for j in range(a.n_variables):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If sampled uniform number if smaller than crossover probability
            if r1 < Cr:
                # Gathers the position from the selected agent
                a.position[j] = copy.deepcopy(agents[m].position[j])

        return a

    def _mutation(self, agents: List[Agent], idx: int) -> Agent:
        """Performs the mutation between selected agent and randomly agents (eq. 22).

        Args:
            agents: List of agents.
            idx: Selected agent.

        Returns:
            (Agent): An agent after suffering a mutation operator.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agents[idx])

        # Samples random integers
        p = r.generate_integer_random_number(0, len(agents), exclude_value=idx)
        q = r.generate_integer_random_number(0, len(agents), exclude_value=idx)

        # Calculates the current mutation probability
        Mu = self.Mu / (
            (agents[idx].fit - agents[0].fit)
            / (agents[-1].fit - agents[0].fit + c.EPSILON)
            + c.EPSILON
        )

        # Iterates through all variables
        for j in range(a.n_variables):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If sampled uniform number if smaller than mutation probability
            if r1 < Mu:
                # Generates another uniform random number
                r2 = r.generate_uniform_random_number()

                # Mutates the current position
                a.position[j] = agents[0].position[j] + r2 * (
                    agents[p].position[j] - agents[q].position[j]
                )

        return a

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps motion and genetic updates over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Calculates the food location (eq. 12)
        food = self._food_location(space.agents, function)

        # Iterates through all agents
        for i, _ in enumerate(space.agents):
            # Updates current agent's position
            space.agents[i].position = self._update_position(
                space.agents,
                i,
                iteration,
                n_iterations,
                food,
                self.motion[i],
                self.foraging[i],
            )

            # Performs the crossover and mutation
            space.agents[i] = self._crossover(space.agents, i)
            space.agents[i] = self._mutation(space.agents, i)
