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

        super(KH, self).__init__()

        self.N_max = 0.01
        self.w_n = 0.42

        self.NN = 5

        self.V_f = 0.02
        self.w_f = 0.38
        self.D_max = 0.002
        self.C_t = 0.5

        self.Cr = 0.2
        self.Mu = 0.05

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

        food = copy.deepcopy(agents[0])

        sum_fitness_pos = np.sum(
            [1 / (agent.fit + c.EPSILON) * agent.position for agent in agents], axis=0
        )
        sum_fitness = np.sum([1 / (agent.fit + c.EPSILON) for agent in agents])

        food.position = sum_fitness_pos / sum_fitness
        food.clip_by_bound()

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

        eucl_distance = [
            g.euclidean_distance(agents[idx].position, agent.position)
            for agent in agents
        ]
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

        neighbours = []

        for i, dist in enumerate(eucl_distance):
            if idx != i and sensing_distance > dist:
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

        fitness = [
            (agent.fit - neighbour.fit) / (worst.fit - best.fit + c.EPSILON)
            for neighbour in neighbours
        ]

        position = [
            (neighbour.position - agent.position)
            / (g.euclidean_distance(neighbour.position, agent.position) + c.EPSILON)
            for neighbour in neighbours
        ]

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

        fitness = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

        position = (best.position - agent.position) / (
            g.euclidean_distance(best.position, agent.position) + c.EPSILON
        )

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

        # Calculates the local alpha (eq. 4)
        neighbours = self._get_neighbours(agents, idx, sensing_distance, eucl_distance)
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

        fitness = (agent.fit - food.fit) / (worst.fit - best.fit + c.EPSILON)

        position = (food.position - agent.position) / (
            g.euclidean_distance(food.position, agent.position) + c.EPSILON
        )

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

        fitness = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

        position = (best.position - agent.position) / (
            g.euclidean_distance(best.position, agent.position) + c.EPSILON
        )

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

        # Calculates the physical diffusion (eq. 17)
        r1 = r.generate_uniform_random_number(-1, 1, size=(n_variables, n_dimensions))
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

        neighbour_motion = self._neighbour_motion(
            agents, idx, iteration, n_iterations, motion
        )

        foraging_motion = self._foraging_motion(
            agents, idx, iteration, n_iterations, food, foraging
        )

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

        a = copy.deepcopy(agents[idx])
        m = r.generate_integer_random_number(0, len(agents), exclude_value=idx)

        Cr = self.Cr * (
            (agents[idx].fit - agents[0].fit)
            / (agents[-1].fit - agents[0].fit + c.EPSILON)
        )

        for j in range(a.n_variables):
            r1 = r.generate_uniform_random_number()
            if r1 < Cr:
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

        a = copy.deepcopy(agents[idx])

        p = r.generate_integer_random_number(0, len(agents), exclude_value=idx)
        q = r.generate_integer_random_number(0, len(agents), exclude_value=idx)

        Mu = self.Mu / (
            (agents[idx].fit - agents[0].fit)
            / (agents[-1].fit - agents[0].fit + c.EPSILON)
            + c.EPSILON
        )

        for j in range(a.n_variables):
            r1 = r.generate_uniform_random_number()
            if r1 < Mu:
                r2 = r.generate_uniform_random_number()
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

        space.agents.sort(key=lambda x: x.fit)

        # Calculates the food location (eq. 12)
        food = self._food_location(space.agents, function)

        for i, _ in enumerate(space.agents):
            space.agents[i].position = self._update_position(
                space.agents,
                i,
                iteration,
                n_iterations,
                food,
                self.motion[i],
                self.foraging[i],
            )

            space.agents[i] = self._crossover(space.agents, i)
            space.agents[i] = self._mutation(space.agents, i)
