"""Lion Optimization Algorithm.
"""

import copy
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Agent, Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Lion(Agent):
    """A Lion class complements its inherited parent with additional information neeeded by
    the Lion Optimization Algorithm.

    """

    def __init__(
        self,
        n_variables: int,
        n_dimensions: int,
        lower_bound: Union[List, Tuple, np.ndarray],
        upper_bound: Union[List, Tuple, np.ndarray],
        position: np.ndarray,
        fit: float,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            position: Position array.
            fit: Fitness value.

        """

        # Overrides its parent class with the receiving params
        super(Lion, self).__init__(n_variables, n_dimensions, lower_bound, upper_bound)

        # Copies the current position and fitness to overrided object
        self.position = copy.deepcopy(position)
        self.fit = copy.deepcopy(fit)

        # Best position
        self.best_position = copy.deepcopy(position)

        # Previous fitness
        self.p_fit = copy.deepcopy(fit)

        # Whether lion is nomad or not
        self.nomad = False

        # Whether lion is female or not
        self.female = False

        # Index of pride
        self.pride = 0

        # Index of hunting group
        self.group = 0

    @property
    def best_position(self) -> np.ndarray:
        """N-dimensional array of best positions."""

        return self._best_position

    @best_position.setter
    def best_position(self, best_position: np.ndarray) -> None:
        if not isinstance(best_position, np.ndarray):
            raise e.TypeError("`best_position` should be a numpy array")

        self._best_position = best_position

    @property
    def p_fit(self) -> float:
        """Previous fitness value."""

        return self._p_fit

    @p_fit.setter
    def p_fit(self, p_fit: float) -> None:
        if not isinstance(p_fit, (float, int, np.int32, np.int64)):
            raise e.TypeError("`p_fit` should be a float or integer")

        self._p_fit = p_fit

    @property
    def nomad(self) -> bool:
        """bool: Whether lion is nomad or not."""

        return self._nomad

    @nomad.setter
    def nomad(self, nomad: bool) -> None:
        if not isinstance(nomad, bool):
            raise e.TypeError("`nomad` should be a boolean")

        self._nomad = nomad

    @property
    def female(self) -> bool:
        """Whether lion is female or not."""

        return self._female

    @female.setter
    def female(self, female: bool) -> None:
        if not isinstance(female, bool):
            raise e.TypeError("`female` should be a boolean")

        self._female = female

    @property
    def pride(self) -> int:
        """Index of pride."""

        return self._pride

    @pride.setter
    def pride(self, pride: int) -> None:
        if not isinstance(pride, int):
            raise e.TypeError("`pride` should be an integer")
        if pride < 0:
            raise e.ValueError("`pride` should be > 0")

        self._pride = pride

    @property
    def group(self) -> int:
        """Index of hunting group."""

        return self._group

    @group.setter
    def group(self, group: int) -> None:
        if not isinstance(group, int):
            raise e.TypeError("`group` should be an integer")
        if group < 0:
            raise e.ValueError("`group` should be > 0")

        self._group = group


class LOA(Optimizer):
    """An LOA class, inherited from Optimizer.

    This is the designed class to define LOA-related
    variables and methods.

    References:
        M. Yazdani and F. Jolai. Lion Optimization Algorithm (LOA): A nature-inspired metaheuristic algorithm.
        Journal of Computational Design and Engineering (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> LOA.")

        # Overrides its parent class with the receiving params
        super(LOA, self).__init__()

        # Percentage of nomad lions
        self.N = 0.2

        # Number of prides
        self.P = 4

        # Percentage of female lions
        self.S = 0.8

        # Percentage of roaming lions
        self.R = 0.2

        # Immigrate rate
        self.I = 0.4

        # Mating probability
        self.Ma = 0.3

        # Mutation probability
        self.Mu = 0.2

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def N(self) -> float:
        """Percentage of nomad lions."""

        return self._N

    @N.setter
    def N(self, N: float) -> None:
        if not isinstance(N, (float, int)):
            raise e.TypeError("`N` should be a float or integer")
        if N < 0 or N > 1:
            raise e.ValueError("`N` should be between 0 and 1")

        self._N = N

    @property
    def P(self) -> int:
        """Number of prides."""

        return self._P

    @P.setter
    def P(self, P: int) -> None:
        if not isinstance(P, int):
            raise e.TypeError("`P` should be an integer")
        if P <= 0:
            raise e.ValueError("`P` should be > 0")

        self._P = P

    @property
    def S(self) -> float:
        """Percentage of female lions."""

        return self._S

    @S.setter
    def S(self, S: float) -> None:
        if not isinstance(S, (float, int)):
            raise e.TypeError("`S` should be a float or integer")
        if S < 0 or S > 1:
            raise e.ValueError("`S` should be between 0 and 1")

        self._S = S

    @property
    def R(self) -> float:
        """Percentage of roaming lions."""

        return self._R

    @R.setter
    def R(self, R: float) -> None:
        if not isinstance(R, (float, int)):
            raise e.TypeError("`R` should be a float or integer")
        if R < 0 or R > 1:
            raise e.ValueError("`R` should be between 0 and 1")

        self._R = R

    @property
    def I(self) -> float:
        """Immigrate rate."""

        return self._I

    @I.setter
    def I(self, I: float) -> None:
        if not isinstance(I, (float, int)):
            raise e.TypeError("`I` should be a float or integer")
        if I < 0 or I > 1:
            raise e.ValueError("`I` should be between 0 and 1")

        self._I = I

    @property
    def Ma(self) -> float:
        """Mating probability."""

        return self._Ma

    @Ma.setter
    def Ma(self, Ma: float) -> None:
        if not isinstance(Ma, (float, int)):
            raise e.TypeError("`Ma` should be a float or integer")
        if Ma < 0 or Ma > 1:
            raise e.ValueError("`Ma` should be between 0 and 1")

        self._Ma = Ma

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

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Replaces the current agents with a derived Lion structure
        space.agents = [
            Lion(
                agent.n_variables,
                agent.n_dimensions,
                agent.lb,
                agent.ub,
                agent.position,
                agent.fit,
            )
            for agent in space.agents
        ]

        # Calculates the number of nomad lions and their genders
        n_nomad = int(self.N * space.n_agents)
        nomad_gender = d.generate_bernoulli_distribution(1 - self.S, n_nomad)

        # Iterates through all possible nomads
        for i, agent in enumerate(space.agents[:n_nomad]):
            # Toggles to `True` the nomad property
            agent.nomad = True

            # Defines the gender according to Bernoulli distribution
            agent.female = bool(nomad_gender[i])

        # Calculates the gender of pride lions
        pride_gender = d.generate_bernoulli_distribution(
            self.S, space.n_agents - n_nomad
        )

        # Iterates through all possible prides
        for i, agent in enumerate(space.agents[n_nomad:]):
            # Defines the gender according to Bernoulli distribution
            agent.female = bool(pride_gender[i])

            # Allocates to the corresponding pride
            agent.pride = i % self.P

    def _get_nomad_lions(self, agents: List[Lion]) -> List[Lion]:
        """Gets all nomad lions.

        Args:
            agents: Agents.

        Returns:
            (List[Lion]): A list of nomad lions.

        """

        # Returns a list of nomad lions
        return [agent for agent in agents if agent.nomad]

    def _get_pride_lions(self, agents: List[Lion]) -> List[List[Lion]]:
        """Gets all non-nomad (pride) lions.

        Args:
            agents: Agents.

        Returns:
            (List[List[Lion]]): A list of lists, where each one indicates a particular pride with its lions.

        """

        # Gathers all non-nomad lions
        agents = [agent for agent in agents if not agent.nomad]

        # Returns a list of lists of prides
        return [[agent for agent in agents if agent.pride == i] for i in range(self.P)]

    def _hunting(self, prides: List[Lion], function: Function) -> None:
        """Performs the hunting procedure (s. 2.2.2).

        Args:
            prides: List of prides holding their corresponding lions.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all prides
        for pride in prides:
            # Iterates through all agents in pride
            for agent in pride:
                # If agent is female
                if agent.female:
                    # Allocates to a random hunting group
                    agent.group = r.generate_integer_random_number(high=4)

                # If agent is male
                else:
                    # Allocates to a null group (no-hunting)
                    agent.group = 0

            # Calculates the fitness sum of first, second and third groups
            first_group = np.sum([agent.fit for agent in pride if agent.group == 1])
            second_group = np.sum([agent.fit for agent in pride if agent.group == 2])
            third_group = np.sum([agent.fit for agent in pride if agent.group == 3])

            # Averages the position of the prey (lions in group 0)
            prey = np.mean(
                [agent.position for agent in pride if agent.group == 0], axis=0
            )

            # Calculates the group indexes and their corresponding
            # positions: center, left and right
            groups_idx = np.argsort([first_group, second_group, third_group]) + 1
            center = groups_idx[0]
            left = groups_idx[1]
            right = groups_idx[2]

            # Iterates through all agents in pride
            for agent in pride:
                # If agent belongs to the center group
                if agent.group == center:
                    # Iterates through all decision variables
                    for j in range(agent.n_variables):
                        # If agent's position is smaller than prey's
                        if agent.position[j] < prey[j]:
                            # Updates its position (eq. 5 - top)
                            agent.position[j] = r.generate_uniform_random_number(
                                agent.position[j], prey[j]
                            )
                        else:
                            # Updates its position (eq. 5 - bottom)
                            agent.position[j] = r.generate_uniform_random_number(
                                prey[j], agent.position[j]
                            )

                # If agent belongs to the left or right groups
                if agent.group in [left, right]:
                    # Iterates through all decision variables
                    for j in range(agent.n_variables):
                        # Calculates the encircling position
                        encircling = 2 * prey[j] - agent.position[j]

                        # If encircling's position is smaller than prey's
                        if encircling < prey[j]:
                            # Updates its position (eq. 4 - top)
                            agent.position[j] = r.generate_uniform_random_number(
                                encircling, prey[j]
                            )
                        else:
                            # Updates its position (eq. 4 - bottom)
                            agent.position[j] = r.generate_uniform_random_number(
                                prey[j], encircling
                            )

                # Clips their limits
                agent.clip_by_bound()

                # Defines the previous fitness and calculates the newer one
                agent.p_fit = copy.deepcopy(agent.fit)
                agent.fit = function(agent.position)

                # If new fitness is better than old one
                if agent.fit < agent.p_fit:
                    # Updates its best position
                    agent.best_position = copy.deepcopy(agent.position)

                    # Calculates the probability of improvement
                    p_improvement = agent.fit / agent.p_fit

                    # Updates the prey's position
                    r1 = r.generate_uniform_random_number()
                    prey += r1 * p_improvement * (prey - agent.position)

    def _moving_safe_place(self, prides: List[Lion]) -> None:
        """Move prides to safe locations (s. 2.2.3).

        Args:
            prides: List of prides holding their corresponding lions.

        """

        # Iterates through all prides
        for pride in prides:
            # Calculates the number of improved lions (eq. 7)
            n_improved = np.sum([1 for agent in pride if agent.fit < agent.p_fit])

            # Calculates the fitness of lions (eq. 8)
            fitnesses = [agent.fit for agent in pride]

            # Calculates the size of tournament (eq. 9)
            tournament_size = np.maximum(2, int(np.ceil(n_improved / 2)))

            # Iterates through all agents in pride
            for agent in pride:
                # If agent is female and belongs to group 0
                if agent.group == 0 and agent.female:
                    # Gathers the winning lion from tournament selection
                    w = g.tournament_selection(fitnesses, 1, tournament_size)[0]

                    # Calculates the distance between agent and winner
                    distance = g.euclidean_distance(agent.position, pride[w].position)

                    # Generates random numbers
                    rand = r.generate_uniform_random_number()
                    u = r.generate_uniform_random_number(-1, 1)
                    theta = r.generate_uniform_random_number(-np.pi / 6, np.pi / 6)

                    # Calculates both `R1` and `R2` vectors
                    R1 = pride[w].position - agent.position
                    R2 = np.random.randn(*R1.T.shape)
                    R2 = R2.T - R2.dot(R1) * R1 / (np.linalg.norm(R1) ** 2 + c.EPSILON)

                    # Updates agent's position (eq. 6)
                    agent.position += (
                        2 * distance * rand * R1 + u * np.tan(theta) * distance * R2
                    )

    def _roaming(self, prides: List[Lion], function: Function) -> None:
        """Performs the roaming procedure (s. 2.2.4).

        Args:
            prides: List of prides holding their corresponding lions.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all prides
        for pride in prides:
            # Calculates the number of roaming lions
            n_roaming = int(len(pride) * self.P)

            # Selects `n_roaming` lions
            selected = r.generate_integer_random_number(high=len(pride), size=n_roaming)

            # Iterates through all agents in pride
            for agent in pride:
                # If agent is male
                if not agent.female:
                    # Iterates through selected roaming lions
                    for s in selected:
                        # Calculates the direction angle
                        theta = r.generate_uniform_random_number(-np.pi / 6, np.pi / 6)

                        # Calculates the distance between selected lion and current one
                        distance = g.euclidean_distance(
                            pride[s].best_position, agent.position
                        )

                        # Generates the step (eq. 10)
                        step = r.generate_uniform_random_number(0, 2 * distance)

                        # Updates the agent's position
                        agent.position += step * np.tan(theta)

                        # Clip the agent's limits
                        agent.clip_by_bound()

                        # Defines the previous fitness and calculates the newer one
                        agent.p_fit = copy.deepcopy(agent.fit)
                        agent.fit = function(agent.position)

                        # If new fitness is better than old one
                        if agent.fit < agent.p_fit:
                            # Updates its best position
                            agent.best_position = copy.deepcopy(agent.position)

    def _mating_operator(
        self, agent: List[Lion], males: List[Lion], function: Function
    ) -> Tuple[Lion, Lion]:
        """Wraps the mating operator.

        Args:
            agent: Current agent.
            males: List of males that will be breed.
            function: A Function object that will be used as the objective function.

        Returns:
            (Tuple[Lion, Lion]): A pair of offsprings that resulted from mating.

        """

        # Calculates the males average position
        males_average = np.mean([male.position for male in males], axis=0)

        # Generates a gaussian random number
        beta = r.generate_gaussian_random_number(0.5, 0.1)

        # Copies current agent into two offsprings
        a1, a2 = copy.deepcopy(agent), copy.deepcopy(agent)

        # Updates first offspring position (eq. 13)
        a1.position = beta * a1.position + (1 - beta) * males_average

        # Updates second offspring position (eq. 14)
        a2.position = (1 - beta) * a2.position + beta * males_average

        # Iterates though all decision variables
        for j in range(agent.n_variables):
            # Generates random numbers
            r2 = r.generate_uniform_random_number()
            r3 = r.generate_uniform_random_number()

            # If first random number is smaller tha mutation probability
            if r2 < self.Mu:
                # Mutates the first offspring
                a1.position[j] = r.generate_uniform_random_number(a1.lb[j], a1.ub[j])

            # If second random number is smaller tha mutation probability
            if r3 < self.Mu:
                # Mutates the second offspring
                a2.position[j] = r.generate_uniform_random_number(a2.lb[j], a2.ub[j])

        # Clips both offspring bounds
        a1.clip_by_bound()
        a2.clip_by_bound()

        # Updates first offspring properties
        a1.best_position = copy.deepcopy(a1.position)
        a1.female = bool(beta >= 0.5)
        a1.fit = function(a1.position)

        # Updates second offspring properties
        a2.best_position = copy.deepcopy(a2.position)
        a2.female = bool(beta >= 0.5)
        a2.fit = function(a2.position)

        return a1, a2

    def _mating(self, prides: List[Lion], function: Function) -> Lion:
        """Generates offsprings from mating (s. 2.2.5).

        Args:
            prides: List of prides holding their corresponding lions.
            function: A Function object that will be used as the objective function.

        Returns:
            (Lion): Cubs generated from the mating procedure.

        """

        # Creates a list of prides offsprings
        prides_cubs = []

        # Iterates through all prides
        for pride in prides:
            # Creates a list of current pride offsprings
            cubs = []

            # Iterates through all agents in pride
            for agent in pride:
                # If agent is female
                if agent.female:
                    # Generates a random number
                    r1 = r.generate_uniform_random_number()

                    # If random number is smaller than mating probability
                    if r1 < self.Ma:
                        # Gathers a list of male lions that belongs to current pride
                        males = [agent for agent in pride if not agent.female]

                        # Performs the mating
                        a1, a2 = self._mating_operator(agent, males, function)

                        # Merges current pride offsprings
                        cubs += [a1, a2]

            # Appends pride offspring into prides list
            prides_cubs.append(cubs)

        return prides_cubs

    def _defense(
        self, nomads: List[Lion], prides: List[List[Lion]], cubs: List[Lion]
    ) -> Tuple[List[Lion], List[List[Lion]]]:
        """Performs the defense procedure (s. 2.2.6).

        Args:
            nomads: Nomad lions.
            prides: List of prides holding their corresponding lions.
            cubs: List of cubs holding their corresponding lions.

        Returns:
            (Tuple[List[Lion], List[List[Lion]]]): Both updated nomad and pride lions.

        """

        # Instantiate lists of new nprides lions
        new_prides = []

        for pride, cub in zip(prides, cubs):
            # Gathers the females and males from current pride
            pride_female = [agent for agent in pride if agent.female]
            pride_male = [agent for agent in pride if not agent.female]

            # Gathers the female and male cubs from current pride
            cub_female = [agent for agent in cub if agent.female]
            cub_male = [agent for agent in cub if not agent.female]

            # Sorts the males from current pride
            pride_male.sort(key=lambda x: x.fit)

            # Gathers the new pride by merging pride's females, cub's females,
            # cub's males and non-beaten pride's males
            new_pride = (
                pride_female + cub_female + cub_male + pride_male[: -len(cub_male)]
            )
            new_prides.append(new_pride)

            # Gathers the new nomads
            nomads += pride_male[-len(cub_male) :]

        return nomads, new_prides

    def _nomad_roaming(self, nomads: List[Lion], function: Function) -> None:
        """Performs the roaming procedure for nomad lions (s. 2.2.4).

        Args:
            nomads: Nomad lions.
            function: A Function object that will be used as the objective function.

        """

        # Sorts nomads
        nomads.sort(key=lambda x: x.fit)

        # Iterates through all nomad agents
        for agent in nomads:
            # Gathers the best nomad fitness
            best_fit = nomads[0].fit

            # Calculates the roaming probability (eq. 12)
            prob = 0.1 + np.minimum(
                0.5, (agent.fit - best_fit) / (best_fit + c.EPSILON)
            )

            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than roaming probability
            if r1 < prob:
                # Iterates through all decision variables
                for j in range(agent.n_variables):
                    # Updates the agent's position (eq. 11 - bottom)
                    agent.position[j] = r.generate_uniform_random_number(
                        agent.lb[j], agent.ub[j]
                    )

            # Clip the agent's limits
            agent.clip_by_bound()

            # Defines the previous fitness and calculates the newer one
            agent.p_fit = copy.deepcopy(agent.fit)
            agent.fit = function(agent.position)

            # If new fitness is better than old one
            if agent.fit < agent.p_fit:
                # Updates its best position
                agent.best_position = copy.deepcopy(agent.position)

    def _nomad_mating(self, nomads: List[Lion], function: Function) -> List[Lion]:
        """Generates offsprings from nomad lions mating (s. 2.2.5).

        Args:
            nomads: Nomad lions.
            function: A Function object that will be used as the objective function.

        Returns:
            (List[Lion]): Updated nomad lions.

        """

        # Creates a list of offsprings
        cubs = []

        # Iterates through all nomad agents
        for agent in nomads:
            # If agent is female
            if agent.female:
                # Generates a random number
                r1 = r.generate_uniform_random_number()

                # If random number is smaller than mating probability
                if r1 < self.Ma:
                    # Gathers a list of male lions that belongs to current pride
                    # and samples a random integer
                    males = [agent for agent in nomads if not agent.female]

                    # If there is at least a male
                    if len(males) > 0:
                        # Gathers a random index
                        idx = r.generate_integer_random_number(high=len(males))

                        # Performs the mating
                        a1, a2 = self._mating_operator(agent, [males[idx]], function)

                        # Merges current pride offsprings
                        cubs += [a1, a2]

        # Merges both initial nomads and cubs
        nomads += cubs

        return nomads

    def _nomad_attack(
        self, nomads: List[Lion], prides: List[List[Lion]]
    ) -> Tuple[List[Lion], List[List[Lion]]]:
        """Performs the nomad's attacking procedure (s. 2.2.6).

        Args:
            nomads: Nomad lions.
            prides: List of prides holding their corresponding lions.

        Returns:
            (Tuple[List[Lion], List[List[Lion]]]): Both updated nomad and pride lions.

        """

        # Iterates through all nomads
        for agent in nomads:
            # If current agent is female
            if agent.female:
                # Generates a binary array of prides to be attacked
                attack_prides = r.generate_binary_random_number(self.P)

                # Iterates through every pride
                for i, pride in enumerate(prides):
                    # If pride is supposed to be attacked
                    if attack_prides[i]:
                        # Gathers all the males in the pride
                        males = [agent for agent in pride if not agent.female]

                        # If there is at least a male
                        if len(males) > 0:
                            # If current nomad agent is better than male in pride
                            if agent.fit < males[0].fit:
                                # Swaps them
                                agent, males[0] = copy.deepcopy(
                                    males[0]
                                ), copy.deepcopy(agent)

        return nomads, prides

    def _migrating(
        self, nomads: List[Lion], prides: List[List[Lion]]
    ) -> Tuple[List[Lion], List[List[Lion]]]:
        """Performs the nomad's migration procedure (s. 2.2.7).

        Args:
            nomads: Nomad lions.
            prides: List of prides holding their corresponding lions.

        Returns:
            (Tuple[List[Lion], List[List[Lion]]]): Both updated nomad and pride lions.

        """

        # Creates a list to hold the updated prides
        new_prides = []

        # Iterates through all prides
        for pride in prides:
            # Calculates the number of females to be migrated
            n_migrating = int(len(pride) * self.I)

            # Selects `n_migrating` lions
            selected = r.generate_integer_random_number(
                high=len(pride), size=n_migrating
            )

            # Iterates through selected pride lions
            for s in selected:
                # If current agent is female
                if pride[s].female:
                    # Migrates the female to nomads and defines its property
                    n = copy.deepcopy(pride[s])
                    n.nomad = True

                    # Appends the new nomad lion to the corresponding list
                    nomads.append(n)

            # Appends non-selected lions to the new pride positions
            new_prides.append(
                [agent for i, agent in enumerate(pride) if i not in selected]
            )

        return nomads, new_prides

    def _equilibrium(
        self, nomads: List[Lion], prides: List[List[Lion]], n_agents: List[Agent]
    ) -> Tuple[List[Lion], List[List[Lion]]]:
        """Performs the population's equilibrium procedure (s. 2.2.8).

        Args:
            nomads: Nomad lions.
            prides: List of prides holding their corresponding lions.

        Returns:
            (Tuple[List[Lion], List[List[Lion]]]): Both updated nomad and pride lions.

        """

        # Splits the nomad's population into females and males
        nomad_female = [agent for agent in nomads if agent.female]
        nomad_male = [agent for agent in nomads if not agent.female]

        # Sorts both female and male nomads
        nomad_female.sort(key=lambda x: x.fit)
        nomad_male.sort(key=lambda x: x.fit)

        # Calculates the correct size of each pride
        correct_pride_size = int((1 - self.N) * n_agents / self.P)

        # Iterates through all prides
        for i in range(self.P):
            # While pride is bigger than the correct size
            while len(prides[i]) > correct_pride_size:
                # Removes an agent
                del prides[i][-1]

            # While pride is smaller than correct size
            while len(prides[i]) < correct_pride_size:
                # Gathers the best female nomad and transform into a pride-based lion
                nomad_female[0].pride = i
                nomad_female[0].nomad = False

                # Appens the female to the pride
                prides[i].append(copy.deepcopy(nomad_female[0]))

                # Removes from the nomads
                del nomad_female[0]

        # Merges both female and male nomads into a single population
        # and sorts its according to their fitness
        nomads = nomad_female + nomad_male
        nomads.sort(key=lambda x: x.fit)

        return nomads, prides

    def _check_prides_for_males(self, prides: List[List[Lion]]) -> None:
        """Checks if there is at least one male per pride.

        Args:
            prides: List of prides holding their corresponding lions.

        """

        # Gathers the amount of males per pride
        males_prides = [
            len([agent for agent in pride if not agent.female]) for pride in prides
        ]

        # Iterates through all prides
        for males_per_pride, pride in zip(males_prides, prides):
            # If there is no male in current pride
            if males_per_pride == 0:
                # Selects a random index and turns into a male
                idx = r.generate_integer_random_number(high=len(pride))
                pride[idx].female = False

    def update(self, space: Space, function: Function) -> None:
        """Wraps Lion Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Gets nomad and non-nomad (pride) lions
        nomads = self._get_nomad_lions(space.agents)
        prides = self._get_pride_lions(space.agents)

        # Performs the hunting procedure, moving, roaming,
        # mating and defending for pride lions (step 3)
        self._hunting(prides, function)
        self._moving_safe_place(prides)
        self._roaming(prides, function)
        pride_cubs = self._mating(prides, function)
        nomads, prides = self._defense(nomads, prides, pride_cubs)

        # Performs roaming, mating and attacking
        # for nomad lions (step 4)
        self._nomad_roaming(nomads, function)
        nomads = self._nomad_mating(nomads, function)
        nomads, prides = self._nomad_attack(nomads, prides)

        # Migrates females lions from prides (step 5)
        nomads, prides = self._migrating(nomads, prides)

        # Equilibrates the nomads and prides population (step 6)
        nomads, prides = self._equilibrium(nomads, prides, space.n_agents)

        # Checks if there is at least one male per pride
        self._check_prides_for_males(prides)

        # Defines the correct splitting point, so
        # the agents in space can be correctly updated
        correct_nomad_size = int(self.N * space.n_agents)

        # Updates the nomads
        space.agents[:correct_nomad_size] = copy.deepcopy(nomads[:correct_nomad_size])

        # Updates the prides
        space.agents[correct_nomad_size:] = copy.deepcopy(
            list(itertools.chain.from_iterable(prides))
        )
