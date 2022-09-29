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

        super(Lion, self).__init__(n_variables, n_dimensions, lower_bound, upper_bound)

        self.position = copy.deepcopy(position)
        self.best_position = copy.deepcopy(position)

        self.fit = copy.deepcopy(fit)
        self.p_fit = copy.deepcopy(fit)

        self.nomad = False
        self.female = False

        self.pride = 0
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

        super(LOA, self).__init__()

        self.N = 0.2
        self.P = 4

        self.S = 0.8
        self.R = 0.2
        self.I = 0.4

        self.Ma = 0.3
        self.Mu = 0.2

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

        n_nomad = int(self.N * space.n_agents)
        nomad_gender = d.generate_bernoulli_distribution(1 - self.S, n_nomad)

        for i, agent in enumerate(space.agents[:n_nomad]):
            agent.nomad = True
            agent.female = bool(nomad_gender[i])

        pride_gender = d.generate_bernoulli_distribution(
            self.S, space.n_agents - n_nomad
        )

        for i, agent in enumerate(space.agents[n_nomad:]):
            agent.female = bool(pride_gender[i])
            agent.pride = i % self.P

    def _get_nomad_lions(self, agents: List[Lion]) -> List[Lion]:
        """Gets all nomad lions.

        Args:
            agents: Agents.

        Returns:
            (List[Lion]): A list of nomad lions.

        """

        return [agent for agent in agents if agent.nomad]

    def _get_pride_lions(self, agents: List[Lion]) -> List[List[Lion]]:
        """Gets all non-nomad (pride) lions.

        Args:
            agents: Agents.

        Returns:
            (List[List[Lion]]): A list of lists, where each one indicates a particular pride with its lions.

        """

        agents = [agent for agent in agents if not agent.nomad]

        return [[agent for agent in agents if agent.pride == i] for i in range(self.P)]

    def _hunting(self, prides: List[Lion], function: Function) -> None:
        """Performs the hunting procedure (s. 2.2.2).

        Args:
            prides: List of prides holding their corresponding lions.
            function: A Function object that will be used as the objective function.

        """

        for pride in prides:
            for agent in pride:
                if agent.female:
                    agent.group = r.generate_integer_random_number(high=4)
                else:
                    agent.group = 0

            first_group = np.sum([agent.fit for agent in pride if agent.group == 1])
            second_group = np.sum([agent.fit for agent in pride if agent.group == 2])
            third_group = np.sum([agent.fit for agent in pride if agent.group == 3])

            prey = np.mean(
                [agent.position for agent in pride if agent.group == 0], axis=0
            )

            groups_idx = np.argsort([first_group, second_group, third_group]) + 1
            center = groups_idx[0]
            left = groups_idx[1]
            right = groups_idx[2]

            for agent in pride:
                if agent.group == center:
                    for j in range(agent.n_variables):
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

                if agent.group in [left, right]:
                    for j in range(agent.n_variables):
                        encircling = 2 * prey[j] - agent.position[j]

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

                agent.clip_by_bound()

                agent.p_fit = copy.deepcopy(agent.fit)
                agent.fit = function(agent.position)
                if agent.fit < agent.p_fit:
                    agent.best_position = copy.deepcopy(agent.position)

                    p_improvement = agent.fit / agent.p_fit

                    r1 = r.generate_uniform_random_number()
                    prey += r1 * p_improvement * (prey - agent.position)

    def _moving_safe_place(self, prides: List[Lion]) -> None:
        """Move prides to safe locations (s. 2.2.3).

        Args:
            prides: List of prides holding their corresponding lions.

        """

        for pride in prides:
            # Calculates the number of improved lions (eq. 7)
            n_improved = np.sum([1 for agent in pride if agent.fit < agent.p_fit])

            # Calculates the fitness of lions (eq. 8)
            fitnesses = [agent.fit for agent in pride]

            # Calculates the size of tournament (eq. 9)
            tournament_size = np.maximum(2, int(np.ceil(n_improved / 2)))

            for agent in pride:
                if agent.group == 0 and agent.female:
                    w = g.tournament_selection(fitnesses, 1, tournament_size)[0]

                    distance = g.euclidean_distance(agent.position, pride[w].position)

                    rand = r.generate_uniform_random_number()
                    u = r.generate_uniform_random_number(-1, 1)
                    theta = r.generate_uniform_random_number(-np.pi / 6, np.pi / 6)

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

        for pride in prides:
            n_roaming = int(len(pride) * self.P)

            selected = r.generate_integer_random_number(high=len(pride), size=n_roaming)

            for agent in pride:
                if not agent.female:
                    for s in selected:
                        theta = r.generate_uniform_random_number(-np.pi / 6, np.pi / 6)

                        distance = g.euclidean_distance(
                            pride[s].best_position, agent.position
                        )

                        # Generates the step (eq. 10)
                        step = r.generate_uniform_random_number(0, 2 * distance)
                        agent.position += step * np.tan(theta)
                        agent.clip_by_bound()

                        agent.p_fit = copy.deepcopy(agent.fit)
                        agent.fit = function(agent.position)
                        if agent.fit < agent.p_fit:
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

        males_average = np.mean([male.position for male in males], axis=0)
        beta = r.generate_gaussian_random_number(0.5, 0.1)

        a1, a2 = copy.deepcopy(agent), copy.deepcopy(agent)

        # Updates first offspring position (eq. 13)
        a1.position = beta * a1.position + (1 - beta) * males_average

        # Updates second offspring position (eq. 14)
        a2.position = (1 - beta) * a2.position + beta * males_average

        for j in range(agent.n_variables):
            r2 = r.generate_uniform_random_number()
            if r2 < self.Mu:
                a1.position[j] = r.generate_uniform_random_number(a1.lb[j], a1.ub[j])

            r3 = r.generate_uniform_random_number()
            if r3 < self.Mu:
                a2.position[j] = r.generate_uniform_random_number(a2.lb[j], a2.ub[j])

        a1.clip_by_bound()
        a2.clip_by_bound()

        a1.best_position = copy.deepcopy(a1.position)
        a1.female = bool(beta >= 0.5)
        a1.fit = function(a1.position)

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

        prides_cubs = []
        for pride in prides:
            cubs = []

            for agent in pride:
                if agent.female:
                    r1 = r.generate_uniform_random_number()
                    if r1 < self.Ma:
                        males = [agent for agent in pride if not agent.female]

                        a1, a2 = self._mating_operator(agent, males, function)
                        cubs += [a1, a2]

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

        new_prides = []
        for pride, cub in zip(prides, cubs):
            pride_female = [agent for agent in pride if agent.female]
            pride_male = [agent for agent in pride if not agent.female]

            cub_female = [agent for agent in cub if agent.female]
            cub_male = [agent for agent in cub if not agent.female]

            pride_male.sort(key=lambda x: x.fit)

            new_pride = (
                pride_female + cub_female + cub_male + pride_male[: -len(cub_male)]
            )
            new_prides.append(new_pride)

            nomads += pride_male[-len(cub_male) :]

        return nomads, new_prides

    def _nomad_roaming(self, nomads: List[Lion], function: Function) -> None:
        """Performs the roaming procedure for nomad lions (s. 2.2.4).

        Args:
            nomads: Nomad lions.
            function: A Function object that will be used as the objective function.

        """

        nomads.sort(key=lambda x: x.fit)
        for agent in nomads:
            best_fit = nomads[0].fit

            # Calculates the roaming probability (eq. 12)
            prob = 0.1 + np.minimum(
                0.5, (agent.fit - best_fit) / (best_fit + c.EPSILON)
            )

            r1 = r.generate_uniform_random_number()
            if r1 < prob:
                for j in range(agent.n_variables):
                    # Updates the agent's position (eq. 11 - bottom)
                    agent.position[j] = r.generate_uniform_random_number(
                        agent.lb[j], agent.ub[j]
                    )

            agent.clip_by_bound()

            agent.p_fit = copy.deepcopy(agent.fit)
            agent.fit = function(agent.position)
            if agent.fit < agent.p_fit:
                agent.best_position = copy.deepcopy(agent.position)

    def _nomad_mating(self, nomads: List[Lion], function: Function) -> List[Lion]:
        """Generates offsprings from nomad lions mating (s. 2.2.5).

        Args:
            nomads: Nomad lions.
            function: A Function object that will be used as the objective function.

        Returns:
            (List[Lion]): Updated nomad lions.

        """

        cubs = []

        for agent in nomads:
            if agent.female:
                r1 = r.generate_uniform_random_number()
                if r1 < self.Ma:
                    males = [agent for agent in nomads if not agent.female]

                    if len(males) > 0:
                        idx = r.generate_integer_random_number(high=len(males))

                        a1, a2 = self._mating_operator(agent, [males[idx]], function)
                        cubs += [a1, a2]

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

        for agent in nomads:
            if agent.female:
                attack_prides = r.generate_binary_random_number(self.P)

                for i, pride in enumerate(prides):
                    if attack_prides[i]:
                        males = [agent for agent in pride if not agent.female]
                        if len(males) > 0:
                            if agent.fit < males[0].fit:
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

        new_prides = []

        for pride in prides:
            n_migrating = int(len(pride) * self.I)

            selected = r.generate_integer_random_number(
                high=len(pride), size=n_migrating
            )
            for s in selected:
                if pride[s].female:
                    n = copy.deepcopy(pride[s])
                    n.nomad = True

                    nomads.append(n)

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

        nomad_female = [agent for agent in nomads if agent.female]
        nomad_male = [agent for agent in nomads if not agent.female]

        nomad_female.sort(key=lambda x: x.fit)
        nomad_male.sort(key=lambda x: x.fit)

        correct_pride_size = int((1 - self.N) * n_agents / self.P)

        for i in range(self.P):
            while len(prides[i]) > correct_pride_size:
                del prides[i][-1]

            while len(prides[i]) < correct_pride_size:
                nomad_female[0].pride = i
                nomad_female[0].nomad = False

                prides[i].append(copy.deepcopy(nomad_female[0]))

                del nomad_female[0]

        nomads = nomad_female + nomad_male
        nomads.sort(key=lambda x: x.fit)

        return nomads, prides

    def _check_prides_for_males(self, prides: List[List[Lion]]) -> None:
        """Checks if there is at least one male per pride.

        Args:
            prides: List of prides holding their corresponding lions.

        """

        males_prides = [
            len([agent for agent in pride if not agent.female]) for pride in prides
        ]

        for males_per_pride, pride in zip(males_prides, prides):
            if males_per_pride == 0:
                idx = r.generate_integer_random_number(high=len(pride))
                pride[idx].female = False

    def update(self, space: Space, function: Function) -> None:
        """Wraps Lion Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

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
        self._check_prides_for_males(prides)

        correct_nomad_size = int(self.N * space.n_agents)
        space.agents[:correct_nomad_size] = copy.deepcopy(nomads[:correct_nomad_size])
        space.agents[correct_nomad_size:] = copy.deepcopy(
            list(itertools.chain.from_iterable(prides))
        )
