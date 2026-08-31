"""Lion Optimization Algorithm."""

import copy
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import opytimizer.math.general as g
import opytimizer.utils.constant as c
from opytimizer.core import Agent, Optimizer
from opytimizer.core.space import Space


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

        super(LOA, self).__init__()

        self.N = 0.2
        self.P = 4

        self.S = 0.8
        self.R = 0.2
        self.I = 0.4

        self.Ma = 0.3
        self.Mu = 0.2

        self.build(params)

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
        nomad_gender = np.random.binomial(1, 1 - self.S, n_nomad)

        for i, agent in enumerate(space.agents[:n_nomad]):
            agent.nomad = True
            agent.female = bool(nomad_gender[i])

        pride_gender = np.random.binomial(1, self.S, space.n_agents - n_nomad)

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

    def _hunting(self, prides: List[Lion], function: Callable) -> None:
        """Performs the hunting procedure (s. 2.2.2).

        Args:
            prides: List of prides holding their corresponding lions.
            function: A callable that will be used as the objective function.

        """

        for pride in prides:
            for agent in pride:
                if agent.female:
                    agent.group = np.random.randint(0, 4, None)
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
                            agent.position[j] = np.random.uniform(
                                agent.position[j], prey[j], 1
                            )
                        else:
                            # Updates its position (eq. 5 - bottom)
                            agent.position[j] = np.random.uniform(
                                prey[j], agent.position[j], 1
                            )

                if agent.group in [left, right]:
                    for j in range(agent.n_variables):
                        encircling = 2 * prey[j] - agent.position[j]

                        if encircling < prey[j]:
                            # Updates its position (eq. 4 - top)
                            agent.position[j] = np.random.uniform(
                                encircling, prey[j], 1
                            )
                        else:
                            # Updates its position (eq. 4 - bottom)
                            agent.position[j] = np.random.uniform(
                                prey[j], encircling, 1
                            )

                agent.clip_by_bound()

                agent.p_fit = copy.deepcopy(agent.fit)
                agent.fit = function(agent.position)
                if agent.fit < agent.p_fit:
                    agent.best_position = copy.deepcopy(agent.position)

                    p_improvement = agent.fit / agent.p_fit

                    r1 = np.random.uniform(0.0, 1.0, 1)
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

                    distance = np.linalg.norm(agent.position - pride[w].position)

                    rand = np.random.uniform(0.0, 1.0, 1)
                    u = np.random.uniform(-1, 1, 1)
                    theta = np.random.uniform(-np.pi / 6, np.pi / 6, 1)

                    R1 = pride[w].position - agent.position
                    R2 = np.random.randn(*R1.T.shape)
                    R2 = R2.T - R2.dot(R1) * R1 / (np.linalg.norm(R1) ** 2 + c.EPSILON)

                    # Updates agent's position (eq. 6)
                    agent.position += (
                        2 * distance * rand * R1 + u * np.tan(theta) * distance * R2
                    )

    def _roaming(self, prides: List[Lion], function: Callable) -> None:
        """Performs the roaming procedure (s. 2.2.4).

        Args:
            prides: List of prides holding their corresponding lions.
            function: A callable that will be used as the objective function.

        """

        for pride in prides:
            n_roaming = int(len(pride) * self.P)

            selected = np.random.randint(0, len(pride), n_roaming)

            for agent in pride:
                if not agent.female:
                    for s in selected:
                        theta = np.random.uniform(-np.pi / 6, np.pi / 6, 1)

                        distance = np.linalg.norm(
                            pride[s].best_position - agent.position
                        )

                        # Generates the step (eq. 10)
                        step = np.random.uniform(0, 2 * distance, 1)
                        agent.position += step * np.tan(theta)
                        agent.clip_by_bound()

                        agent.p_fit = copy.deepcopy(agent.fit)
                        agent.fit = function(agent.position)
                        if agent.fit < agent.p_fit:
                            agent.best_position = copy.deepcopy(agent.position)

    def _mating_operator(
        self, agent: List[Lion], males: List[Lion], function: Callable
    ) -> Tuple[Lion, Lion]:
        """Wraps the mating operator.

        Args:
            agent: Current agent.
            males: List of males that will be breed.
            function: A callable that will be used as the objective function.

        Returns:
            (Tuple[Lion, Lion]): A pair of offsprings that resulted from mating.

        """

        males_average = np.mean([male.position for male in males], axis=0)
        beta = np.random.normal(0.5, 0.1, 1)

        a1, a2 = copy.deepcopy(agent), copy.deepcopy(agent)

        # Updates first offspring position (eq. 13)
        a1.position = beta * a1.position + (1 - beta) * males_average

        # Updates second offspring position (eq. 14)
        a2.position = (1 - beta) * a2.position + beta * males_average

        for j in range(agent.n_variables):
            r2 = np.random.uniform(0.0, 1.0, 1)
            if r2 < self.Mu:
                a1.position[j] = np.random.uniform(a1.lb[j], a1.ub[j], 1)

            r3 = np.random.uniform(0.0, 1.0, 1)
            if r3 < self.Mu:
                a2.position[j] = np.random.uniform(a2.lb[j], a2.ub[j], 1)

        a1.clip_by_bound()
        a2.clip_by_bound()

        a1.best_position = copy.deepcopy(a1.position)
        a1.female = bool(beta >= 0.5)
        a1.fit = function(a1.position)

        a2.best_position = copy.deepcopy(a2.position)
        a2.female = bool(beta >= 0.5)
        a2.fit = function(a2.position)

        return a1, a2

    def _mating(self, prides: List[Lion], function: Callable) -> Lion:
        """Generates offsprings from mating (s. 2.2.5).

        Args:
            prides: List of prides holding their corresponding lions.
            function: A callable that will be used as the objective function.

        Returns:
            (Lion): Cubs generated from the mating procedure.

        """

        prides_cubs = []
        for pride in prides:
            cubs = []

            for agent in pride:
                if agent.female:
                    r1 = np.random.uniform(0.0, 1.0, 1)
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

    def _nomad_roaming(self, nomads: List[Lion], function: Callable) -> None:
        """Performs the roaming procedure for nomad lions (s. 2.2.4).

        Args:
            nomads: Nomad lions.
            function: A callable that will be used as the objective function.

        """

        nomads.sort(key=lambda x: x.fit)
        for agent in nomads:
            best_fit = nomads[0].fit

            # Calculates the roaming probability (eq. 12)
            prob = 0.1 + np.minimum(
                0.5, (agent.fit - best_fit) / (best_fit + c.EPSILON)
            )

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < prob:
                for j in range(agent.n_variables):
                    # Updates the agent's position (eq. 11 - bottom)
                    agent.position[j] = np.random.uniform(agent.lb[j], agent.ub[j], 1)

            agent.clip_by_bound()

            agent.p_fit = copy.deepcopy(agent.fit)
            agent.fit = function(agent.position)
            if agent.fit < agent.p_fit:
                agent.best_position = copy.deepcopy(agent.position)

    def _nomad_mating(self, nomads: List[Lion], function: Callable) -> List[Lion]:
        """Generates offsprings from nomad lions mating (s. 2.2.5).

        Args:
            nomads: Nomad lions.
            function: A callable that will be used as the objective function.

        Returns:
            (List[Lion]): Updated nomad lions.

        """

        cubs = []

        for agent in nomads:
            if agent.female:
                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 < self.Ma:
                    males = [agent for agent in nomads if not agent.female]

                    if len(males) > 0:
                        idx = np.random.randint(0, len(males), None)

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
                attack_prides = np.random.randint(0, 2, self.P)

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

            selected = np.random.randint(0, len(pride), n_migrating)
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
                idx = np.random.randint(0, len(pride), None)
                pride[idx].female = False

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Lion Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

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
