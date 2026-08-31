"""Forest Optimization Algorithm."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class FOA(Optimizer):
    """A FOA class, inherited from Optimizer.

    This is the designed class to define FOA-related
    variables and methods.

    References:
        M. Ghaemi, Mohammad-Reza F.-D. Forest Optimization Algorithm.
        Expert Systems with Applications (2014).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(FOA, self).__init__()

        self.life_time = 6
        self.area_limit = 30
        self.LSC = 1
        self.GSC = 1
        self.transfer_rate = 0.1

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.age = [0] * space.n_agents

    def _local_seeding(self, space: Space, function: Callable) -> None:
        """Performs the local seeding on zero-aged trees.

        Args:
            space: A Space object containing meta-information.
            function: A callable that will be used as the objective function.

        """

        new_agents = []
        for i, agent in enumerate(space.agents):
            if self.age[i] == 0:
                for _ in range(self.LSC):
                    child = copy.deepcopy(agent)

                    j = np.random.randint(0, child.n_variables, None)
                    child.position[j] += np.random.uniform(child.lb[j], child.ub[j], 1)
                    child.clip_by_bound()

                    child.fit = function(child.position)

                    new_agents.append(child)

        self.age = [age + 1 for age in self.age]

        space.agents += new_agents

        self.age += [0] * len(new_agents)

    def _population_limiting(self, space: Space) -> List[Agent]:
        """Limits the population by removing old trees.

        Args:
            space: A Space object containing meta-information.

        Returns:
            (List[Agent]): A list of candidate trees that were removed from the forest.

        """

        candidate = []

        for i, _ in enumerate(space.agents):
            if self.age[i] > self.life_time:
                agent = space.agents.pop(i)
                self.age.pop(i)

                candidate.append(agent)

        space.agents, self.age = map(
            list, zip(*sorted(zip(space.agents, self.age), key=lambda x: x[0].fit))
        )

        if len(space.agents) > self.area_limit:
            candidate += space.agents[self.area_limit :]

            space.agents = space.agents[: self.area_limit]
            self.age = self.age[: self.area_limit]

        return candidate

    def _global_seeding(
        self, space: Space, function: Callable, candidate: List[Agent]
    ) -> None:
        """Performs the global seeding.

        Args:
            space: A Space object containing meta-information.
            function: A callable that will be used as the objective function.
            candidate: Candidate trees.

        """

        new_agents = []

        n_candidate = int(len(candidate) * self.transfer_rate)
        for agent in candidate[:n_candidate]:
            a = copy.deepcopy(agent)

            for _ in range(self.GSC):
                j = np.random.randint(0, a.n_variables, None)

                a.position[j] += np.random.uniform(a.lb[j], a.ub[j], 1)
                a.clip_by_bound()

                a.fit = function(a.position)

                new_agents.append(a)

        space.agents += new_agents

        self.age += [0] * len(new_agents)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Forest Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        self._local_seeding(space, function)
        candidate = self._population_limiting(space)
        self._global_seeding(space, function, candidate)

        space.agents, self.age = map(
            list, zip(*sorted(zip(space.agents, self.age), key=lambda x: x[0].fit))
        )

        self.age[0] = 0
