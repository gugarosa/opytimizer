"""Forest Optimization Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Overriding class: Optimizer -> FOA.")

        super(FOA, self).__init__()

        self.life_time = 6
        self.area_limit = 30
        self.LSC = 1
        self.GSC = 1
        self.transfer_rate = 0.1

        self.build(params)

        logger.info("Class overrided.")

    @property
    def life_time(self) -> int:
        """Maximum age of trees."""

        return self._life_time

    @life_time.setter
    def life_time(self, life_time: int) -> None:
        if not isinstance(life_time, int):
            raise e.TypeError("`life_time` should be an integer")
        if life_time <= 0:
            raise e.ValueError("`life_time` should be > 0")

        self._life_time = life_time

    @property
    def area_limit(self) -> int:
        """Maximum number of trees in the florest."""

        return self._area_limit

    @area_limit.setter
    def area_limit(self, area_limit: int) -> None:
        if not isinstance(area_limit, int):
            raise e.TypeError("`area_limit` should be an integer")
        if area_limit <= 0:
            raise e.ValueError("`area_limit` should be > 0")

        self._area_limit = area_limit

    @property
    def LSC(self) -> int:
        """Local Seeding Changes."""

        return self._LSC

    @LSC.setter
    def LSC(self, LSC: int) -> None:
        if not isinstance(LSC, int):
            raise e.TypeError("`LSC` should be an integer")
        if LSC <= 0:
            raise e.ValueError("`LSC` should be > 0")

        self._LSC = LSC

    @property
    def GSC(self) -> int:
        """Global Seeding Changes."""

        return self._GSC

    @GSC.setter
    def GSC(self, GSC: int) -> None:
        if not isinstance(GSC, int):
            raise e.TypeError("`GSC` should be an integer")
        if GSC <= 0:
            raise e.ValueError("`GSC` should be > 0")

        self._GSC = GSC

    @property
    def transfer_rate(self) -> float:
        """Global seeding percentage."""

        return self._transfer_rate

    @transfer_rate.setter
    def transfer_rate(self, transfer_rate: float) -> None:
        if not isinstance(transfer_rate, (float, int)):
            raise e.TypeError("`transfer_rate` should be a float or integer")
        if transfer_rate < 0 or transfer_rate > 1:
            raise e.ValueError("`transfer_rate` should be between 0 and 1")

        self._transfer_rate = transfer_rate

    @property
    def age(self) -> List[int]:
        """Trees ages."""

        return self._age

    @age.setter
    def age(self, age: List[int]) -> None:
        if not isinstance(age, list):
            raise e.TypeError("`age` should be a list")

        self._age = age

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.age = [0] * space.n_agents

    def _local_seeding(self, space: Space, function: Function) -> None:
        """Performs the local seeding on zero-aged trees.

        Args:
            space: A Space object containing meta-information.
            function: A Function object that will be used as the objective function.

        """

        new_agents = []
        for i, agent in enumerate(space.agents):
            if self.age[i] == 0:
                for _ in range(self.LSC):
                    child = copy.deepcopy(agent)

                    j = r.generate_integer_random_number(high=child.n_variables)
                    child.position[j] += r.generate_uniform_random_number(
                        child.lb[j], child.ub[j]
                    )
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
        self, space: Space, function: Function, candidate: List[Agent]
    ) -> None:
        """Performs the global seeding.

        Args:
            space: A Space object containing meta-information.
            function: A Function object that will be used as the objective function.
            candidate: Candidate trees.

        """

        new_agents = []

        n_candidate = int(len(candidate) * self.transfer_rate)
        for agent in candidate[:n_candidate]:
            a = copy.deepcopy(agent)

            for _ in range(self.GSC):
                j = r.generate_integer_random_number(high=a.n_variables)

                a.position[j] += r.generate_uniform_random_number(a.lb[j], a.ub[j])
                a.clip_by_bound()

                a.fit = function(a.position)

                new_agents.append(a)

        space.agents += new_agents

        self.age += [0] * len(new_agents)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Forest Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        self._local_seeding(space, function)
        candidate = self._population_limiting(space)
        self._global_seeding(space, function, candidate)

        space.agents, self.age = map(
            list, zip(*sorted(zip(space.agents, self.age), key=lambda x: x[0].fit))
        )

        self.age[0] = 0
