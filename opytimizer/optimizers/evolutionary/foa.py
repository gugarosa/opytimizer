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

        # Overrides its parent class with the receiving params
        super(FOA, self).__init__()

        # Maximum age of trees
        self.life_time = 6

        # Maximum number of trees in the florest
        self.area_limit = 30

        # Local Seeding Changes
        self.LSC = 1

        # Global Seeding Changes
        self.GSC = 1

        # Global seeding percentage
        self.transfer_rate = 0.1

        # Builds the class
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

        # Lists of ages
        self.age = [0] * space.n_agents

    def _local_seeding(self, space: Space, function: Function) -> None:
        """Performs the local seeding on zero-aged trees.

        Args:
            space: A Space object containing meta-information.
            function: A Function object that will be used as the objective function.

        """

        # Creates a list of temporary agents (trees)
        new_agents = []

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Checks if current agent has zero age
            if self.age[i] == 0:
                # Iterates through all possible local changes
                for _ in range(self.LSC):
                    # Copies a temporary agent
                    child = copy.deepcopy(agent)

                    # Selects a random decision variable
                    j = r.generate_integer_random_number(high=child.n_variables)

                    # Updates the temporary agent's position and clips its bounds
                    child.position[j] += r.generate_uniform_random_number(
                        child.lb[j], child.ub[j]
                    )
                    child.clip_by_bound()

                    # Evaluates new position
                    child.fit = function(child.position)

                    # Appends to the list of children
                    new_agents.append(child)

        # Increases the age for the old trees
        self.age = [age + 1 for age in self.age]

        # Merges both new and old populations
        space.agents += new_agents

        # Adds the age of new trees as zero
        self.age += [0] * len(new_agents)

    def _population_limiting(self, space: Space) -> List[Agent]:
        """Limits the population by removing old trees.

        Args:
            space: A Space object containing meta-information.

        Returns:
            (List[Agent]): A list of candidate trees that were removed from the forest.

        """

        # List of candidate trees
        candidate = []

        # Iterates through all agents
        for i, _ in enumerate(space.agents):
            # Checks whether current agent has exceed its life time
            if self.age[i] > self.life_time:
                # Removes the tree and its corresponding age from forest
                agent = space.agents.pop(i)
                self.age.pop(i)

                # Adds the removed agent to the candidate list
                candidate.append(agent)

        # Sorts agents and their corresponding ages
        space.agents, self.age = map(
            list, zip(*sorted(zip(space.agents, self.age), key=lambda x: x[0].fit))
        )

        # If the population exceeds the forest limits
        if len(space.agents) > self.area_limit:
            # Adds extra trees to the candidate list
            candidate += space.agents[self.area_limit :]

            # Removes the extra trees and their corresponding ages from forest
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

        # Creates a list of temporary agents (trees)
        new_agents = []

        # Calculates the number of candidates
        n_candidate = int(len(candidate) * self.transfer_rate)

        # Iterates through all selected trees
        for agent in candidate[:n_candidate]:
            # Makes a copy of current agent
            a = copy.deepcopy(agent)

            # Iterates through all possible global changes
            for _ in range(self.GSC):
                # Selects a random variable
                j = r.generate_integer_random_number(high=a.n_variables)

                # Updates the temporary agent's position
                a.position[j] += r.generate_uniform_random_number(a.lb[j], a.ub[j])
                a.clip_by_bound()

                # Evaluates its fitness
                a.fit = function(a.position)

                # Appends to the list of children
                new_agents.append(a)

        # Merges both new and old populations
        space.agents += new_agents

        # Adds the age of new trees as zero
        self.age += [0] * len(new_agents)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Forest Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Performs the local seeding
        self._local_seeding(space, function)

        # Limits the population
        candidate = self._population_limiting(space)

        # Performs the global seeding
        self._global_seeding(space, function, candidate)

        # Sorts agents and their corresponding ages
        space.agents, self.age = map(
            list, zip(*sorted(zip(space.agents, self.age), key=lambda x: x[0].fit))
        )

        # Sets the best tree's age to zero
        self.age[0] = 0
