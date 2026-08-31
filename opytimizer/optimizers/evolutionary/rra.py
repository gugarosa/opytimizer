"""Runner-Root Algorithm."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class RRA(Optimizer):
    """An RRA class, inherited from Optimizer.

    This is the designed class to define RRA-related
    variables and methods.

    References:
        F. Merrikh-Bayat.
        The runner-root algorithm: A metaheuristic for solving unimodal and
        multimodal optimization problems inspired by runners and roots of plants in nature.
        Applied Soft Computing (2015).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(RRA, self).__init__()

        self.d_runner = 2
        self.d_root = 0.01
        self.tol = 0.01

        self.max_stall = 1000
        self.n_stall = 0

        self.last_best_fit = c.FLOAT_MAX

        self.build(params)

    def _stalling_search(
        self,
        daughters: List[Agent],
        function: Callable,
        is_large: bool = True,
    ) -> None:
        """Performs the stalling random larrge or small search (eq. 4 and 5).

        Args:
            daughters: Daughters.
            function: A callable that will be used as the objective function.
            is_large: Whether to perform the large or small search.

        """

        for _ in range(len(daughters) - 1):
            temp_daughter = copy.deepcopy(daughters[0])

            j = np.random.randint(0, temp_daughter.n_variables, None)

            if is_large:
                # Disturbs a selected temporary daughter's position (eq. 4)
                r1 = np.random.normal(0.0, 1.0, 1)
                temp_daughter.position[j] += self.d_runner * r1
            else:
                # Disturbs a selected temporary daughter's position (eq. 5)
                r1 = np.random.uniform(-0.5, 0.5, 1)
                temp_daughter.position[j] += self.d_root * r1

            temp_daughter.clip_by_bound()

            temp_daughter.fit = function(temp_daughter.position)
            if temp_daughter.fit < daughters[0].fit:
                daughters[0].position = copy.deepcopy(temp_daughter.position)
                daughters[0].fit = copy.deepcopy(temp_daughter.fit)

    def _roulette_selection(self, fitness: List[float], a: float = 0.1) -> int:
        """Performs a roulette selection on the population (eq. 8).

        Args:
            fitness: A fitness list of every agent.
            a: Selection regularizer.

        Returns:
            (int): The selected index of the population.

        """

        min_fitness = np.min(fitness)

        # Re-arrange the list of fitness by inverting it (eq. 7)
        inv_fitness = [1 / (a + fit - min_fitness) for fit in fitness]
        total_fitness = np.sum(inv_fitness)

        # Calculates the probability of each inverted fitness (eq. 8)
        probs = [fit / total_fitness for fit in inv_fitness]
        selected = np.random.choice(len(probs), 1, p=probs, replace=False)

        return selected[0]

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Runner-Root Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        space.agents.sort(key=lambda x: x.fit)

        self.last_best_fit = space.agents[0].fit

        daughters = copy.deepcopy(space.agents)
        for daughter in daughters[1:]:
            r1 = np.random.uniform(-0.5, 0.5, 1)

            # Updates the daughter's position and clips its bounds (eq. 2)
            daughter.position += self.d_runner * r1
            daughter.clip_by_bound()

            daughter.fit = function(daughter.position)

        daughters.sort(key=lambda x: x.fit)

        # Checks the new positions' effectiviness (eq. 3)
        effectiveness = np.fabs(
            (self.last_best_fit - daughters[0].fit) / (self.last_best_fit + c.EPSILON)
        )
        if effectiveness < self.tol:
            # Performs the stalling large search (eq. 4)
            self._stalling_search(daughters, function, is_large=True)

            # Performs the stalling small search (eq. 5)
            self._stalling_search(daughters, function, is_large=False)

        # Performs the elite selection (eq. 6)
        space.agents[0] = copy.deepcopy(daughters[0])

        daughters_fit = [daughter.fit for daughter in daughters]
        for agent in space.agents[1:]:
            idx = self._roulette_selection(daughters_fit)
            agent = copy.deepcopy(daughters[idx])

        # Checks again the positions' effectiviness (eq. 3)
        effectiveness = np.fabs(
            (self.last_best_fit - daughters[0].fit) / (self.last_best_fit + c.EPSILON)
        )
        if effectiveness < self.tol:
            self.n_stall += 1
        else:
            self.n_stall = 0

        if self.n_stall == self.max_stall:
            for agent in space.agents:
                agent.fill_with_uniform()

            self.n_stall = 0
