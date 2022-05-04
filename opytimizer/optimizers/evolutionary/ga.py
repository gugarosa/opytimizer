"""Genetic Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.math.distribution as d
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


class GA(Optimizer):
    """An GA class, inherited from Optimizer.

    This is the designed class to define GA-related
    variables and methods.

    References:
        M. Mitchell. An introduction to genetic algorithms. MIT Press (1998).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(GA, self).__init__()

        # Probability of selection
        self.p_selection = 0.75

        # Probability of mutation
        self.p_mutation = 0.25

        # Probability of crossover
        self.p_crossover = 0.5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def p_selection(self) -> float:
        """Probability of selection."""

        return self._p_selection

    @p_selection.setter
    def p_selection(self, p_selection: float) -> None:
        if not isinstance(p_selection, (float, int)):
            raise e.TypeError("`p_selection` should be a float or integer")
        if p_selection < 0 or p_selection > 1:
            raise e.ValueError("`p_selection` should be between 0 and 1")

        self._p_selection = p_selection

    @property
    def p_mutation(self) -> float:
        """Probability of mutation."""

        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if not isinstance(p_mutation, (float, int)):
            raise e.TypeError("`p_mutation` should be a float or integer")
        if p_mutation < 0 or p_mutation > 1:
            raise e.ValueError("`p_mutation` should be between 0 and 1")

        self._p_mutation = p_mutation

    @property
    def p_crossover(self) -> float:
        """Probability of crossover."""

        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, p_crossover: float) -> None:
        if not isinstance(p_crossover, (float, int)):
            raise e.TypeError("`p_crossover` should be a float or integer")
        if p_crossover < 0 or p_crossover > 1:
            raise e.ValueError("`p_crossover` should be between 0 and 1")

        self._p_crossover = p_crossover

    def _roulette_selection(self, n_agents: int, fitness: List[float]) -> List[int]:
        """Performs a roulette selection on the population (p. 8).

        Args:
            n_agents: Number of agents allowed in the space.
            fitness: A fitness list of every agent.

        Returns:
            (List[int]): The selected indexes of the population.

        """

        # Calculates the number of selected individuals
        n_individuals = int(n_agents * self.p_selection)

        # Checks if `n_individuals` is an odd number
        if n_individuals % 2 != 0:
            # If it is, increase it by one
            n_individuals += 1

        # Defines the maximum fitness of current generation
        max_fitness = np.max(fitness)

        # Re-arrange the list of fitness by inverting it
        # Note that we apply a trick due to it being designed for minimization
        # f'(x) = f_max - f(x)
        inv_fitness = [max_fitness - fit + c.EPSILON for fit in fitness]

        # Calculates the total inverted fitness
        total_fitness = np.sum(inv_fitness)

        # Calculates the probability of each inverted fitness
        probs = [fit / total_fitness for fit in inv_fitness]

        # Performs the selection process
        selected = d.generate_choice_distribution(n_agents, probs, n_individuals)

        return selected

    def _crossover(self, father: Agent, mother: Agent) -> Tuple[Agent, Agent]:
        """Performs the crossover between a pair of parents (p. 8).

        Args:
            father: Father to produce the offsprings.
            mother: Mother to produce the offsprings.

        Returns:
            (Tuple[Agent, Agent]): Two generated offsprings based on parents.

        """

        # Makes a deep copy of father and mother
        alpha, beta = copy.deepcopy(father), copy.deepcopy(mother)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # If random number is smaller than crossover probability
        if r1 < self.p_crossover:
            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # Calculates the crossover based on a linear combination between father and mother
            alpha.position = r2 * father.position + (1 - r2) * mother.position

            # Calculates the crossover based on a linear combination between father and mother
            beta.position = r2 * mother.position + (1 - r2) * father.position

        return alpha, beta

    def _mutation(self, alpha: Agent, beta: Agent) -> Tuple[Agent, Agent]:
        """Performs the mutation over offsprings (p. 8).

        Args:
            alpha: First offspring.
            beta: Second offspring.

        Returns:
            (Tuple[Agent, Agent]): Two mutated offsprings.

        """

        # For every decision variable
        for j in range(alpha.n_variables):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than probability of mutation
            if r1 < self.p_mutation:
                # Mutates the offspring
                alpha.position[j] += r.generate_gaussian_random_number()

            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # If random number is smaller than probability of mutation
            if r2 < self.p_mutation:
                # Mutates the offspring
                beta.position[j] += r.generate_gaussian_random_number()

        return alpha, beta

    def update(self, space: Space, function: Function) -> None:
        """Wraps Genetic Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Creates a list to hold the new population
        new_agents = []

        # Retrieves the number of agents
        n_agents = len(space.agents)

        # Calculates a list of fitness from every agent
        fitness = [agent.fit + c.EPSILON for agent in space.agents]

        # Selects the parents
        selected = self._roulette_selection(n_agents, fitness)

        # For every pair of selected parents
        for s in g.n_wise(selected):
            # Performs the crossover and mutation
            alpha, beta = self._crossover(space.agents[s[0]], space.agents[s[1]])
            alpha, beta = self._mutation(alpha, beta)

            # Checking `alpha` and `beta` limits
            alpha.clip_by_bound()
            beta.clip_by_bound()

            # Calculates new fitness for `alpha` and `beta`
            alpha.fit = function(alpha.position)
            beta.fit = function(beta.position)

            # Appends the mutated agents to the children
            new_agents.extend([alpha, beta])

        # Joins both populations, sort agents and gathers best `n_agents`
        space.agents += new_agents
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]
