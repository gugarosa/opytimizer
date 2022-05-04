"""Passing Vehicle Search.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class PVS(Optimizer):
    """A PVS class, inherited from Optimizer.

    This is the designed class to define PVS-related
    variables and methods.

    References:
        P. Savsani and V. Savsani. Passing vehicle search (PVS): A novel metaheuristic algorithm.
        Applied Mathematical Modelling (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> PVS.")

        # Overrides its parent class with the receiving params
        super(PVS, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    def update(self, space: Space, function: Function) -> None:
        """Wraps Passing Vehicle Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Creates a list of equal indexes
            R = [0, 0]

            # Keeps generating indexes until they are different
            while R[0] == R[1]:
                # Generates random indexes
                R = r.generate_integer_random_number(0, space.n_agents, i, 2)

            # Calculates the selected agents distances (eq. 16)
            D1 = 1 / space.n_agents * agent.fit
            D2 = 1 / space.n_agents * space.agents[R[0]].fit
            D3 = 1 / space.n_agents * space.agents[R[1]].fit

            # Calculates the selected agents velocities (eq. 17)
            V1 = r.generate_uniform_random_number() * (1 - D1)
            V2 = r.generate_uniform_random_number() * (1 - D2)
            V3 = r.generate_uniform_random_number() * (1 - D3)

            # Calculates both `x` and `y` distance differences (eq. 18 and 19)
            x = np.fabs(D3 - D1)
            y = np.fabs(D3 - D2)

            # Calculates both `x1` and `y1` constraints (eq. 4 and 7)
            x1 = (V3 * x) / (V1 - V3)
            y1 = (V2 * x) / (V1 - V3)

            # Generates a random uniform number
            rnd = r.generate_uniform_random_number()

            # If third agent's velocity is smaller than first agent's velocity
            if V3 < V1:
                # If difference between `y` gaps is bigger than `x1`
                if (y - y1) > x1:
                    # Calculates the condition velocity (eq. 23)
                    Vco = V1 / (V1 - V3)

                    # Updates the temporary agent's position accordingly (eq. 20)
                    a.position += Vco * rnd * (a.position - space.agents[R[1]].position)

                # If difference between `y` gaps is smaller than `x1`
                else:
                    # Updates the temporary agent's position accordingly (eq. 21)
                    a.position += rnd * (a.position - space.agents[R[0]].position)

            # If third agent's velocity is bigger than first agent's velocity
            else:
                # Updates the temporary agent's position accordingly (eq. 22)
                a.position += rnd * (space.agents[R[1]].position - a.position)

            # Clips the temporary agent
            a.clip_by_bound()

            # Re-evaluates its fitness
            a.fit = function(a.position)

            # If temporary agent is better than old one
            if a.fit < agent.fit:
                # Copies its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
