"""Magnetic Optimization Algorithm."""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class MOA(Optimizer):
    """An MOA class, inherited from Optimizer.

    This is the designed class to define MOA-related
    variables and methods.

    References:
        M.-H. Tayarani and M.-R. Akbarzadeh. Magnetic-inspired optimization algorithms: Operators and structures.
        Swarm and Evolutionary Computation (2014).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(MOA, self).__init__()

        self.alpha = 1.0
        self.rho = 2.0

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        if not np.sqrt(space.n_agents).is_integer():
            raise ValueError("`n_agents` should have a perfect square")

    def update(self, space: Space) -> None:
        """Wraps Magnetic Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        space.agents.sort(key=lambda x: x.fit)

        # Gathers the best and worst agents and calculates a list of normalized fitness (eq. 2)
        best, worst = space.agents[0], space.agents[-1]
        fitness = [
            (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)
            for agent in space.agents
        ]

        # Calculates the masses (eq. 3)
        mass = [self.alpha + self.rho * fit for fit in fitness]

        for i, agent in enumerate(space.agents):
            # Gathers the agents neighbours (eq. 4)
            root = np.sqrt(space.n_agents)
            north = int((i - root) % space.n_agents)
            south = int((i + root) % space.n_agents)
            west = int((i - 1) + ((i + root - 1) % root) // (root - 1) * root)
            east = int((i + 1) - (i % root) // (root - 1) * root)
            neighbours = [north, south, west, east]

            force = 0

            for n in neighbours:
                # Calculates the distance between current agent and neighbour (eq. 7)
                distance = np.linalg.norm(agent.position - space.agents[n].position)

                # Calculates the force between agents (eq. 5)
                force += (
                    (space.agents[n].position - agent.position)
                    * fitness[n]
                    / (distance + c.EPSILON)
                )

            force = np.mean(force)

            # Updates the agent's velocity(eq. 9)
            r1 = np.random.uniform(0.0, 1.0, 1)
            velocity = force / mass[i] * r1

            # Updates the agent's position (eq. 10)
            agent.position += velocity
            agent.clip_by_bound()
