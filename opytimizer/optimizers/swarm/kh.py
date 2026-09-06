# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Krill Herd.

References:
    A. Gandomi and A. Alavi. Krill herd: A new bio-inspired optimization algorithm.
    Communications in Nonlinear Science and Numerical Simulation (2012).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class KH(Optimizer):
    """Search with krill neighborhood motion, foraging, diffusion, and genetic operators.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure krill movement and genetic operator scales.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``N_max`` (0.01) scales induced motion, and ``w_n`` (0.42) weights its previous value.
            ``NN`` (5) divides the mean inter-agent distance to define the sensing radius.
            ``V_f`` (0.02) scales foraging speed, and ``w_f`` (0.38) weights previous foraging.
            ``D_max`` (0.002) scales physical diffusion, and ``C_t`` (0.5) scales the position-update time step.
            ``Cr`` (0.2) and ``Mu`` (0.05) scale fitness-adjusted crossover and mutation probabilities.
            Compilation zeros ``motion`` and ``foraging`` with shape ``(n_agents, n_variables, n_dimensions)``.

        """

        super(KH, self).__init__()

        self.N_max = 0.01
        self.w_n = 0.42

        self.NN = 5

        self.V_f = 0.02
        self.w_f = 0.38
        self.D_max = 0.002
        self.C_t = 0.5

        self.Cr = 0.2
        self.Mu = 0.05

        self.build(params)

    def compile(self, space: Space) -> None:
        self.motion = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))
        self.foraging = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def _food_location(self, agents: list[Agent], function: Callable) -> Agent:
        food = copy.deepcopy(agents[0])

        sum_fitness_pos = np.sum([1 / (agent.fit + c.EPSILON) * agent.position for agent in agents], axis=0)
        sum_fitness = np.sum([1 / (agent.fit + c.EPSILON) for agent in agents])

        food.position = sum_fitness_pos / sum_fitness
        food.clip_by_bound()

        food.fit = function(food.position)

        return food

    def _sensing_distance(self, agents: list[Agent], idx: int) -> tuple[float, float]:
        eucl_distance = [np.linalg.norm(agents[idx].position - agent.position) for agent in agents]
        distance = np.sum(eucl_distance) / (self.NN * len(agents))

        return distance, eucl_distance

    def _get_neighbours(
        self,
        agents: list[Agent],
        idx: int,
        sensing_distance: float,
        eucl_distance: list[float],
    ) -> list[Agent]:
        neighbours = []

        for i, dist in enumerate(eucl_distance):
            if idx != i and sensing_distance > dist:
                neighbours.append(agents[i])

        return neighbours

    def _local_alpha(self, agent: Agent, worst: Agent, best: Agent, neighbours: list[Agent]) -> float:
        fitness = [(agent.fit - neighbour.fit) / (worst.fit - best.fit + c.EPSILON) for neighbour in neighbours]

        position = [
            (neighbour.position - agent.position) / (np.linalg.norm(neighbour.position - agent.position) + c.EPSILON)
            for neighbour in neighbours
        ]

        alpha = np.sum([fit * pos for (fit, pos) in zip(fitness, position)], axis=0)

        return alpha

    def _target_alpha(self, agent: Agent, worst: Agent, best: Agent, C_best: float) -> float:
        fitness = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

        position = (best.position - agent.position) / (np.linalg.norm(best.position - agent.position) + c.EPSILON)

        alpha = C_best * fitness * position

        return alpha

    def _neighbour_motion(
        self,
        agents: list[Agent],
        idx: int,
        iteration: int,
        n_iterations: int,
        motion: np.ndarray,
    ) -> np.ndarray:
        # Calculates the sensing distance (eq. 7)
        sensing_distance, eucl_distance = self._sensing_distance(agents, idx)

        # Calculates the local alpha (eq. 4)
        neighbours = self._get_neighbours(agents, idx, sensing_distance, eucl_distance)
        alpha_l = self._local_alpha(agents[idx], agents[-1], agents[0], neighbours)

        # Calculates the effective coefficient (eq. 9)
        C_best = 2 * (np.random.uniform(0.0, 1.0, 1) + iteration / n_iterations)

        # Calculates the target alpha (eq. 8)
        alpha_t = self._target_alpha(agents[idx], agents[-1], agents[0], C_best)

        # Calculates the neighbour motion (eq. 2)
        neighbour_motion = self.N_max * (alpha_l + alpha_t) + self.w_n * motion

        return neighbour_motion

    def _food_beta(self, agent: Agent, worst: Agent, best: Agent, food: np.ndarray, C_food: float) -> np.ndarray:
        fitness = (agent.fit - food.fit) / (worst.fit - best.fit + c.EPSILON)

        position = (food.position - agent.position) / (np.linalg.norm(food.position - agent.position) + c.EPSILON)

        beta = C_food * fitness * position

        return beta

    def _best_beta(self, agent: Agent, worst: Agent, best: Agent) -> np.ndarray:
        fitness = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

        position = (best.position - agent.position) / (np.linalg.norm(best.position - agent.position) + c.EPSILON)

        beta = fitness * position

        return beta

    def _foraging_motion(
        self,
        agents: list[Agent],
        idx: int,
        iteration: int,
        n_iterations: int,
        food: np.ndarray,
        foraging: np.ndarray,
    ) -> np.ndarray:
        # Calculates the food coefficient (eq. 14)
        C_food = 2 * (1 - iteration / n_iterations)

        # Calculates the food attraction (eq. 13)
        beta_f = self._food_beta(agents[idx], agents[-1], agents[0], food, C_food)

        # Calculates the best attraction (eq. 15)
        beta_b = self._best_beta(agents[idx], agents[-1], agents[0])

        # Calculates the foraging motion (eq. 10)
        foraging_motion = self.V_f * (beta_f + beta_b) + self.w_f * foraging

        return foraging_motion

    def _physical_diffusion(self, n_variables: int, n_dimensions: int, iteration: int, n_iterations: int) -> float:
        # Physical diffusion decays with the iteration budget (eq. 16-17)
        r1 = np.random.uniform(-1, 1, (n_variables, n_dimensions))
        physical_diffusion = self.D_max * (1 - iteration / n_iterations) * r1

        return physical_diffusion

    def _update_position(
        self,
        agents: list[Agent],
        idx: int,
        iteration: int,
        n_iterations: int,
        food: np.ndarray,
        motion: np.ndarray,
        foraging: np.ndarray,
    ) -> np.ndarray:
        neighbour_motion = self._neighbour_motion(agents, idx, iteration, n_iterations, motion)

        foraging_motion = self._foraging_motion(agents, idx, iteration, n_iterations, food, foraging)

        physical_diffusion = self._physical_diffusion(
            agents[idx].n_variables, agents[idx].n_dimensions, iteration, n_iterations
        )

        # Calculates the delta (eq. 19)
        delta_t = self.C_t * np.sum(agents[idx].ub - agents[idx].lb)

        # Updates the current agent's position (eq. 18)
        new_position = agents[idx].position + delta_t * (neighbour_motion + foraging_motion + physical_diffusion)

        return new_position

    def _crossover(self, agents: list[Agent], idx: int) -> Agent:
        a = copy.deepcopy(agents[idx])
        m = r.integer(0, len(agents), exclude=idx, size=None)

        # Fitness-adjusted crossover probability (eq. 21)
        Cr = self.Cr * ((agents[idx].fit - agents[0].fit) / (agents[-1].fit - agents[0].fit + c.EPSILON))

        for j in range(a.n_variables):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < Cr:
                a.position[j] = copy.deepcopy(agents[m].position[j])

        return a

    def _mutation(self, agents: list[Agent], idx: int) -> Agent:
        a = copy.deepcopy(agents[idx])

        p = r.integer(0, len(agents), exclude=idx, size=None)
        q = r.integer(0, len(agents), exclude=idx, size=None)

        # Fitness-adjusted mutation probability (eq. 22)
        Mu = self.Mu / ((agents[idx].fit - agents[0].fit) / (agents[-1].fit - agents[0].fit + c.EPSILON) + c.EPSILON)

        for j in range(a.n_variables):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < Mu:
                r2 = np.random.uniform(0.0, 1.0, 1)
                a.position[j] = agents[0].position[j] + r2 * (agents[p].position[j] - agents[q].position[j])

        return a

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        space.agents.sort(key=lambda x: x.fit)

        # Calculates the food location (eq. 12)
        food = self._food_location(space.agents, function)

        for i, _ in enumerate(space.agents):
            space.agents[i].position = self._update_position(
                space.agents,
                i,
                iteration,
                n_iterations,
                food,
                self.motion[i],
                self.foraging[i],
            )

            space.agents[i] = self._crossover(space.agents, i)
            space.agents[i] = self._mutation(space.agents, i)
