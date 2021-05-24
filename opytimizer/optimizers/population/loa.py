"""Lion Optimization Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer
from opytimizer.core import Agent

logger = l.get_logger(__name__)


class Lion(Agent):
    """A Lion class complements its inherited parent with additional information neeeded by
    the Lion Optimization Algorithm.

    """
    
    def __init__(self, n_variables, n_dimensions, lower_bound, upper_bound, position, fit):
        """Initialization method.

        Args:
            n_variables (int): Number of decision variables.
            n_dimensions (int): Number of dimensions.
            lower_bound (list, tuple, np.array): Minimum possible values.
            upper_bound (list, tuple, np.array): Maximum possible values.
            position (np.array): Position array.
            fit (float): Fitness value.

        """

        # Overrides its parent class with the receiving params
        super(Lion, self).__init__(n_variables, n_dimensions, lower_bound, upper_bound)

        # Copies the current position and fitness to overrided object
        self.position = copy.deepcopy(position)
        self.fit = copy.deepcopy(fit)

        # Best position
        self.best_position = copy.deepcopy(position)
        
        # Previous fitness
        self.p_fit = copy.deepcopy(fit)

        # Whether lion is nomad or not
        self.nomad = False

        # Whether lion is female or not
        self.female = False

        # Index of pride
        self.pride = 0

        # Index of hynting group
        self.group = 0


class LOA(Optimizer):
    """An LOA class, inherited from Optimizer.

    This is the designed class to define LOA-related
    variables and methods.

    References:
        M. Yazdani and F. Jolai. Lion Optimization Algorithm (LOA): A nature-inspired metaheuristic algorithm.
        Journal of Computational Design and Engineering (2016).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> LOA.')

        # Overrides its parent class with the receiving params
        super(LOA, self).__init__()

        #
        self.N = 0.2

        #
        self.P = 4

        #
        self.S = 0.8

        #
        self.R = 0.2

        #
        self.Ma = 0.3

        #
        self.Mu = 0.2

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def compile(self, space):
        space.agents = [Lion(agent.n_variables, agent.n_dimensions, agent.lb, agent.ub, agent.position, agent.fit) for agent in space.agents]

        for agent in space.agents:
            is_nomad = r.generate_uniform_random_number()
            if is_nomad < self.N:
                agent.nomad = True
                is_female = r.generate_uniform_random_number()
                if is_female < (1 - self.S):
                    agent.female = True
            else:
                agent.nomad = False
                agent.pride = r.generate_integer_random_number(high=self.P)
                is_female = r.generate_uniform_random_number()
                if is_female < self.S:
                    agent.female = True
        # self.n_nomad = int(self.N * space.n_agents)
        # self.n_pride = (space.n_agents - self.n_nomad) // self.P

        # self.gender = [1] * space.n_agents

    def _get_nomad_lions(self, agents):
        return [agent for agent in agents if agent.nomad is True]

    def _get_pride_lions(self, agents):
        agents = [agent for agent in agents if agent.nomad is False]
        return [[agent for agent in agents if agent.pride == i] for i in range(self.P)]

    # def _get_pride_lions(self, agents):
    #     prides, genders = [], []
    #     for i in range(self.P):
    #         start, end = i * self.n_pride, (i + 1) * self.n_pride
    #         prides.append(agents[start:end])
    #         genders.append(self.gender[start:end])
    #     return prides, genders

    def _hunting(self, prides, function):
        for pride in prides:
            for agent in pride:
                if agent.female:
                    agent.group = r.generate_integer_random_number(high=4)
                else:
                    agent.group = 0

            first_group = np.sum([agent.fit for agent in pride if agent.group == 1])
            second_group = np.sum([agent.fit for agent in pride if agent.group == 2])
            third_group = np.sum([agent.fit for agent in pride if agent.group == 3])

            prey = np.mean([agent.position for agent in pride if agent.group == 0], axis=0)

            fits = [first_group, second_group, third_group]
            idx = np.argsort(fits)

            center = idx[0] + 1
            left = idx[1] + 1
            right = idx[2] + 1

            # print(first_group, second_group, third_group)
            # print(prey)

            # print(center, left, right)

            for agent in pride:
                if agent.group == center:
                    for j in range(agent.n_variables):
                        if agent.position[j] < prey[j]:
                            agent.position[j] = r.generate_uniform_random_number(agent.position[j], prey[j])
                        else:
                            agent.position[j] = r.generate_uniform_random_number(prey[j], agent.position[j])

                if agent.group in [left, right]:
                    for j in range(agent.n_variables):
                        if 2 * prey[j] - agent.position[j] < prey[j]:
                            agent.position[j] = r.generate_uniform_random_number(2 * prey[j] - agent.position[j], prey[j])
                        else:
                            agent.position[j] = r.generate_uniform_random_number(prey[j], 2 * prey[j] - agent.position[j])

                agent.clip_by_bound()

                agent.p_fit = copy.deepcopy(agent.fit)
                agent.fit = function(agent.position)

                if agent.fit < agent.p_fit:
                    p_improvement = agent.fit / agent.fit
                    agent.best_position = copy.deepcopy(agent.position)
                    r1 = r.generate_uniform_random_number()
                    prey += r1 * p_improvement * (prey - agent.position)

    def _moving_safe_place(self, prides):
        for pride in prides:
            n_improved = np.sum([1 for agent in pride if agent.fit < agent.p_fit])
            fits = [agent.fit for agent in pride]
            tournament_size = np.maximum(2, int(np.ceil(n_improved / 2)))
            
            for agent in pride:
                if agent.group == 0 and agent.female is True:
                    winner = g.tournament_selection(fits, 1, tournament_size)[0]
                    distance = g.euclidean_distance(agent.position, pride[winner].position)
                    r1 = r.generate_uniform_random_number()
                    u = r.generate_uniform_random_number(-1, 1)
                    theta = np.random.uniform(-np.pi/6, np.pi/6)
                    R1 = pride[winner].position - agent.position
                    R2 = np.random.randn(*R1.T.shape)
                    R2 = R2.T - R2.dot(R1) * R1 / (np.linalg.norm(R1) ** 2 + c.EPSILON)
                    agent.position += 2 * distance * r1 * R1 + u * np.tan(theta) * R2 * distance

    def _roaming(self, prides, function):
        for pride in prides:
            # probs = [self.P] * len(pride)
            n_territory = int(len(pride) * self.P)
            choices = r.generate_integer_random_number(high=len(pride), size=n_territory)
            # print(choices)
            for agent in pride:
                if agent.female is not True:
                    for choice in choices:
                        angle = np.random.uniform(-np.pi/6, np.pi/6)
                        distance = g.euclidean_distance(pride[choice].best_position, agent.position)
                        step = r.generate_uniform_random_number(0, 2 * distance)
                        agent.position += step * np.tan(angle)
                        agent.clip_by_bound()

                        agent.p_fit = copy.deepcopy(agent.fit)
                        agent.fit = function(agent.position)

                        if agent.fit < agent.p_fit:
                            agent.best_position = copy.deepcopy(agent.position)


    def _mating(self, prides, function):
        offspring = []
        for pride in prides:
            offspring_p = []
            for agent in pride:
                if agent.female:
                    r1 = r.generate_uniform_random_number()
                    if r1 < self.Ma:
                        males = [agent for agent in pride if agent.female is not True and r.generate_uniform_random_number() < 0.5]
                        
                        beta = r.generate_gaussian_random_number(0.5, 0.1)

                        males_avg = np.mean([male.position for male in males], axis=0)

                        a1 = copy.deepcopy(agent)
                        a2 = copy.deepcopy(agent)

                        a1.position = beta * a1.position + (1 - beta) * males_avg
                        a1.position = (1 - beta) * a1.position + beta * males_avg

                        for j in range(agent.n_variables):
                            r2 = r.generate_uniform_random_number()
                            r3 = r.generate_uniform_random_number()

                            if r2 < self.Mu:
                                a1.position[j] = r.generate_uniform_random_number(a1.lb[j], a1.ub[j])

                            if r3 < self.Mu:
                                a2.position[j] = r.generate_uniform_random_number(a2.lb[j], a2.ub[j])

                        a1.clip_by_bound()
                        a2.clip_by_bound()

                        a1.best_position = copy.deepcopy(a1.position)
                        a1.female = (beta >= 0.5)
                        a1.fit = function(a1.position)
                        a2.best_position = copy.deepcopy(a2.position)
                        a2.female = (beta >= 0.5)
                        a2.fit = function(a2.position)

                        offspring_p.append(a1)
                        offspring_p.append(a2)
            offspring.append(offspring_p)

        return offspring


    def update(self, space, function):
        # print(self._get_nomad_lions(space.agents))
        prides = self._get_pride_lions(space.agents)

        self._hunting(prides, function)

        self._moving_safe_place(prides)

        self._roaming(prides, function)

        offspring = self._mating(prides, function)

        

        # for agent in space.agents:
        #     print(agent.group)

