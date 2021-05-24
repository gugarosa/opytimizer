"""Lion Optimization Algorithm.
"""

import copy

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Agent, Optimizer

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
        super(Lion, self).__init__(n_variables,
                                   n_dimensions, lower_bound, upper_bound)

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

        # Index of hunting group
        self.group = 0

    @property
    def best_position(self):
        """np.array: N-dimensional array of best positions.

        """

        return self._best_position

    @best_position.setter
    def best_position(self, best_position):
        if not isinstance(best_position, np.ndarray):
            raise e.TypeError('`best_position` should be a numpy array')

        self._best_position = best_position

    @property
    def p_fit(self):
        """float: Previous fitness value.

        """

        return self._p_fit

    @p_fit.setter
    def p_fit(self, p_fit):
        if not isinstance(p_fit, (float, int, np.int32, np.int64)):
            raise e.TypeError('`p_fit` should be a float or integer')

        self._p_fit = p_fit

    @property
    def nomad(self):
        """bool: Whether lion is nomad or not.

        """

        return self._nomad

    @nomad.setter
    def nomad(self, nomad):
        if not isinstance(nomad, bool):
            raise e.TypeError('`nomad` should be a boolean')

        self._nomad = nomad

    @property
    def female(self):
        """bool: Whether lion is female or not.

        """

        return self._female

    @female.setter
    def female(self, female):
        if not isinstance(female, bool):
            raise e.TypeError('`female` should be a boolean')

        self._female = female

    @property
    def pride(self):
        """int: Index of pride.

        """

        return self._pride

    @pride.setter
    def pride(self, pride):
        if not isinstance(pride, int):
            raise e.TypeError('`pride` should be an integer')
        if pride < 0:
            raise e.ValueError('`pride` should be > 0')

        self._pride = pride

    @property
    def group(self):
        """int: Index of hunting group.

        """

        return self._group

    @group.setter
    def group(self, group):
        if not isinstance(group, int):
            raise e.TypeError('`group` should be an integer')
        if group < 0:
            raise e.ValueError('`group` should be > 0')

        self._group = group


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

        # Percentage of nomad lions
        self.N = 0.2

        # Number of prides
        self.P = 4

        # Percentage of female lions
        self.S = 0.8

        # Percentage of roaming lions
        self.R = 0.2

        # Mating probability
        self.Ma = 0.3

        # Mutation probability
        self.Mu = 0.2

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def N(self):
        """float: Percentage of nomad lions.

        """

        return self._N

    @N.setter
    def N(self, N):
        if not isinstance(N, (float, int)):
            raise e.TypeError('`N` should be a float or integer')
        if N < 0 or N > 1:
            raise e.ValueError('`N` should be between 0 and 1')

        self._N = N

    @property
    def P(self):
        """int: Number of prides.

        """

        return self._P

    @P.setter
    def P(self, P):
        if not isinstance(P, int):
            raise e.TypeError('`P` should be an integer')
        if P <= 0:
            raise e.ValueError('`P` should be > 0')

        self._P = P

    @property
    def S(self):
        """float: Percentage of female lions.

        """

        return self._S

    @S.setter
    def S(self, S):
        if not isinstance(S, (float, int)):
            raise e.TypeError('`S` should be a float or integer')
        if S < 0 or S > 1:
            raise e.ValueError('`S` should be between 0 and 1')

        self._S = S

    @property
    def R(self):
        """float: Percentage of roaming lions.

        """

        return self._R

    @R.setter
    def R(self, R):
        if not isinstance(R, (float, int)):
            raise e.TypeError('`R` should be a float or integer')
        if R < 0 or R > 1:
            raise e.ValueError('`R` should be between 0 and 1')

        self._R = R

    @property
    def Ma(self):
        """float: Mating probability.

        """

        return self._Ma

    @Ma.setter
    def Ma(self, Ma):
        if not isinstance(Ma, (float, int)):
            raise e.TypeError('`Ma` should be a float or integer')
        if Ma < 0 or Ma > 1:
            raise e.ValueError('`Ma` should be between 0 and 1')

        self._Ma = Ma

    @property
    def Mu(self):
        """float: Mutation probability.

        """

        return self._Mu

    @Mu.setter
    def Mu(self, Mu):
        if not isinstance(Mu, (float, int)):
            raise e.TypeError('`Mu` should be a float or integer')
        if Mu < 0 or Mu > 1:
            raise e.ValueError('`Mu` should be between 0 and 1')

        self._Mu = Mu

    def compile(self, space):
        """Compiles additional information that is used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Replaces the current agents with a derived Lion structure
        space.agents = [Lion(agent.n_variables, agent.n_dimensions, agent.lb,
                             agent.ub, agent.position, agent.fit) for agent in space.agents]

        # Calculates the number of nomad lions and their genders
        n_nomad = int(self.N * space.n_agents)
        nomad_gender = d.generate_bernoulli_distribution(1 - self.S, n_nomad)

        # Iterates through all possible nomads
        for i, agent in enumerate(space.agents[:n_nomad]):
            # Toggles to `True` the nomad property
            agent.nomad = True

            # Defines the gender according to Bernoulli distribution
            agent.female = bool(nomad_gender[i])

        # Calculates the gender of pride lions
        pride_gender = d.generate_bernoulli_distribution(
            self.S, space.n_agents - n_nomad)

        # Iterates through all possible prides
        for i, agent in enumerate(space.agents[n_nomad:]):
            # Defines the gender according to Bernoulli distribution
            agent.female = bool(pride_gender[i])

            # Allocates to the corresponding pride
            agent.pride = i % self.P

    def _get_nomad_lions(self, agents):
        """Gets all nomad lions.

        Args:
            agents (list): Agents.

        Returns:
            A list of nomad lions.

        """

        # Returns a list of nomad lions
        return [agent for agent in agents if agent.nomad is True]

    def _get_pride_lions(self, agents):
        """Gets all non-nomad (pride) lions.

        Args:
            agents (list): Agents.

        Returns:
            A list of lists, where each one indicates a particular pride with its lions.

        """

        # Gathers all non-nomad lions
        agents = [agent for agent in agents if agent.nomad is False]

        # Returns a list of lists of prides
        return [[agent for agent in agents if agent.pride == i] for i in range(self.P)]

    def _hunting(self, prides, function):
        """Performs the hunting procedure (s. 2.2.2).

        Args:
            prides (list): List of prides holding their corresponding lions.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterates through all prides
        for pride in prides:
            # Iterates through all agents in pride
            for agent in pride:
                # If agent is female
                if agent.female:
                    # Allocates to a random hunting group
                    agent.group = r.generate_integer_random_number(high=4)

                # If agent is male
                else:
                    # Allocates to a null group (no-hunting)
                    agent.group = 0

            # Calculates the fitness sum of first, second and third groups
            first_group = np.sum(
                [agent.fit for agent in pride if agent.group == 1])
            second_group = np.sum(
                [agent.fit for agent in pride if agent.group == 2])
            third_group = np.sum(
                [agent.fit for agent in pride if agent.group == 3])

            # Averages the position of the prey (lions in group 0)
            prey = np.mean(
                [agent.position for agent in pride if agent.group == 0], axis=0)

            # Calculates the group indexes and their corresponding
            # positions: center, left and right
            groups_idx = np.argsort(
                [first_group, second_group, third_group]) + 1
            center = groups_idx[0]
            left = groups_idx[1]
            right = groups_idx[2]

            # Iterates through all agents in pride
            for agent in pride:
                # If agent belongs to the center group
                if agent.group == center:
                    # Iterates through all decision variables
                    for j in range(agent.n_variables):
                        # If agent's position is smaller than prey's
                        if agent.position[j] < prey[j]:
                            # Updates its position (eq. 5 - top)
                            agent.position[j] = r.generate_uniform_random_number(
                                agent.position[j], prey[j])
                        else:
                            # Updates its position (eq. 5 - bottom)
                            agent.position[j] = r.generate_uniform_random_number(
                                prey[j], agent.position[j])

                # If agent belongs to the left or right groups
                if agent.group in [left, right]:
                    # Iterates through all decision variables
                    for j in range(agent.n_variables):
                        # Calculates the encircling position
                        encircling = 2 * prey[j] - agent.position[j]

                        # If encircling's position is smaller than prey's
                        if encircling < prey[j]:
                            # Updates its position (eq. 4 - top)
                            agent.position[j] = r.generate_uniform_random_number(
                                encircling, prey[j])
                        else:
                            # Updates its position (eq. 4 - bottom)
                            agent.position[j] = r.generate_uniform_random_number(
                                prey[j], encircling)

                # Clips their limits
                agent.clip_by_bound()

                # Defines the previous fitness and calculates the newer one
                agent.p_fit = copy.deepcopy(agent.fit)
                agent.fit = function(agent.position)

                # If new fitness is better than old one
                if agent.fit < agent.p_fit:
                    # Updates its best position
                    agent.best_position = copy.deepcopy(agent.position)

                    # Calculates the probability of improvement
                    p_improvement = agent.fit / agent.fit

                    # Updates the prey's position
                    r1 = r.generate_uniform_random_number()
                    prey += r1 * p_improvement * (prey - agent.position)

    def _moving_safe_place(self, prides):
        for pride in prides:
            n_improved = np.sum(
                [1 for agent in pride if agent.fit < agent.p_fit])
            fits = [agent.fit for agent in pride]
            tournament_size = np.maximum(2, int(np.ceil(n_improved / 2)))

            for agent in pride:
                if agent.group == 0 and agent.female is True:
                    winner = g.tournament_selection(
                        fits, 1, tournament_size)[0]
                    distance = g.euclidean_distance(
                        agent.position, pride[winner].position)
                    r1 = r.generate_uniform_random_number()
                    u = r.generate_uniform_random_number(-1, 1)
                    theta = np.random.uniform(-np.pi/6, np.pi/6)
                    R1 = pride[winner].position - agent.position
                    R2 = np.random.randn(*R1.T.shape)
                    R2 = R2.T - R2.dot(R1) * R1 / \
                        (np.linalg.norm(R1) ** 2 + c.EPSILON)
                    agent.position += 2 * distance * r1 * \
                        R1 + u * np.tan(theta) * R2 * distance

    def _roaming(self, prides, function):
        for pride in prides:
            # probs = [self.P] * len(pride)
            n_territory = int(len(pride) * self.P)
            choices = r.generate_integer_random_number(
                high=len(pride), size=n_territory)
            # print(choices)
            for agent in pride:
                if agent.female is not True:
                    for choice in choices:
                        angle = np.random.uniform(-np.pi/6, np.pi/6)
                        distance = g.euclidean_distance(
                            pride[choice].best_position, agent.position)
                        step = r.generate_uniform_random_number(
                            0, 2 * distance)
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
                        males = [
                            agent for agent in pride if agent.female is not True and r.generate_uniform_random_number() < 0.5]

                        beta = r.generate_gaussian_random_number(0.5, 0.1)

                        males_avg = np.mean(
                            [male.position for male in males], axis=0)

                        a1 = copy.deepcopy(agent)
                        a2 = copy.deepcopy(agent)

                        a1.position = beta * a1.position + \
                            (1 - beta) * males_avg
                        a1.position = (1 - beta) * \
                            a1.position + beta * males_avg

                        for j in range(agent.n_variables):
                            r2 = r.generate_uniform_random_number()
                            r3 = r.generate_uniform_random_number()

                            if r2 < self.Mu:
                                a1.position[j] = r.generate_uniform_random_number(
                                    a1.lb[j], a1.ub[j])

                            if r3 < self.Mu:
                                a2.position[j] = r.generate_uniform_random_number(
                                    a2.lb[j], a2.ub[j])

                        a1.clip_by_bound()
                        a2.clip_by_bound()

                        a1.best_position = copy.deepcopy(a1.position)
                        a1.female = bool(beta >= 0.5)
                        a1.fit = function(a1.position)
                        a2.best_position = copy.deepcopy(a2.position)
                        a2.female = bool(beta >= 0.5)
                        a2.fit = function(a2.position)

                        offspring_p.append(a1)
                        offspring_p.append(a2)
            offspring.append(offspring_p)

        return offspring

    def update(self, space, function):
        """Wraps Lion Optimization Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.

        """

        # Gets the non-nomad (pride) lions
        prides = self._get_pride_lions(space.agents)

        # Performs the hunting procedure
        self._hunting(prides, function)

        # Move prides to safe locations
        self._moving_safe_place(prides)

        # Performs the roaming procedure
        self._roaming(prides, function)

        # Generates offsprings from mating
        offspring = self._mating(prides, function)
