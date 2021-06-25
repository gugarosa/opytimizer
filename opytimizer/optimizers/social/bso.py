"""Brain Storm Optimization.
"""

import copy

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer

logger = l.get_logger(__name__)


class BSO(Optimizer):
    """A BSO class, inherited from Optimizer.

    This is the designed class to define BSO-related
    variables and methods.

    References:
        Y. Shi. Brain Storm Optimization Algorithm.
        International Conference in Swarm Intelligence (2011).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BSO.')

        # Overrides its parent class with the receiving params
        super(BSO, self).__init__()

        # Number of clusters
        self.k = 5

        # Probability of selecting a single cluster
        self.p_single_cluster = 0.3

        # Probability of selecting the best idea from a single cluster
        self.p_single_best = 0.4

        # Probability of selecting the best idea from a pair of clusters
        self.p_double_best = 0.3

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def k(self):
        """int: Number of clusters.

        """

        return self._k

    @k.setter
    def k(self, k):
        if not isinstance(k, int):
            raise e.TypeError('`k` should be an integer')
        if k <= 0:
            raise e.ValueError('`k` should be > 0')

        self._k = k

    @property
    def p_single_cluster(self):
        """float: Probability of selecting a single cluster.

        """

        return self._p_single_cluster

    @p_single_cluster.setter
    def p_single_cluster(self, p_single_cluster):
        if not isinstance(p_single_cluster, (float, int)):
            raise e.TypeError('`p_single_cluster` should be a float or integer')
        if p_single_cluster < 0 or p_single_cluster > 1:
            raise e.ValueError('`p_single_cluster` should be between 0 and 1')

        self._p_single_cluster = p_single_cluster

    @property
    def p_single_best(self):
        """float: Probability of selecting the best idea from a single cluster.

        """

        return self._p_single_best

    @p_single_best.setter
    def p_single_best(self, p_single_best):
        if not isinstance(p_single_best, (float, int)):
            raise e.TypeError('`p_single_best` should be a float or integer')
        if p_single_best < 0 or p_single_best > 1:
            raise e.ValueError('`p_single_best` should be between 0 and 1')

        self._p_single_best = p_single_best

    @property
    def p_double_best(self):
        """float: Probability of selecting the best idea from a pair of clusters.

        """

        return self._p_double_best

    @p_double_best.setter
    def p_double_best(self, p_double_best):
        if not isinstance(p_double_best, (float, int)):
            raise e.TypeError('`p_double_best` should be a float or integer')
        if p_double_best < 0 or p_double_best > 1:
            raise e.ValueError('`p_double_best` should be between 0 and 1')

        self._p_double_best = p_double_best

    def _clusterize(self, agents):
        """Performs the clusterization over the agents' positions.

        Args:
            agents (list): List of agents.

        Returns:
            Agents indexes and best agent index per cluster.

        """

        # Gathers current agents' positions (ideas)
        ideas = np.array([agent.position for agent in agents])

        # Performs the K-means clustering
        labels = g.kmeans(ideas, self.k)

        # Creates lists to ideas and best idea indexes per cluster
        ideas_idx_per_cluster, best_idx_per_cluster = [], []

        # Iterates through all possible clusters
        for i in range(self.k):
            # Gathers ideas that belongs to current cluster
            ideas_idx = np.where(labels == i)[0]

            # If there are any ideas
            if len(ideas_idx) > 0:
                # Appends them to the corresponding list
                ideas_idx_per_cluster.append(ideas_idx)

            # If not
            else:
                # Just appends an empty list for compatibility purposes
                ideas_idx_per_cluster.append([])

            # Gathers a tuple of sorted agents and their index for the current cluster
            ideas_per_cluster = [(agents[j], j) for j in ideas_idx_per_cluster[i]]
            ideas_per_cluster.sort(key=lambda x: x[0].fit)

            # If there are any ideas
            if len(ideas_per_cluster) > 0:
                # Appends the best index to the corresponding list
                best_idx_per_cluster.append(ideas_per_cluster[0][1])

            # If not
            else:
                # Just appends a `-1` for compatibility purposes
                best_idx_per_cluster.append(-1)

        return ideas_idx_per_cluster, best_idx_per_cluster

    def update(self, space, function, iteration, n_iterations):
        """Wraps Brain Storm Optimization over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Current iteration.

        """

        # Clusterizes the current agents
        ideas_idx_per_cluster, best_idx_per_cluster = self._clusterize(space.agents)

        # Iterates through all agents
        for agent in space.agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If probability of selecting a single cluster is smaller than random number
            if self.p_single_cluster < r1:
                # Randomly selects a cluster
                c = r.generate_integer_random_number(0, self.k)

                # If there are available ideas in the cluster
                if len(ideas_idx_per_cluster[c]) > 0:
                    # Generates a random number
                    r2 = r.generate_uniform_random_number()

                    # If selection should come from best cluster
                    if self.p_single_best < r2:
                        # Updates the temporary agent's position
                        a.position = copy.deepcopy(space.agents[best_idx_per_cluster[c]].position)

                    # If selection should come from a random agent in cluster
                    else:
                        # Gathers an index from agent in cluster
                        j = r.generate_integer_random_number(0, len(ideas_idx_per_cluster[c]))

                        # Updates the temporary agent's position
                        a.position = copy.deepcopy(space.agents[ideas_idx_per_cluster[c][j]].position)

            # If probability of selecting a single cluster is bigger than random number
            else:
                # Checks if there are 2+ available clusters
                if self.k > 1:
                    # Selects two different clusters
                    c1 = r.generate_integer_random_number(0, self.k)
                    c2 = r.generate_integer_random_number(0, self.k, c1)

                    # If both clusters have at least one idea
                    if len(ideas_idx_per_cluster[c1]) > 0 and len(ideas_idx_per_cluster[c2]) > 0:
                        # Generates a new set of random numbers
                        r3 = r.generate_uniform_random_number()
                        r4 = r.generate_uniform_random_number()

                        ## If selection should come from best clusters
                        if self.p_double_best < r3:
                            # Updates the temporary agent's position
                            a.position = r4 * space.agents[best_idx_per_cluster[c1]].position + \
                                         (1 - r4) * space.agents[best_idx_per_cluster[c2]].position

                        # If selection should come from random agents in clusters
                        else:
                            # Gathers indexes from agents in clusters
                            u = r.generate_integer_random_number(0, len(ideas_idx_per_cluster[c1]))
                            v = r.generate_integer_random_number(0, len(ideas_idx_per_cluster[c2]))

                            # Updates the temporary agent's position
                            a.position = r4 * space.agents[ideas_idx_per_cluster[c1][u]].position + \
                                         (1 - r4) * space.agents[ideas_idx_per_cluster[c2][v]].position

            # Generates a random noise and activates it with a sigmoid function
            noise = (0.5 * n_iterations - iteration) / agent.n_variables
            r5 = r.generate_uniform_random_number() * (1 / (1 + np.exp(-noise)))

            # Updates the temporary agent's position
            a.position += r5 * r.generate_gaussian_random_number()

            # Checks agent's limits
            a.clip_by_bound()

            # Re-evaluates the temporary agent
            a.fit = function(a.position)

            # If temporary agent's fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Replace its position and fitness
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
