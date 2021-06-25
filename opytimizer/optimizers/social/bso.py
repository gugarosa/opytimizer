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

        # Probability of selecting an idea from a single cluster
        self.p_single_idea = 0.4

        # Probability of selecting an idea from a pair of clusters
        self.p_double_idea = 0.3

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
    def p_single_idea(self):
        """float: Probability of selecting an idea from a single cluster.

        """

        return self._p_single_idea

    @p_single_idea.setter
    def p_single_idea(self, p_single_idea):
        if not isinstance(p_single_idea, (float, int)):
            raise e.TypeError('`p_single_idea` should be a float or integer')
        if p_single_idea < 0 or p_single_idea > 1:
            raise e.ValueError('`p_single_idea` should be between 0 and 1')

        self._p_single_idea = p_single_idea

    @property
    def p_double_idea(self):
        """float: Probability of selecting an idea from a pair of clusters.

        """

        return self._p_double_idea

    @p_double_idea.setter
    def p_double_idea(self, p_double_idea):
        if not isinstance(p_double_idea, (float, int)):
            raise e.TypeError('`p_double_idea` should be a float or integer')
        if p_double_idea < 0 or p_double_idea > 1:
            raise e.ValueError('`p_double_idea` should be between 0 and 1')

        self._p_double_idea = p_double_idea

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

        #
        ideas_idx_per_cluster = []
        best_idx_per_cluster = []

        #
        for i in range(self.k):
            #
            ideas_idx = np.where(labels == i)[0]

            if len(ideas_idx) > 0:
                #
                ideas_idx_per_cluster.append(ideas_idx)
            else:
                ideas_idx_per_cluster.append([])

            #
            ideas_per_cluster = [(agents[j], j)
                                 for j in ideas_idx_per_cluster[i]]
            ideas_per_cluster.sort(key=lambda x: x[0].fit)

            if len(ideas_per_cluster) > 0:
                #
                best_idx_per_cluster.append(ideas_per_cluster[0][1])
            else:
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

            # If probability of selecting a cluster is bigger than random number
            if self.p_single_cluster > r1:
                # Randomly selects a cluster
                c = r.generate_integer_random_number(0, self.k)

                # If there are available ideas in the cluster
                if len(ideas_idx_per_cluster[c]) > 0:
                    # Generates a random number
                    r2 = r.generate_uniform_random_number()

                    # If selection should come from best cluster
                    if self.p_single_idea > r2:
                        # Updates the temporary agent's position with cluster's position
                        a.position = copy.deepcopy(
                            space.agents[best_idx_per_cluster[c]].position)

                    # If selection should come from a random agent in cluster
                    else:
                        j = r.generate_integer_random_number(
                            0, len(ideas_idx_per_cluster[c]))
                        a.position = copy.deepcopy(
                            space.agents[ideas_idx_per_cluster[c][j]].position)
            else:
                if self.k > 1:
                    c1 = r.generate_integer_random_number(0, self.k)
                    c2 = r.generate_integer_random_number(0, self.k, c1)

                    if len(ideas_idx_per_cluster[c1]) > 0 and len(ideas_idx_per_cluster[c2]) > 0:

                        r3 = r.generate_uniform_random_number()
                        r4 = r.generate_uniform_random_number()

                        if self.p_double_idea > r3:
                            a.position = r4 * space.agents[best_idx_per_cluster[c1]].position + (
                                1 - r4) * space.agents[best_idx_per_cluster[c2]].position
                        else:
                            u = r.generate_integer_random_number(
                                0, len(ideas_idx_per_cluster[c1]))
                            v = r.generate_integer_random_number(
                                0, len(ideas_idx_per_cluster[c2]))
                            a.position = r4 * space.agents[ideas_idx_per_cluster[c1][u]].position + (
                                1 - r4) * space.agents[ideas_idx_per_cluster[c2][v]].position

            noise = (0.5 * n_iterations - iteration) / agent.n_variables
            r5 = r.generate_uniform_random_number() * (1 / (1 + np.exp(-noise)))

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
