"""Henry Gas Solubility Optimization.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class HGSO(Optimizer):
    """An HGSO class, inherited from Optimizer.

    This is the designed class to define HGSO-related
    variables and methods.

    References:
        F. Hashim et al. Henry gas solubility optimization: A novel physics-based algorithm.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, algorithm='HGSO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> HGSO.')

        # Override its parent class with the receiving hyperparams
        super(HGSO, self).__init__(algorithm)

        # Number of clusters
        self.n_clusters = 2

        # Henry's coefficient constant
        self.l1 = 0.0005

        # Partial pressure constant
        self.l2 = 100

        # Constant
        self.l3 = 0.001

        # Influence of gases
        self.alpha = 1.0

        # Gas constant
        self.beta = 1.0

        # Solubility constant
        self.K = 1.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def n_clusters(self):
        """int: Number of clusters.

        """

        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        if not isinstance(n_clusters, int):
            raise e.TypeError('`n_clusters` should be an integer')
        if n_clusters <= 0:
            raise e.ValueError('`n_clusters` should be > 0')

        self._n_clusters = n_clusters

    @property
    def l1(self):
        """float: Henry's coefficient constant.

        """

        return self._l1

    @l1.setter
    def l1(self, l1):
        if not isinstance(l1, (float, int)):
            raise e.TypeError('`l1` should be a float or integer')
        if l1 < 0:
            raise e.ValueError('`l1` should be >= 0')

        self._l1 = l1

    @property
    def l2(self):
        """int: Partial pressure constant.

        """

        return self._l2

    @l2.setter
    def l2(self, l2):
        if not isinstance(l2, int):
            raise e.TypeError('`l2` should be an integer')
        if l2 <= 0:
            raise e.ValueError('`l2` should be > 0')

        self._l2 = l2

    @property
    def l3(self):
        """float: Constant.

        """

        return self._l3

    @l3.setter
    def l3(self, l3):
        if not isinstance(l3, (float, int)):
            raise e.TypeError('`l3` should be a float or integer')
        if l3 < 0:
            raise e.ValueError('`l3` should be >= 0')

        self._l3 = l3

    @property
    def alpha(self):
        """float: Influence of gases.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0:
            raise e.ValueError('`alpha` should be >= 0')

        self._alpha = alpha

    @property
    def beta(self):
        """float: Gas constant.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    @property
    def K(self):
        """float: Solubility constant.

        """

        return self._K

    @K.setter
    def K(self, K):
        if not isinstance(K, (float, int)):
            raise e.TypeError('`K` should be a float or integer')
        if K < 0:
            raise e.ValueError('`K` should be >= 0')

        self._K = K

    def _update_position(self, agent, cluster_agent, best_agent, solubility):
        """Updates the position of a single gas (eq. 10).

        Args:
            agent (Agent): Current agent.
            cluster_agent (Agent): Best cluster's agent.
            best_agent (Agent): Best agent.
            solubility (float): Solubility for current agent.

        Returns:
            An updated position.

        """

        # Calculates `gamma`
        gamma = self.beta * np.exp(-(best_agent.fit + 0.05) / (agent.fit + 0.05))

        # Generates a flag that provides diversity
        flag = np.sign(r.generate_uniform_random_number(-1, 1))

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Updates the position
        new_position = agent.position + flag * r1 * gamma * \
            (cluster_agent.position - agent.position) + flag * r1 * \
            self.alpha * (solubility * best_agent.position - agent.position)

        return new_position

    def _update(self, agents, best_agent, function, coefficient, pressure, constant, iteration, n_iterations):
        """Method that wraps Henry Gas Solubility Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            coefficient (np.array): Henry's coefficient array.
            pressure (np.array): Partial pressure array.
            constant (np.array): Constants array.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Creates n-wise clusters
        clusters = g.n_wise(agents, pressure.shape[1])

        # Iterates through all clusters
        for i, cluster in enumerate(clusters):
            # Calculates the system's current temperature (eq. 8)
            T = np.exp(-iteration / n_iterations)

            # Updates Henry's coefficient (eq. 8)
            coefficient[i] *= np.exp(-constant[i] * (1 / T - 1 / 298.15))

            # Transforms the cluster into a list and sorts it
            cluster = list(cluster)
            cluster.sort(key=lambda x: x.fit)

            # Iterates through all agents in cluster
            for j, agent in enumerate(cluster):
                # Calculates agent's solubility (eq. 9)
                solubility = self.K * coefficient[i] * pressure[i][j]

                # Updates agent's position (eq. 10)
                agent.position = self._update_position(agent, cluster[0], best_agent, solubility)

                # Clips agent's limits
                agent.clip_limits()

                # Re-calculates its fitness
                agent.fit = function(agent.position)

        # Re-sorts the whole space
        agents.sort(key=lambda x: x.fit)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates the number of worst agents (eq. 11)
        N = int(len(agents) * (r1 * (0.2 - 0.1) + 0.1))

        # Iterates through every bad agent
        for agent in agents[-N:]:
            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # Updates bad agent's position (eq. 12)
            agent.position = agent.lb + r2 * (agent.ub - agent.lb)

    def run(self, space, function, store_best_only=False, pre_evaluation=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Calculates the number of agents per cluster
        n_agents_per_cluster = int(len(space.agents) / self.n_clusters)

        # Instantiates a coefficients' array
        coefficient = self.l1 * r.generate_uniform_random_number(size=self.n_clusters)

        # Creates a pressures' array
        pressure = self.l2 * r.generate_uniform_random_number(size=(self.n_clusters, n_agents_per_cluster))

        # And finally, creates a constants' array
        constant = self.l3 * r.generate_uniform_random_number(size=self.n_clusters)

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent, function,
                             coefficient, pressure, constant, t, space.n_iterations)

                # Checking if agents meet the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history
