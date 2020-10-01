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

    def _build(self, hyperparams):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'n_clusters' in hyperparams:
                self.n_clusters = hyperparams['n_clusters']
            if 'l1' in hyperparams:
                self.l1 = hyperparams['l1']
            if 'l2' in hyperparams:
                self.l2 = hyperparams['l2']
            if 'l3' in hyperparams:
                self.l3 = hyperparams['l3']
            if 'alpha' in hyperparams:
                self.alpha = hyperparams['alpha']
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']
            if 'K' in hyperparams:
                self.K = hyperparams['K']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s | Hyperparameters: n_clusters = %s, l1 = %s, l2 = %s, l3 = %s, '
                     'alpha = %s, beta = %s, K = %s | Built: %s.',
                     self.algorithm, self.n_clusters, self.l1, self.l2, self.l3,
                     self.alpha, self.beta, self.K, self.built)

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
        gamma = self.beta * \
            np.exp(-(best_agent.fit + 0.05) / (agent.fit + 0.05))

        # Generates a flag that provides diversity
        flag = np.sign(r.generate_uniform_random_number(-1, 1))

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Updates the position
        new_position = agent.position + flag * r1 * gamma * \
            (cluster_agent.position - agent.position) + flag * r1 * \
            self.alpha * (solubility * best_agent.position - agent.position)

        return new_position

    def _update(self, agents, best_agent, coefficient, pressure, constant, iteration, n_iterations):
        """Method that wraps Henry Gas Solubility Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        #
        n_agents_type = int(len(agents) / self.n_clusters)

        #
        for i, agents_type in enumerate(g.pairwise(agents, n_agents_type)):
            #
            T = np.exp(-iteration / n_iterations)

            #
            coefficient[i] *= np.exp(-constant[i] * (1 / T - 1 / 298.15))

            #
            agents_type = list(agents_type)

            #
            agents_type.sort(key=lambda x: x.fit)

            #
            for j, agent in enumerate(agents_type):
                #
                solubility = self.K * coefficient[i] * pressure[i][j]

                #
                agent.position = self._update_position(
                    agent, agents_type[0], best_agent, solubility)

        #
        r1 = r.generate_uniform_random_number()

        #
        N = int(len(agents) * (r1 * (0.2 - 0.1) + 0.1))

        #
        agents.sort(key=lambda x: x.fit)

        for agent in agents[-N:]:
            #
            r2 = r.generate_uniform_random_number()

            #
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

        #
        n_agents = int(len(space.agents) / self.n_clusters)

        #
        coefficient = self.l1 * \
            r.generate_uniform_random_number(size=self.n_clusters)

        #
        pressure = self.l2 * \
            r.generate_uniform_random_number(size=(self.n_clusters, n_agents))

        #
        constant = self.l3 * \
            r.generate_uniform_random_number(size=self.n_clusters)

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
                self._update(space.agents, space.best_agent,
                             coefficient, pressure, constant, t, space.n_iterations)

                # Checking if agents meets the bounds limits
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
