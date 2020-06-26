import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class ES(Optimizer):
    """An ES class, inherited from Optimizer.

    This is the designed class to define ES-related
    variables and methods.

    References:
        T. Bäck and H.–P. Schwefel. An Overview of Evolutionary Algorithms for Parameter Optimization.
        Evolutionary Computation (1993).

    """

    def __init__(self, algorithm='ES', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(ES, self).__init__(algorithm)

        # Ratio of children in the population
        self.child_ratio = 0.5

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def child_ratio(self):
        """float: Ratio of children in the population.

        """

        return self._child_ratio

    @child_ratio.setter
    def child_ratio(self, child_ratio):
        if not (isinstance(child_ratio, float) or isinstance(child_ratio, int)):
            raise e.TypeError('`child_ratio` should be a float or integer')
        if child_ratio < 0 or child_ratio > 1:
            raise e.ValueError('`child_ratio` should be between 0 and 1')

        self._child_ratio = child_ratio

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
            if 'child_ratio' in hyperparams:
                self.child_ratio = hyperparams['child_ratio']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: child_ratio = {self.child_ratio} | Built: {self.built}.')

    def _mutate_parent(self, agent, function, strategy):
        """Mutates a parent into a new child.

        Args:
            agent (Agent): An agent instance to be rESroduced.
            function (Function): A Function object that will be used as the objective function.
            strategy (np.array): An array holding the strategies that conduct the searching process.

        Returns:
            A mutated child.

        """

        # Makea a deepcopy on selected agent
        a = copy.deepcopy(agent)

        # Generates a uniform random number
        r1 = r.generate_gaussian_random_number()

        # Updates its position
        a.position += strategy * r1

        # Clips its limits
        a.clip_limits()

        # Calculates its fitness
        a.fit = function(a.position)

        return a

    def _update_strategy(self, strategy):
        """Updates the strategy.

        Args:
            strategy (np.array): An strategy array to be updated.

        Returns:
            The updated strategy.

        """

        # Calculates the number of variables and dimensions
        n_variables, n_dimensions = strategy.shape[0], strategy.shape[1]

        # Calculates the mutation strength
        tau = 1 / np.sqrt(2 * n_variables)

        # Calculates the mutation strength complementary
        tau_p = 1 / np.sqrt(2 * np.sqrt(n_variables))

        # Generates a uniform random number
        r1 = r.generate_gaussian_random_number(
            size=(n_variables, n_dimensions))

        # Generates another uniform random number
        r2 = r.generate_gaussian_random_number(
            size=(n_variables, n_dimensions))

        # Calculates the new strategy
        new_strategy = strategy * np.exp(tau_p * r1 + tau * r2)

        return new_strategy

    def _update(self, agents, n_agents, function, n_children, strategy):
        """Method that wraps evolution over all agents and variables.

        Args:
            agents (list): List of agents.
            n_agents (int): Number of possible agents in the space.
            function (Function): A Function object that will be used as the objective function.
            n_children (int): Number of possible children in the space.
            strategy (np.array): An strategy array.

        """

        # Creating a list for the produced children
        children = []

        # Iterate through all children
        for i in range(n_children):
            # Mutates a parent and generates a new child
            a = self._mutate_parent(agents[i], function, strategy[i])

            # Updates the strategy
            strategy[i] = self._update_strategy(strategy[i])

            # Appends the mutated agent to the children
            children.append(a)

        # Joins both populations
        agents += children

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        return agents[:n_agents]

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

        # Calculates the number of possible children
        n_children = int(space.n_agents * self.child_ratio)

        # Instantiate an array of strategies
        strategy = np.zeros(
            (n_children, space.n_variables, space.n_dimensions))

        # Iterate through all possible children
        for i in range(n_children):
            # For every decision variable
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                # Initializes the strategy array with the proposed ES distance
                strategy[i][j] = 0.05 * r.generate_uniform_random_number(0, ub - lb, size=space.agents[i].n_dimensions)

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
                space.agents = self._update(
                    space.agents, space.n_agents, function, n_children, strategy)

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
