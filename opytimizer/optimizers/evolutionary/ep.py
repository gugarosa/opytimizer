import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class EP(Optimizer):
    """An EP class, inherited from Optimizer.

    This is the designed class to define EP-related
    variables and methods.

    References:
        D. B. Fogel. Evolutionary computation: toward a new philosophy of machine intelligence.
        Vol. 1. John Wiley & Sons (2006).

    """

    def __init__(self, algorithm='EP', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(EP, self).__init__(algorithm)

        # Size of bout during the tournament selection
        self.bout_size = 0.1

        # Clipping ratio to helps the algorithm's convergence
        self.clip_ratio = 0.05

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def bout_size(self):
        """float: Size of bout during the tournament selection.

        """

        return self._bout_size

    @bout_size.setter
    def bout_size(self, bout_size):
        if not (isinstance(bout_size, float) or isinstance(bout_size, int)):
            raise e.TypeError('`bout_size` should be a float or integer')
        if bout_size < 0 or bout_size > 1:
            raise e.ValueError('`bout_size` should be between 0 and 1')

        self._bout_size = bout_size

    @property
    def clip_ratio(self):
        """float: Clipping ratio to helps the algorithm's convergence.

        """

        return self._clip_ratio

    @clip_ratio.setter
    def clip_ratio(self, clip_ratio):
        if not (isinstance(clip_ratio, float) or isinstance(clip_ratio, int)):
            raise e.TypeError('`clip_ratio` should be a float or integer')
        if clip_ratio < 0 or clip_ratio > 1:
            raise e.ValueError('`clip_ratio` should be between 0 and 1')

        self._clip_ratio = clip_ratio

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
            if 'bout_size' in hyperparams:
                self.bout_size = hyperparams['bout_size']
            if 'clip_ratio' in hyperparams:
                self.clip_ratio = hyperparams['clip_ratio']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: bout_size = {self.bout_size}, clip_ratio = {self.clip_ratio} | '
            f'Built: {self.built}.')

    def _mutate_parent(self, agent, function, strategy):
        """Mutates a parent into a new child.

        Args:
            agent (Agent): An agent instance to be reproduced.
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

    def _update_strategy(self, strategy, lower_bound, upper_bound):
        """Updates the strategy and performs a clipping process to help its convergence.

        Args:
            strategy (np.array): An strategy array to be updated.
            lower_bound (np.array): An array holding the lower bounds.
            upper_bound (np.array): An array holding the upper bounds.

        Returns:
            The updated strategy.

        """

        # Calculates the number of variables and dimensions
        n_variables, n_dimensions = strategy.shape[0], strategy.shape[1]

        # Generates a uniform random number
        r1 = r.generate_gaussian_random_number(size=(n_variables, n_dimensions))

        # Calculates the new strategy
        new_strategy = strategy + r1 * (np.sqrt(np.abs(strategy)))

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
            # Uses the clip ratio to help the convergence
            new_strategy[j] = np.clip(
                new_strategy[j], lb, ub) * self.clip_ratio

        return new_strategy

    def _update(self, agents, n_agents, function, strategy):
        """Method that wraps evolution over all agents and variables.

        Args:
            agents (list): List of agents.
            n_agents (int): Number of possible agents in the space.
            function (Function): A Function object that will be used as the objective function.
            strategy (np.array): An array of strategies.

        Returns:
            A new population with more fitted individuals.

        """

        # Creating a list for the produced children
        children = []

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Mutates a parent and generates a new child
            a = self._mutate_parent(agent, function, strategy[i])

            # Updates the strategy
            strategy[i] = self._update_strategy(
                strategy[i], agent.lb, agent.ub)

            # Appends the mutated agent to the children
            children.append(a)

        # Joins both populations
        agents += children

        # Number of individuals to be selected
        n_individuals = int(n_agents * self.bout_size)

        # Creates an empty array of wins
        wins = np.zeros(len(agents))

        # Iterate through all agents in the new population
        for i in range(len(agents)):
            # Iterate through all tournament individuals
            for _ in range(n_individuals):
                # Gathers a random index
                index = int(r.generate_uniform_random_number(0, len(agents)))

                # If current agent's fitness is smaller than selected one
                if agents[i].fit < agents[index].fit:
                    # Increases its winning by one
                    wins[i] += 1

        # Sorts the agents list based on its winnings
        agents = [agents for _, agents in sorted(
            zip(wins, agents), key=lambda pair: pair[0], reverse=True)]

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

        # Instantiate an array of strategies
        strategy = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # Iterate through all agents
        for i in range(space.n_agents):
            # For every decision variable
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                # Initializes the strategy array with the proposed EP distance
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
                    space.agents, space.n_agents, function, strategy)

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
