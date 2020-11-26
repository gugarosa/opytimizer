"""Univariate Marginal Distribution Algorithm.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class UMDA(Optimizer):
    """An UMDA class, inherited from Optimizer.

    This is the designed class to define UMDA-related variables and methods.

    References:
        H. Mühlenbein. The equation for response to selection and its use for prediction.
        Evolutionary Computation (1997).

    """

    def __init__(self, algorithm='UMDA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(UMDA, self).__init__(algorithm)

        # Probability of selection
        self.p_selection = 0.75

        # Distribution lower bound
        self.lower_bound = 0.05

        # Distribution upper bound
        self.upper_bound = 0.95

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def p_selection(self):
        """float: Probability of selection.

        """

        return self._p_selection

    @p_selection.setter
    def p_selection(self, p_selection):
        if not isinstance(p_selection, (float, int)):
            raise e.TypeError('`p_selection` should be a float or integer')
        if p_selection < 0 or p_selection > 1:
            raise e.ValueError('`p_selection` should be between 0 and 1')

        self._p_selection = p_selection

    @property
    def lower_bound(self):
        """float: Distribution lower bound.

        """

        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        if not isinstance(lower_bound, (float, int)):
            raise e.TypeError('`lower_bound` should be a float or integer')
        if lower_bound < 0 or lower_bound > 1:
            raise e.ValueError('`lower_bound` should be between 0 and 1')

        self._lower_bound = lower_bound

    @property
    def upper_bound(self):
        """float: Distribution upper bound.

        """

        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        if not isinstance(upper_bound, (float, int)):
            raise e.TypeError('`upper_bound` should be a float or integer')
        if upper_bound < 0 or upper_bound > 1:
            raise e.ValueError('`upper_bound` should be between 0 and 1')
        if upper_bound < self.lower_bound:
            raise e.ValueError(
                '`upper_bound` should be greater than `lower_bound')

        self._upper_bound = upper_bound

    def _calculate_probability(self, agents):
        """Calculates probabilities based on pre-selected agents' variables occurrence (eq. 47).

        Args:
            agents (list): List of pre-selected agents.

        Returns:
            Probability of variables occurence.

        """

        # Creates an empty array of probabilities
        probs = np.zeros((agents[0].n_variables, agents[0].n_dimensions))

        # For every pre-selected agent
        for agent in agents:
            # Increases if feature is selected
            probs += agent.position

        # Normalizes into real probabilities
        probs /= len(agents)

        # Clips between pre-defined lower and upper bounds
        probs = np.clip(probs, self.lower_bound, self.upper_bound)

        return probs

    def _sample_position(self, probs):
        """Samples new positions according to their probability of ocurrence (eq. 53).

        Args:
            probs (np.array): Array of probabilities.

        Returns:
            New sampled position.

        """

        # Creates a uniform random array with the same shape as `probs`
        r1 = r.generate_uniform_random_number(size=(probs.shape[0], probs.shape[1]))

        # Samples new positions
        new_position = np.where(probs < r1, True, False)

        return new_position

    def _update(self, agents):
        """Method that wraps selection, probability calculation and
        position sampling over all agents and variables.

        Args:
            agents (list): List of agents.

        """
        # Retrieving the number of agents
        n_agents = len(agents)

        # Selects the individuals through ranking
        n_selected = int(n_agents * self.p_selection)

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Calculates the probability of ocurrence from selected agents
        probs = self._calculate_probability(agents[:n_selected])

        # Iterates through every agents
        for agent in agents:
            # Samples new agent's position
            agent.position = self._sample_position(probs)

            # Checking its limits
            agent.clip_limits()

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
                self._update(space.agents)

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
