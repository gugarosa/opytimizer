import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class DE(Optimizer):
    """A DE class, inherited from Optimizer.

    This is the designed class to define DE-related
    variables and methods.

    References:
        R. Storn. On the usage of differential evolution for function optimization.
        Proceedings of North American Fuzzy Information Processing (1996).

    """

    def __init__(self, algorithm='DE', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(DE, self).__init__(algorithm)

        # Crossover probability
        self.CR = 0.9

        # Differential weight
        self.F = 0.7

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def CR(self):
        """float: Crossover probability.

        """

        return self._CR

    @CR.setter
    def CR(self, CR):
        if not (isinstance(CR, float) or isinstance(CR, int)):
            raise e.TypeError('`CR` should be a float or integer')
        if CR < 0 or CR > 1:
            raise e.ValueError('`CR` should be between 0 and 1')

        self._CR = CR

    @property
    def F(self):
        """float: Differential weight.

        """

        return self._F

    @F.setter
    def F(self, F):
        if not (isinstance(F, float) or isinstance(F, int)):
            raise e.TypeError('`F` should be a float or integer')
        if F < 0 or F > 2:
            raise e.ValueError('`F` should be between 0 and 2')

        self._F = F

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
            if 'CR' in hyperparams:
                self.CR = hyperparams['CR']
            if 'F' in hyperparams:
                self.F = hyperparams['F']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: CR = {self.CR}, F = {self.F} | '
            f'Built: {self.built}.')

    def _mutate_agent(self, agent, alpha, beta, gamma):
        """Mutates a new agent based on pre-picked distinct agents.

        Args:
            agent (Agent): Current agent.
            alpha (Agent): 1st picked agent.
            beta (Agent): 2nd picked agent.
            gamma (Agent): 3rd picked agent.

        Returns:
            A mutated agent.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Generates a random index for further comparison
        R = int(r.generate_uniform_random_number(0, agent.n_variables))

        # For every decision variable
        for j in range(a.n_variables):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than crossover or `j` equals to the sampled index
            if r1 < self.CR or j == R:
                # Updates the mutated agent position
                a.position[j] = alpha.position[j] + self.F * \
                    (beta.position[j] - gamma.position[j])

        return a

    def _update(self, agents, best_agent, function):
        """Method that wraps selection and mutation updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Randomly picks three distinct other agents, not including current one
            C = d.generate_choice_distribution(
                np.setdiff1d(range(0, len(agents)), i), size=3)

            # Mutates the current agent
            a = self._mutate_agent(
                agent, agents[C[0]], agents[C[1]], agents[C[2]])

            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)

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
                self._update(space.agents, space.best_agent, function)

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
