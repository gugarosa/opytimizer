import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class ABC(Optimizer):
    """An ABC class, inherited from Optimizer.

    This is the designed class to define ABC-related
    variables and methods.

    References:
        D. Karaboga and B. Basturk.
        A powerful and efficient algorithm for numerical function optimization: Artificial bee colony (ABC) algorithm.
        Journal of Global Optimization (2007). 

    """

    def __init__(self, algorithm='ABC', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ABC.')

        # Override its parent class with the receiving hyperparams
        super(ABC, self).__init__(algorithm)

        # Number of trial limits
        self.n_trials = 10

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def n_trials(self):
        """int: Number of trial limits.

        """

        return self._n_trials

    @n_trials.setter
    def n_trials(self, n_trials):
        if not isinstance(n_trials, int):
            raise e.TypeError('`n_trials` should be an integer')
        if n_trials <= 0:
            raise e.ValueError('`n_trials` should be > 0')

        self._n_trials = n_trials

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
            if 'n_trials' in hyperparams:
                self.n_trials = hyperparams['n_trials']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Algorithm: {self.algorithm} | Hyperparameters: n_trials = {self.n_trials}.')

    def _evaluate_location(self, agent, neighbour, function, trial):
        """Evaluates a food source location and update its value if possible.

        Args:
            agent (Agent): An agent.
            neighbour (Agent): A neightbour agent.
            function (Function): A function object.
            trial (int): A trial counter.

        Returns:
            The number of trials for the current food source.

        """

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number(-1, 1)

        # Copies actual food source location
        a = copy.deepcopy(agent)

        # Change its location according to equation 2.2
        a.position = agent.position + (agent.position - neighbour.position) * r1

        # Check agent limits
        a.clip_limits()

        # Evaluating its fitness
        a.fit = function(a.position)

        # Check if fitness is improved
        if a.fit < agent.fit:
            # If yes, reset the number of trials for this particular food source
            trial = 0

            # Copies the new position
            agent.position = copy.deepcopy(a.position)

            # And also the new fitness
            agent.fit = copy.deepcopy(a.fit)

        # If not
        else:
            # We increse the trials counter
            trial += 1

        return trial

    def _send_employee(self, agents, function, trials):
        """Sends employee bees onto food source to evaluate its nectar.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            trials (np.array): Array of trials counter.

        """

        # Iterate through all food sources
        for i, agent in enumerate(agents):
            # Gathering a random source to be used
            source = int(r.generate_uniform_random_number(0, len(agents)))

            # Measuring food source location
            trials[i] = self._evaluate_location(
                agent, agents[source], function, trials[i])

    def _send_onlooker(self, agents, function, trials):
        """Sends onlooker bees to select new food sources.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            trials (np.array): Array of trials counter.

        """

        # Calculating the fitness somatory
        total = sum(agent.fit for agent in agents)

        # Defining food sources' counter
        k = 0

        # While counter is less than the amount of food sources
        while k < len(agents):
            # We iterate through every agent
            for i, agent in enumerate(agents):
                # Creates a random uniform number
                r1 = r.generate_uniform_random_number()

                # Calculates the food source's probability
                probs = (agent.fit / (total + c.EPSILON)) + 0.1

                # If the random number is smaller than food source's probability
                if r1 < probs:
                    # We need to increment the counter
                    k += 1

                    # Gathers a random source to be used
                    source = int(
                        r.generate_uniform_random_number(0, len(agents)))

                    # Evaluate its location
                    trials[i] = self._evaluate_location(
                        agent, agents[source], function, trials[i])

    def _send_scout(self, agents, function, trials):
        """Sends scout bees to scout for new possible food sources.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            trials (np.array): Array of trials counter.

        """

        # Calculating the maximum trial counter value and index
        max_trial, max_index = np.max(trials), np.argmax(trials)

        # If maximum trial is bigger than number of possible trials
        if max_trial > self.n_trials:
            # Resets the trial counter
            trials[max_index] = 0

            # Copies the current agent
            a = copy.deepcopy(agents[max_index])

            # Updates its position with a random shakeness
            a.position += r.generate_uniform_random_number(-1, 1)

            # Check agent limits
            a.clip_limits()

            # Recalculates its fitness
            a.fit = function(a.position)

            # If fitness is better
            if a.fit < agents[max_index].fit:
                # We copy the temporary agent to the current one
                agents[max_index] = copy.deepcopy(a)

    def _update(self, agents, function, trials):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            trials (np.array): Array of trials counter.

        """

        # Sending employee bees step
        self._send_employee(agents, function, trials)

        # Sending onlooker bees step
        self._send_onlooker(agents, function, trials)

        # Sending scout bees step
        self._send_scout(agents, function, trials)

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

        # Instanciating array of trials counter
        trials = np.zeros(space.n_agents)

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
                self._update(space.agents, function, trials)

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
