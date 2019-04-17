import copy

import numpy as np
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class ABC(Optimizer):
    """A ABC class, inherited from Optimizer.

    This will be the designed class to define ABC-related
    variables and methods.

    References:

    """

    def __init__(self, algorithm='ABC', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): A string holding optimizer's algorithm name.
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ABC.')

        # Override its parent class with the receiving hyperparams
        super(ABC, self).__init__(algorithm=algorithm)

        # Number of trial limits
        self._n_trials = 10

        # Food sources' trials
        self._trials = None

        # Food sources' probabilites
        self._probs = None

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
        self._n_trials = n_trials

    @property
    def trials(self):
        """np.array: Particles' number of trials.

        """

        return self._trials

    @trials.setter
    def trials(self, trials):
        self._trials = trials

    @property
    def probs(self):
        """np.array: Particles' probabilites.

        """

        return self._probs

    @probs.setter
    def probs(self, probs):
        self._probs = probs

    def _build(self, hyperparams):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

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
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: n_trials = {self.n_trials}.')

    def _measure_location(self, agent, neighbour, function, trial):
        """Measures a food source location and update its value if possible.

        Args:
            agent (Agent): An agent.
            neighbour (Agent): A neightbour agent.
            function (Function): A function object.
            trial (int): A trial counter.

        """

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number(-1, 1)

        # Copies actual food source location
        a = copy.deepcopy(agent)

        # Change its location according to equation 2.2
        a.position = agent.position + (agent.position - neighbour.position) * r1

        # Evaluating its fitness
        a.fit = function.pointer(a.position)

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

    def _send_employee(self, agents, function, trials):
        """Sends employee bees onto food source to measure its nectar.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            trials (np.array): Array of trials counter.

        """

        # Iterate through all food sources
        for i, agent in enumerate(agents):
            # Gathering a random bee to be used
            bee = int(r.generate_uniform_random_number(0, len(agents)))

            # Measuring food source location
            self._measure_location(agent, agents[bee], function, trials[i])

    def _send_onlooker(self, agents, function, trials, probs):
        """Sends onlooker bees to select new food sources.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            trials (np.array): Array of trials counter.
            probs (np.array): Array of probabilites.

        """

        # Zeroing the fitness somatory
        sum_fit = 0

        # Iterating through agents
        for agent in agents:
            # Calculate the fitness sum
            sum_fit += agent.fit

        # Iterating through all agents
        for i, agent in enumerate(agents):
            # Calculating their probabilites according to equation 2.1
            probs[i] = agent.fit / sum_fit


        i = 0
        k = 0

        # 
        while k < len(agents):
            #
            for i, agent in enumerate(agents):
                #
                r1 = r.generate_uniform_random_number(0, 1)
                #
                if r1 < probs[i]:
                    #
                    k += 1

                    # Gathering a random bee to be used
                    bee = int(r.generate_uniform_random_number(0, len(agents)))

                    #
                    self._measure_location(agent, agents[bee], function, trials[i])





    # def _send_scout(self, agents, function, trials):
    #     """Sends scout bees to scout for new possible food sources.

    #     Args:
    #         agents (list): List of agents.
    #         function (Function): A function object.
    #         trials (np.array): Array of trials counter.

    #     """

    def _update(self, agents, best_agent, function, trials, probs):
        """Method that wraps Bat Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            trials (np.array): Array of trials counter.
            probs (np.array): Array of probabilites.

        """

        # Sending employee bees step
        self._send_employee(agents, function, trials)

        # Sending onlooker bees step
        self._send_onlooker(agents, function, trials, probs)

        # Sending scout bees step
        #self._send_scout()

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Instanciating array of trials counter
        self.trials = np.zeros(space.n_agents)

        # Instanciating array of probabilites
        self.probs = np.zeros(space.n_agents)
        
        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function, self.trials, self.probs)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history
