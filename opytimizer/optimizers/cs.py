import copy

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class CS(Optimizer):
    """A CS class, inherited from Optimizer.

    This will be the designed class to define CS-related
    variables and methods.

    References:
        X.-S. Yang and D. Suash. Cuckoo search via Lévy flights. World Congress on Nature & Biologically Inspired Computing (2009).

    """

    def __init__(self, algorithm='CS', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> CS.')

        # Override its parent class with the receiving hyperparams
        super(CS, self).__init__(algorithm=algorithm)

        # Step size
        self._alpha = 1

        # Lévy distribution parameter
        self._beta = 1.5

        # Probability of replacing worst nests
        self._p = 0.2

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: Step size.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def beta(self):
        """float: Lévy distribution parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def p(self):
        """float: Probability of replacing worst nests.

        """

        return self._p

    @p.setter
    def p(self, p):
        self._p = p

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
            if 'alpha' in hyperparams:
                self.alpha = hyperparams['alpha']
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']
            if 'p' in hyperparams:
                self.p = hyperparams['p']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: alpha = {self.alpha}, beta = {self.beta}, p = {self.p} | Built: {self.built}.')

    def _generate_new_nests(self, agents, best_agent):
        """Generate new nests according to Yang's implementation.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Current best agent.

        Returns:
            A new list of agents which can be seen as new nests (Equation 1).

        """

        # Makes a temporary copy of current agents
        new_agents = copy.deepcopy(agents)

        # Then, we iterate for every agent
        for new_agent in new_agents:
            # Calculating the Lévy distribution
            step = d.generate_levy_distribution(
                self.beta, new_agent.n_variables)

            # Expanding its dimension to perform entrywise multiplication
            step = np.expand_dims(step, axis=1)

            # Calculating the difference vector between local and best positions
            # Alpha controls the intensity of the step size
            step_size = self.alpha * step * \
                (new_agent.position - best_agent.position)

            # Generates a random normal distribution
            g = r.generate_gaussian_random_number(
                size=new_agent.n_variables)

            # Expanding its dimension to perform entrywise multiplication
            g = np.expand_dims(g, axis=1)

            # Acutally performs the random walk / flight
            new_agent.position += step_size * g

        return new_agents

    def _generate_abandoned_nests(self, agents, prob):
        """Generate a fraction of nests to be replaced according to Yang's implementation.

        Args:
            agents (list): List of agents.
            prob (float): Probability of replacing worst nests.

        Returns:
            A new list of agents which can be seen as the new nests to be replaced.

        """

        # Makes a temporary copy of current agents
        new_agents = copy.deepcopy(agents)

        # Generates a bernoulli distribution array
        # It will be used to replace or not a certain nest
        b = d.generate_bernoulli_distribution(prob, len(agents))

        # Iterating through every new agent
        for j, new_agent in enumerate(new_agents):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number(0, 1)

            # Then, we select two random nests
            k = int(r.generate_uniform_random_number(0, len(agents)-1))
            l = int(r.generate_uniform_random_number(0, len(agents)-1))

            # Calculating the random walk between these two nests
            step_size = r1 * (agents[k].position - agents[l].position)

            # Finally, we replace the old nest
            # Note it will only be replaced if 'b' is 1
            new_agent.position += (step_size * b[j])

        return new_agents

    def _evaluate_nests(self, agents, new_agents, function):
        """Evaluate new nests according to a fitness function.

        Args:
            agents (list): List of current agents.
            new_agents (list): List of new agents to be evaluated.
            function (Function): Fitness function used to evaluate.

        """

        # Iterating through each agent and new agent
        for agent, new_agent in zip(agents, new_agents):
            # Calculates the new agent fitness
            new_agent.fit = function.pointer(new_agent.position)

            # If new agent's fitness is better than agent's
            if new_agent.fit < agent.fit:
                # Replace its position
                agent.position = copy.deepcopy(new_agent.position)

                # And also, its fitness
                agent.fit = copy.deepcopy(new_agent.fit)

    def _update(self, agents, best_agent, function):
        """Method that wraps Cuckoo Search algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.

        """

        # Generate new nests
        new_agents = self._generate_new_nests(agents, best_agent)

        # Evaluate new generated nests
        self._evaluate_nests(agents, new_agents, function)

        # Generate new nests to be replaced
        new_agents = self._generate_abandoned_nests(agents, 1 - self.p)

        # Evaluate new generated nests for further replacement
        self._evaluate_nests(agents, new_agents, function)

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents, space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history
