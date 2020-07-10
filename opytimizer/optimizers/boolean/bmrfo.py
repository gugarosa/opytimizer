import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class BMRFO(Optimizer):
    """A BMRFO class, inherited from Optimizer.

    This is the designed class to define boolean MRFO-related
    variables and methods.

    References:
        Not yet published.

    """

    def __init__(self, algorithm='BMRFO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BMRFO.')

        # Override its parent class with the receiving hyperparams
        super(BMRFO, self).__init__(algorithm=algorithm)

        # Somersault foraging
        self.S = np.array([1])

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def S(self):
        """float: Somersault foraging.
        """

        return self._S

    @S.setter
    def S(self, S):
        if not isinstance(S, np.ndarray):
            raise e.TypeError('`S` should be a numpy array')

        self._S = S

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
            if 'S' in hyperparams:
                self.S = hyperparams['S']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: S = {self.S} | Built: {self.built}.')

    def _cyclone_foraging(self, agents, best_position, i, iteration, n_iterations):
        """Performs the cyclone foraging procedure.

        Args:
            agents (list): List of agents.
            best_position (np.array): Global best position.
            i (int): Current agent's index.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        Returns:
            A new cyclone foraging.

        """

        # Generates binary random numbers
        r1 = r.generate_binary_random_number(best_position.shape)
        r2 = r.generate_binary_random_number(best_position.shape)
        beta = r.generate_binary_random_number(best_position.shape)

        # Generates a uniform random number
        u = r.generate_uniform_random_number()

        # Checks if current iteration proportion is smaller than random generated number
        if iteration / n_iterations < u:
            # Generates binary random positions
            r_position = r.generate_binary_random_number(size=(agents[i].n_variables, agents[i].n_dimensions))

            # Checks if the index is equal to zero
            if (i == 0):
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r2, np.logical_xor(r_position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(r_position, agents[i].position))
                cyclone_foraging = np.logical_and(r_position, np.logical_and(partial_one, partial_two))

            # If index is different than zero
            else:
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r2, np.logical_xor(agents[i - 1].position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(r_position, agents[i].position))
                cyclone_foraging = np.logical_and(r_position, np.logical_and(partial_one, partial_two))

        # If current iteration proportion is bigger than random generated number
        else:
            # Checks if the index is equal to zero
            if (i == 0):
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r2, np.logical_xor(best_position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(best_position, agents[i].position))
                cyclone_foraging = np.logical_and(best_position, np.logical_and(partial_one, partial_two))

            # If index is different than zero
            else:
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r2, np.logical_xor(agents[i - 1].position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(best_position, agents[i].position))
                cyclone_foraging = np.logical_and(best_position, np.logical_and(partial_one, partial_two))

        return cyclone_foraging

    def _chain_foraging(self, agents, best_position, i):
        """Performs the chain foraging procedure.

        Args:
            agents (list): List of agents.
            best_position (np.array): Global best position.
            i (int): Current agent's index.

        Returns:
            A new chain foraging.

        """

        # Generates binary random numbers
        r1 = r.generate_binary_random_number(best_position.shape)
        r2 = r.generate_binary_random_number(best_position.shape)
        alpha = r.generate_binary_random_number(best_position.shape)

        # Checks if the index is equal to zero
        if i == 0:
            # Calculates the chain foraging
            partial_one = np.logical_and(r2, np.logical_xor(best_position, agents[i].position))
            partial_two = np.logical_and(alpha, np.logical_xor(best_position, agents[i].position))
            chain_foraging = np.logical_or(agents[i].position, np.logical_or(partial_one, partial_two))

        # If index is different than zero
        else:
            # Calculates the chain foraging
            partial_one = np.logical_and(r2, np.logical_xor(agents[i - 1].position, agents[i].position))
            partial_two = np.logical_and(alpha, np.logical_xor(best_position, agents[i].position))
            chain_foraging = np.logical_or(agents[i].position, np.logical_or(partial_one, partial_two))

        return chain_foraging

    def _somersault_foraging(self, position, best_position):
        """Performs the somersault foraging procedure.

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.

        Returns:
            A new somersault foraging.

        """

        # Generates binary random numbers
        r1 = r.generate_binary_random_number(best_position.shape)
        r2 = r.generate_binary_random_number(best_position.shape)

        # Calculates the somersault foraging
        somersault_foraging = np.logical_or(position, np.logical_and(self.S, np.logical_xor(np.logical_xor(r1, best_position), np.logical_xor(r2, position))))

        return somersault_foraging

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps chain, cyclone and somersault foraging updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than 1/2
            if r1 < 0.5:
                # Performs the cyclone foraging
                agent.position = self._cyclone_foraging(agents, best_agent.position, i, iteration, n_iterations)

            # If random number is bigger than 1/2
            else:
                # Performs the chain foraging
                agent.position = self._chain_foraging(agents, best_agent.position, i)

            # Clips the agent's limits
            agent.clip_limits()

            # Evaluates the agent
            agent.fit = function(agent.position)

            # If new agent's fitness is better than best
            if agent.fit < best_agent.fit:
                # Replace the best agent's position with its copy
                best_agent.position = copy.deepcopy(agent.position)

                # Also replace its fitness
                best_agent.fit = copy.deepcopy(agent.fit)

        # Iterate through all agents
        for agent in agents:
            # Performs the somersault foraging
            agent.position = self._somersault_foraging(agent.position, best_agent.position)

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
                self._update(space.agents, space.best_agent, function, t, space.n_iterations)

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