import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core import agent
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)

class MRFO(Optimizer):
    """A MRFO class, inherited from Optimizer.

    This is the designed class to define MRFO-related
    variables and methods.

    References:
        W. Zhao, Z. Zhang and L. Wang.
        Manta Ray Foraging Optimization: An effective bio-inspired optimizer for engineering applications.
        Engineering Applications of Artificial Intelligence (2020). 

    """
    def __init__(self, algorithm='MRFO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MRFO.')

        # Override its parent class with the receiving hyperparams
        super(MRFO, self).__init__(algorithm=algorithm)

        # Somersault foraging
        self.s = 2.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')
    
    @property
    def s(self):
        """float: Somersault foraging.

        """

        return self._s

    @s.setter
    def s(self, s):
        if not (isinstance(s, float) or isinstance(s, int)):
            raise e.TypeError('`s` should be a float or integer')
        if s < 0:
            raise e.ValueError('`s` should be >= 0')

        self._s = s

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
            if 's' in hyperparams:
                self.w = hyperparams['s']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: s = {self.s} | Built: {self.built}.')
    
    def _cyclone_foraging(self, agents, best_position, i, n_iterations, t):
        """
        """

        # Generates an uniform random number
        rand = r.generate_uniform_random_number()
        
        beta = 2 * np.exp(rand * (n_iterations - t+1) / n_iterations) * np.sin(2 * np.pi * rand)
        
        if (t/n_iterations < r.generate_uniform_random_number()):

            rand_position = np.zeros((agents[i].n_variables, 1))
            for j, (lb, ub) in enumerate(zip(agents[i].lb, agents[i].ub)):
                    rand_position[j] = r.generate_uniform_random_number(
                        lb, ub, size=agents[i].n_dimensions)

            if (i == 0):
                cyclone_foraging = rand_position + r.generate_uniform_random_number() * \
                (rand_position - agents[i].position) + \
                beta * (rand_position - agents[i].position)
            else:
                cyclone_foraging = rand_position + r.generate_uniform_random_number() * \
                (agents[i-1].position - agents[i].position) + \
                beta * (rand_position - agents[i].position)
        else:
            if (i == 0):
                cyclone_foraging = best_position + r.generate_uniform_random_number() * \
                (best_position - agents[i].position) + \
                beta * (best_position - agents[i].position)
            else:
                cyclone_foraging = best_position + r.generate_uniform_random_number() * \
                (agents[i-1].position - agents[i].position) + \
                beta * (best_position - agents[i].position)
        
        return cyclone_foraging
        
    
    def _chain_foraging(self, agents, best_position, i):
        """
        """

        # Generates an uniform random number
        rand = r.generate_uniform_random_number()
        
        alpha = 2 * rand * np.sqrt(np.abs(np.log(rand)))
        
        if (i == 0):
            chain_foraging = agents[i].position + r.generate_uniform_random_number() * \
                (best_position - agents[i].position) + alpha *\
                (best_position - agents[i].position)
        else:
            chain_foraging = agents[i].position + r.generate_uniform_random_number() * \
                (agents[i-1].position - agents[i].position) + alpha * \
                (best_position - agents[i].position)
        
        return chain_foraging

    def _somersault_foraging(self, agent_position, best_position):
        """
        """
        
        somersault_foraging = agent_position + \
            2 * (r.generate_uniform_random_number() * \
                 best_position - r.generate_uniform_random_number() * agent_position)
        
        return somersault_foraging
    
    def _update(self, agents, best_agent, function, n_iterations, t):
        """Method that wraps chain, cyclone and somersault foraging updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            cyclone = np.zeros((agent.n_variables, 1))
            chain = np.zeros((agent.n_variables, 1))
            # Generates an uniform random number
            rand = r.generate_uniform_random_number()
            if (rand < 0.5):
                cyclone = self._cyclone_foraging(agents, best_agent.position, i, n_iterations, t)
            else:
                chain = self._chain_foraging(agents, best_agent.position, i)
            
            agent.position = agent.position + cyclone + chain
            
            # Checking agents limits
            agent.clip_limits()
            
            # Evaluates agent
            agent.fit = function.pointer(agent.position)
            
            # If new agent's fitness is better than best
            if agent.fit < best_agent.fit:
                # Swap their positions
                agent.position, best_agent.position = best_agent.position, agent.position

                # Also swap their fitness
                agent.fit, best_agent.fit = best_agent.fit, agent.fit

        # Iterate through all agents
        for agent in agents:
            somersault = self._somersault_foraging(agent.position, best_agent.position)

            # Updates current agent positions
            agent.position = agent.position + somersault

    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (boolean): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Check if there is a pre-evaluation hook
        if pre_evaluation_hook:
            # Applies the hook
            pre_evaluation_hook(self, space, function)

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function, space.n_iterations, t)

            # Checking if agents meets the bounds limits
            space.clip_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents, local positions and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history