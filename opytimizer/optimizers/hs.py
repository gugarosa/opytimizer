import copy

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class HS(Optimizer):
    """A HS class, inherited from Optimizer.

    This is the designed class to define HS-related
    variables and methods.

    References:
        Z. W. Geem, J. H. Kim, and G. V. Loganathan. A new heuristic optimization algorithm: Harmony search. Simulation (2001). 

    """

    def __init__(self, algorithm='HS', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> HS.')

        # Override its parent class with the receiving hyperparams
        super(HS, self).__init__(algorithm=algorithm)

        # Harmony memory considering rate
        self.HMCR = 0.7

        # Pitch adjusting rate
        self.PAR = 0.7

        # Bandwidth parameter
        self.bw = 1.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def HMCR(self):
        """float: Harmony memory considering rate.

        """

        return self._HMCR

    @HMCR.setter
    def HMCR(self, HMCR):
        if not (isinstance(HMCR, float) or isinstance(HMCR, int)):
            raise e.TypeError('`HMCR` should be a float or integer')
        if HMCR < 0 or HMCR > 1:
            raise e.ValueError('`HMCR` should be between 0 and 1')

        self._HMCR = HMCR

    @property
    def PAR(self):
        """float: Pitch adjusting rate.

        """

        return self._PAR

    @PAR.setter
    def PAR(self, PAR):
        if not (isinstance(PAR, float) or isinstance(PAR, int)):
            raise e.TypeError('`PAR` should be a float or integer')
        if PAR < 0 or PAR > 1:
            raise e.ValueError('`PAR` should be between 0 and 1')

        self._PAR = PAR

    @property
    def bw(self):
        """float: Bandwidth parameter.

        """

        return self._bw

    @bw.setter
    def bw(self, bw):
        if not (isinstance(bw, float) or isinstance(bw, int)):
            raise e.TypeError('`bw` should be a float or integer')
        if bw < 0:
            raise e.ValueError('`bw` should be >= 0')

        self._bw = bw

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
            if 'HMCR' in hyperparams:
                self.HMCR = hyperparams['HMCR']
            if 'PAR' in hyperparams:
                self.PAR = hyperparams['PAR']
            if 'bw' in hyperparams:
                self.bw = hyperparams['bw']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: HMCR = {self.HMCR}, PAR = {self.PAR}, bw = {self.bw} | Built: {self.built}.')

    def _generate_new_harmony(self, agent):
        """It generates a new harmony.

        Args:
            agent (Agent): An agent class instance.

        Returns:
            A new agent (harmony) based on music generation process.

        """

        # Mimics its position
        a = copy.deepcopy(agent)

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number(0, 1)

        # Using harmony memory
        if r1 < self.HMCR:
            # Generates a new uniform random number
            r2 = r.generate_uniform_random_number(0, 1)

            # Checks if it needs a pitch adjusting
            if r2 < self.PAR:
                # Generates a final random number
                r3 = r.generate_uniform_random_number(-1, 1)

                # Updates harmony position
                a.position += (r3 * self.bw)

        # If harmony memory is not used
        else:
            # Generates a new random harmony
            for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
                # For each decision variable, we generate uniform random numbers
                a.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=agent.n_dimensions)

        return a

    def _update(self, agents, function):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.

        """

        # Calculates a random index
        i = int(r.generate_uniform_random_number(0, len(agents)))

        # Generates a new harmony
        agent = self._generate_new_harmony(agents[i])

        # Calculates the new harmony fitness
        agent.fit = function.pointer(agent.position)

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # If newly generated agent fitness is better
        if agent.fit < agents[-1].fit:
            # Updates the corresponding agent's object
            agents[-1] = copy.deepcopy(agent)

    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (boolean): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (function): A function that receives the optimizer, space and function
                and returns None. This function is executed before evaluating the function being optimized.

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
            self._update(space.agents, function)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history
