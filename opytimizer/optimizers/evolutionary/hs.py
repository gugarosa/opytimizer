"""Harmony Search-based algorithms.
"""

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


class HS(Optimizer):
    """A HS class, inherited from Optimizer.

    This is the designed class to define HS-related
    variables and methods.

    References:
        Z. W. Geem, J. H. Kim, and G. V. Loganathan.
        A new heuristic optimization algorithm: Harmony search. Simulation (2001).

    """

    def __init__(self, algorithm='HS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> HS.')

        # Override its parent class with the receiving hyperparams
        super(HS, self).__init__(algorithm)

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
        if not isinstance(HMCR, (float, int)):
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
        if not isinstance(PAR, (float, int)):
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
        if not isinstance(bw, (float, int)):
            raise e.TypeError('`bw` should be a float or integer')
        if bw < 0:
            raise e.ValueError('`bw` should be >= 0')

        self._bw = bw

    def _generate_new_harmony(self, agents):
        """It generates a new harmony.

        Args:
            agents (list): List of agents.

        Returns:
            A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Using the harmony memory
            if r1 <= self.HMCR:
                # Generates a random index
                k = r.generate_integer_random_number(0, len(agents))

                # Replaces the position with agent `k`
                a.position[j] = agents[k].position[j]

                # Generates a new uniform random number
                r2 = r.generate_uniform_random_number()

                # Checks if it needs a pitch adjusting
                if r2 <= self.PAR:
                    # Generates a final random number
                    r3 = r.generate_uniform_random_number(-1, 1)

                    # Updates harmony position
                    a.position[j] += (r3 * self.bw)

            # If harmony memory is not used
            else:
                # Generate a uniform random number
                a.position[j] = r.generate_uniform_random_number(lb, ub, size=a.n_dimensions)

        return a

    def _update(self, agents, function):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.

        """

        # Generates a new harmony
        agent = self._generate_new_harmony(agents)

        # Checking agent limits
        agent.clip_limits()

        # Calculates the new harmony fitness
        agent.fit = function(agent.position)

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # If newly generated agent fitness is better
        if agent.fit < agents[-1].fit:
            # Updates the corresponding agent's position
            agents[-1].position = copy.deepcopy(agent.position)

            # And its fitness as well
            agents[-1].fit = copy.deepcopy(agent.fit)

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
                self._update(space.agents, function)

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


class IHS(HS):
    """An IHS class, inherited from HS.

    This is the designed class to define IHS-related
    variables and methods.

    References:
        M. Mahdavi, M. Fesanghary, and E. Damangir.
        An improved harmony search algorithm for solving optimization problems.
        Applied Mathematics and Computation (2007).

    """

    def __init__(self, algorithm='IHS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: HS -> IHS.')

        # Minimum pitch adjusting rate
        self.PAR_min = 0

        # Maximum pitch adjusting rate
        self.PAR_max = 1

        # Minimum bandwidth parameter
        self.bw_min = 1

        # Maximum bandwidth parameter
        self.bw_max = 10

        # Override its parent class with the receiving hyperparams
        super(IHS, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    @property
    def PAR_min(self):
        """float: Minimum pitch adjusting rate.

        """

        return self._PAR_min

    @PAR_min.setter
    def PAR_min(self, PAR_min):
        if not isinstance(PAR_min, (float, int)):
            raise e.TypeError('`PAR_min` should be a float or integer')
        if PAR_min < 0 or PAR_min > 1:
            raise e.ValueError('`PAR_min` should be between 0 and 1')

        self._PAR_min = PAR_min

    @property
    def PAR_max(self):
        """float: Maximum pitch adjusting rate.

        """

        return self._PAR_max

    @PAR_max.setter
    def PAR_max(self, PAR_max):
        if not isinstance(PAR_max, (float, int)):
            raise e.TypeError('`PAR_max` should be a float or integer')
        if PAR_max < 0 or PAR_max > 1:
            raise e.ValueError('`PAR_max` should be between 0 and 1')
        if PAR_max < self.PAR_min:
            raise e.ValueError('`PAR_max` should be >= `PAR_min`')

        self._PAR_max = PAR_max

    @property
    def bw_min(self):
        """float: Minimum bandwidth parameter.

        """

        return self._bw_min

    @bw_min.setter
    def bw_min(self, bw_min):
        if not isinstance(bw_min, (float, int)):
            raise e.TypeError('`bw_min` should be a float or integer')
        if bw_min < 0:
            raise e.ValueError('`bw_min` should be >= 0')

        self._bw_min = bw_min

    @property
    def bw_max(self):
        """float: Maximum bandwidth parameter.

        """

        return self._bw_max

    @bw_max.setter
    def bw_max(self, bw_max):
        if not isinstance(bw_max, (float, int)):
            raise e.TypeError('`bw_max` should be a float or integer')
        if bw_max < 0:
            raise e.ValueError('`bw_max` should be >= 0')
        if bw_max < self.bw_min:
            raise e.ValueError('`bw_max` should be >= `bw_min`')

        self._bw_max = bw_max

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

                # Updating pitch adjusting rate
                self.PAR = self.PAR_min + \
                    (((self.PAR_max - self.PAR_min) / space.n_iterations) * t)

                # Updating bandwidth parameter
                self.bw = self.bw_max * \
                    np.exp((np.log(self.bw_min / self.bw_max) /
                            space.n_iterations) * t)

                # Updating agents
                self._update(space.agents, function)

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


class GHS(IHS):
    """A GHS class, inherited from IHS.

    This is the designed class to define GHS-related
    variables and methods.

    References:
        M. Omran and M. Mahdavi. Global-best harmony search.
        Applied Mathematics and Computation (2008).

    """

    def __init__(self, algorithm='GHS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: IHS -> GHS.')

        # Override its parent class with the receiving hyperparams
        super(GHS, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    def _generate_new_harmony(self, agents):
        """It generates a new harmony.

        Args:
            agents (list): List of agents.

        Returns:
            A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Using the harmony memory
            if r1 <= self.HMCR:
                # Generates a random index
                k = r.generate_integer_random_number(0, len(agents))

                # Replaces the position with agent `k`
                a.position[j] = agents[k].position[j]

                # Generates a new uniform random number
                r2 = r.generate_uniform_random_number()

                # Checks if it needs a pitch adjusting
                if r2 <= self.PAR:
                    # Generates a random index
                    z = r.generate_integer_random_number(0, a.n_variables)

                    # Updates harmony position
                    a.position[j] = agents[0].position[z]

            # If harmony memory is not used
            else:
                # Generate a uniform random number
                a.position[j] = r.generate_uniform_random_number(lb, ub, size=a.n_dimensions)

        return a


class SGHS(HS):
    """A SGHS class, inherited from HS.

    This is the designed class to define SGHS-related
    variables and methods.

    References:
        Q.-K. Pan, P. Suganthan, M. Tasgetiren and J. Liang.
        A self-adaptive global best harmony search algorithm for continuous optimization problems.
        Applied Mathematics and Computation (2010).

    """

    def __init__(self, algorithm='SGHS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: HS -> SGHS.')

        # Learning period
        self.LP = 100

        # Mean harmony memory considering rate
        self.HMCRm = 0.98

        # Mean pitch adjusting rate
        self.PARm = 0.9

        # Minimum bandwidth parameter
        self.bw_min = 1

        # Maximum bandwidth parameter
        self.bw_max = 10

        # Override its parent class with the receiving hyperparams
        super(SGHS, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    @property
    def HMCR(self):
        """float: Harmony memory considering rate.

        """

        return self._HMCR

    @HMCR.setter
    def HMCR(self, HMCR):
        if not isinstance(HMCR, (float, int)):
            raise e.TypeError('`HMCR` should be a float or integer')

        self._HMCR = HMCR

    @property
    def PAR(self):
        """float: Pitch adjusting rate.

        """

        return self._PAR

    @PAR.setter
    def PAR(self, PAR):
        if not isinstance(PAR, (float, int)):
            raise e.TypeError('`PAR` should be a float or integer')

        self._PAR = PAR

    @property
    def LP(self):
        """int: Learning period.

        """

        return self._LP

    @LP.setter
    def LP(self, LP):
        if not isinstance(LP, int):
            raise e.TypeError('`LP` should be a integer')
        if LP <= 0:
            raise e.ValueError('`LP` should be > 0')

        self._LP = LP

    @property
    def HMCRm(self):
        """float: Mean harmony memory considering rate

        """

        return self._HMCRm

    @HMCRm.setter
    def HMCRm(self, HMCRm):
        if not isinstance(HMCRm, (float, int)):
            raise e.TypeError('`HMCRm` should be a float or integer')
        if HMCRm < 0 or HMCRm > 1:
            raise e.ValueError('`HMCRm` should be between 0 and 1')

        self._HMCRm = HMCRm

    @property
    def PARm(self):
        """float: Mean pitch adjusting rate.

        """

        return self._PARm

    @PARm.setter
    def PARm(self, PARm):
        if not isinstance(PARm, (float, int)):
            raise e.TypeError('`PARm` should be a float or integer')
        if PARm < 0 or PARm > 1:
            raise e.ValueError('`PARm` should be between 0 and 1')

        self._PARm = PARm

    @property
    def bw_min(self):
        """float: Minimum bandwidth parameter.

        """

        return self._bw_min

    @bw_min.setter
    def bw_min(self, bw_min):
        if not isinstance(bw_min, (float, int)):
            raise e.TypeError('`bw_min` should be a float or integer')
        if bw_min < 0:
            raise e.ValueError('`bw_min` should be >= 0')

        self._bw_min = bw_min

    @property
    def bw_max(self):
        """float: Maximum bandwidth parameter.

        """

        return self._bw_max

    @bw_max.setter
    def bw_max(self, bw_max):
        if not isinstance(bw_max, (float, int)):
            raise e.TypeError('`bw_max` should be a float or integer')
        if bw_max < 0:
            raise e.ValueError('`bw_max` should be >= 0')
        if bw_max < self.bw_min:
            raise e.ValueError('`bw_max` should be >= `bw_min`')

        self._bw_max = bw_max

    def _generate_new_harmony(self, agents):
        """It generates a new harmony.

        Args:
            agents (list): List of agents.

        Returns:
            A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Using the harmony memory
            if r1 <= self.HMCR:
                # Generates a uniform random number
                r2 = r.generate_uniform_random_number(-1, 1)

                # Updates harmony position
                a.position[j] += (r2 * self.bw)

                # Generates a new uniform random number
                r3 = r.generate_uniform_random_number()

                # Checks if it needs a pitch adjusting
                if r3 <= self.PAR:
                    # Updates harmony position
                    a.position[j] = agents[0].position[j]

            # If harmony memory is not used
            else:
                # Generate a uniform random number
                a.position[j] = r.generate_uniform_random_number(lb, ub, size=a.n_dimensions)

        return a

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

        # Initializing lists of HMCR and PAR
        HMCR, PAR = [], []

        # Initializing the learning period
        lp = 1

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating harmony memory considering rate
                self.HMCR = r.generate_gaussian_random_number(self.HMCRm, 0.01)[0]

                # Updating pitch adjusting rate
                self.PAR = r.generate_gaussian_random_number(self.PARm, 0.05)[0]

                # Storing both values
                HMCR.append(self.HMCR)
                PAR.append(self.PAR)

                # If current iteration is smaller than half
                if t < space.n_iterations // 2:
                    # Updates the bandwidth parameter
                    self.bw = self.bw_max - ((self.bw_max - self.bw_min) / space.n_iterations) * 2 * t

                # If is bigger than half
                else:
                    # Replaces by the minimum bandwidth
                    self.bw = self.bw_min

                # Updating agents
                self._update(space.agents, function)

                # Checking if agents meet the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Checks if learning period has reached its maximum
                if lp == self.LP:
                    # Re-calculates the mean HMCR
                    self.HMCRm = np.mean(HMCR)

                    # Re-calculates the mean PAR
                    self.PARm = np.mean(PAR)

                    # Resets the learning period to one
                    lp = 1

                # If has not reached
                else:
                    # Increase it by one
                    lp += 1

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history


class NGHS(HS):
    """A NGHS class, inherited from HS.

    This is the designed class to define NGHS-related
    variables and methods.

    References:
        D. Zou, L. Gao, J. Wu and S. Li.
        Novel global harmony search algorithm for unconstrained problems.
        Neurocomputing (2010).

    """

    def __init__(self, algorithm='NGHS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: HS -> NGHS.')

        # Mutation probability
        self.pm = 0.1

        # Override its parent class with the receiving hyperparams
        super(NGHS, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    @property
    def pm(self):
        """float: Mutation probability.

        """

        return self._pm

    @pm.setter
    def pm(self, pm):
        if not isinstance(pm, (float, int)):
            raise e.TypeError('`pm` should be a float or integer')
        if pm < 0 or pm > 1:
            raise e.ValueError('`pm` should be between 0 and 1')

        self._pm = pm

    def _generate_new_harmony(self, best, worst):
        """It generates a new harmony.

        Args:
            best (Agent): Best agent.
            worst (Agent): Worst agent.

        Returns:
            A new agent (harmony) based on music generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(best)

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Updates the harmony position
            new_position = 2 * (best.position[j] - worst.position[j])

            # Clips the harmony position between lower and upper bounds
            new_position = np.clip(new_position, lb, ub)

            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # Updates current agent's position
            a.position[j] = worst.position[j] + r1 * (new_position - worst.position[j])

            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # Checks if is supposed to be mutated
            if r2 <= self.pm:
                # Mutates the position
                a.position[j] = r.generate_uniform_random_number(lb, ub, size=a.n_dimensions)

        return a

    def _update(self, agents, function):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.

        """

        # Generates a new harmony
        agent = self._generate_new_harmony(agents[0], agents[-1])

        # Checking agent limits
        agent.clip_limits()

        # Calculates the new harmony fitness
        agent.fit = function(agent.position)

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Updates the worst agent's position
        agents[-1].position = copy.deepcopy(agent.position)

        # And its fitness as well
        agents[-1].fit = copy.deepcopy(agent.fit)


class GOGHS(NGHS):
    """A GOGHS class, inherited from NGHS.

    This is the designed class to define GOGHS-related
    variables and methods.

    References:
        Z. Guo, S. Wang, X. Yue and H. Yang.
        Global harmony search with generalized opposition-based learning.
        Soft Computing (2017).

    """

    def __init__(self, algorithm='GOGHS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: NGHS -> GOGHS.')

        # Override its parent class with the receiving hyperparams
        super(GOGHS, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    def _generate_opposition_harmony(self, new_agent, agents):
        """It generates a new opposition-based harmony.

        Args:
            new_agent (Agent): Newly created agent.
            agents (list): List of agents.

        Returns:
            A new agent (harmony) based on opposition generation process.

        """

        # Mimics an agent position
        a = copy.deepcopy(agents[0])

        # Creating pseudo-harmonies
        A = np.zeros((a.n_variables))
        B = np.zeros((a.n_variables))

        # Generates a new uniform random number
        k = r.generate_uniform_random_number()

        # Iterates over every variable
        for j in range(a.n_variables):
            # Defines to `A` and `B` maximum and minimum values, respectively
            A[j], B[j] = c.FLOAT_MAX, -c.FLOAT_MAX

            # Iterates over every agent
            for agent in agents:
                # If `A` is bigger than agent's position
                if A[j] > agent.position[j]:
                    # Replaces its value
                    A[j] = agent.position[j]

                # If `B` is smaller than agent's position
                elif B[j] < agent.position[j]:
                    # Replaces its value
                    B[j] = agent.position[j]

            # Calculates new agent's position
            a.position[j] = k * (A[j] + B[j]) - new_agent.position[j]

        return a

    def _update(self, agents, function):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.

        """

        # Generates new harmonies
        agent = self._generate_new_harmony(agents[0], agents[-1])
        opp_agent = self._generate_opposition_harmony(agent, agents)

        # Checking agents limits
        agent.clip_limits()
        opp_agent.clip_limits()

        # Calculates harmonies fitness
        agent.fit = function(agent.position)
        opp_agent.fit = function(opp_agent.position)

        # Checking if oppisition-based is better than agent
        if opp_agent.fit < agent.fit:
            # Copies the agent
            agent = copy.deepcopy(opp_agent)

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # If generated agent fitness is better
        if agent.fit < agents[-1].fit:
            # Updates the corresponding agent's position
            agents[-1].position = copy.deepcopy(agent.position)

            # And its fitness as well
            agents[-1].fit = copy.deepcopy(agent.fit)
