import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class EFO(Optimizer):
    """An EFO class, inherited from Optimizer.

    This is the designed class to define EFO-related
    variables and methods.

    References:
        H. Abedinpourshotorban, et al.
        Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
        Swarm and Evolutionary Computation (2016).

    """

    def __init__(self, algorithm='EFO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(EFO, self).__init__(algorithm)

        # Positive field proportion
        self.positive_field = 0.1

        # Negative field proportion
        self.negative_field = 0.5

        # Probability of selecting eletromagnets
        self.ps_ratio = 0.1

        # Probability of selecting a random eletromagnet
        self.r_ratio = 0.4

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def positive_field(self):
        """float: Positive field proportion.

        """

        return self._positive_field

    @positive_field.setter
    def positive_field(self, positive_field):
        if not (isinstance(positive_field, float) or isinstance(positive_field, int)):
            raise e.TypeError('`positive_field` should be a float or integer')
        if positive_field < 0 or positive_field > 1:
            raise e.ValueError('`positive_field` should be between 0 and 1')

        self._positive_field = positive_field

    @property
    def negative_field(self):
        """float: Negative field proportion.

        """

        return self._negative_field

    @negative_field.setter
    def negative_field(self, negative_field):
        if not (isinstance(negative_field, float) or isinstance(negative_field, int)):
            raise e.TypeError('`negative_field` should be a float or integer')
        if negative_field < 0 or negative_field > 1:
            raise e.ValueError('`negative_field` should be between 0 and 1')
        if negative_field + self.positive_field > 1:
            raise e.ValueError(
                '`negative_field` + `positive_field` should not exceed 1')

        self._negative_field = negative_field

    @property
    def ps_ratio(self):
        """float: Probability of selecting eletromagnets.

        """

        return self._ps_ratio

    @ps_ratio.setter
    def ps_ratio(self, ps_ratio):
        if not (isinstance(ps_ratio, float) or isinstance(ps_ratio, int)):
            raise e.TypeError('`ps_ratio` should be a float or integer')
        if ps_ratio < 0 or ps_ratio > 1:
            raise e.ValueError('`ps_ratio` should be between 0 and 1')

        self._ps_ratio = ps_ratio

    @property
    def r_ratio(self):
        """float: Probability of selecting a random eletromagnet.

        """

        return self._r_ratio

    @r_ratio.setter
    def r_ratio(self, r_ratio):
        if not (isinstance(r_ratio, float) or isinstance(r_ratio, int)):
            raise e.TypeError('`r_ratio` should be a float or integer')
        if r_ratio < 0 or r_ratio > 1:
            raise e.ValueError('`r_ratio` should be between 0 and 1')

        self._r_ratio = r_ratio

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
            if 'positive_field' in hyperparams:
                self.positive_field = hyperparams['positive_field']
            if 'negative_field' in hyperparams:
                self.negative_field = hyperparams['negative_field']
            if 'ps_ratio' in hyperparams:
                self.ps_ratio = hyperparams['ps_ratio']
            if 'r_ratio' in hyperparams:
                self.r_ratio = hyperparams['r_ratio']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: positive_field = {self.positive_field}, negative_field = {self.negative_field}, '
            f'ps_ratio = {self.ps_ratio}, r_ratio = {self.r_ratio} | '
            f'Built: {self.built}.')

    def _calculate_indexes(self, n_agents):
        """Calculates the indexes of positive, negative and neutral particles.

        Args:
            n_agents (int): Number of agents in the space.

        Returns:
            Positive, negative and neutral particles' indexes.

        """

        # Calculates a positive particle's index
        positive_index = int(r.generate_uniform_random_number(
            0, n_agents * self.positive_field))

        # Calculates a negative particle's index
        negative_index = int(r.generate_uniform_random_number(
            n_agents * (1 - self.negative_field), n_agents))

        # Calculates a neutral particle's index
        neutral_index = int(r.generate_uniform_random_number(
            n_agents * self.positive_field, n_agents * (1 - self.negative_field)))

        return positive_index, negative_index, neutral_index

    def _update(self, agents, function, phi, RI):
        """Method that wraps global and local pollination updates over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            phi (float): Golden ratio constant.
            RI (int): Index of particle's eletromagnet.

        """

        # Sorts agents according to their fitness
        agents.sort(key=lambda x: x.fit)

        # Gathers the number of total agents
        n_agents = len(agents)

        # Making a deepcopy of current's best agent
        agent = copy.deepcopy(agents[0])

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Generates a uniform random number for the force
        force = r.generate_uniform_random_number()

        # For every decision variable
        for j in range(agent.n_variables):
            # Calculates the index of positive, negative and neutral particles
            pos, neg, neu = self._calculate_indexes(n_agents)

            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # If random number is smaller than the probability of selecting eletromagnets
            if r2 < self.ps_ratio:
                # Applies agent's position as positive particle's position
                agent.position[j] = agents[pos].position[j]

            # If random number is bigger
            else:
                # Calculates the new agent's position
                agent.position[j] = agents[neg].position[j] + phi * force * (
                    agents[pos].position[j] - agents[neu].position[j]) - force * (agents[neg].position[j] - agents[neu].position[j])

        # Clips the agent's position to its limits
        agent.clip_limits()

        # Generates a third uniform random number
        r3 = r.generate_uniform_random_number()

        # If random number is smaller than probability of changing a random eletromagnet
        if r3 < self.r_ratio:
            # Update agent's position based on RI
            agent.position[RI] = r.generate_uniform_random_number(
                agent.lb[RI], agent.ub[RI])

            # Increases RI by one
            RI += 1

            # If RI exceeds the number of variables
            if RI >= agent.n_variables:
                # Resets it to one
                RI = 1

        # Calculates the agent's fitness
        agent.fit = function(agent.position)

        # If newly generated agent fitness is better than worst fitness
        if agent.fit < agents[-1].fit:
            # Updates the corresponding agent's object
            agents[-1] = copy.deepcopy(agent)

        return RI

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

        # Defines the golden ratio
        phi = (1 + np.sqrt(5)) / 2

        # Defines the eletromagnetic index
        RI = 0

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
                RI = self._update(space.agents, function, phi, RI)

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
