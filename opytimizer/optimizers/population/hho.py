"""Harris Hawks Optimization.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class HHO(Optimizer):
    """An HHO class, inherited from Optimizer.

    This is the designed class to define HHO-related
    variables and methods.

    References:
        A. Heidari et al. Harris hawks optimization: Algorithm and applications.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, algorithm='HHO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> HHO.')

        # Override its parent class with the receiving hyperparams
        super(HHO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _calculate_initial_coefficients(self, iteration, n_iterations):
        """Calculates the initial coefficients, i.e., energy and jump's strength.

        Args:
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        Returns:
            Absolute value of energy and jump's strength.

        """

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates initial jump energy
        E_0 = 2 * r1 - 1

        # Calculates the jump strength
        J = 2 * (1 - r1)

        # Calculates the energy (Eq. 3)
        E = 2 * E_0 * (1 - (iteration / n_iterations))

        return np.fabs(E), J

    def _exploration_phase(self, agents, current_agent, best_agent):
        """Performs the exploration phase.

        Args:
            agents (list): List of agents.
            current_agent (Agent): Current agent to be updated (or not).
            best_agent (Agent): Best population's agent.

        Returns:
            A location vector containing the updated position.

        """

        # Generates a uniform random number
        q = r.generate_uniform_random_number()

        # Checks if random number is bigger or equal to 0.5
        if q >= 0.5:
            # Samples a random integer
            j = r.generate_integer_random_number(0, len(agents))

            # Generates two uniform random numbers
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates the location vector (Eq. 1 - part 1)
            location_vector = agents[j].position - r1 * \
                np.fabs(agents[j].position - 2 * r2 * current_agent.position)

        # If random number is smaller than 0.5
        else:
            # Averages the population's position
            average = np.mean([agent.position for agent in agents], axis=0)

            # Generates uniform random numbers
            r3 = r.generate_uniform_random_number()
            r4 = r.generate_uniform_random_number()

            # Expand the dimensions on lower and upper bounds
            lb = np.expand_dims(current_agent.lb, -1)
            ub = np.expand_dims(current_agent.ub, -1)

            # Updates the location vector (Eq. 1 - part 2)
            location_vector = (best_agent.position - average) - r3 * (lb + r4 * (ub - lb))

        return location_vector

    def _exploitation_phase(self, energy, jump, agents, current_agent, best_agent, function):
        """Performs the exploitation phase.

        Args:
            energy (float): Energy coefficient.
            jump (float): Jump's strength.
            agents (list): List of agents.
            current_agent (Agent): Current agent to be updated (or not).
            best_agent (Agent): Best population's agent.
            function (Function): A function object.

        Returns:
            A location vector containing the updated position.

        """

        # Generates a uniform random number
        w = r.generate_uniform_random_number()

        # Soft besiege
        if w >= 0.5 and energy >= 0.5:
            # Calculates the delta's position
            delta = best_agent.position - current_agent.position

            # Calculates the location vector (Eq. 4)
            location_vector = delta - energy * \
                np.fabs(jump * best_agent.position - current_agent.position)

            return location_vector

        # Hard besiege
        if w >= 0.5 and energy < 0.5:
            # Calculates the delta's position
            delta = best_agent.position - current_agent.position

            # Calculates the location vector (Eq. 6)
            location_vector = best_agent.position - energy * np.fabs(delta)

            return location_vector

        # Soft besiege with rapid dives
        if w < 0.5 and energy >= 0.5:
            # Calculates the `Y` position (Eq. 7)
            Y = best_agent.position - energy * \
                np.fabs(jump * best_agent.position - current_agent.position)

            # Generates the Lévy's flight and random array (Eq. 9)
            LF = d.generate_levy_distribution(1.5, (current_agent.n_variables, current_agent.n_dimensions))
            S = r.generate_uniform_random_number(size=(current_agent.n_variables, current_agent.n_dimensions))

            # Calculates the `Z` position (Eq. 8)
            Z = Y + S * LF

            # Evaluates new positions
            Y_fit = function(Y)
            Z_fit = function(Z)

            # If `Y` position is better than current agent's one (Eq. 10 - part 1)
            if Y_fit < current_agent.fit:
                return Y

            # If `Z` position is better than current agent's one (Eq. 10 - part 2)
            if Z_fit < current_agent.fit:
                return Z

        # Hard besiege with rapid dives
        else:
            # Averages the population's position
            average = np.mean([x.position for x in agents], axis=0)

            # Calculates the `Y` position (Eq. 12)
            Y = best_agent.position - energy * \
                np.fabs(jump * best_agent.position - average)

            # Generates the Lévy's flight and random array (Eq. 9)
            LF = d.generate_levy_distribution(1.5, (current_agent.n_variables, current_agent.n_dimensions))
            S = r.generate_uniform_random_number(size=(current_agent.n_variables, current_agent.n_dimensions))

            # Calculates the `Z` position (Eq. 13)
            Z = Y + S * LF

            # Evaluates new positions
            Y_fit = function(Y)
            Z_fit = function(Z)

            # If `Y` position is better than current agent's one (Eq. 11 - part 1)
            if Y_fit < current_agent.fit:
                return Y

            # If `Z` position is better than current agent's one (Eq. 11 - part 2)
            if Z_fit < current_agent.fit:
                return Z

        return current_agent.position

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps the Harris Hawks Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Iterates through all agents
        for agent in agents:
            # Calculates the prey's energy and jump's stength
            E, J = self._calculate_initial_coefficients(iteration, n_iterations)

            # Checks if energy is bigger or equal to one
            if E >= 1:
                # Performs the exploration phase
                agent.position = self._exploration_phase(agents, agent, best_agent)

            # If energy is smaller than one
            else:
                # Performs the exploitation phase
                agent.position = self._exploitation_phase(E, J, agents, agent, best_agent, function)

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
                self._update(space.agents, space.best_agent,
                             function, t, space.n_iterations)

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
