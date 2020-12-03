"""Atom Search Optimization.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class ASO(Optimizer):
    """An ASO class, inherited from Optimizer.

    This is the designed class to define ASO-related
    variables and methods.

    References:
        W. Zhao, L. Wang and Z. Zhang.
        A novel atom search optimization for dispersion coefficient estimation in groundwater.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, algorithm='ASO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ASO.')

        # Override its parent class with the receiving hyperparams
        super(ASO, self).__init__(algorithm)

        # Depth weight
        self.alpha = 50.0

        # Multiplier weight
        self.beta = 0.2

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: Depth weight.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')

        self._alpha = alpha

    @property
    def beta(self):
        """float: Multiplier weight.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0 or beta > 1:
            raise e.ValueError('`beta` should be between 0 and 1')

        self._beta = beta

    def _calculate_mass(self, agents):
        """Calculates the atoms' masses (Eq. 17 and 18).

        Args:
            agents (list): List of agents.

        Returns:
            A list holding the atoms' masses.

        """

        # Sorts the agents
        agents.sort(key=lambda x: x.fit)

        # Defines worst and best fitness
        worst = agents[-1].fit
        best = agents[0].fit

        # Calculates the total fitness
        total_fit = np.sum([np.exp(-(agent.fit - best) / (worst - best + c.EPSILON)) for agent in agents])

        # Calculates the masses
        mass = [np.exp(-(agent.fit - best) / (worst - best + c.EPSILON)) / total_fit for agent in agents]

        return mass

    def _calculate_potential(self, agent, K_agent, average, iteration, n_iterations):
        """Calculates the potential of an agent based on its neighbour and average positioning.

        Args:
            agent (Agent): Agent to have its potential calculated.
            K_agent (Agent): Neighbour agent.
            average (np.array): Array of average positions.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the distance between agent's position and average position
        distance = np.linalg.norm(agent.position - average)

        # Calculates the radius between agent's and its neighbour
        radius = np.linalg.norm(agent.position - K_agent.position)

        # Defines the `rsmin` and `rsmax` coefficients
        rsmin = 1.1 + 0.1 * np.sin((iteration + 1) / n_iterations * np.pi / 2)
        rsmax = 1.24

        # If ratio between radius and distance is smaller than `rsmin`
        if radius / (distance + c.EPSILON) < rsmin:
            # Defines `rs` as `rsmin`
            rs = rsmin

        # If ratio between radius and distance is bigger than `rsmin`
        else:
            # If ratio is bigger than `rsmax`
            if radius / (distance + c.EPSILON) > rsmax:
                # Defines `rs` as `rsmax`
                rs = rsmax

            # If ratio is smaller than `rsmax`
            else:
                # Defines `rs` as the ratio
                rs = radius / (distance + c.EPSILON)

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates the potential
        coef = (1 - iteration / n_iterations) ** 3
        potential = coef * (12 * (-rs) ** (-13) - 6 * (-rs) ** (-7)) * \
            r1 * ((K_agent.position - agent.position) / (radius + c.EPSILON))

        return potential

    def _calculate_acceleration(self, agents, best_agent, mass, iteration, n_iterations):
        """Calculates the atoms' acceleration.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            mass (np.array): Array of masses.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.

        Returns:
            An array holding the atoms' acceleration.

        """

        # Instantiates an array of accelerations
        acceleration = np.zeros((len(agents), best_agent.n_variables, best_agent.n_dimensions))

        # Calculates the gravitational force
        G = np.exp(-20.0 * iteration / n_iterations)

        # Calculates the number of best agents
        K = int(len(agents) - (len(agents) - 2) * np.sqrt(iteration / n_iterations))

        # Sorts the agents according to their masses
        K_agents, _ = map(list, zip(*sorted(zip(agents, mass), key=lambda x: x[1], reverse=True)[:K]))

        # Calculates the average position
        average = np.mean([agent.position for agent in K_agents])

        # Iterates through every agent
        for i, agent in enumerate(agents):
            # Creates an array for holding the total potential
            total_potential = np.zeros((agent.n_variables, agent.n_dimensions))

            # Iterates through every neighbour agent
            for K_agent in K_agents:
                # Sums up the current potential to the total one
                total_potential += self._calculate_potential(agent, K_agent, average, iteration, n_iterations)

            # Finally, calculates the acceleration (Eq. 16)
            acceleration[i] = G * self.alpha * total_potential + \
                self.beta * (best_agent.position - agent.position) / mass[i]

        return acceleration

    def _update_velocity(self, velocity, acceleration):
        """Updates an atom's velocity (Eq. 21).

        Args:
            velocity (np.array): Agent's velocity.
            acceleration (np.array): Agent's acceleration.

        Returns:
            An updated velocity.

        """

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates the new velocity
        new_velocity = r1 * velocity + acceleration

        return new_velocity

    def _update_position(self, position, velocity):
        """Updates an atom's position (Eq. 22).

        Args:
            position (np.array): Agent's position.
            velocity (np.array): Agent's velocity.

        Returns:
            An updated position.

        """

        # Calculates the new position
        new_position = position + velocity

        return new_position

    def _update(self, agents, best_agent, velocity, iteration, n_iterations):
        """Method that wraps the Atom Search Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            velocity (np.array): Array of velocities.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the masses (Eq. 17 and 18)
        mass = self._calculate_mass(agents)

        # Calculates the acceleration (Eq. 16)
        acceleration = self._calculate_acceleration(agents, best_agent, mass, iteration, n_iterations)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Updates current agent's velocity (Eq. 21)
            velocity[i] = self._update_velocity(velocity[i], acceleration[i])

            # Updates current agent's position (Eq. 22)
            agent.position = self._update_position(agent.position, velocity[i])

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

        # Instantiates an array of velocities
        velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

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
                self._update(space.agents, space.best_agent, velocity, t, space.n_iterations)

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
