import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.optimizers.pso import PSO

logger = l.get_logger(__name__)


class RPSO(PSO):
    """An RPSO class, inherited from Optimizer.

    This is the designed class to define RPSO-related
    variables and methods.

    References:


    """

    def __init__(self, algorithm='RPSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> RPSO.')

        # Override its parent class with the receiving hyperparams
        super(RPSO, self).__init__(
            algorithm=algorithm, hyperparams=hyperparams)

        logger.info('Class overrided.')

    def _update_velocity(self, agent_position, best_position, local_position, max_velocity, velocity, mass):
        """Updates a single particle velocity (over a single variable).

        Args:
            agent_position (float): Agent's current position.
            best_position (float): Global best position.
            local_position (float): Agent's local best position.
            max_velocity (float): Maximum velocity of all agents.
            velocity (float): Agent's current velocity.
            mass (float): Agent's mass.

        Returns:
            A new velocity based on relativistic speed proposal.

        """

        # Generating first random number
        r1 = r.generate_uniform_random_number()

        # Generating second random number
        r2 = r.generate_uniform_random_number()

        # Calculating gamma parameter
        gamma = 1 / np.sqrt(1 - (max_velocity ** 2 / 300000 ** 2))

        # Calculates new velocity
        new_velocity = mass * velocity * gamma + self.c1 * r1 * \
            (local_position - agent_position) + self.c2 * \
            r2 * (best_position - agent_position)

        return new_velocity

    def _update(self, agents, best_agent, local_position, velocity, mass):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.
            mass (np.array): Array of agents' masses.

        """

        # Calculating the maximum velocity
        max_velocity = np.max(velocity)

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Updates current agent velocities
            velocity[i] = self._update_velocity(
                agent.position, best_agent.position, local_position[i], max_velocity, velocity[i], mass[i])

            # Updates current agent positions
            agent.position = self._update_position(agent.position, velocity[i])

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

        # Instanciating array of local positions
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # An array of velocities
        velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And finally, an array of masses
        mass = r.generate_uniform_random_number(
            size=(space.n_agents, space.n_variables, space.n_dimensions))

        # Check if there is a pre-evaluation hook
        if pre_evaluation_hook:
            # Applies the hook
            pre_evaluation_hook(self, space, function)

        # Initial search space evaluation
        self._evaluate(space, function, local_position)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent,
                         local_position, velocity, mass)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function, local_position)

            # Every iteration, we need to dump agents, local positions and best agent
            history.dump(agents=space.agents, local=local_position,
                         best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history
