import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class AEO(Optimizer):
    """An AEO class, inherited from Optimizer.

    This is the designed class to define AEO-related
    variables and methods.

    References:
        W. Zhao, L. Wang, and Z. Zhang. 
        Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm.
        Neural Computing and Applications (2019).

    """

    def __init__(self, algorithm='AEO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(AEO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build()

        logger.info('Class overrided.')

    def _build(self):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        """

        logger.debug('Running private method: build().')

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Built: {self.built}.')

    def _production(self, agent, best_agent, iteration, n_iterations):
        """Performs the producer update.

        Args:
            agent (Agent): Current agent.
            best_agent (Agent): Best agent.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.

        Returns:
            An updated producer according to AEO's paper equation 1.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Calculates the alpha factor (equation 2)
        alpha = (1 - iteration / n_iterations) * r.generate_uniform_random_number()

        # For every possible decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Updates its position
            a.position[j] = (1 - alpha) * best_agent.position[j] + alpha * r.generate_uniform_random_number(lb, ub, a.n_dimensions)

        return a

    def _herbivore_consumption(self, agent, producer, C):
        """Performs the consumption update by a herbivore.

        Args:
            agent (Agent): Current agent.
            producer (Agent): Producer agent.
            C (float): Consumption factor.

        Returns:
            An updated consumption by a herbivore according to AEO's paper equation 6.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Updates its position
        a.position += C * (agent.position - producer.position)

        return a

    def _omnivore_consumption(self, agent, producer, consumer, C):
        """Performs the consumption update by an omnivore.

        Args:
            agent (Agent): Current agent.
            producer (Agent): Producer agent.
            consumer (Agent): Consumer agent.
            C (float): Consumption factor.

        Returns:
            An updated consumption by an omnivore according to AEO's paper equation 8.

        """
        
        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Generates the second random number
        r2 = r.generate_uniform_random_number()

        # Updates its position
        a.position += C * r2 * (a.position - producer.position) + (1 - r2) * (a.position - consumer.position)

        return a

    def _carnivore_consumption(self, agent, consumer, C):
        """Performs the consumption update by a carnivore.

        Args:
            agent (Agent): Current agent.
            consumer (Agent): Consumer agent.
            C (float): Consumption factor.

        Returns:
            An updated consumption by a carnivore according to AEO's paper equation 7.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Updates its position
        a.position += C * (a.position - consumer.position) 

        return a
        
    def _update_composition(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps production and consumption updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Number of current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Sorting agents according to their energy
        agents.sort(key=lambda x: x.fit, reverse=True)

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # If it is the first agent
            if i == 0:
                # It will surely be a producer
                a = self._production(agent, best_agent, iteration, n_iterations)
            
            # If it is not the first agent
            else:
                # Generates the first random number
                r1 = r.generate_uniform_random_number()

                # Generates a gaussian random number
                v1 = r.generate_gaussian_random_number()

                # Generates another gaussian random number
                v2 = r.generate_gaussian_random_number()
                
                # Calculates the consumption factor (equation 4)
                C = 0.5 * v1 / np.abs(v2)

                # If random number lies in the first third
                if r1 < 1/3:
                    # It will surely be a herbivore
                    a = self._herbivore_consumption(agent, agents[0], C)
                
                # If random number lies in the second third
                elif 1/3 <= r1 <= 2/3:
                    # Generates a random index from the population
                    j = int(r.generate_uniform_random_number(1, i))

                    # It will surely be a omnivore
                    a = self._omnivore_consumption(agent, agents[0], agents[j], C)
                
                # If random number lies in the last third
                else:
                    # Generates a random index from the population
                    j = int(r.generate_uniform_random_number(1, i))

                    # It will surely be a carnivore
                    a = self._carnivore_consumption(agent, agents[j], C)
            
            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)  

    def _update_decomposition(self, agents, best_agent, function):
        """Method that wraps decomposition updates over all agents and variables according to AEO's paper equation 9.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterate through all agents
        for agent in agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Calculates the decomposition factor (equation 10)
            D = 3 * r.generate_gaussian_random_number()

            # Generates the third random number
            r3 = r.generate_uniform_random_number()

            # First weight coefficient (equation 11)
            e = r3 * int(r.generate_uniform_random_number(1, 2)) - 1

            # Second weight coefficient (equation 12)
            h = 2 * r3 - 1

            # Updates the new agent position
            a.position = best_agent.position + D * (e * best_agent.position - h * agent.position)

            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)     

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

                # Updating agents within the composition step
                self._update_composition(space.agents, space.best_agent, function, t, space.n_iterations)

                # Updating agents within the decomposition step
                self._update_decomposition(space.agents, space.best_agent, function)

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
