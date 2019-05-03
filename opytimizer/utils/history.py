import pickle


class History:
    """A History class is responsible for saving each iteration's output.

    One can configure, if necessary, different properties or outputs that
    can be saved. Currently, we only save agents' and best agent's position and fitness.

    """

    def __init__(self):
        """Initialization method.

        """

        # Our agents property will be first set as an empty list
        self._agents = []

        # Also, an empty list for the best agent property
        self._best_agent = []

    @property
    def agents(self):
        """list: An agents property to hold agents' position and fitness.

        """

        return self._agents

    @agents.setter
    def agents(self, agents):
        self._agents = agents

    @property
    def best_agent(self):
        """list: A best agent property to hold best agent's position and fitness.

        """

        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent):
        self._best_agent = best_agent

    def dump(self, agents, best_agent):
        """Dumps agents and best agent into the object

        Args:
            agents (list): List of agents.
            best_agent (Agent): An instance of the best agent.

        """

        # Declaring an auxiliary empty list
        a = []

        # For each agent
        for agent in agents:
            # We append its position as a list and its fitness
            a.append((agent.position.tolist(), agent.fit))

        # Finally, we can append the current iteration agents to our property
        self.agents.append(a)

        # Appending the best agent as well
        self.best_agent.append((best_agent.position.tolist(), best_agent.fit))

    def show(self):
        """Prints in a formatted way the history of agents' and best agent's 
        position and fitness.
        
        """

        # For every iteration
        for i, (agents, best) in enumerate(zip(self.agents, self.best_agent)):
            print(f'\nIteration: {i+1}\n')

            # Iterating through every agent
            for j, agent in enumerate(agents):
                print(f'Agent[{j}]: {agent[0]} | Fitness: {agent[1]}')

            print(f'Best agent: {best[0]} | Fitness: {best[1]}')

    def save(self, file_name):
        """Saves the object to a pickle encoding.

        Args:
            file_name (str): String holding the file's name that will be saved.

        """

        # Opening the file in write mode
        f = open(file_name, 'wb')

        # Dumps to a pickle file
        pickle.dump(self, f)

        # Close the file
        f.close()

    def load(self, file_name):
        """Loads the object from a pickle encoding.

        Args:
            file_name (str): String containing pickle's file path.

        """

        # Opens the desired file in read mode
        f = open(file_name, "rb")

        # Loads using pickle
        h = pickle.load(f)

        # Resetting current object state to loaded state
        self.__dict__.update(h.__dict__)
