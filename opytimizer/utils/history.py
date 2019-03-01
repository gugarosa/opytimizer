import pandas
import pickle

class History:
    """A History class is responsible for saving each iteration's output.

    One can configure, if necessary, different properties or outputs that
    can be saved. Currently, we only save each agent's position and its fitness.

    Attributes:
        history (list): A list to hold agent's position and fitness.

    Methods:
        dump(agents): Dumps a list of agents into the object.
        show(): Prints the object in a formatted way.
        save(file_name): Saves the object to a pickle encoding.
        load(file_name): Loads the object from a pickle encoding.

    """

    def __init__(self):
        """Initialization method.

        """

        # Our history property will be first set as an empty list
        self._history = []

    @property
    def history(self):
        """A history property to hold agent's position and fitness.

        """

        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    
    def dump(self, agents):
        """Dumps a list of agents into the object.

        Args:
            agents (list): List of agents.

        """

        # Declaring an auxiliary empty list
        a = []

        # For each agent
        for agent in agents:
            # We append its position as a list and its fitness
            a.append((agent.position.tolist(), agent.fit))
            
        # Finally, we can append the current iteration list to our property
        self.history.append(a)

    def show(self):
        """Prints in a formatted way the history of agent's position
        and fitness.
        
        """

        # For every iteration
        for i, iteration in enumerate(self.history):
            print(f'\nIteration: {i+1}\n')
            # And for every agent
            for j, agent in enumerate(iteration):
                print(f'Agent[{j}]: {agent[0]} | Fitness: {agent[1]}')

    def save(self, file_name):
        """Saves the object to a pickle encoding.

        Args:
            file_name (str): String holding the file's name that will be saved.

        """

        # Opening the file in write mode
        f = open(file_name, 'wb')

        # Dumps to a pickle file
        pickle.dump(self.history, f)

        # Close the file
        f.close()


    def load(self, file_name):
        """Loads the object from a pickle encoding.

        Args:
            file_name (str): String containing pickle's file path.

        Returns:
            The content of a History object loaded from a pickle file.

        """

        # Opens the desired file in read mode
        f = open(file_name, "rb")

        # Loads using pickle
        history = pickle.load(f)

        return history