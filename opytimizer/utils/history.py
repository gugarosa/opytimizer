import pickle

import opytimizer.utils.constants as c


class History:
    """A History class is responsible for saving each iteration's output.

    Note that you can use dump() and parse() for whatever your needs. Our default
    is only for agents, best agent and best agent's index.

    """

    def __str__(self):
        """Prints in a formatted way the history of agents' throughout the
        optimization task.

        """

        # For every iteration
        for i, (agents, best) in enumerate(zip(self.agents, self.best)):
            print(f'\nIteration {i+1}/{len(self.agents)}')

            # Iterating through every agent
            for j, agent in enumerate(agents):
                # Prints an agent
                print(f'Agent[{j}]: {agent[0]} | Fitness: {agent[1]}')

            # Prints the best agent so far
            print(f'\nBest: {best[0]} | Fitness: {best[1]}')

        return ''

    def dump(self, **kwargs):
        """Dumps key-value pairs into lists attributes.

        Note that if an attribute already exists, it will be appended
        in the list.

        """

        # For every key-value pair
        for (k, v) in kwargs.items():
            # Checks if current key has a specific rule
            if k in c.HISTORY_KEYS:
                # Parses the information according to the key
                out = self.parse(k, v)

            # If there is no specific rule
            else:
                # Just applies the information
                out = v

            # If there is no attribute
            if not hasattr(self, k):
                # Sets its initial value as a list
                setattr(self, k, [out])

            # If there is already an attribute
            else:
                # Appends the new value to the attribute
                getattr(self, k).append(out)

    def parse(self, key, value):
        """Parses a value according to the key's requirement.

        Args:
            key (str): Key's identifier.
            value (any): Any possible value.

        Returns:
            The parsed (formatted) value according to the key.

        """

        # Checks if the key is `agents`
        if key == 'agents':
            # Returns a list of agents' tuples (position, fit)
            return [(v.position.tolist(), v.fit) for v in value]

        # Checks if the key is `best`
        elif key == 'best':
            # Returns the best agent's tuple (position, fit)
            return (value.position.tolist(), value.fit)

        # Checks if the key is `local`
        elif key == 'local':
            # Returns a list of local positions
            return [v.tolist() for v in value]

    def save(self, file_name):
        """Saves the object to a pickle encoding.

        Args:
            file_name (str): File's name to be saved.

        """

        # Opening a destination file
        with open(file_name, 'wb') as dest_file:
            # Dumping History to file
            pickle.dump(self, dest_file)

    def load(self, file_name):
        """Loads the object from a pickle encoding.

        Args:
            file_name (str): Pickle's file path to be loaded.

        """

        # Trying to open the file
        with open(file_name, "rb") as origin_file:
            # Loading History from file
            h = pickle.load(origin_file)

            # Updating all values
            self.__dict__.update(h.__dict__)
