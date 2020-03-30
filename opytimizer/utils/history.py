import pickle

import numpy as np

import opytimizer.utils.constants as c
import opytimizer.utils.exception as e


class History:
    """A History class is responsible for saving each iteration's output.

    Note that you can use dump() and parse() for whatever your needs. Our default
    is only for agents, best agent and best agent's index.

    """

    def __init__(self, store_best_only=False):
        """Initialization method.

        Args:
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.

        """

        # Whether only the best agent should be stored or not
        self.store_best_only = store_best_only

    @property
    def store_best_only(self):
        """bool: Whether only the best agent should be stored in the class or not.

        """

        return self._store_best_only

    @store_best_only.setter
    def store_best_only(self, store_best_only):
        if not isinstance(store_best_only, bool):
            raise e.TypeError('`store_best_only` should be a boolean')

        self._store_best_only = store_best_only

    def __str__(self):
        """Prints in a formatted way the history of best agents throughout the
        optimization task.

        """

        # For every iteration
        for i, best in enumerate(self.best_agent):
            print(f'\nIteration {i+1}/{len(self.best_agent)}')
            print(f'\nPosition: {best[0]} | Fitness: {best[1]}')

        return ''

    def _parse(self, key, value):
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

        # Checks if the key is `best_agent`
        elif key == 'best_agent':
            # Returns the best agent's tuple (position, fit)
            return (value.position.tolist(), value.fit)

        # Checks if the key is `local`
        elif key == 'local':
            # Returns a list of local positions
            return [v.tolist() for v in value]

    def dump(self, **kwargs):
        """Dumps key-value pairs into lists attributes.

        Note that if an attribute already exists, it will be appended
        in the list.

        """

        # For every key-value pair
        for (k, v) in kwargs.items():
            # Checks if current key has a specific rule
            if k in c.HISTORY_KEYS:
                # Checks if it is supposed to only store the best agent
                if k != 'best_agent' and self.store_best_only:
                    continue

                # Parses information using specific rules, if defined
                out = self._parse(k, v)
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

    def get(self, key, index):
        """Gets the desired key based on the input index.

        Args:
            key (str): Key's name to be retrieved.
            index (tuple): A tuple indicating which indexes should be retrieved.

        Returns:
            All key's values based on the input index. Note that this method returns all records, i.e.,
            all values from the `t` iterations.

        """

        # Checks if index is a tuple
        if not isinstance(index, tuple):
            raise e.TypeError('`index` should be a tuple')

        # Gathers the numpy array from the attribute
        attr = np.asarray(getattr(self, key))

        # Checks if attribute's dimensions are equal to the length of input index
        # We use `- 1` as the method retrieves values from all iterations
        if attr.ndim - 1 != len(index):
            raise e.SizeError(
                f'`index` = {len(index)} should have one less dimension than `key` = {attr.ndim}')

        # Slices the array based on the input index
        # Again, slice(None) will retrieve values from all iterations
        attr = attr[(slice(None),) + index]

        # We use hstack to horizontally concatenate the axis,
        # allowing an easier input to the visualization package
        attr = np.hstack(attr)

        return attr

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
        with open(file_name, "rb") as input_file:
            # Loading History from file
            h = pickle.load(input_file)

            # Updating all values
            self.__dict__.update(h.__dict__)
