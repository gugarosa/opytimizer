"""History-based object that helps in saving the optimization history.
"""

import numpy as np

import opytimizer.utils.constant as c
import opytimizer.utils.exception as e


class History:
    """A History class is responsible for saving each iteration's output.

    Note that you can use dump() and parse() for whatever your needs. Our default
    is only for agents, best agent and best agent's index.

    """

    def __init__(self, store_best_only=False):
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

    def _parse(self, key, value):
        """Parses a value according to the key.

        Args:
            key (str): Key's identifier.
            value (any): Value.

        Returns:
            Parsed value according to the key.

        """

        # Checks if the key is `agents`
        if key == 'agents':
            # Returns a list of tuples (position, fit)
            return [(v.position.tolist(), v.fit) for v in value]

        # Checks if the key is `best_agent`
        if key == 'best_agent':
            # Returns a tuple (position, fit)
            return (value.position.tolist(), value.fit)

        # Checks if the key is `local_position`
        if key == 'local_position':
            # Returns a list of local positions
            return [v.tolist() for v in value]

    def dump(self, **kwargs):
        """Dumps key-value pairs into class attributes.

        """

        # Iterates through all key-word arguments
        for (key, value) in kwargs.items():
            # Checks if only the best agent is supposed to be saved
            if self.store_best_only:
                # Checks if key is different from `best_agent` or `time`
                if key not in ['best_agent', 'time']:
                    # Breaks the current loop
                    continue

            # Checks if current key has a specific rule
            if key in c.HISTORY_KEYS:
                # Parses information using specific rules, if defined
                out = self._parse(key, value)
            else:
                # Just applies the information
                out = value

            # If there is no attribute
            if not hasattr(self, key):
                # Sets its initial value as a list
                setattr(self, key, [out])

            # If there is already an attribute
            else:
                # Appends the new value to the attribute
                getattr(self, key).append(out)

    def get(self, key, index):
        """Gets the desired key based on the input index.

        Args:
            key (str): Key's name to be retrieved.
            index (tuple): A tuple indicating which indexes should be retrieved.

        Returns:
            All key's values based on the input index.
            Note that this method returns all records, i.e., all values from the `t` iterations.

        """

        # Checks if index is a tuple
        if not isinstance(index, tuple):
            raise e.TypeError('`index` should be a tuple')

        # Gathers the numpy array from the attribute
        attr = np.asarray(getattr(self, key), dtype=list)

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
