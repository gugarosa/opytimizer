"""History-based object that helps in saving the optimization history.
"""

import numpy as np

import opytimizer.utils.exception as e


class History:
    """A History class is responsible for saving each iteration's output.

    Note that you can use dump() and parse() for whatever your needs. Our default
    is only for agents, best agent and best agent's index.

    """

    def __init__(self, save_agents=False):
        """Initialization method.

        Args:
            save_agents (bool): Saves all agents in the search space.

        """

        # Stores only the best agent
        self.save_agents = save_agents

    @property
    def save_agents(self):
        """bool: Saves all agents in the search space.

        """

        return self._save_agents

    @save_agents.setter
    def save_agents(self, save_agents):
        if not isinstance(save_agents, bool):
            raise e.TypeError('`save_agents` should be a boolean')

        self._save_agents = save_agents

    def _parse(self, key, value):
        """Parses incoming values with specified formats.

        Args:
            key (str): Key.
            value (any): Value.

        Returns:
            Parsed value according to the specified format.

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
        """Dumps keyword pairs into self-class attributes.

        """

        # Iterates through all keyword arguments
        for (key, value) in kwargs.items():
            # If current `key` is `agents` and they should not be saved,
            # we skip this loop iteration
            if key == 'agents' and not self.save_agents:
                continue

            # If current `key` has a specific parsing rule,
            # we need to parse it accordingly
            if key in ['agents', 'best_agent', 'local_position']:
                output = self._parse(key, value)
            else:
                output = value

            # If class still does not have a `key` property,
            # we need to set its initial value as a list
            if not hasattr(self, key):
                setattr(self, key, [output])
            else:
                getattr(self, key).append(output)

    def get_convergence(self, key, index=0):
        """Gets the convergence list of a specified key.

        Args:
            key (str): Key to be retrieved.
            index (tuple): Index to be retrieved.

        Returns:
            Values based on key and index.

        """

        # Gathers the numpy array from the attribute
        attr = np.asarray(getattr(self, key), dtype=list)

        # Checks if the key is `agents`
        if key in ['agents']:
            # Gathers positions and fitnesses
            attr_pos = np.hstack(attr[(slice(None), index, 0)])
            attr_fit = np.hstack(attr[(slice(None), index, 1)])

            return attr_pos, attr_fit

        # Checks if the key is `best_agent`
        if key in ['best_agent']:
            # Gathers positions and fitnesses
            attr_pos = np.hstack(attr[(slice(None), 0)])
            attr_fit = np.hstack(attr[(slice(None), 1)])

            return attr_pos, attr_fit

        # Checks if the key is `local_position`
        if key in ['local_position']:
            # Gathers positions
            attr_pos = np.hstack(attr[(slice(None), index)])

            return attr_pos

        # Gathers the attribute
        attr = np.hstack(attr[(slice(None))])

        return attr
