"""Block.
"""

from inspect import signature
import opytimizer.utils.exception as e


class Block:
    """A Block serves as the foundation class for all graph-based optimization techniques.

    """

    def __init__(self, type, pointer):
        """Initialization method.

        Args:
            type (str): Type of the block.
            pointer (callable): Any type of callable to be applied when block is called.

        """

        # Type of the block
        self.type = type

        # Callable applied when block is called
        self.pointer = pointer
    
    def __call__(self, *args):
        """Callable to avoid using the `pointer` property.

        Returns:
            Input arguments applied to callable `pointer`.

        """

        return self.pointer(*args)

    @property
    def type(self):
        """str: Type of the block.

        """

        return self._type

    @type.setter
    def type(self, type):
        if type not in ['input', 'intermediate', 'output']:
            raise e.ValueError('`type` should be `input`, `intermediate` or `output`')

        self._type = type

    @property
    def pointer(self):
        """callable: Points to the actual function when block is called.

        """

        return self._pointer

    @pointer.setter
    def pointer(self, pointer):
        if not callable(pointer):
            raise e.TypeError('`pointer` should be a callable')
        if len(signature(pointer).parameters) < 1:
            raise e.ArgumentError('`pointer` should have at least 1 argument')

        self._pointer = pointer
