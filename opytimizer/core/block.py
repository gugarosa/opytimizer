"""Block.
"""

import opytimizer.utils.exception as e


class Block:
    """A Block serves as the foundation class for all graph-based optimization techniques."""

    def __init__(self, type, pointer, n_input, n_output):
        """Initialization method.

        Args:
            type (str): Type of the block.
            pointer (callable): Any type of callable to be applied when block is called.
            n_input (int): Number of input arguments.
            n_output (int): Number of output arguments.

        """

        # Type of the block
        self.type = type

        # Callable applied when block is called
        self.pointer = pointer

        # Number of input arguments
        self.n_input = n_input

        # Number of output arguments
        self.n_output = n_output

    def __call__(self, *args):
        """Callable to avoid using the `pointer` property.

        Returns:
            Input arguments applied to callable `pointer`.

        """

        return self.pointer(*args)

    @property
    def type(self):
        """str: Type of the block."""

        return self._type

    @type.setter
    def type(self, type):
        if type not in ["input", "inner", "output"]:
            raise e.ValueError("`type` should be `input`, `inner` or `output`")

        self._type = type

    @property
    def pointer(self):
        """callable: Points to the actual function when block is called."""

        return self._pointer

    @pointer.setter
    def pointer(self, pointer):
        if not callable(pointer):
            raise e.TypeError("`pointer` should be a callable")

        self._pointer = pointer

    @property
    def n_input(self):
        """int: Number of input arguments."""

        return self._n_input

    @n_input.setter
    def n_input(self, n_input):
        if not isinstance(n_input, int):
            raise e.TypeError("`n_input` should be an integer")
        if n_input <= 0:
            raise e.ValueError("`n_input` should be > 0")

        self._n_input = n_input

    @property
    def n_output(self):
        """int: Number of output arguments."""

        return self._n_output

    @n_output.setter
    def n_output(self, n_output):
        if not isinstance(n_output, int):
            raise e.TypeError("`n_output` should be an integer")
        if n_output <= 0:
            raise e.ValueError("`n_output` should be > 0")

        self._n_output = n_output


class InputBlock(Block):
    """An InputBlock defines a block that is only used for entry points."""

    def __init__(self, n_input, n_output):
        """Initialization method.

        Args:
            n_input (int): Number of input arguments.
            n_output (int): Number of output arguments.

        """

        super().__init__("input", lambda *args: args, n_input, n_output)


class InnerBlock(Block):
    """An InnerBlock defines a block that is used for inner points (between input and output)."""

    def __init__(self, pointer, n_input, n_output):
        """Initialization method.

        Args:
            pointer (callable): Any type of callable to be applied when block is called.
            n_input (int): Number of input arguments.
            n_output (int): Number of output arguments.

        """

        super().__init__("inner", pointer, n_input, n_output)


class OutputBlock(Block):
    """An OutputBlock defines a block that is only used for output points."""

    def __init__(self, n_input, n_output):
        """Initialization method.

        Args:
            n_input (int): Number of input arguments.
            n_output (int): Number of output arguments.

        """

        super().__init__("output", lambda *args: args, n_input, n_output)
