""" This is the function's structure and its basic functions module.
"""

import py_expression_eval as math_parser

import opytimizer.utils.math as math
from opytimizer.utils.exception import ArgumentException


class Function(object):
    """ A function class for all meta-heuristic optimization techniques.

        # Arguments
            expression: Mathematical expression to be evaluated. Please define variables as {
                x0, x1, x2, ..., xn}.
            }
            lower_bound: Lower bound of expression's variables.
            upper_bound: Upper bound of expression's variables.

        # Properties
            _called: Boolean value to check whether object was called or not.
            expression: Mathematical expression to be evaluated.
            variables: Dictionary contaning (variable, value) of parsed expression.
            lower_bound: Lower bound of expression's variables.
            upper_bound: Upper bound of expression's variables.

        # Methods
            call(): Calls a function object.
            evaluate(data_type, position): Evaluates an input position vector or tensor.
    """

    def __init__(self, **kwargs):

        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'expression',
                          'lower_bound',
                          'upper_bound'
                         }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception('Keyword argument not understood:', kwarg)

        # Define called as 'False'
        self._called = False

        # Define all class variables as 'None'
        self.expression = None
        self.lower_bound = None
        self.upper_bound = None
        self.variables = None

        # Check if arguments are supplied
        if 'expression' not in kwargs:
            raise ArgumentException('expression')
        if 'lower_bound' not in kwargs:
            raise ArgumentException('lower_bound')
        if 'upper_bound' not in kwargs:
            raise ArgumentException('upper_bound')

        # Apply arguments to class variables
        self.expression = kwargs['expression']
        self.lower_bound = kwargs['lower_bound']
        self.upper_bound = kwargs['upper_bound']

    def call(self):
        """ Calls a function object.
            In other words, it parses the input expression
            and saves to 'variables' a dictionary contaning (variable, value).
        """

        # Creates a parser object
        parser = math_parser.Parser()

        # Gathers all current variables
        expression_variables = parser.parse(self.expression).variables()

        # Creates a dictionary and iterate through it, setting everyone to 0
        variables = {}
        for expression_variable in expression_variables:
            variables[expression_variable] = 0

        # Stores the dictionary into function's object
        self.variables = variables

        # Set internal called variable to 'True'
        self._called = True

    def evaluate(self, data_type=None, position=None):
        """ Evaluates an input position vector or tensor.
            It stores the input variables to the function's dictionary, then it evaluates these dictionary
            based on the expression's mathematical function.

            # Arguments
            data_type: 'vector' or 'tensor'.
            position: vector or tensor to be evaluated.

            # Returns
            fitness: The function value according to its input variables
        """

        # Check if object was called
        if not self._called:
            raise Exception(
                " Method 'call()' should be called prior to this method.")

        # Check if the amount of function variables is equal to input number of variables
        if len(self.variables) != position.shape[0]:
            raise Exception(
                'The number of expression variables must match to the number of input variables.')

        # Check the input data type and call the corresponding function
        if data_type == 'vector':
            position = math.check_bounds(
                vector=position, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
            for i, (key, value) in enumerate(self.variables.items()):
                self.variables[key] = position[i]
        elif data_type == 'tensor':
            for i, (key, value) in enumerate(self.variables.items()):
                position[i] = math.check_unitary(position[i])
                self.variables[key] = math.span(
                    vector=position[i], lower_bound=self.lower_bound[i], upper_bound=self.upper_bound[i])

        # Creates a parser object
        parser = math_parser.Parser()

        # Evaluate the input variables and returns to a fitness variable
        fitness = float(parser.parse(self.expression).evaluate(self.variables))

        return fitness
