""" This is the function's structure and its basic functions module.
"""

import py_expression_eval as math_parser
import opytimizer.utils.math as math


class Function(object):
    """ A function class for all meta-heuristic optimization techniques.

        # Arguments
            expression: Mathematical expression to be evaluated. Please define variables as {
                x0, x1, x2, ..., xn}.
            }
            lower_bound: Lower bound of expression's variables.
            upper_bound: Upper bound of expression's variables.

        # Properties
            expression: Mathematical expression to be evaluated.
            variables: Dictionary contaning (variable, value) of parsed expression.
            lower_bound: Lower bound of expression's variables.
            upper_bound: Upper bound of expression's variables.

        # Methods
            instanciate(): Instanciate a function object.
            evaluate(agent): Evaluates an agent object.
            check_limits(position, lower_bound, upper_bound): Check if vector 'position'
            is between lower and upper bounds.
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'expression',
                          'lower_bound',
                          'upper_bound'
                         }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Iterate through all properties and set the remaining ones.
        self.expression = None
        self.lower_bound = None
        self.upper_bound = None
        self.variables = None

        # Check if arguments are supplied
        if 'expression' not in kwargs:
            raise TypeError('You must input an expression to this object.')
        if 'lower_bound' not in kwargs:
            raise TypeError(
                'You must input the expression variables lower bound.')
        if 'upper_bound' not in kwargs:
            raise TypeError(
                'You must input the expression variables upper bound.')
        if 'expression' in kwargs:
            expression = kwargs['expression']
            self.expression = expression
        if 'lower_bound' in kwargs:
            lower_bound = kwargs['lower_bound']
            self.lower_bound = lower_bound
        if 'upper_bound' in kwargs:
            upper_bound = kwargs['upper_bound']
            self.upper_bound = upper_bound

    def instanciate(self):
        """ Instanciate a function object.
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

    def evaluate(self, data_type=None, position=None):
        """ Evaluates an input position vector or tensor
            It stores the input variables to the function's dictionary, then it evaluates these dictionary
            based on the expression's mathematical function.

            # Arguments
            type: 'vector' or 'tensor'
            position: position vector or tensor to be evaluated
        """
        # Check if the amount of function variables is equal to agent's number of variables
        if len(self.variables) != position.shape[0]:
            raise Exception(
                'The number of expression variables must match to the number of input variables.')
        # Check the input data type and call the corresponding function
        if data_type == 'vector':
            for i, (key, value) in enumerate(self.variables.items()):
                self.variables[key] = position[i]
        elif data_type == 'tensor':
            for i, (key, value) in enumerate(self.variables.items()):
                self.variables[key] = math.span(vector=position[i], lower_bound=self.lower_bound[i], upper_bound=self.upper_bound[i])
        # Creates a parser object
        parser = math_parser.Parser()
        # Evaluate the agent's variables and store in its fit's property
        fitness = float(parser.parse(self.expression).evaluate(self.variables))

        return fitness
