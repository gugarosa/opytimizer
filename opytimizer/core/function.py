""" This is the function's structure and its basic functions module.
"""

import py_expression_eval as math_parser


class Function(object):
    """ A function class for all meta-heuristic optimization techniques.

        # Arguments
            expression: Mathematical expression to be evaluated. Please define variables as {
                x0, x1, x2, ..., xn}.
            }

        # Properties
            expression: Mathematical expression to be evaluated.
            variables: Dictionary contaning (variable, value) of parsed expression.

        # Methods
            instanciate():
            evaluate(agent):
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'expression',
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Iterate through all properties and set the remaining ones.
        self.expression = None
        self.variables = None

        # Check if arguments are supplied
        if 'expression' not in kwargs:
            raise TypeError('You must input an expression to this object.')
        if 'expression' in kwargs:
            expression = kwargs['expression']
            self.expression = expression

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

    def evaluate(self, agent):
        """ Evaluates a function object.
            It applies the norm function over a agent's variable components and stores to the function's dictionary,
            then it evaluates these dictionary based on the expression's mathematical function.

            # Arguments
            agent: Agent object contaning variables to be evaluated.
        """
        # Check if the amount of function variables is equal to agent's number of variables
        if len(self.variables) != agent.n_variables:
            raise TypeError(
                'The number of expression variables must match to the number of agent variables.')
        # If that assumption is true,
        # iterate through all variables and stores the norm function of corresponding variable
        if len(self.variables) == agent.n_variables:
            for i, (key, value) in enumerate(self.variables.items()):
                if agent.n_dimensions == 1:
                    self.variables[key] = agent.norm(i)
                if agent.n_dimensions > 1:
                    self.variables[key] = agent.span(i)
        # Creates a parser object
        parser = math_parser.Parser()
        # Evaluate the agent's variables and store in its fit's property
        agent.fit = parser.parse(self.expression).evaluate(self.variables)
