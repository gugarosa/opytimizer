import opytimizer.utils.logging as l
import py_expression_eval as math_parser

logger = l.get_logger(__name__)



def build_internal(expression='x + 2'):
    """
    """

    logger.info('Learning Internal ...')

    # Creates a parser object
    parser = math_parser.Parser()

    # Gathers all current variables
    expression_variables = parser.parse(expression).variables()

    # Creates a dictionary and iterate through it, setting everyone to 0
    variables = {}
    for expression_variable in expression_variables:
        variables[expression_variable] = 0

    # Stores the dictionary into function's object
    return variables