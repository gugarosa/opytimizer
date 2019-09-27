import sys

# A constant value used to avoid division by zero, zero logarithms
# and any possible mathematical error
EPSILON = 10e-10

# When the agents are initialized, their fitness is defined as
# the maximum float value possible
FLOAT_MAX = sys.float_info.max

# If necessary, one can apply custom rules to keys' dumping
# when using the History object
HISTORY_KEYS = ['agents', 'best', 'local']

# When using Genetic Programming, each function node needs an unique number of arguments,
# which is defined by this dictionary
N_ARGS_FUNCTION = {
    'SUM': 2,
    'SUB': 2,
    'MUL': 2,
    'DIV': 2,
    'EXP': 1,
    'SQRT': 1,
    'LOG': 1,
    'ABS': 1
}

# A test passes if the best solution found by the agent in the target function
# is smaller than this value. 
TEST_EPSILON = 5
