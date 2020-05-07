import sys

# A constant value used to avoid division by zero, zero logarithms
# and any possible mathematical error
EPSILON = 1e-32

# When the agents are initialized, their fitness is defined as
# the maximum float value possible
FLOAT_MAX = sys.float_info.max

# If necessary, one can apply custom rxules to keys' dumping
# when using the History object
HISTORY_KEYS = ['agents', 'best_agent', 'local']

# When working with relativity theories, it is necessary
# to defined a constant for the speed of light
LIGHT_SPEED = 3e5

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
    'ABS': 1,
    'SIN': 1,
    'COS': 1
}

# A test passes if the best solution found by the agent in the target function
# is smaller than this value
TEST_EPSILON = 100

# When using the Tournament Selection, one must provide the size of rounds,
# where individuals will compete among themselves
TOURNAMENT_SIZE = 2
