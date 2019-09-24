# A constant value used to avoid division by zero, zero logarithms
# and any possible mathematical error
EPSILON = 10e-10

# A test passes if the best solution found by the agent in the target function
# (which is always the sphere function, i.e. sum(x^2) and has global minimum at 0)
# is smaller than this value. We use a rather large margin to avoid tests failing
# in CI caused by bad initialization points by the agents
TEST_EPSILON = 2
