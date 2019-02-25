import numpy as np

from opytimizer import Opytimizer
from opytimizer.core.space import Space
from opytimizer.functions.internal import Internal
from opytimizer.optimizers.fpa import FPA


def test(x):
    sum = 0

    for array in x:
        value = np.linalg.norm(array)
        sum += value ** 2

    return sum


# Creating Internal's function
f = Internal(pointer=test)

# Number of agents
n_agents = 2

# Number of decision variables
n_variables = 2

# Number of dimensions
n_dimensions = 1

# Number of running iterations
n_iterations = 10

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creating the Space class
s = Space(n_agents=n_agents, n_iterations=n_iterations,
          n_variables=n_variables, n_dimensions=n_dimensions,
          lower_bound=lower_bound, upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'beta': 1.5,
    'eta': 0.2,
    'p': 0.8
}

# Creating FPA's optimizer
p = FPA(hyperparams=hyperparams)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
o.start()
