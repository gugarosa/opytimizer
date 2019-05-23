import numpy as np
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.pso import PSO
from opytimizer.spaces.search import SearchSpace
import opf_wrapper as wp

def optimum_path_forest(opytimizer):
    # Instanciating an OPF class
    opf = wp.OPF()

    wp._cluster(opf, 'training.dat', 1, 0.2)
    wp._test(opf, 'testing.dat')
    acc = wp._acc(opf, 'testing.dat')

    return 1 - acc

# Creating Function's object
f = Function(pointer=optimum_path_forest)

# Number of agents
n_agents = 2

# Number of decision variables
n_variables = 1

# Number of running iterations
n_iterations = 2

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [0.00001]
upper_bound = [10]

# Creating the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(hyperparams=hyperparams)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
