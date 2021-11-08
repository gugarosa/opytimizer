import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 3
n_variables = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=True)

# Runs the optimization task
opt.start(n_iterations=10)

# Saves the optimization task
opt.save('opt_task.pkl')

# One can load the optimization task from disk or work directly with the attribute that is saved
# History keys are saved as lists, where the last dimension stands for their iteration
# opt = Opytimizer.load('opt_task.pkl')

# Prints the last iteration best agent and checks that it matches the best agent in space
# Also prints a random iteration best agent
best_agent_pos, best_agent_fit = opt.history.get_convergence('best_agent')
print(f'Best agent (position, fit): ({best_agent_pos[:, -1]}, {best_agent_fit[-1]})')
print(f'Best agent (position, fit): ({opt.space.best_agent.position}, {opt.space.best_agent.fit})')
print(f'Iter 4 - Best agent (position, fit): ({best_agent_pos[:, 3]}, {best_agent_fit[3]})')

# As `save_agents` was passed as True to Opytimizer(),
# we can also inspect the convergence of the agents itself
agent_0_pos, agent_0_fit = opt.history.get_convergence('agents', index=0)
print(f'Agent[0] (position, fit): ({agent_0_pos[:, -1]}, {agent_0_fit[-1]})')
print(f'Iter 4 - Agent[0] (position, fit): ({agent_0_pos[:, 3]}, {agent_0_fit[3]})')