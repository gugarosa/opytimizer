import numpy as np

import opytimizer.visualization.convergence as c
from opytimizer.utils.history import History

# Creating the history object
history = History()

# Loading saved optimization task
history.load('')

# Gathering desired keys from the object
# In this case, we will the first agent's position and fitness
agent_pos = history.get(key='agents', index=(0, 0))
agent_fit = history.get(key='agents', index=(0, 1))

# We will also gather the best agent's position and fitness
best_agent_pos = history.get(key='best_agent', index=(0,))
best_agent_fit = history.get(key='best_agent', index=(1,))

# Plotting convergence graphs
# Plotting the convergence of agent's positions
c.plot(agent_pos[0], agent_pos[1], labels=['$x_0$', '$x_1$'],
       title='Sphere Function: $x^2 \mid x \in [-10, 10]$', subtitle='Agent: 0 | Algorithm: Particle Swarm Optimization')

# Plotting the convergence of best agent's positions
c.plot(best_agent_pos[0], best_agent_pos[1], labels=['$x^*_0$', '$x^*_1$'],
       title='Sphere Function: $x^2 \mid x \in [-10, 10]$', subtitle="Agent: Best | Algorithm: Particle Swarm Optimization")

# Plotting the convergence of agent's and best agent's fitness
c.plot(agent_fit, best_agent_fit, labels=[
       '$f(x)$', '$f(x^{*})$'], title='Sphere Function: $x^2 \mid x \in [-10, 10]$', subtitle="Agents: 0 and Best | Algorithm: Particle Swarm Optimization")
