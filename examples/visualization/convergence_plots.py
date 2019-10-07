import numpy as np

import opytimizer.visualization.convergence as c
from opytimizer.utils.history import History

# Creating the history object
history = History()

# Loading saved optimization task
history.load('pso.pkl')

# Gathering desired keys from the object
# In this case, we will the first agent's position and fitness
agent_pos = history.get(key='agents', index=(0, 0))
agent_fit = history.get(key='agents', index=(0, 1))

# We will also gather the best agent's position and fitness
best_agent_pos = history.get(key='best_agent', index=(0,))
best_agent_fit = history.get(key='best_agent', index=(1,))

# Plotting convergence graphs
# Plotting the convergence of agent's positions
c.plot(agent_pos[0], agent_pos[1], labels=['agent[0]', 'agent[1]'],
       title='Convergence Plot', subtitle="Agent's positioning throughout $1000$ iterations")

# Plotting the convergence of best agent's positions
c.plot(best_agent_pos[0], best_agent_pos[1], labels=['best_agent[0]', 'best_agent[1]'],
       title='Convergence Plot', subtitle="Best Agent's positioning throughout $1000$ iterations")

# Plotting the convergence of agent's and best agent's fitness
c.plot(agent_fit, best_agent_fit, labels=['agent', 'best_agent'], title='Convergence Plot',
       subtitle="Agent's and best agent's fitness throughout $1000$ iterations")
