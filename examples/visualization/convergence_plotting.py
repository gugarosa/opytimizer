import opytimizer.visualization.convergence as c

# Defines agent's position and fitness
agent_pos = [[0.5, 0.4, 0.3], [0.5, 0.4, 0.3]]
agent_fit = [0.5, 0.32, 0.18]

# Defines best agent's position and fitness
best_agent_pos = [[0.01, 0.005, 0.0001], [0.01, 0.005, 0.0001]]
best_agent_fit = [0.0002, 0.00005, 0.00002]

# Plotting the convergence of agent's positions
c.plot(agent_pos[0], agent_pos[1], labels=['$x_0$', '$x_1$'],
       title='Sphere Function: $x^2 \mid x \in [-10, 10]$',
       subtitle='Agent: 0 | Algorithm: Particle Swarm Optimization')

# Plotting the convergence of best agent's positions
c.plot(best_agent_pos[0], best_agent_pos[1], labels=['$x^*_0$', '$x^*_1$'],
       title='Sphere Function: $x^2 \mid x \in [-10, 10]$',
       subtitle="Agent: Best | Algorithm: Particle Swarm Optimization")

# Plotting the convergence of agent's and best agent's fitness
c.plot(agent_fit, best_agent_fit, labels=['$f(x)$', '$f(x^{*})$'],
       title='Sphere Function: $x^2 \mid x \in [-10, 10]$',
       subtitle="Agents: 0 and Best | Algorithm: Particle Swarm Optimization")
