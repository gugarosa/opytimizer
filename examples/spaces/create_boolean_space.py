from opytimizer.spaces import BooleanSpace

# Defines the number of agents and decision variables
n_agents = 2
n_variables = 5

# Creates the BooleanSpace
s = BooleanSpace(n_agents, n_variables)

# Prints out some properties
print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)
