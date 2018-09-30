from opytimizer.core.space import Space

# Creates a new Space
s = Space(n_agents=2)

# Prior to use the Space, you need to build so,
# its agents are disponible for further use
s.build(n_variables=5, n_dimensions=1)

# You can print out the whole Space or acess
# individual agents and its properties
print(s.agents[0].position)
