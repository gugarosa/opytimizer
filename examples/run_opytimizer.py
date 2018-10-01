from opytimizer import Opytimizer
from opytimizer.core.space import Space
from opytimizer.functions.internal import Internal
from opytimizer.optimizers.pso import PSO


def test(x):
    sum = 0

    for value in x:
        sum += value
    
    return sum


# Input parameters
n_agents = 5
n_variables = 5
n_dimensions = 1
n_iterations = 5

# Bounds parameters
# Note that it has to have the same size as n_variables
lower_bound = [0.1, 0.3, 0.5, 0.5, 0.5]
upper_bound = [0.2, 0.4, 2.0, 2.0, 2.0]

# Building up Space
s = Space(n_agents=n_agents, n_variables=n_variables,
          n_dimensions=n_dimensions, n_iterations=n_iterations)
s.build(lower_bound=lower_bound, upper_bound=upper_bound)

# Building up PSO's optimizer
p = PSO()
p.build()

# Building up Internal function
f = Internal()
f.build(test)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

o.evaluate()

print(o.space.best_agent.position)
print(o.space.best_agent.fit)
print('\n')

for agent in o.space.agents:
    print(agent.position)
    print(agent.fit)
    print('\n')
