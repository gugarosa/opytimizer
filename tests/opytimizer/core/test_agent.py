import opytimizer.core.agent as Agent
import numpy as np

# Instanciate a new agent
a = Agent.Agent(n_variables=2, n_dimensions=1)

# Test check_limits method
LB = np.zeros(a.n_variables)
UB = np.zeros(a.n_variables)

for i in range(a.n_variables):
    LB[i] = 1
    UB[i] = 3
    
a.check_limits(LB, UB)
print(a.position)