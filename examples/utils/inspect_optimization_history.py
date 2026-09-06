# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace


def sphere(x: np.ndarray) -> float:
    """Evaluate the sphere objective.

    Args:
        x: Candidate position array.

    Returns:
        Sum of squared decision variables.

    """

    return np.sum(x**2)


# Random seed for experimental consistency
np.random.seed(0)

n_agents = 3
n_variables = 2

lower_bound = [-10, -10]
upper_bound = [10, 10]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, sphere, save_agents=True)

opt.start(n_iterations=10)

opt.save("opt_task.pkl")

# The same history is available after loading a trusted checkpoint with Opytimizer.load
# get_convergence concatenates positions, giving one column per iteration in SearchSpace
best_agent_pos, best_agent_fit = opt.history.get_convergence("best_agent")
print(f"Best agent (position, fit): ({best_agent_pos[:, -1]}, {best_agent_fit[-1]})")
print(f"Best agent (position, fit): ({opt.space.best_agent.position}, {opt.space.best_agent.fit})")
print(f"Iter 4 - Best agent (position, fit): ({best_agent_pos[:, 3]}, {best_agent_fit[3]})")

# Population histories are available because save_agents is True
agent_0_pos, agent_0_fit = opt.history.get_convergence("agents", index=0)
print(f"Agent[0] (position, fit): ({agent_0_pos[:, -1]}, {agent_0_fit[-1]})")
print(f"Iter 4 - Agent[0] (position, fit): ({agent_0_pos[:, 3]}, {agent_0_fit[3]})")
